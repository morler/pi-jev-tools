import fs from "node:fs";
import path from "node:path";
import type { ExtensionAPI, ExtensionCommandContext, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { noulProbability, type JevClient } from "./jev.js";
import type { QuestionConfig } from "./types.js";

/** Single activation cutoff for Jev probabilities. Raise to reduce noise, lower for recall. */
export const JEV_THRESHOLD = 0.65;

export interface SkillMetadata {
  name: string;
  description: string;
  location?: string;
}

export interface LoadedSkill {
  name: string;
  location?: string;
  content?: string;
  error?: string;
}

export interface SkillRouterResult {
  query: string;
  candidates: string[];
  recommended: Array<{
    name: string;
    description: string;
    location?: string;
    probability: number;
  }>;
  fallbackUsed: boolean;
  elapsedMs: number;
}

export function normalizeSkillName(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9]+/g, "");
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export class SkillRouter {
  private pi: ExtensionAPI;
  private jevClient: JevClient;

  constructor(pi: ExtensionAPI, jevClient: JevClient) {
    this.pi = pi;
    this.jevClient = jevClient;
  }

  public getAvailableSkills(ctx?: ExtensionContext | ExtensionCommandContext): SkillMetadata[] {
    const skillsMap = new Map<string, SkillMetadata>();

    if (ctx && "getSystemPromptOptions" in ctx) {
      try {
        const opts = (ctx as ExtensionCommandContext).getSystemPromptOptions();
        if (opts.skills && Array.isArray(opts.skills)) {
          for (const raw of opts.skills) {
            if (!raw || typeof raw !== "object") continue;
            // SAFETY: Pi exposes skill metadata as a structural SDK value at runtime.
            const skill = raw as unknown as Record<string, unknown>;
            const name = typeof skill.name === "string" ? skill.name : "";
            const description = typeof skill.description === "string" ? skill.description : "";
            if (name && description) {
              const location = typeof skill.location === "string"
                ? skill.location
                : typeof skill.path === "string" ? skill.path : undefined;
              skillsMap.set(name, { name, description, location });
            }
          }
        }
      } catch (error: unknown) {
        // A malformed prompt context should not hide command-discovered skills.
        console.warn(`[jev] failed to read skill prompt metadata: ${errorMessage(error)}`);
      }
    }

    for (const cmd of this.pi.getCommands()) {
      if (cmd.source === "skill" && !skillsMap.has(cmd.name)) {
        skillsMap.set(cmd.name, {
          name: cmd.name,
          description: cmd.description || `Skill for ${cmd.name}`,
          location: cmd.sourceInfo?.path,
        });
      }
    }

    return Array.from(skillsMap.values());
  }

  public loadSkills(names: string[], ctx?: ExtensionContext | ExtensionCommandContext): LoadedSkill[] {
    const byName = new Map(
      this.getAvailableSkills(ctx).map((skill) => [normalizeSkillName(skill.name), skill])
    );

    return names.slice(0, 5).map((requested) => {
      const skill = byName.get(normalizeSkillName(requested));
      if (!skill) return { name: requested, error: "Skill not found" };
      if (!skill.location) return { name: skill.name, error: "Skill location is unavailable" };

      const file = skill.location.endsWith("SKILL.md")
        ? skill.location
        : path.join(skill.location, "SKILL.md");
      try {
        return { name: skill.name, location: file, content: fs.readFileSync(file, "utf8") };
      } catch (error: unknown) {
        return { name: skill.name, location: file, error: errorMessage(error) };
      }
    });
  }

  public shortlist(skills: SkillMetadata[], query: string, limit = 10): SkillMetadata[] {
    const terms = query.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
    if (terms.length === 0) return skills.slice(0, limit);

    const scored = skills.map((skill) => {
      const text = `${skill.name} ${skill.description}`.toLowerCase();
      const score = terms.filter((term) => text.includes(term)).length;
      return { skill, score };
    });

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, limit).map(({ skill }) => skill);
  }

  public async findSkills(
    query: string,
    threshold = JEV_THRESHOLD,
    ctx?: ExtensionContext,
    signal?: AbortSignal
  ): Promise<SkillRouterResult> {
    const startTime = Date.now();
    const candidates = this.shortlist(this.getAvailableSkills(ctx), query, 12);
    const candidateNames = candidates.map((candidate) => candidate.name);
    if (candidates.length === 0) {
      return { query, candidates: [], recommended: [], fallbackUsed: false, elapsedMs: Date.now() - startTime };
    }

    const recommended: SkillRouterResult["recommended"] = [];
    let fallbackUsed = false;

    if (this.jevClient.isConfigured()) {
      try {
        const questions: Record<string, QuestionConfig> = {};
        for (const skill of candidates) {
          questions[skill.name] = {
            type: "noul",
            instructions: `Using \`task\` and \`available_skills\`, does the skill '${skill.name}' (${skill.description}) provide direct guidance or specialized domain steps for the task?`,
          };
        }

        const result = await this.jevClient.evaluate(
          { state: { task: query, available_skills: candidates }, questions },
          signal
        );
        for (const skill of candidates) {
          const answer = result.answers[skill.name];
          const probability = noulProbability(answer?.raw) ?? 0;
          if (probability >= threshold) {
            recommended.push({ ...skill, probability });
          }
        }
        recommended.sort((a, b) => b.probability - a.probability);
      } catch {
        fallbackUsed = true;
      }
    } else {
      fallbackUsed = true;
    }

    if (fallbackUsed) {
      const terms = query.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
      for (const skill of candidates) {
        const text = `${skill.name} ${skill.description}`.toLowerCase();
        if (terms.some((term) => text.includes(term))) recommended.push({ ...skill, probability: 0 });
      }
    }

    return {
      query,
      candidates: candidateNames,
      recommended,
      fallbackUsed,
      elapsedMs: Date.now() - startTime,
    };
  }
}
