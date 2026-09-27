import type { ExtensionAPI, ExtensionCommandContext } from "@earendil-works/pi-coding-agent";
import type { JevClient } from "./jev.js";
import type { SkillRouter } from "./skills.js";
import type { AutoJev } from "./auto.js";
import type { AutoModelRouter } from "./model-router.js";
import type { JevCompactor } from "./compact.js";
import type { AgentOrchestrator } from "./orchestrator.js";
import type { ToolGuard } from "./tool-guard.js";
import { designEvaluation } from "./designer.js";
import type { JevEvaluationRequest } from "./types.js";
import { JEV_TOOL_NAMES, isJevTool } from "./types.js";
import { JEV_THRESHOLD } from "./skills.js";
import { credentialHint } from "pi-jev-core";
import { SWITCHES, configPath, envOverrides, envShadowed, type JevConfigKey } from "./config.js";

/** What the in-place pruning controls need from a command or tool context. */
export interface PruneContext {
  cwd: string;
  /** Only the branch is read, and its entry types are a union with no common shape. */
  sessionManager: { getBranch?(): unknown[] };
  signal?: AbortSignal;
}

/** Shown when the extension was loaded without the pruning controls. */
const PRUNE_UNAVAILABLE = "Jev in-place pruning is not available in this session.";

/** The pruning path, exposed to `/jev compact status|reset|now` and the `jev_compact_now` tool. */
export interface PruneControl {
  status(): string;
  /** One line for `/jev status`: the counts of the last summary compaction. */
  counts(): string;
  /** Reports whether the score cache was actually cleared; false means the file was unwritable. */
  reset(ctx: PruneContext): boolean;
  now(ctx: PruneContext): Promise<string>;
}

export function registerJevCommands(
  pi: ExtensionAPI,
  jevClient: JevClient,
  skillRouter: SkillRouter,
  auto: AutoJev,
  autoModel?: AutoModelRouter,
  compactor?: JevCompactor,
  agents?: AgentOrchestrator,
  persistSwitch?: (key: JevConfigKey, value: boolean) => boolean,
  prune?: PruneControl,
  toolGuard?: ToolGuard,
  searchGate?: { enabled: boolean; setEnabled(enabled: boolean): void },
  skillStrip?: { enabled: boolean; setEnabled(enabled: boolean): void }
): void {
  const agentMode = agents ?? { enabled: false, setEnabled: () => {}, dispatch: async () => ({ accepted: false, error: "disabled" }) };
  const compactMode = compactor ?? { enabled: false, setEnabled: () => {} };
  const modelMode = autoModel ?? { enabled: false, setEnabled: () => {} };
  const stripMode = skillStrip ?? { enabled: false, setEnabled: () => {} };

  /** Persists a toggle and returns the note to append to the notice. */
  const saveNotice = (key: JevConfigKey, value: boolean): string => {
    if (!persistSwitch) return "";
    // A write failure must not be reported as a save: the toggle then lives in this session only.
    if (persistSwitch(key, value) === false) return " Saved for this session only (the config file is not writable).";
    // An env var that disagrees with what we just saved wins on the next start: say so now.
    return envShadowed(key, value)
      ? ` Saved, but $${SWITCHES[key]} is set and overrides it.`
      : " Saved to the global config.";
  };

  const guardMode = toolGuard ?? { enabled: false, setEnabled: () => {} };
  const gateMode = searchGate ?? { enabled: false, setEnabled: () => {} };
  pi.registerCommand("jev", {
    description:
      "Manage TypeSafe Jev (status, enable, auto, auto-model, compact, tool-guard, search-gate, skills, agents)",
    handler: async (args: string, ctx: ExtensionCommandContext) => {
      const tokens = args.trim().split(/\s+/).filter(Boolean);
      const sub = (tokens[0] ?? "").toLowerCase();
      const rest = tokens.slice(1).join(" ");
      const usage =
        "Available options: /jev status, /jev skills [query], /jev test [prompt], /jev enable, /jev disable, /jev auto [on|off], /jev auto-model [on|off], /jev compact [on|off|status|reset|now], /jev auto-agents [on|off], /jev tool-guard [on|off], /jev search-gate [on|off], /jev skill-strip [on|off], /jev agents [task]";

      if (sub === "status" || sub === "") {
        const origin = jevClient.getKeyOrigin();
        const activeTools = pi.getActiveTools();
        const allTools = pi.getAllTools();
        const activeSet = new Set(activeTools);
        const routable = allTools.filter(
          (t: any) => !activeSet.has(t.name) && !isJevTool(t.name)
        ).length;
        const overrides = envOverrides();

        ctx.ui.notify(
          `Jev Status:\n` +
            `• Platform: ${jevClient.platform}\n` +
            `• Configured: ${origin ? `Yes (from ${origin})` : "No"}\n` +
            `• Requests in session: ${jevClient.stats.requestsCount}\n` +
            `• Total tokens used: ${jevClient.stats.totalTokens}\n` +
            `• Auto mode: ${auto.enabled ? "on" : "off"}${auto.enabled && !origin ? " (inactive: Jev unconfigured)" : ""}\n` +
            `• Auto-model: ${modelMode.enabled ? "on" : "off"}\n` +
            `• Tool guard: ${guardMode.enabled ? "on" : "off"}\n` +
            `• Search gate: ${gateMode.enabled ? "on" : "off"}\n` +
            `• Skill strip: ${stripMode.enabled ? "on" : "off"}\n` +
            `• Jev compaction: ${compactMode.enabled ? "on" : "off"}\n` +
            (prune ? `• ${prune.counts()}\n` : "") +
            `• Agent orchestration: ${agentMode.enabled ? "on" : "off"}\n` +
            `• Saved config: ${configPath()}\n` +
            (overrides.length ? `• Env override (wins over saved): ${overrides.join(", ")}\n` : "") +
            `• Active tools: ${activeTools.length} / Available: ${allTools.length} (${routable} routable)\n` +
            (jevClient.stats.lastError ? `• Last error: ${jevClient.stats.lastError}` : ""),
          "info"
        );
        return;
      }

      if (sub === "help") {
        ctx.ui.notify(`Jev commands:\n${usage}`, "info");
        return;
      }

      if (sub === "test" || sub === "eval" || sub === "evaluate") {
        if (!jevClient.isConfigured()) {
          ctx.ui.notify(
            `Cannot run evaluation: no Jev API key for the ${jevClient.platform} platform. ${credentialHint(jevClient.platform)}.`,
            "error"
          );
          return;
        }

        // With a prompt: the active model designs the evaluation. Without one: fixed smoke test.
        let request: JevEvaluationRequest;
        if (rest) {
          ctx.ui.notify(`Designing a Jev evaluation for: "${rest}"...`, "info");
          try {
            request = await designEvaluation(ctx, rest, ctx.signal);
          } catch (err: any) {
            ctx.ui.notify(`Could not design evaluation: ${err?.message || err}`, "error");
            return;
          }
          ctx.ui.notify(
            `Designed ${Object.keys(request.questions).length} question(s): ${Object.keys(request.questions).join(", ")}\nSending to Jev (${jevClient.platform})...`,
            "info"
          );
        } else {
          ctx.ui.notify(`Sending test evaluation request to Jev (${jevClient.platform})...`, "info");
          request = {
            state: { message: "Payment processing failed due to credit card expiration." },
            questions: {
              is_billing: {
                type: "noul" as const,
                instructions: "Using `message`, is this message related to a billing issue?",
              },
              category: {
                type: "choice" as const,
                instructions: "Using `message`, which category does this issue fall into?",
                criteria: {
                  billing: "Billing, invoices, card issues",
                  bug: "Software bug or crash",
                  other: "General questions",
                },
              },
            },
          };
        }

        try {
          const res = await jevClient.evaluate(request);

          ctx.ui.notify(
            (rest ? `Jev Evaluation (${res.elapsedMs}ms):\n` : `Jev Test Successful (${res.elapsedMs}ms):\n`) +
              Object.entries(res.answers)
                .map(([id, ans]) => {
                  const shown = ans.value ?? "no answer";
                  const value =
                    ans.type === "noul"
                      ? `${shown}${typeof ans.value === "number" ? ` (${(ans.value * 100).toFixed(0)}% yes)` : ""}`
                      : `${shown}${ans.confidence !== undefined ? ` (confidence: ${ans.confidence})` : ""}`;
                  return `• ${id}: ${value}`;
                })
                .join("\n"),
            "info"
          );
        } catch (err: any) {
          ctx.ui.notify(`Jev Evaluation Failed: ${err?.message || err}`, "error");
        }
        return;
      }

      if (sub === "skills" || sub === "skill") {
        const query = rest;
        if (!query) {
          const available = skillRouter.getAvailableSkills(ctx);
          ctx.ui.notify(
            `Available skills (${available.length}):\n` +
              available.map((s) => `• ${s.name}: ${s.description.slice(0, 80)}...`).join("\n"),
            "info"
          );
          return;
        }

        ctx.ui.notify(`Searching skills for: "${query}"...`, "info");
        const res = await skillRouter.findSkills(query, JEV_THRESHOLD, ctx);
        if (res.recommended.length === 0) {
          ctx.ui.notify(`No skills matched "${query}".`, "info");
          return;
        }

        ctx.ui.notify(
          `Matching skills for "${query}":\n` +
            res.recommended
              .map((r) => `• /skill:${r.name} (P=${r.probability.toFixed(2)}) - ${r.description}`)
              .join("\n") +
            (res.fallbackUsed
              ? "\n(Note: Jev unconfigured/offline — local keyword shortlist, probabilities are not Jev judgments)"
              : ""),
          res.fallbackUsed ? "warning" : "info"
        );
        return;
      }

      if (sub === "agents" || sub === "orchestrate") {
        if (!rest) {
          ctx.ui.notify("Usage: /jev agents <task>", "warning");
          return;
        }
        const result = await agentMode.dispatch(rest, ctx);
        if (!result.accepted) ctx.ui.notify(`Agent orchestration unavailable: ${result.error ?? "unknown error"}`, "warning");
        return;
      }

      if (sub === "tool-guard" || sub === "toolguard" || sub === "guard") {
        const arg = rest.toLowerCase();
        if (arg !== "" && arg !== "on" && arg !== "off") {
          ctx.ui.notify(`Unknown /jev tool-guard argument "${rest}". ${usage}`, "warning");
          return;
        }
        const enabled = arg === "on" ? true : arg === "off" ? false : !guardMode.enabled;
        guardMode.setEnabled(enabled);
        ctx.ui.notify(`Jev tool guard ${enabled ? "enabled" : "disabled"}.${saveNotice("toolGuard", enabled)}`, "info");
        return;
      }

      if (sub === "search-gate" || sub === "searchgate" || sub === "gate") {
        const arg = rest.toLowerCase();
        if (arg !== "" && arg !== "on" && arg !== "off") {
          ctx.ui.notify(`Unknown /jev search-gate argument "${rest}". ${usage}`, "warning");
          return;
        }
        const enabled = arg === "on" ? true : arg === "off" ? false : !gateMode.enabled;
        gateMode.setEnabled(enabled);
        ctx.ui.notify(`Jev search gate ${enabled ? "enabled" : "disabled"}.${saveNotice("searchGate", enabled)}`, "info");
        return;
      }

      if (sub === "skill-strip" || sub === "skillstrip" || sub === "strip") {
        const arg = rest.toLowerCase();
        if (arg !== "" && arg !== "on" && arg !== "off") {
          ctx.ui.notify(`Unknown /jev skill-strip argument "${rest}". ${usage}`, "warning");
          return;
        }
        const enabled = arg === "on" ? true : arg === "off" ? false : !stripMode.enabled;
        stripMode.setEnabled(enabled);
        ctx.ui.notify(`Jev skill strip ${enabled ? "enabled" : "disabled"}.${saveNotice("skillStrip", enabled)}`, "info");
        return;
      }

      if (sub === "compact") {
        const arg = rest.toLowerCase();

        if (arg === "status") {
          ctx.ui.notify(prune?.status() ?? PRUNE_UNAVAILABLE, "info");
          return;
        }

        if (arg === "reset") {
          if (!prune) {
            ctx.ui.notify(PRUNE_UNAVAILABLE, "warning");
            return;
          }
      const written = prune.reset(ctx);
      ctx.ui.notify(
        written
          ? "Jev compaction: frozen scores, applied decisions, breaker, and pressure state cleared."
          : "Jev compaction: state cleared for this session, but the score cache file could not be written.",
        written ? "info" : "warning"
      );
      return;
        }

        if (arg === "now") {
          if (!prune) {
            ctx.ui.notify(PRUNE_UNAVAILABLE, "warning");
            return;
          }
          // No optimistic progress line: `now` is the only thing that knows whether it will score.
          ctx.ui.notify(await prune.now(ctx), "info");
          return;
        }

        if (arg !== "" && arg !== "on" && arg !== "off") {
          ctx.ui.notify(`Unknown /jev compact argument "${rest}". ${usage}`, "warning");
          return;
        }
        const enabled = arg === "on" ? true : arg === "off" ? false : !compactMode.enabled;
        compactMode.setEnabled(enabled);
        ctx.ui.notify(`Jev compaction ${enabled ? "enabled" : "disabled"}.${saveNotice("compact", enabled)} Use /compact to run it.`, "info");
        return;
      }

      if (sub === "auto-agents" || sub === "autoagents") {
        const arg = rest.toLowerCase();
        if (arg !== "" && arg !== "on" && arg !== "off") {
          ctx.ui.notify(`Unknown /jev auto-agents argument "${rest}". ${usage}`, "warning");
          return;
        }
        const enabled = arg === "on" ? true : arg === "off" ? false : !agentMode.enabled;
        agentMode.setEnabled(enabled);
        ctx.ui.notify(`Automatic agent orchestration ${enabled ? "enabled" : "disabled"}.${saveNotice("agents", enabled)}`, "info");
        return;
      }

      if (sub === "auto-model" || sub === "automodel") {
        const arg = rest.toLowerCase();
        if (arg !== "" && arg !== "on" && arg !== "off") {
          ctx.ui.notify(`Unknown /jev auto-model argument "${rest}". ${usage}`, "warning");
          return;
        }
        const enabled = arg === "on" ? true : arg === "off" ? false : !modelMode.enabled;
        modelMode.setEnabled(enabled);
        ctx.ui.notify(`Jev auto-model mode ${enabled ? "enabled" : "disabled"}.${saveNotice("autoModel", enabled)}`, "info");
        return;
      }

      if (sub === "auto") {
        const arg = rest.toLowerCase();
        if (arg !== "" && arg !== "on" && arg !== "off") {
          ctx.ui.notify(`Unknown /jev auto argument "${rest}". ${usage}`, "warning");
          return;
        }
        const enabled = arg === "on" ? true : arg === "off" ? false : !auto.enabled;
        auto.setEnabled(enabled);
        ctx.ui.notify(
          (enabled
            ? "Jev auto mode enabled: each prompt routes tools and suggests skills. Costs one Jev request per prompt."
            : "Jev auto mode disabled.") + saveNotice("auto", enabled),
          "info"
        );
        return;
      }

      if (sub === "enable") {
        pi.setActiveTools([...new Set([...pi.getActiveTools(), ...JEV_TOOL_NAMES])]);
        ctx.ui.notify(`Jev tools (${JEV_TOOL_NAMES.join(", ")}) enabled for this session.`, "info");
        return;
      }

      if (sub === "disable") {
        pi.setActiveTools(pi.getActiveTools().filter((t) => !isJevTool(t)));
        ctx.ui.notify("Jev tools disabled for this session.", "info");
        return;
      }

      ctx.ui.notify(`Unknown command /jev ${sub}.\n${usage}`, "warning");
    },
  });
}
