import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { SkillRouter, type SkillMetadata } from "../src/skills.js";
import { JevClient } from "../src/jev.js";

function mockPi(commands: object[]): ExtensionAPI {
  // SAFETY: tests provide only the getCommands member consumed by SkillRouter.
  return { getCommands: () => commands } as unknown as ExtensionAPI;
}

test("SkillRouter shortlists skills based on query terms", () => {
  const mockSkills: SkillMetadata[] = [
    { name: "tdd", description: "Test-driven development and unit testing" },
    { name: "frontend-design", description: "Create distinctive production-grade UI interfaces" },
    { name: "resolving-merge-conflicts", description: "Resolve git rebase and merge conflicts" },
    { name: "accessibility", description: "Audit and improve WCAG accessibility" },
  ];

  const router = new SkillRouter(mockPi([]), new JevClient());
  const candidates = router.shortlist(mockSkills, "fix git rebase conflicts");
  assert.equal(candidates.length, 4);
  assert.equal(candidates[0].name, "resolving-merge-conflicts");
});

test("SkillRouter loads enabled SKILL.md files with normalized names", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "pi-jev-skill-"));
  const skillDir = path.join(root, "fleet");
  fs.mkdirSync(skillDir);
  fs.writeFileSync(path.join(skillDir, "SKILL.md"), "# Fleet workflow");

  const router = new SkillRouter(
    mockPi([{
      name: "fleet",
      description: "Fleet workflow",
      source: "skill",
      sourceInfo: { path: skillDir },
    }]),
    { isConfigured: () => false } as unknown as JevClient,
  );

  const loaded = router.loadSkills(["Fleet"]);
  assert.equal(loaded[0].content, "# Fleet workflow");
  assert.equal(loaded[0].error, undefined);
});

test("SkillRouter fallback does not claim a Jev probability", async () => {
  const mockSkills: SkillMetadata[] = [
    { name: "tdd", description: "Test-driven development" },
    { name: "accessibility", description: "Audit web accessibility" },
  ];

  const router = new SkillRouter(
    mockPi(mockSkills.map((skill) => ({
      name: skill.name,
      description: skill.description,
      source: "skill",
    }))),
    { isConfigured: () => false } as unknown as JevClient,
  );

  const result = await router.findSkills("make web accessible");
  assert.equal(result.fallbackUsed, true);
  assert.equal(result.recommended.length, 1);
  assert.equal(result.recommended[0].name, "accessibility");
  assert.equal(result.recommended[0].probability, 0);
});

test("SkillRouter preserves string probabilities from Jev", async () => {
  const router = new SkillRouter(
    mockPi([{
      name: "typesafe-ai",
      description: "Build with TypeSafe AI",
      source: "skill",
    }]),
    {
      isConfigured: () => true,
      evaluate: async () => ({
        answers: { "typesafe-ai": { type: "noul", value: 0.91 as unknown as string, raw: { probability: "0.91" } } },
      }),
    } as unknown as JevClient,
  );

  const result = await router.findSkills("build with TypeSafe AI");
  assert.equal(result.fallbackUsed, false);
  assert.equal(result.recommended[0].probability, 0.91);
});
