import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "@sinclair/typebox";
import { JevClient } from "../src/jev.js";
import { ToolRouter } from "../src/router.js";
import { SkillRouter } from "../src/skills.js";
import { AutoJev } from "../src/auto.js";
import { registerJevTools, registerSearchGateTool } from "../src/tools.js";
import { registerJevCommands, type PruneControl } from "../src/commands.js";
import { AutoModelRouter } from "../src/model-router.js";
import { JevCompactor, branchMessagesOf, pruneSchedule } from "../src/compact.js";
import { applyPrune, cancelThresholdCompaction, realContextBoundary, reconcilePressure, type ContextBoundary, type Decision, type PressureState } from "../src/prune.js";
import { AgentOrchestrator } from "../src/orchestrator.js";
import { JevAgentHandler } from "../src/agent.js";
import { loadConfig, loadModelPool, resolveSwitch, saveConfig, type JevConfigKey } from "../src/config.js";
import { ToolGuard } from "../src/tool-guard.js";
import { stripSkillCatalogFromEvent } from "../src/skill-strip.js";

export default function (pi: ExtensionAPI) {
  const saved = loadConfig();
  /** CLI flag > env var (when set at all) > saved global config > the switch's default (compact on, the rest off). */
  const flagDefault = (key: JevConfigKey): boolean => resolveSwitch(key, saved);

  const jevClient = new JevClient();
  const router = new ToolRouter(pi, jevClient);
  const skillRouter = new SkillRouter(pi, jevClient);

  pi.registerFlag("jev-agents", {
    description: "Enable explicit and automatic orchestration of available agents",
    type: "boolean",
    default: flagDefault("agents"),
  });

  pi.registerFlag("jev-tool-guard", {
    description: "Validate tool calls with Jev System One to prevent hallucinations",
    type: "boolean",
    default: flagDefault("toolGuard"),
  });

  pi.registerFlag("jev-search-gate", {
    description: "Offer jev_search_gate: Jev ranks web-search results, filters prompt injection, and judges sufficiency",
    type: "boolean",
    default: flagDefault("searchGate"),
  });

  pi.registerFlag("jev-compact", {
    description: "Replace /compact's summary with Jev compaction (on by default; PI_JEV_COMPACT=0 or /jev compact off restores Pi's built-in)",
    type: "boolean",
    default: flagDefault("compact"),
  });

  pi.registerFlag("jev-auto-model", {
    description: "Automatically choose a model for each prompt based on task needs",
    type: "boolean",
    default: flagDefault("autoModel"),
  });

  pi.registerFlag("jev-skill-strip", {
    description: "Remove the Agent Skills catalog from the system prompt and point discovery at jev_find_skill",
    type: "boolean",
    default: flagDefault("skillStrip"),
  });

  pi.registerFlag("jev-auto", {
    description:
      "Automatically route Pi tools and suggest skills with Jev on every prompt (also via PI_JEV_AUTO=1)",
    type: "boolean",
    default: flagDefault("auto"),
  });

  const auto = new AutoJev(
    jevClient,
    router,
    skillRouter,
    Boolean(pi.getFlag("jev-auto"))
  );
  const autoModel = new AutoModelRouter(pi, Boolean(pi.getFlag("jev-auto-model")), loadModelPool(), jevClient);
  const compactor = new JevCompactor(jevClient, Boolean(pi.getFlag("jev-compact")));
  const agents = new AgentOrchestrator(pi, jevClient, Boolean(pi.getFlag("jev-agents")));
  agents.installCompletionNotice();

  const toolGuard = new ToolGuard(pi, jevClient, Boolean(pi.getFlag("jev-tool-guard")));
  toolGuard.install();

  /** The gate fails closed on Jev outages; the switch only gates tool availability. */
  const searchGateControl = {
    enabled: Boolean(pi.getFlag("jev-search-gate")),
    setEnabled(value: boolean) {
      this.enabled = value;
    },
  };
  registerSearchGateTool(pi, jevClient, () => searchGateControl.enabled);

  /** Catalog removal is prompt hygiene; it runs whether or not auto mode is on. */
  const skillStrip = {
    enabled: Boolean(pi.getFlag("jev-skill-strip")),
    setEnabled(value: boolean) {
      this.enabled = value;
    },
  };

  const agentHandler = new JevAgentHandler(pi, jevClient);
  agentHandler.install();

  /** Save a runtime toggle so the next Pi session starts the same way. Reports whether it stuck. */
  const persistSwitch = (key: JevConfigKey, value: boolean): boolean => {
    try {
      saveConfig({ [key]: value });
      return true;
    } catch {
      // Unwritable config dir: keep the in-session toggle rather than failing the command.
      return false;
    }
  };

  // ---------- Jev-guided in-place pruning ----------
  // Judging runs in the background and only writes scores. The context hook is the single place
  // messages change, and it refreshes its decisions only at a checkpoint where a prefix-cache miss
  // is already paid or free. Between checkpoints the same decisions are re-applied, so the prompt
  // prefix stays byte-stable.
  //
  // These helpers reach pi's payloads through `any` on purpose: the host types are not usable here,
  // and they are called with both an ExtensionContext and the narrower PruneContext, so the common
  // shape is structural rather than a shared SDK type.
  let lastResponseAt = 0;
  let noCacheStreak = 0;
  let activeRun = false;
  let applied: Map<string, Decision> | null = null;
  let lastJudgeAt = 0;
  /** The counts of the last summary compaction, for `/jev status`. */
  let lastCompact: { kept: number; truncated: number; dropped: number } | null = null;
  let pressure: PressureState = "armed";

  const pruningOn = (schedule = pruneSchedule()): boolean =>
    compactor.enabled && schedule.enabled && jevClient.isConfigured();

  /** Consecutive turns with no cache read and no cache write: the provider never writes cache. */
  const NO_CACHE_STREAK = 3;

  const branchMessages = (ctx: any): any[] => branchMessagesOf(ctx?.sessionManager?.getBranch?.() ?? []);

  // Pi's own compaction settings sit behind SettingsManager, reachable only at runtime. When they
  // cannot be read the safe-input ceiling is unknown, and an unknown boundary cancels nothing —
  // treating it as reserve 0 would widen the ceiling to the whole window and silently cancel Pi's own
  // threshold compaction. The cold and no-cache checkpoints work regardless.
  let settingsModule: any;
  const loadSettingsModule = async (): Promise<any> => {
    if (settingsModule !== undefined) return settingsModule;
    try {
      const mod: any = await import("@earendil-works/pi-coding-agent");
      // Only a successful load is cached: a transient import failure must not poison every later call.
      if (mod?.SettingsManager && mod?.getAgentDir) settingsModule = mod;
    } catch {
      // Fall through: this call reports unknown, the next one tries again.
    }
    return settingsModule ?? null;
  };

  /** Where real usage sits against Pi's own safe-input ceiling; overCeiling is null when unknown. */
  const boundary = async (ctx: any): Promise<ContextBoundary> => {
    const mod = await loadSettingsModule();
    let reserveTokens: number | null = null;
    if (mod) {
      try {
        const manager = mod.SettingsManager.create(ctx?.cwd, mod.getAgentDir(), {
          projectTrusted: ctx?.isProjectTrusted?.() ?? false,
        });
        const settings = manager.getCompactionSettings(
          ctx?.model ? { provider: ctx.model.provider, id: ctx.model.id } : undefined
        );
        const reserve = settings?.reserveTokens;
        if (typeof reserve === "number" && Number.isFinite(reserve)) reserveTokens = Math.max(0, reserve);
      } catch {
        reserveTokens = null; // settings unreadable: unknown ceiling, so nothing is cancelled
      }
    }
    let usage: { tokens?: number | null; contextWindow?: number } | undefined;
    try {
      usage = ctx?.getContextUsage?.();
    } catch {
      usage = undefined; // unknown usage: the boundary stays unknown, so nothing acts on it
    }
    if (reserveTokens === null) return { overCeiling: null };
    return realContextBoundary({
      tokens: usage?.tokens,
      contextWindow: usage?.contextWindow ?? ctx?.model?.contextWindow,
      reserveTokens,
    });
  };

  const pruneControl: PruneControl = {
    status() {
      const schedule = pruneSchedule();
      return [
        `Jev in-place pruning: ${pruningOn() ? "on" : "off"}${schedule.enabled ? "" : " (JEV_COMPACT_PRUNE=0)"}`,
        `Pressure: ${pressure} · agent run: ${activeRun ? "active" : "idle"} · no-cache streak: ${noCacheStreak}/${NO_CACHE_STREAK}`,
        `Pending changes: ${applied?.size ?? 0} · last judged: ${lastJudgeAt ? new Date(lastJudgeAt).toISOString() : "never"}`,
        `Bands: keep >= ${schedule.keepThreshold} · truncate >= ${schedule.dropThreshold} · head ${schedule.headChars} chars`,
        `Cache TTL: ${schedule.cacheTtlMs}ms · judge gap: ${schedule.minGapMs}ms`,
      ].join("\n");
    },

    counts() {
      return lastCompact
        ? `Last compaction: ${lastCompact.kept} kept · ${lastCompact.truncated} trunc · ${lastCompact.dropped} drop`
        : "Last compaction: none this session";
    },

    reset(ctx) {
      const written = compactor.reset(ctx.cwd);
      applied = null;
      pressure = "armed";
      lastJudgeAt = 0;
      noCacheStreak = 0;
      lastResponseAt = 0;
      // The caller must not report a clear the cache file never took.
      return written;
    },

    async now(ctx) {
      if (!pruningOn()) return "Jev compaction is off — enable it with /jev compact on.";
      const messages = branchMessages(ctx);
      if (messages.length === 0) return "Jev compaction: this session has no messages to score.";

      try {
        await compactor.judge(messages, ctx.cwd, ctx.signal);
      } catch (error) {
        return `Jev compaction failed: ${error instanceof Error ? error.message : String(error)}`;
      }
      // A manual run forces a checkpoint: the user accepts the one-time cache miss. The rewrite itself
      // happens in the context hook, so report the decisions rather than dry-running it here.
      applied = compactor.planPrune(messages, ctx.cwd, activeRun);
      lastJudgeAt = Date.now();

      const decisions = [...applied.values()];
      const drops = decisions.filter((decision) => decision === "drop").length;
      const truncations = decisions.filter((decision) => decision === "truncate").length;
      return `Jev compaction: ${drops} to drop · ${truncations} to truncate — applies to the next request.`;
    },
  };

  registerJevTools(pi, router, skillRouter);
  registerJevCommands(pi, jevClient, skillRouter, auto, autoModel, compactor, agents, persistSwitch, pruneControl, toolGuard, searchGateControl, skillStrip);

  pi.registerTool({
    name: "jev_compact_now",
    label: "Jev Compact Now",
    description:
      "Score the current history with Jev and apply the pruning decisions immediately, accepting one prompt-cache miss. Use when the context feels heavy or before a long task.",
    parameters: Type.Object({}),
    async execute(_toolCallId, _params, _signal, _onUpdate, ctx) {
      const text = await pruneControl.now(ctx);
      return { content: [{ type: "text", text }], details: { outcome: text } };
    },
  });

  // ---------- handlers ----------

  pi.on("session_start", (_event, ctx) => {
    if (!jevClient.isConfigured()) {
      ctx.ui.setStatus("jev", "jev: unconfigured");
      return;
    }
    ctx.ui.setStatus(
      "jev",
      autoModel.enabled ? "jev: auto-model" : auto.enabled ? "jev: auto" : "jev: ready"
    );
  });

  pi.on("session_before_compact", async (event, ctx) => {
    // Pi's threshold compaction is its own early reaction to pressure; pruning covers that ground
    // while it can. Manual and overflow compactions always pass through.
    if (event.reason === "threshold") {
      const pruning = pruningOn();
      const overCeiling = pruning ? (await boundary(ctx)).overCeiling : null;
      if (cancelThresholdCompaction({ pruning, pressure, overCeiling })) return { cancel: true };
    }

    const result = await compactor.compact(event, ctx);
    if (!result.summary) {
      // Fail-open: Pi's summarizer takes over. Only an unexpected failure is worth a shout; an off
      // switch or a missing key is the user's own configuration, and a history with no text to judge
      // is not a failure either.
      if (result.skipped === "error") ctx.ui.setStatus("jev", "jev: compact failed → Pi's summarizer ran");
      else if (result.skipped === "empty") ctx.ui.setStatus("jev", "jev: nothing to compact → Pi's summarizer ran");
      return;
    }
    lastCompact = { kept: result.kept, truncated: result.truncated, dropped: result.dropped };
    ctx.ui.setStatus("jev", `jev: compact ${result.kept} kept · ${result.truncated} trunc · ${result.dropped} drop`);
    return {
      compaction: {
        summary: result.summary,
        firstKeptEntryId: event.preparation.firstKeptEntryId,
        tokensBefore: event.preparation.tokensBefore,
      },
    };
  });

  pi.on("agent_start", () => {
    activeRun = true;
  });

  pi.on("agent_settled", async (_event, ctx) => {
    activeRun = false;
    const schedule = pruneSchedule();
    if (!pruningOn(schedule)) return;
    if (Date.now() - lastJudgeAt < schedule.minGapMs) return;
    try {
      await compactor.judge(branchMessages(ctx), ctx.cwd, ctx.signal);
      lastJudgeAt = Date.now();
    } catch {
      // Best effort: a Jev failure must not reject the settled event.
    }
  });

  pi.on("turn_end", async (event, ctx) => {
    const usage = (event.message as any)?.usage;
    if (usage) {
      // A zero cacheRead is not on its own a signal: right after a cache write it reads 0.
      noCacheStreak = (usage.cacheRead ?? 0) === 0 && (usage.cacheWrite ?? 0) === 0 ? noCacheStreak + 1 : 0;
    }
    // Nothing consumes the pressure state while pruning is off, and reading the boundary costs a
    // settings lookup.
    if (pruningOn()) pressure = reconcilePressure(pressure, (await boundary(ctx)).overCeiling);
  });

  pi.on("after_provider_response", (event, ctx) => {
    // Only a served request re-caches the prefix: counting a failure would postpone the next cold
    // checkpoint by a whole TTL, stalling pruning after provider errors.
    if (event.status < 400) lastResponseAt = Date.now();
    const kind = autoModel.recordProviderResponse(event.status, ctx.model);
    if (kind) ctx.ui.setStatus("jev", `jev: ${kind} → fallback next prompt`);
  });

  pi.on("context", async (event, ctx) => {
    const schedule = pruneSchedule();
    if (!pruningOn(schedule)) return;

    const messages: any[] = event.messages ?? [];
    // A refresh changes the prompt prefix, which costs a full-prefix cache miss. Wait for a
    // checkpoint where that miss is already paid (stale cache) or free: a provider that writes no
    // cache, or usage past Pi's own ceiling, where the next request misses anyway.
    const noCache = noCacheStreak >= NO_CACHE_STREAK;
    const cold = lastResponseAt === 0 || Date.now() - lastResponseAt > schedule.cacheTtlMs;
    const bypass = !cold && !noCache && pressure === "armed" && (await boundary(ctx)).overCeiling === true;

    if (cold || noCache || bypass) applied = compactor.planPrune(messages, ctx.cwd, activeRun);
    if (bypass && applied && applied.size > 0) pressure = "awaiting_validation";
    if (!applied || applied.size === 0) return;

    const { messages: next, stats } = applyPrune(messages, applied, schedule.headChars);
    if (stats.dropped === 0 && stats.truncated === 0) return; // nothing to change: leave the array alone
    ctx.ui.setStatus("jev", `jev: prune -${Math.max(0, stats.charsBefore - stats.charsAfter)} ch`);
    return { messages: next };
  });

  pi.on("before_agent_start", async (event, ctx) => {
    // Catalog removal runs regardless of auto mode; without the note the model would
    // not know skills exist, so the note is what keeps jev_find_skill discoverable.
    let systemPrompt = event.systemPrompt;
    if (skillStrip.enabled) {
      // Handles both generations: string strip on pi <= 0.85, in-place
      // systemPromptOptions.skills=[] (+ note section) on pi >= 0.87.
      const stripped = stripSkillCatalogFromEvent(event);
      if (stripped.systemPrompt !== undefined) systemPrompt = stripped.systemPrompt;
    }
    const promptChanged = systemPrompt !== event.systemPrompt;

    // Auto-model routes independently of auto mode: each opt-in switch gates only itself.
    let modelResult;
    try {
      modelResult = await autoModel.route(event.prompt, ctx, { hasImages: Boolean(event.images?.length) });
    } catch {
      modelResult = { changed: false };
    }
    if (modelResult.changed) {
      ctx.ui.setStatus("jev", `jev: ${modelResult.tier} → ${modelResult.model?.id ?? "model"}`);
    }

    if (!auto.enabled) return promptChanged ? { systemPrompt } : undefined;

    if (agents.enabled && /\b(architecture|refactor|security review|entire repo|parallel|multiple agents|complex migration)\b/i.test(event.prompt)) {
      try {
        await agents.dispatch(event.prompt, ctx, true);
      } catch {
        // Best effort: agent orchestration must not reject prompt startup.
      }
    }

    let result;
    try {
      result = await auto.route(event.prompt, ctx, ctx.signal);
    } catch {
      result = { ran: false, activated: [], skills: [], elapsedMs: 0 };
    }
    if (!result.ran) return promptChanged ? { systemPrompt } : undefined;

    if (result.activated.length > 0) {
      ctx.ui.setStatus("jev", `jev: auto (+${result.activated.length} tools)`);
    }

    if (result.skills.length === 0) return promptChanged ? { systemPrompt } : undefined;

    return {
      systemPrompt: promptChanged ? systemPrompt : undefined,
      message: {
        customType: "jev-auto",
        display: true,
        content:
          "Jev auto-matched skill(s) for this task. Load the matching SKILL.md before proceeding:\n" +
          result.skills
            .map((s) => `• /skill:${s.name.replace(/^(?:skill:)+/, "")} (P=${s.probability.toFixed(2)})`)
            .join("\n"),
      },
    };
  });
}
