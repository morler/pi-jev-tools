import fs from "node:fs";
import os from "node:os";
import path from "node:path";

/** Switches that can be saved globally, each with the env var that overrides it. */
export const SWITCHES = {
  auto: "PI_JEV_AUTO",
  autoModel: "PI_JEV_AUTO_MODEL",
  compact: "PI_JEV_COMPACT",
  agents: "PI_JEV_AGENTS",
  toolGuard: "PI_JEV_TOOL_GUARD",
  searchGate: "PI_JEV_SEARCH_GATE",
  skillStrip: "PI_JEV_SKILL_STRIP",
} as const;

export type JevConfigKey = keyof typeof SWITCHES;

/** Built-in switch defaults: what a fresh install starts with when no env var and no saved file says anything. */
const SWITCH_DEFAULTS: Record<JevConfigKey, boolean> = {
  auto: false,
  autoModel: false,
  compact: true,
  agents: false,
  toolGuard: false,
  searchGate: false,
  skillStrip: false,
};

export type JevConfig = Partial<Record<JevConfigKey, boolean>>;

const TRUTHY = new Set(["1", "true", "yes", "on"]);

/** Whether an env var holds a truthy value. An unset var is not truthy; a blank one is not either. */
export function isTruthy(value: string | undefined): boolean {
  return value !== undefined && TRUTHY.has(value.trim().toLowerCase());
}

/** True for a plain JSON object: not null, not an array. */
export function isJsonObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

/** Global config file, beside Pi's own settings. Honors PI_CODING_AGENT_DIR like Pi does. */
export function configPath(): string {
  const dir = process.env.PI_CODING_AGENT_DIR?.trim() || path.join(os.homedir(), ".pi", "agent");
  return path.join(dir, "pi-jev.json");
}

/** Read a JSON object; a missing, corrupt, or non-object file reads as empty. Never throws. */
export function readJsonObject(file: string): Record<string, unknown> {
  try {
    const parsed = JSON.parse(fs.readFileSync(file, "utf8"));
    return isJsonObject(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

/** Write a JSON object, creating its directory. Throws only if the path is unwritable. */
export function writeJson(file: string, value: unknown): void {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const temp = `${file}.${process.pid}.${Date.now()}.tmp`;
  try {
    fs.writeFileSync(temp, `${JSON.stringify(value, null, 2)}\n`);
    fs.renameSync(temp, file);
  } catch (error) {
    try { fs.rmSync(temp, { force: true }); } catch { /* preserve original write error */ }
    throw error;
  }
}

/** Never throws: a missing or corrupt file just means "nothing saved". */
export function loadConfig(): JevConfig {
  return readJsonObject(configPath()) as JevConfig;
}

/** Merge switches into the global file. Throws only if the path is unwritable. */
export function saveConfig(patch: JevConfig): JevConfig {
  const next = { ...loadConfig(), ...patch };
  writeJson(configPath(), next);
  return next;
}

/**
 * A switch default: an env var that is set at all wins (so PI_JEV_COMPACT=0 is a hard off),
 * then the saved file, then the switch's built-in default (compact on, the rest off). An explicit
 * CLI flag still overrides both.
 */
export function resolveSwitch(key: JevConfigKey, saved: JevConfig): boolean {
  const raw = process.env[SWITCHES[key]];
  if (raw !== undefined) return isTruthy(raw);
  return resolveSaved(key, saved);
}

/** The saved file's value for a switch (coerced), else its built-in default. Env is not consulted. */
function resolveSaved(key: JevConfigKey, saved: JevConfig): boolean {
  // The file is hand-editable: coerce the way the env path does, so `"compact": "off"` cannot enable a
  // switch just by being a truthy string while "on" still means on.
  const value = saved[key];
  if (typeof value === "boolean") return value;
  if (typeof value === "string") return isTruthy(value);
  return SWITCH_DEFAULTS[key];
}

/** True when a set env var would override `value` on the next start, so a toggle must say so now. */
export function envShadowed(key: JevConfigKey, value: boolean): boolean {
  return process.env[SWITCHES[key]] !== undefined && isTruthy(process.env[SWITCHES[key]]) !== value;
}

/** Env vars that actually change a switch from the saved file, for `/jev status`. */
export function envOverrides(): string[] {
  const saved = loadConfig();
  return (Object.keys(SWITCHES) as JevConfigKey[]).flatMap((key) =>
    envShadowed(key, resolveSaved(key, saved)) ? [`$${SWITCHES[key]}`] : []
  );
}

/** Pool entry: a model id plus an optional one-line capability note passed to the routing judge. */
export interface PoolEntry { model: string; note?: string }

const DEFAULT_POOL_ENTRIES: PoolEntry[] = [
  { model: "glm-5.5-flash", note: "fine with multi-step tool loops and web search; weak on long-context synthesis" },
  { model: "deepseek-v4-flash", note: "deep reasoning, large-context synthesis, complex vision" },
];

/** Default auto-model candidate pool: [light, heavy]. */
export const DEFAULT_MODEL_POOL = DEFAULT_POOL_ENTRIES;

/**
 * The auto-model candidate pool, saved as an ordered "autoModelPool" array in pi-jev.json:
 * index 0 answers light tasks, index 1 heavy ones. Entries may be a bare model id string
 * (legacy) or `{model, note?}`. A missing, empty, or malformed entry falls back to the
 * built-in default.
 */
export function loadModelPool(): PoolEntry[] {
  const raw = readJsonObject(configPath())["autoModelPool"];
  if (!Array.isArray(raw)) return DEFAULT_MODEL_POOL;
  const pool: PoolEntry[] = [];
  for (const x of raw) {
    if (typeof x === "string" && x.trim() !== "") pool.push({ model: x.trim() });
    else if (x && typeof x === "object" && typeof (x as Record<string, unknown>).model === "string") {
      const model = ((x as Record<string, unknown>).model as string).trim();
      if (!model) continue;
      const noteRaw = (x as Record<string, unknown>).note;
      pool.push(typeof noteRaw === "string" ? { model, note: noteRaw } : { model });
    }
  }
  return pool.length > 0 ? pool : DEFAULT_MODEL_POOL;
}
