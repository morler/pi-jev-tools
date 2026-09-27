import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import type { JevClient } from "./jev.js";

const RPC_REQUEST = "subagents:rpc:v1:request";
const RPC_REPLY = "subagents:rpc:v1:reply:";
const ASYNC_COMPLETE = "subagent:async-complete";

/** Shape of a subagents:rpc:v1 reply payload. */
interface RpcReply {
  success?: boolean;
  data?: { runId?: string; id?: string };
  error?: { message?: string };
}

export type OrchestrationTopology = "implementation" | "research" | "review" | "general";

export interface OrchestrationResult {
  runId?: string;
  accepted: boolean;
  topology?: OrchestrationTopology;
  error?: string;
}

// ---------------------------------------------------------------------------
// Backend-neutral workflow plan (JSON). No script text, no template markers:
// task text flows in verbatim as a { task: true } or literal segment, so user
// content containing braces can never be mistaken for a placeholder.
// ---------------------------------------------------------------------------

/** One piece of a node's prompt: literal text, an upstream node's output, or the task itself. */
export type PlanSegment = { text: string } | { input: string } | { task: true };

export interface PlanNode {
  /** Unique id, doubles as the variable name backends bind (scout, worker, ...). */
  id: string;
  /** Agent the backend should run for this node (may differ from id, e.g. synthesizer -> worker). */
  agent: string;
  label: string;
  /** Upstream node ids that must complete before this node runs. */
  needs: string[];
  /** Prompt pieces in emission order. */
  segments: PlanSegment[];
}

export interface WorkflowPlan {
  task: string;
  topology: OrchestrationTopology;
  /** Topologically ordered: every node's needs precede it. */
  nodes: PlanNode[];
}

/**
 * A subagent backend: everything pi-jev needs from a subagent extension.
 * spawn() receives the plan (not compiled script) and resolves with a runId
 * or throws an Error whose message is shown to the user.
 * onCompleted(), when present, registers a runId completion handler.
 */
export interface OrchestrationBackend {
  spawn(plan: WorkflowPlan): Promise<{ runId?: string }>;
  onCompleted?(handler: (runId: string) => void): void;
}

export function classifyTopologyFallback(task: string): OrchestrationTopology {
  const lower = task.toLowerCase();
  if (/\b(review|audit|security|check|verify|spec)\b/i.test(lower)) {
    return "review";
  }
  if (/\b(research|investigate|explore|how does|find|analyze|architecture)\b/i.test(lower)) {
    return "research";
  }
  if (/\b(fix|bug|implement|refactor|add|build|create|migrate|update|delete)\b/i.test(lower)) {
    return "implementation";
  }
  return "general";
}

export async function determineTopology(
  task: string,
  jevClient?: JevClient,
  signal?: AbortSignal
): Promise<OrchestrationTopology> {
  if (jevClient?.isConfigured()) {
    try {
      const response = await jevClient.evaluate(
        {
          state: { task },
          questions: {
            topology: {
              type: "choice",
              instructions: "Using `task`, what type of workflow is best suited for this task?",
              criteria: {
                implementation: "Code change, bugfix, refactoring, feature implementation, or file modifications",
                research: "Investigating codebase, external research, architectural analysis, or exploration",
                review: "Code review, security audit, checking compliance or reviewing a pull request",
                general: "General question or task not requiring multi-stage implementation",
              },
            },
          },
        },
        signal
      );
      const choiceVal = response.answers["topology"]?.value as OrchestrationTopology;
      if (choiceVal && ["implementation", "research", "review", "general"].includes(choiceVal)) {
        return choiceVal;
      }
    } catch {
      // Fall back on local classifier
    }
  }

  return classifyTopologyFallback(task);
}

export function buildWorkflowPlan(task: string, topology: OrchestrationTopology): WorkflowPlan {
  const t: PlanSegment = { task: true };
  let nodes: PlanNode[];

  switch (topology) {
    case "implementation":
      nodes = [
        {
          id: "scout",
          agent: "scout",
          label: "Scout codebase context",
          needs: [],
          segments: [{ text: "Find all relevant files, functions, and architecture context needed for: " }, t],
        },
        {
          id: "worker",
          agent: "worker",
          label: "Implement changes",
          needs: ["scout"],
          segments: [
            { text: "Implement the requested task using scout findings.\n\nScout findings:\n" },
            { input: "scout" },
            { text: "\n\nTask:\n" },
            t,
          ],
        },
        {
          id: "reviewer",
          agent: "reviewer",
          label: "Review implementation",
          needs: ["worker"],
          segments: [
            { text: "Review the implementation against standards, bugs, and requirements.\n\nTask:\n" },
            t,
            { text: "\n\nWorker output:\n" },
            { input: "worker" },
          ],
        },
      ];
      break;

    case "research":
      nodes = [
        {
          id: "scout",
          agent: "scout",
          label: "Scout repository evidence",
          needs: [],
          segments: [{ text: "Inspect repository files, structure, and code relevant to: " }, t],
        },
        {
          id: "researcher",
          agent: "researcher",
          label: "Research external and technical context",
          needs: [],
          segments: [{ text: "Research technical domain, best practices, and documentation for: " }, t],
        },
        {
          id: "synthesizer",
          agent: "worker",
          label: "Synthesize research report",
          needs: ["scout", "researcher"],
          segments: [
            {
              text:
                "Synthesize local repository findings and external research into an actionable report.\n\nRepository findings:\n",
            },
            { input: "scout" },
            { text: "\n\nExternal research:\n" },
            { input: "researcher" },
            { text: "\n\nTask:\n" },
            t,
          ],
        },
      ];
      break;

    case "review":
      nodes = [
        {
          id: "reviewer",
          agent: "reviewer",
          label: "Code review standards",
          needs: [],
          segments: [{ text: "Perform a code review for quality, bugs, and specifications: " }, t],
        },
        {
          id: "auditor",
          agent: "evidence-auditor",
          label: "Security and evidence audit",
          needs: [],
          segments: [{ text: "Audit security risks, verification evidence, and edge cases for: " }, t],
        },
      ];
      break;

    case "general":
    default:
      nodes = [
        { id: "worker", agent: "worker", label: "Execute task", needs: [], segments: [t] },
        {
          id: "reviewer",
          agent: "reviewer",
          label: "Verify output",
          needs: ["worker"],
          segments: [
            { text: "Verify that the work meets requirements.\n\nTask:\n" },
            t,
            { text: "\n\nWorker output:\n" },
            { input: "worker" },
          ],
        },
      ];
      break;
  }

  return { task, topology, nodes };
}

// ---------------------------------------------------------------------------
// pi-subagents backend: compiles the plan to a pi-subagents workflow script
// and dispatches it over the subagents:rpc:v1 event pair.
// ---------------------------------------------------------------------------

function segmentExpr(seg: PlanSegment, plan: WorkflowPlan, done: ReadonlySet<string>): string {
  if ("text" in seg) return JSON.stringify(seg.text);
  if ("task" in seg) return JSON.stringify(plan.task);
  if (!done.has(seg.input)) {
    throw new Error(`workflow plan segment references "${seg.input}" before it runs`);
  }
  return `${seg.input}.output`;
}

function taskExpr(node: PlanNode, plan: WorkflowPlan, done: ReadonlySet<string>): string {
  return node.segments.map((seg) => segmentExpr(seg, plan, done)).join(" + ");
}

/** Compile a WorkflowPlan to a pi-subagents workflow script. */
export function compilePiSubagentsScript(plan: WorkflowPlan): string {
  const lines: string[] = [];
  const done = new Set<string>();
  const remaining = [...plan.nodes];

  while (remaining.length) {
    // Nodes are topologically ordered; the batch is the ready prefix at batch start,
    // so a node never runs before everything it needs has been bound.
    const batch: PlanNode[] = [];
    for (const node of remaining) {
      if (node.needs.every((n) => done.has(n))) batch.push(node);
      else break;
    }
    if (!batch.length) {
      throw new Error(`workflow plan has unsatisfiable needs: ${remaining.map((n) => n.id).join(", ")}`);
    }
    remaining.splice(0, batch.length);

    if (batch.length === 1) {
      const node = batch[0];
      lines.push(
        `const ${node.id} = await runs.run(${JSON.stringify(node.id)}, {` +
          `\n  agent: ${JSON.stringify(node.agent)},` +
          `\n  label: ${JSON.stringify(node.label)},` +
          `\n  task: ${taskExpr(node, plan, done)}` +
          `\n});`
      );
    } else {
      lines.push(`const [${batch.map((n) => n.id).join(", ")}] = runs.all([`);
      for (const node of batch) {
        lines.push(
          `  {` +
            `\n    key: ${JSON.stringify(node.id)},` +
            `\n    agent: ${JSON.stringify(node.agent)},` +
            `\n    label: ${JSON.stringify(node.label)},` +
            `\n    task: ${taskExpr(node, plan, done)}` +
            `\n  },`
        );
      }
      lines.push(`]);`);
    }
    for (const node of batch) done.add(node.id);
  }

  lines.push(`return { ${plan.nodes.map((n) => `${n.id}: ${n.id}.output`).join(", ")};`);
  return lines.join("\n");
}

/** Kept for back-compat: same script as before, now derived from the JSON plan. */
export function buildWorkflowScript(task: string, topology: OrchestrationTopology): string {
  return compilePiSubagentsScript(buildWorkflowPlan(task, topology));
}

/** Default backend: dispatches through the pi-subagents extension's RPC events. */
export class PiSubagentsBackend implements OrchestrationBackend {
  constructor(private pi: ExtensionAPI) {}

  public async spawn(plan: WorkflowPlan): Promise<{ runId?: string }> {
    const workflowScript = compilePiSubagentsScript(plan);
    const requestId = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;
    const replyEvent = `${RPC_REPLY}${requestId}`;

    const response = await new Promise<RpcReply>((resolve) => {
      let unsubscribe = () => {};
      const timer = setTimeout(() => {
        unsubscribe();
        resolve({ success: false, error: { message: "pi-subagents RPC timeout" } });
      }, 10_000);

      const onReply = (reply: unknown) => {
        clearTimeout(timer);
        unsubscribe();
        resolve(reply as RpcReply);
      };

      unsubscribe = this.pi.events.on(replyEvent, onReply);
      this.pi.events.emit(RPC_REQUEST, {
        version: 1,
        requestId,
        method: "spawn",
        source: { extension: "pi-jev" },
        params: {
          async: true,
          workflowScript,
        },
      });
    });

    if (!response?.success) {
      throw new Error(response?.error?.message ?? "pi-subagents unavailable");
    }
    return { runId: response.data?.runId ?? response.data?.id };
  }

  public onCompleted(handler: (runId: string) => void): void {
    this.pi.events.on(ASYNC_COMPLETE, (event: unknown) => {
      const runId = (event as { runId?: string } | undefined)?.runId;
      if (runId) handler(runId);
    });
  }
}

export class AgentOrchestrator {
  public enabled: boolean;
  private running = false;
  private backend: OrchestrationBackend;

  constructor(
    private pi: ExtensionAPI,
    private jevClient?: JevClient,
    enabled = false,
    backend?: OrchestrationBackend
  ) {
    this.enabled = enabled;
    this.backend = backend ?? new PiSubagentsBackend(pi);
  }

  public setEnabled(enabled: boolean): void {
    this.enabled = enabled;
  }

  public async dispatch(
    task: string,
    ctx: ExtensionContext,
    automatic = false
  ): Promise<OrchestrationResult> {
    if (!this.enabled && automatic) return { accepted: false, error: "disabled" };
    if (this.running) return { accepted: false, error: "busy" };
    if (!task.trim()) return { accepted: false, error: "empty task" };

    this.running = true;
    try {
      const topology = await determineTopology(task, this.jevClient, ctx.signal);
      const plan = buildWorkflowPlan(task, topology);
      const { runId } = await this.backend.spawn(plan);

      ctx.ui.notify(
        `Agent orchestration started (${topology} topology)${runId ? ` [${runId}]` : ""}.`,
        "info"
      );
      return { accepted: true, runId, topology };
    } catch (error) {
      return { accepted: false, error: error instanceof Error ? error.message : String(error) };
    } finally {
      this.running = false;
    }
  }

  public installCompletionNotice(): void {
    this.backend.onCompleted?.((runId) => {
      this.pi.sendMessage({
        customType: "jev-agents",
        display: true,
        content: `Agent orchestration completed: ${runId}`,
      });
    });
  }
}
