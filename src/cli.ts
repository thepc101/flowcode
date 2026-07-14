#!/usr/bin/env node

import OpenAI from "openai";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";
import { createInterface } from "readline";
import { spawn } from "child_process";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const VERSION = "1.3.0";
const HOME = os.homedir();
const CONFIG_PATH = path.join(HOME, ".flow-code-config");
const SESSION_PATH = path.join(HOME, ".flow-code-session");
const ACTIVITY_PATH = path.join(HOME, ".flow-code-activity");
const CONTEXT_WINDOW = 128000;
const MAX_HISTORY_TOKENS = 2000;
const MAX_RESPONSE_TOKENS = 2048;
const TOOL_DEFS_TOKENS = 1500;
const SYSTEM_PROMPT_TOKENS = 200;
const TPM_LIMIT = 6500;
const SHELL_TIMEOUT_MS = 60_000;
const MAX_SHELL_OUTPUT = 4000;
const MAX_WEB_CONTENT = 12000;
const MAX_FILE_READ = 50_000;
const MAX_TOOL_ROUNDS = 25;
const COST_PER_TOKEN = 0.000001;
const MIN_API_KEY_LENGTH = 10;

const BLOCKED_COMMANDS = /\b(rm\s+-rf\s+\/|mkfs|dd\s+if=|:(){ :\|:& };:|chmod\s+-R\s+777\s+\/|wget.*\|\s*sh|curl.*\|\s*sh|sudo\s+rm|shutdown|reboot|halt|init\s+0|killall|pkill\s+-9\s+-u)\b/;

// ANSI
const R = "\x1b[0m";
const B = "\x1b[1m";
const D = "\x1b[2m";
const RED = "\x1b[31m";
const GRN = "\x1b[32m";
const YEL = "\x1b[33m";
const CYN = "\x1b[36m";
const MAG = "\x1b[35m";

const a = {
  bold: (t: string) => `${B}${t}${R}`,
  dim: (t: string) => `${D}${t}${R}`,
  red: (t: string) => `${RED}${t}${R}`,
  green: (t: string) => `${GRN}${t}${R}`,
  yellow: (t: string) => `${YEL}${t}${R}`,
  cyan: (t: string) => `${CYN}${t}${R}`,
  magenta: (t: string) => `${MAG}${t}${R}`,
};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type Intensity = "low" | "medium" | "high";
type Provider = "groq" | "cerebras";
type Role = "system" | "user" | "assistant" | "tool";
type Message = OpenAI.Chat.Completions.ChatCompletionMessageParam;

interface Config {
  apiKey: string;
  provider: Provider;
  defaultModel?: string;
  intensity?: Intensity;
}

interface UsageInfo {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
}

interface SearchResult {
  title: string;
  snippet: string;
  url: string;
}

interface FileToWrite {
  filepath: string;
  content: string;
}

interface ActivityEntry {
  time: number;
  action: string;
}

interface SessionData {
  history: any[];
  model: string;
  provider: Provider;
  intensity: Intensity;
  cwd: string;
  timestamp: number;
}

interface SessionState {
  client: OpenAI;
  model: string;
  intensity: Intensity;
  provider: Provider;
}

interface ToolResult {
  tool_call_id: string;
  output: string;
  error?: boolean;
}

// ---------------------------------------------------------------------------
// Provider configuration
// ---------------------------------------------------------------------------

const PROVIDERS: Record<
  Provider,
  { name: string; baseURL: string; defaultModel: string; modelFilter: RegExp }
> = {
  groq: {
    name: "Groq",
    baseURL: "https://api.groq.com/openai/v1",
    defaultModel: "openai/gpt-oss-120b",
    modelFilter: /llama|mixtral|qwen|gemma|deepseek|gpt-oss/i,
  },
  cerebras: {
    name: "Cerebras",
    baseURL: "https://api.cerebras.ai/v1",
    defaultModel: "zai-glm-4.7",
    modelFilter: /llama|qwen|glm|gpt-oss/i,
  },
};

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 3.5);
}

function getMessageContent(msg: Message): string {
  if (typeof msg.content === "string") return msg.content;
  if (Array.isArray(msg.content)) {
    return msg.content.map((p) => ("text" in p ? p.text : "")).join("");
  }
  return "";
}

function sanitizeModelId(id: string): string {
  return id.replace(/[^a-zA-Z0-9._/-]/g, "").slice(0, 128);
}

function truncate(str: string, max: number): string {
  return str.length > max ? str.slice(0, max - 3) + "..." : str;
}

function timeAgo(ts: number): string {
  const diff = Date.now() - ts;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 7) return `${days}d ago`;
  return `${Math.floor(days / 7)}w ago`;
}

function safePath(p: string): string {
  return path.resolve(p.replace(/\0/g, ""));
}

function isPathInside(child: string, parent: string): boolean {
  const rel = path.relative(parent, child);
  return rel === "" || (!rel.startsWith("..") && !path.isAbsolute(rel));
}

function renderUsageBar(used: number, total: number): string {
  const pct = Math.min(used / total, 1);
  const filled = Math.round(pct * 20);
  const empty = 20 - filled;
  const bar =
    GRN + "\u2588".repeat(filled) + R + D + "\u2591".repeat(empty) + R;
  const pctStr = Math.round(pct * 100) + "%";
  return `  ${a.dim("Context:")} ${bar} ${a.dim(`${used.toLocaleString()} / ${total.toLocaleString()} tokens (${pctStr})`)}`;
}

function printUsage(usage: UsageInfo, history: Message[]): void {
  const totalUsed = history.reduce(
    (acc, msg) => acc + estimateTokens(getMessageContent(msg)),
    0
  );
  console.log("");
  console.log(
    `  ${a.dim("Tokens:")} ${a.bold(String(usage.promptTokens))} prompt + ${a.bold(String(usage.completionTokens))} completion = ${a.bold(String(usage.totalTokens))} total`
  );
  console.log(renderUsageBar(totalUsed, CONTEXT_WINDOW));
  console.log("");
}

// ---------------------------------------------------------------------------
// Activity log
// ---------------------------------------------------------------------------

function logActivity(action: string): void {
  try {
    let entries: ActivityEntry[] = [];
    if (fs.existsSync(ACTIVITY_PATH)) {
      const raw = fs.readFileSync(ACTIVITY_PATH, "utf-8");
      entries = JSON.parse(raw);
      if (!Array.isArray(entries)) entries = [];
    }
    entries.unshift({ time: Date.now(), action });
    entries = entries.slice(0, 20);
    fs.writeFileSync(ACTIVITY_PATH, JSON.stringify(entries, null, 2), {
      encoding: "utf-8",
      mode: 0o600,
    });
  } catch {
    // non-critical
  }
}

function getRecentActivity(): ActivityEntry[] {
  try {
    if (fs.existsSync(ACTIVITY_PATH)) {
      const raw = fs.readFileSync(ACTIVITY_PATH, "utf-8");
      const entries = JSON.parse(raw);
      if (Array.isArray(entries)) return entries.slice(0, 5);
    }
  } catch {
    // ignore
  }
  return [];
}

// ---------------------------------------------------------------------------
// Session save / load
// ---------------------------------------------------------------------------

function saveSession(
  history: Message[],
  model: string,
  provider: Provider,
  intensity: Intensity
): void {
  try {
    const data: SessionData = {
      history: history.map((m) => {
        const msg: any = { role: m.role, content: m.content || "" };
        if ((m as any).tool_calls) msg.tool_calls = (m as any).tool_calls;
        if ((m as any).tool_call_id) msg.tool_call_id = (m as any).tool_call_id;
        return msg;
      }),
      model,
      provider,
      intensity,
      cwd: process.cwd(),
      timestamp: Date.now(),
    };
    fs.writeFileSync(SESSION_PATH, JSON.stringify(data, null, 2), {
      encoding: "utf-8",
      mode: 0o600,
    });
  } catch {
    // non-critical
  }
}

function loadSession(): SessionData | null {
  try {
    if (fs.existsSync(SESSION_PATH)) {
      const raw = fs.readFileSync(SESSION_PATH, "utf-8");
      const data = JSON.parse(raw);
      if (
        data &&
        typeof data === "object" &&
        Array.isArray(data.history) &&
        typeof data.timestamp === "number"
      ) {
        // Sanitize: remove orphaned tool messages (missing tool_call_id)
        // and their preceding assistant messages with tool_calls
        const clean: any[] = [];
        let i = 0;
        while (i < data.history.length) {
          const msg = data.history[i];
          if (msg.role === "tool" && !msg.tool_call_id) {
            // Skip this broken tool message; also skip preceding assistant
            if (clean.length > 0 && clean[clean.length - 1].role === "assistant" && clean[clean.length - 1].tool_calls) {
              clean.pop();
            }
            i++;
            continue;
          }
          clean.push(msg);
          i++;
        }
        data.history = clean;
        return data as SessionData;
      }
    }
  } catch {
    // corrupted
  }
  return null;
}

function hasSavedSession(): boolean {
  try {
    return fs.existsSync(SESSION_PATH);
  } catch {
    return false;
  }
}

function restoreSession(history: Message[]): { count: number; cwd: string; timestamp: number } | null {
  const session = loadSession();
  if (!session) return null;
  history.length = 0;
  for (const msg of session.history) {
    const restored: any = { role: msg.role, content: msg.content || "" };
    if (msg.tool_calls) restored.tool_calls = msg.tool_calls;
    if (msg.tool_call_id) restored.tool_call_id = msg.tool_call_id;
    history.push(restored);
  }
  if (session.cwd && fs.existsSync(session.cwd)) {
    process.chdir(session.cwd);
  }
  return {
    count: history.filter((m) => m.role === "user" || m.role === "assistant").length,
    cwd: process.cwd(),
    timestamp: session.timestamp,
  };
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

function loadConfig(): Config {
  const fallback: Config = { apiKey: "", provider: "groq" };
  try {
    if (!fs.existsSync(CONFIG_PATH)) return fallback;
    const raw = fs.readFileSync(CONFIG_PATH, "utf-8");
    const parsed = JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null) return fallback;
    return {
      apiKey: typeof parsed.apiKey === "string" ? parsed.apiKey : "",
      provider: ["groq", "cerebras"].includes(parsed.provider)
        ? parsed.provider
        : "groq",
      defaultModel:
        typeof parsed.defaultModel === "string"
          ? parsed.defaultModel
          : undefined,
      intensity: ["low", "medium", "high"].includes(parsed.intensity)
        ? parsed.intensity
        : undefined,
    };
  } catch {
    return fallback;
  }
}

function saveConfig(config: Config): void {
  try {
    fs.writeFileSync(CONFIG_PATH, JSON.stringify(config, null, 2), {
      encoding: "utf-8",
      mode: 0o600,
    });
  } catch {
    console.error(a.red("  Failed to save config."));
  }
}

// ---------------------------------------------------------------------------
// Directory scanning
// ---------------------------------------------------------------------------

const IGNORE_DIRS = new Set([
  "node_modules",
  ".git",
  "dist",
  "build",
  ".next",
  ".cache",
  "__pycache__",
  ".vscode",
  ".idea",
  "coverage",
  ".turbo",
  ".vercel",
  ".netlify",
  "target",
  "vendor",
  ".tox",
  "venv",
  ".venv",
]);

const IGNORE_FILES = new Set([
  ".DS_Store",
  "Thumbs.db",
  ".env",
  ".env.local",
  ".env.development",
  ".env.production",
  "package-lock.json",
  "yarn.lock",
  "pnpm-lock.yaml",
]);

function scanDirectory(
  dir: string,
  depth: number = 3,
  prefix: string = ""
): string {
  if (depth < 0) return "";
  const lines: string[] = [];
  try {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    const dirs = entries
      .filter(
        (e) =>
          e.isDirectory() &&
          !IGNORE_DIRS.has(e.name) &&
          !e.name.startsWith(".")
      )
      .sort((a, b) => a.name.localeCompare(b.name));
    const files = entries
      .filter(
        (e) =>
          e.isFile() &&
          !IGNORE_FILES.has(e.name) &&
          !e.name.startsWith(".")
      )
      .sort((a, b) => a.name.localeCompare(b.name));
    const all = [...dirs, ...files];
    for (let i = 0; i < all.length; i++) {
      const entry = all[i];
      const isLast = i === all.length - 1;
      const conn = isLast ? "\\-- " : "|-- ";
      const childPref = isLast ? "    " : "|   ";
      if (entry.isDirectory()) {
        lines.push(`${prefix}${conn}${entry.name}/`);
        const sub = scanDirectory(
          path.join(dir, entry.name),
          depth - 1,
          prefix + childPref
        );
        if (sub) lines.push(sub);
      } else {
        lines.push(`${prefix}${conn}${entry.name}`);
      }
    }
  } catch {
    // permission denied
  }
  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// Web search (DuckDuckGo)
// ---------------------------------------------------------------------------

async function searchWeb(
  query: string,
  numResults: number = 5
): Promise<SearchResult[]> {
  try {
    const url = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`;
    const res = await fetch(url, {
      headers: {
        "User-Agent":
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
      },
      signal: AbortSignal.timeout(10000),
    });
    if (!res.ok) return [];
    const html = await res.text();
    const results: SearchResult[] = [];
    const re =
      /<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>([\s\S]*?)<\/a>[\s\S]*?<a[^>]*class="result__snippet"[^>]*>([\s\S]*?)<\/a>/g;
    let m: RegExpExecArray | null;
    while ((m = re.exec(html)) !== null && results.length < numResults) {
      const rawUrl = m[1];
      const title = m[2].replace(/<[^>]*>/g, "").trim();
      const snippet = m[3].replace(/<[^>]*>/g, "").trim();
      const uddg = rawUrl.match(/uddg=([^&]+)/);
      const finalUrl = uddg ? decodeURIComponent(uddg[1]) : rawUrl;
      if (title && snippet) {
        results.push({ title, snippet, url: finalUrl });
      }
    }
    return results;
  } catch {
    return [];
  }
}

async function fetchUrlContent(url: string): Promise<string> {
  try {
    const res = await fetch(url, {
      headers: {
        "User-Agent":
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        Accept: "text/html,application/xhtml+xml",
      },
      signal: AbortSignal.timeout(15000),
      redirect: "follow",
    });
    if (!res.ok) return `HTTP ${res.status}: ${res.statusText}`;
    const html = await res.text();
    return html
      .replace(/<script[\s\S]*?<\/script>/gi, "")
      .replace(/<style[\s\S]*?<\/style>/gi, "")
      .replace(/<[^>]*>/g, " ")
      .replace(/&nbsp;/g, " ")
      .replace(/&amp;/g, "&")
      .replace(/&lt;/g, "<")
      .replace(/&gt;/g, ">")
      .replace(/&#\d+;/g, "")
      .replace(/\s+/g, " ")
      .trim()
      .slice(0, MAX_WEB_CONTENT);
  } catch (err: unknown) {
    return `Failed: ${err instanceof Error ? err.message : "unknown"}`;
  }
}

// ---------------------------------------------------------------------------
// Tool definitions (OpenAI function calling format)
// ---------------------------------------------------------------------------

const TOOL_DEFINITIONS: OpenAI.Chat.Completions.ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "read_file",
      description:
        "Read the contents of a file. Returns the full file content. Use this to understand existing code before modifying it.",
      parameters: {
        type: "object",
        properties: {
          path: {
            type: "string",
            description: "Absolute or relative path to the file",
          },
        },
        required: ["path"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "write_file",
      description:
        "Write content to a file. Creates the file and any parent directories if they don't exist. Overwrites existing files.",
      parameters: {
        type: "object",
        properties: {
          path: {
            type: "string",
            description: "Absolute or relative path to the file",
          },
          content: {
            type: "string",
            description: "The full content to write to the file",
          },
        },
        required: ["path", "content"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "edit_file",
      description:
        "Edit a file by replacing specific text. Use read_file first to get the exact text to replace.",
      parameters: {
        type: "object",
        properties: {
          path: {
            type: "string",
            description: "Absolute or relative path to the file",
          },
          old_text: {
            type: "string",
            description: "The exact text to find and replace (must match exactly)",
          },
          new_text: {
            type: "string",
            description: "The text to replace it with",
          },
        },
        required: ["path", "old_text", "new_text"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "run_command",
      description:
        "Execute a shell command. Returns stdout/stderr output. Use for building, testing, installing packages, git operations, etc.",
      parameters: {
        type: "object",
        properties: {
          command: {
            type: "string",
            description: "The shell command to execute",
          },
        },
        required: ["command"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "list_directory",
      description:
        "List files and directories at a path. Returns a tree view of the directory structure.",
      parameters: {
        type: "object",
        properties: {
          path: {
            type: "string",
            description:
              "Absolute or relative path. Defaults to current directory.",
          },
          depth: {
            type: "number",
            description: "How many levels deep to scan. Default 2.",
          },
        },
      },
    },
  },
  {
    type: "function",
    function: {
      name: "search_web",
      description:
        "Search the web for current information. Use for latest versions, documentation, error solutions, comparisons, etc.",
      parameters: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: "The search query",
          },
        },
        required: ["query"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "fetch_url",
      description:
        "Fetch and read the content of a URL. Use to read documentation, GitHub issues, Stack Overflow answers, etc.",
      parameters: {
        type: "object",
        properties: {
          url: {
            type: "string",
            description: "The URL to fetch (must start with http:// or https://)",
          },
        },
        required: ["url"],
      },
    },
  },
];

// ---------------------------------------------------------------------------
// Tool execution
// ---------------------------------------------------------------------------

function isInsideCwd(filepath: string): boolean {
  const abs = safePath(path.resolve(process.cwd(), filepath));
  return isPathInside(abs, process.cwd()) || abs === process.cwd();
}

function showTool(name: string, detail: string): void {
  const icon: Record<string, string> = {
    read_file: "READ",
    write_file: "WRITE",
    edit_file: "EDIT",
    run_command: "RUN",
    list_directory: "LIST",
    search_web: "SEARCH",
    fetch_url: "FETCH",
  };
  const label = icon[name] || name.toUpperCase();
  console.log(a.dim(`  >> ${label}: ${truncate(detail, 60)}`));
}

async function execToolAsync(
  name: string,
  args: Record<string, unknown>,
  toolCallId: string
): Promise<ToolResult> {
  const cwd = process.cwd();

  switch (name) {
    case "read_file": {
      const fp = safePath(path.resolve(cwd, String(args.path || "")));
      showTool("read_file", String(args.path || ""));
      if (!fs.existsSync(fp)) {
        return { tool_call_id: toolCallId, output: `File not found: ${args.path}`, error: true };
      }
      if (!isPathInside(fp, cwd)) {
        return { tool_call_id: toolCallId, output: "Access denied: path outside workspace", error: true };
      }
      try {
        const stat = fs.statSync(fp);
        if (stat.size > MAX_FILE_READ) {
          const content = fs.readFileSync(fp, "utf-8").slice(0, MAX_FILE_READ);
          return { tool_call_id: toolCallId, output: content + "\n\n[TRUNCATED]" };
        }
        return { tool_call_id: toolCallId, output: fs.readFileSync(fp, "utf-8") };
      } catch (err: unknown) {
        return { tool_call_id: toolCallId, output: `Error: ${err instanceof Error ? err.message : "unknown"}`, error: true };
      }
    }

    case "write_file": {
      const fp = safePath(path.resolve(cwd, String(args.path || "")));
      showTool("write_file", String(args.path || ""));
      if (!isInsideCwd(String(args.path || ""))) {
        return { tool_call_id: toolCallId, output: "Access denied: can only write inside workspace", error: true };
      }
      const content = String(args.content || "");
      if (!content) {
        return { tool_call_id: toolCallId, output: "Error: content is empty", error: true };
      }
      try {
        const dir = path.dirname(fp);
        if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
        fs.writeFileSync(fp, content, "utf-8");
        const rel = path.relative(cwd, fp);
        logActivity(`Wrote: ${rel}`);
        return { tool_call_id: toolCallId, output: `File written: ${rel} (${content.length} chars)` };
      } catch (err: unknown) {
        return { tool_call_id: toolCallId, output: `Error: ${err instanceof Error ? err.message : "unknown"}`, error: true };
      }
    }

    case "edit_file": {
      const fp = safePath(path.resolve(cwd, String(args.path || "")));
      showTool("edit_file", String(args.path || ""));
      if (!isInsideCwd(String(args.path || ""))) {
        return { tool_call_id: toolCallId, output: "Access denied: can only edit inside workspace", error: true };
      }
      if (!fs.existsSync(fp)) {
        return { tool_call_id: toolCallId, output: `File not found: ${args.path}`, error: true };
      }
      const oldText = String(args.old_text || "");
      const newText = String(args.new_text || "");
      if (!oldText) {
        return { tool_call_id: toolCallId, output: "Error: old_text is required", error: true };
      }
      try {
        let content = fs.readFileSync(fp, "utf-8");
        if (!content.includes(oldText)) {
          return { tool_call_id: toolCallId, output: "Error: old_text not found. Use read_file first.", error: true };
        }
        const matchCount = content.split(oldText).length - 1;
        if (matchCount > 1) {
          content = content.split(oldText).join(newText);
        } else {
          content = content.replace(oldText, newText);
        }
        fs.writeFileSync(fp, content, "utf-8");
        const rel = path.relative(cwd, fp);
        logActivity(`Edited: ${rel}`);
        const extra = matchCount > 1 ? ` (${matchCount} occurrences replaced)` : "";
        return { tool_call_id: toolCallId, output: `File edited: ${rel}${extra}` };
      } catch (err: unknown) {
        return { tool_call_id: toolCallId, output: `Error: ${err instanceof Error ? err.message : "unknown"}`, error: true };
      }
    }

    case "run_command": {
      const cmd = String(args.command || "");
      showTool("run_command", cmd);
      if (!cmd) {
        return { tool_call_id: toolCallId, output: "Error: command is empty", error: true };
      }
      if (BLOCKED_COMMANDS.test(cmd)) {
        return { tool_call_id: toolCallId, output: "Error: command blocked for safety. This command is destructive.", error: true };
      }
      return new Promise((resolve) => {
        let stdout = "";
        let stderr = "";
        try {
          const child = spawn(cmd, [], {
            cwd,
            shell: true,
            stdio: "pipe",
            env: { ...process.env, FORCE_COLOR: "0" },
          });
          const timer = setTimeout(() => child.kill("SIGTERM"), SHELL_TIMEOUT_MS);
          child.stdout?.on("data", (d: Buffer) => {
            stdout += d.toString();
            process.stdout.write(a.dim("  | ") + d.toString());
          });
          child.stderr?.on("data", (d: Buffer) => {
            stderr += d.toString();
            process.stderr.write(a.red("  ! ") + d.toString());
          });
          child.on("close", (code) => {
            clearTimeout(timer);
            const output = (code === 0 ? stdout : stderr || stdout).slice(0, MAX_SHELL_OUTPUT);
            logActivity(`Ran: ${truncate(cmd, 60)}`);
            resolve({
              tool_call_id: toolCallId,
              output: code === 0 ? output || "(no output)" : `Exit ${code}:\n${output}`,
              error: code !== 0,
            });
          });
          child.on("error", (err) => {
            clearTimeout(timer);
            resolve({ tool_call_id: toolCallId, output: err.message, error: true });
          });
        } catch (err: unknown) {
          resolve({
            tool_call_id: toolCallId,
            output: err instanceof Error ? err.message : "spawn error",
            error: true,
          });
        }
      });
    }

    case "list_directory": {
      const dir = args.path ? safePath(path.resolve(cwd, String(args.path))) : cwd;
      const depth = typeof args.depth === "number" ? args.depth : 2;
      showTool("list_directory", String(args.path || cwd));
      if (!fs.existsSync(dir)) {
        return { tool_call_id: toolCallId, output: `Directory not found: ${args.path}`, error: true };
      }
      if (!isPathInside(dir, cwd)) {
        return { tool_call_id: toolCallId, output: "Access denied: path outside workspace", error: true };
      }
      const tree = scanDirectory(dir, depth);
      return { tool_call_id: toolCallId, output: tree || "(empty directory)" };
    }

    case "search_web": {
      const query = String(args.query || "");
      showTool("search_web", query);
      const results = await searchWeb(query, 5);
      if (results.length === 0) {
        return { tool_call_id: toolCallId, output: `No results for "${query}"` };
      }
      const text = results
        .map((r, i) => `${i + 1}. ${r.title}\n   ${r.snippet}\n   ${r.url}`)
        .join("\n\n");
      logActivity(`Searched: ${truncate(query, 60)}`);
      return { tool_call_id: toolCallId, output: text };
    }

    case "fetch_url": {
      const url = String(args.url || "");
      showTool("fetch_url", url);
      if (!url.startsWith("http://") && !url.startsWith("https://")) {
        return { tool_call_id: toolCallId, output: "Error: URL must start with http:// or https://", error: true };
      }
      try {
        const parsed = new URL(url);
        const blocked = ["169.254.169.254", "127.0.0.1", "localhost", "0.0.0.0"];
        if (blocked.includes(parsed.hostname)) {
          return { tool_call_id: toolCallId, output: "Error: URL blocked (private/internal address)", error: true };
        }
      } catch {
        return { tool_call_id: toolCallId, output: "Error: invalid URL", error: true };
      }
      const content = await fetchUrlContent(url);
      logActivity(`Fetched: ${truncate(url, 60)}`);
      return { tool_call_id: toolCallId, output: content };
    }

    default:
      return { tool_call_id: toolCallId, output: `Unknown tool: ${name}`, error: true };
  }
}

// ---------------------------------------------------------------------------
// Code block rendering
// ---------------------------------------------------------------------------

const LANG_LABELS: Record<string, string> = {
  html: "HTML",
  css: "CSS",
  js: "JavaScript",
  javascript: "JavaScript",
  ts: "TypeScript",
  typescript: "TypeScript",
  py: "Python",
  python: "Python",
  json: "JSON",
  yaml: "YAML",
  yml: "YAML",
  md: "Markdown",
  bash: "Bash",
  sh: "Shell",
  shell: "Shell",
  jsx: "JSX",
  tsx: "TSX",
  sql: "SQL",
  rust: "Rust",
  go: "Go",
  java: "Java",
  c: "C",
  cpp: "C++",
  rb: "Ruby",
  php: "PHP",
  swift: "Swift",
  kt: "Kotlin",
  toml: "TOML",
  xml: "XML",
  svg: "SVG",
  dockerfile: "Dockerfile",
  makefile: "Makefile",
  prisma: "Prisma",
  graphql: "GraphQL",
  lua: "Lua",
  r: "R",
  dart: "Dart",
  scala: "Scala",
  ex: "Elixir",
  exs: "Elixir",
  hs: "Haskell",
  clj: "Clojure",
};

function renderCodeBox(lang: string, code: string): string {
  const label = LANG_LABELS[lang.toLowerCase()] || lang.toUpperCase();
  const lines = code.split("\n");
  const maxLen = Math.max(label.length + 4, ...lines.map((l) => l.length));
  const w = Math.min(maxLen + 2, 120);
  const top = `${D}+-- ${label} ${"-".repeat(Math.max(0, w - label.length - 3))}+${R}`;
  const bottom = `${D}+${"-".repeat(w)}+${R}`;
  const body = lines.map((line) => {
    const t = line.length > w ? line.slice(0, w - 3) + "..." : line;
    return `${D}|${R} ${t}`;
  });
  return [top, ...body, bottom].join("\n");
}

function formatResponse(text: string): string {
  return text.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
    const trimmed = code.replace(/\n$/, "");
    return `\n${renderCodeBox(lang || "text", trimmed)}\n`;
  });
}

// ---------------------------------------------------------------------------
// System prompt
// ---------------------------------------------------------------------------

function getSystemPrompt(
  intensity: Intensity,
  cwd: string,
  provider: Provider
): string {
  const pname = PROVIDERS[provider].name;

  return [
    `Flow Code — autonomous coding agent. ${pname} | ${cwd}`,
    "Use ONLY these tools: read_file, write_file, edit_file, run_command, list_directory, search_web, fetch_url. Do NOT invent or call any other tool names.",
    "Never output code blocks — use write_file. Never describe changes — make them with tools.",
    "Read files before editing. Write COMPLETE files. Install deps and start dev server to verify. Fix errors and retry.",
    "Quality: semantic HTML, CSS custom properties, Tailwind, async/await, proper error handling.",
    "Workflow: read → write → install → run → verify. Tell user the URL when done.",
  ].join("\n");
}

// ---------------------------------------------------------------------------
// History management
// ---------------------------------------------------------------------------

function trimHistory(history: Message[]): Message[] {
  if (history.length === 0) return [];

  // Account for system prompt + tool definitions + response budget
  const availableForHistory = TPM_LIMIT - SYSTEM_PROMPT_TOKENS - TOOL_DEFS_TOKENS - 200;

  // Always include system prompt
  let sysIdx = -1;
  const systemMsg = history[0].role === "system" ? history[0] : null;
  if (systemMsg) sysIdx = 0;

  // Collect messages from the end, keeping tool pairs intact
  const collected: Message[] = [];
  let total = 0;
  let i = history.length - 1;

  while (i > sysIdx) {
    const msg = history[i];

    if (msg.role === "tool") {
      // Walk back to find the assistant with tool_calls
      let j = i;
      while (j > sysIdx && history[j].role === "tool") j--;
      const group: Message[] = [];
      for (let k = j; k <= i; k++) {
        group.push(history[k]);
      }
      const groupTokens = group.reduce((sum, m) => sum + estimateTokens(getMessageContent(m)), 0);
      if (total + groupTokens > availableForHistory) break;
      total += groupTokens;
      for (let k = group.length - 1; k >= 0; k--) {
        collected.push(group[k]);
      }
      i = j - 1;
      continue;
    }

    // Skip assistant messages with tool_calls (handled above with their tools)
    if ((msg as any).tool_calls && (msg as any).tool_calls.length > 0) {
      i--;
      continue;
    }

    const tokens = estimateTokens(getMessageContent(msg));
    if (total + tokens > availableForHistory) break;
    total += tokens;
    collected.push(msg);
    i--;
  }

  // collected is in reverse chronological order; reverse to restore order
  collected.reverse();

  const result: Message[] = [];
  if (systemMsg) result.push(systemMsg);
  result.push(...collected);
  return result;
}

// ---------------------------------------------------------------------------
// Readline
// ---------------------------------------------------------------------------

const rl = createInterface({
  input: process.stdin,
  output: process.stdout,
});

function ask(question: string): Promise<string> {
  return new Promise((resolve) => {
    rl.question(question, (answer) => resolve(answer.trim()));
  });
}

function prompt(): string {
  const dir = path.basename(process.cwd());
  return a.cyan(`[${dir}] > `);
}

async function readInput(): Promise<string> {
  const first = await ask(prompt());
  if (!first) return "";
  if (first.endsWith("\\")) {
    const lines: string[] = [first.replace(/\\$/, "")];
    while (true) {
      const line = await ask(a.dim("  ... "));
      if (line === "") break;
      lines.push(line.replace(/\\$/, ""));
    }
    return lines.join("\n").trim();
  }
  return first;
}

// ---------------------------------------------------------------------------
// Thinking indicator
// ---------------------------------------------------------------------------

let thinkingInterval: ReturnType<typeof setInterval> | null = null;
const SPIN = ["|", "/", "-", "\\"];

function showThinking(): void {
  let i = 0;
  process.stdout.write(a.dim("  Thinking... \r"));
  thinkingInterval = setInterval(() => {
    process.stdout.write(a.dim(`  Thinking ${SPIN[i % SPIN.length]} \r`));
    i++;
  }, 150);
}

function clearThinking(): void {
  if (thinkingInterval) {
    clearInterval(thinkingInterval);
    thinkingInterval = null;
  }
  process.stdout.write(" ".repeat(40) + "\r");
}

// ---------------------------------------------------------------------------
// Streaming + Tool loop
// ---------------------------------------------------------------------------

async function streamWithTools(
  client: OpenAI,
  model: string,
  messages: Message[],
  temperature: number,
  history: Message[]
): Promise<{ content: string; usage: UsageInfo }> {
  let rounds = 0;
  let totalUsage: UsageInfo = { promptTokens: 0, completionTokens: 0, totalTokens: 0 };

  // Work on a copy to avoid mutating caller's array
  const apiMessages: Message[] = [...messages];

  // Sanitize: ensure every tool message has a preceding assistant with matching tool_calls
  const sanitized: Message[] = [];
  for (const m of apiMessages) {
    if (m.role === "tool") {
      const tm = m as any;
      if (!tm.tool_call_id) continue;
      if (sanitized.length === 0 || sanitized[sanitized.length - 1].role !== "assistant") continue;
      const prev = sanitized[sanitized.length - 1] as any;
      if (!prev.tool_calls || !prev.tool_calls.some((tc: any) => tc.id === tm.tool_call_id)) continue;
    }
    sanitized.push(m);
  }
  apiMessages.length = 0;
  sanitized.forEach((m) => apiMessages.push(m));

  while (rounds < MAX_TOOL_ROUNDS) {
    rounds++;

    // Dynamically cap response tokens to stay within TPM
    const inputTokens = apiMessages.reduce((sum, m) => sum + estimateTokens(getMessageContent(m)), 0) + TOOL_DEFS_TOKENS + SYSTEM_PROMPT_TOKENS;
    const dynamicMaxTokens = Math.min(MAX_RESPONSE_TOKENS, Math.max(512, TPM_LIMIT - inputTokens - 200));

    // Call API
    showThinking();
    let response: OpenAI.Chat.Completions.ChatCompletion;
    try {
      response = await client.chat.completions.create({
        model,
        messages: apiMessages,
        temperature,
        top_p: 0.95,
        max_tokens: dynamicMaxTokens,
        tools: TOOL_DEFINITIONS,
        tool_choice: "auto",
        stream: false,
      });
    } catch (err: unknown) {
      clearThinking();
      const errMsg = err instanceof Error ? err.message : String(err);
      // If model hallucinated a tool name, retry without tools to get text response
      if (errMsg.includes("tool call validation") || errMsg.includes("not in request.tools")) {
        try {
          response = await client.chat.completions.create({
            model,
            messages: apiMessages,
            temperature,
            top_p: 0.95,
            max_tokens: dynamicMaxTokens,
            stream: false,
          });
        } catch (retryErr) {
          throw retryErr;
        }
      } else {
        throw err;
      }
    }
    clearThinking();

    // Accumulate usage across rounds
    if (response.usage) {
      totalUsage.promptTokens += response.usage.prompt_tokens;
      totalUsage.completionTokens += response.usage.completion_tokens;
      totalUsage.totalTokens += response.usage.total_tokens;
    }

    const choice = response.choices[0];
    const msg = choice.message;

    // If no tool calls, we're done
    if (!msg.tool_calls || msg.tool_calls.length === 0) {
      const content = msg.content || "";
      if (content) {
        process.stdout.write(formatResponse(content) + "\n");
      }
      return { content, usage: totalUsage };
    }

    // Process tool calls
    apiMessages.push(msg as Message);
    history.push(msg as Message);

    for (const tc of msg.tool_calls) {
      const fnName = tc.function.name;
      let args: Record<string, unknown> = {};
      try {
        args = JSON.parse(tc.function.arguments);
      } catch {
        // malformed args
      }

      const result = await execToolAsync(fnName, args, tc.id);

      const toolMsg: Message = {
        role: "tool",
        tool_call_id: tc.id,
        content: result.error ? `ERROR: ${result.output}` : result.output,
      };
      apiMessages.push(toolMsg);
      history.push(toolMsg);
    }
  }

  console.log(a.yellow("  (max tool rounds reached)"));
  return { content: "", usage: totalUsage };
}

// ---------------------------------------------------------------------------
// ASCII Art Logo
// ---------------------------------------------------------------------------

const FLOW_CODE_LOGO = [
  "  ███████╗██╗      ██████╗ ██╗    ██╗     ██████╗ ██████╗ ██████╗ ███████╗",
  "  ██╔════╝██║     ██╔═══██╗██║    ██║    ██╔════╝██╔═══██╗██╔══██╗██╔════╝",
  "  █████╗  ██║     ██║   ██║██║ █╗ ██║    ██║     ██║   ██║██║  ██║█████╗  ",
  "  ██╔══╝  ██║     ██║   ██║██║███╗██║    ██║     ██║   ██║██║  ██║██╔══╝  ",
  "  ██║     ███████╗╚██████╔╝╚███╔███╔╝    ╚██████╗╚██████╔╝██████╔╝███████╗",
  "  ╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝      ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝",
];

// ---------------------------------------------------------------------------
// Banner + Dashboard
// ---------------------------------------------------------------------------

function printBanner(): void {
  console.clear();
  const mg = (t: string) => `${B}${MAG}${t}${R}`;
  const dm = (t: string) => `${D}${t}${R}`;

  const W = 78;
  const top = dm("\u2554" + "\u2550".repeat(W - 2) + "\u2557");
  const bot = dm("\u255A" + "\u2550".repeat(W - 2) + "\u255D");
  const side = dm("\u2551");

  console.log("");
  console.log(top);
  for (const line of FLOW_CODE_LOGO) {
    console.log(`${side}${mg(line)}${" ".repeat(W - 2 - line.length)}${side}`);
  }
  console.log(`${side}${dm("  v" + VERSION)}${" ".repeat(W - 2 - VERSION.length - 4)}${side}`);
  console.log(bot);
  console.log("");
}

function printDashboard(
  config: Config,
  usage?: { totalTokens: number; totalCost: number }
): void {
  console.clear();
  const provider = PROVIDERS[config.provider || "groq"];
  const model = config.defaultModel || provider.defaultModel;
  const cwd = process.cwd();
  const activity = getRecentActivity();

  const dm = (t: string) => `${D}${t}${R}`;
  const bd = (t: string) => `${B}${t}${R}`;
  const yl = (t: string) => `${YEL}${t}${R}`;
  const cy = (t: string) => `${CYN}${t}${R}`;
  const gr = (t: string) => `${GRN}${t}${R}`;
  const mg = (t: string) => `${MAG}${t}${R}`;

  const W = 62;
  const border = dm("+" + "-".repeat(W - 2) + "+");
  const inner = (left: string, right: string) => {
    const l = left.padEnd(W / 2 - 2);
    const r = right.padEnd(W / 2 - 2);
    return `  ${dm("|")} ${l}${dm("|")} ${r}${dm("|")}`;
  };
  const row = (text: string) =>
    `  ${dm("|")} ${text.padEnd(W - 4)} ${dm("|")}`;

  console.log("");
  console.log(border);
  console.log(inner(`  ${bd(cy("Flow Code"))} ${dm("v" + VERSION)}`, `${bd(yl("Recent activity"))}`));

  const leftLines: string[] = [];
  leftLines.push("");
  leftLines.push(`  ${dm(`${provider.name} ${dm("*")} ${model}`)}`);
  leftLines.push(`  ${dm(cwd)}`);

  if (usage && usage.totalTokens > 0) {
    leftLines.push("");
    leftLines.push(`  ${dm("Session:")} ${bd(String(usage.totalTokens.toLocaleString()))} tokens`);
    leftLines.push(`  ${dm("Cost:")} ${gr("$" + usage.totalCost.toFixed(4))}`);
  }

  const rightLines: string[] = [];
  if (activity.length > 0) {
    for (const entry of activity.slice(0, 5)) {
      const ago = timeAgo(entry.time).padEnd(8);
      const action = truncate(entry.action, 30);
      rightLines.push(`  ${dm(ago)} ${action}`);
    }
    if (activity.length > 5) {
      rightLines.push(`  ${dm("...")} ${dm("/resume for more")}`);
    }
  } else {
    rightLines.push(`  ${dm("No recent activity")}`);
  }

  const maxRows = Math.max(leftLines.length, rightLines.length);
  for (let i = 0; i < maxRows; i++) {
    const left = (leftLines[i] || "").padEnd(W / 2 - 2);
    const right = (rightLines[i] || "").padEnd(W / 2 - 2);
    console.log(`  ${dm("|")} ${left}${dm("|")} ${right}${dm("|")}`);
  }

  console.log(border);
  console.log("");
  console.log(
    `  ${dm("/cmds")} commands  ${dm("/help")} help  ${dm("exit")} quit`
  );
  console.log("");
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

async function setup(): Promise<{
  client: OpenAI;
  model: string;
  intensity: Intensity;
  provider: Provider;
}> {
  const config = loadConfig();

  let provider: Provider = config.provider || "groq";
  if (!config.apiKey) {
    console.log(a.bold("  [1/4] Select AI Provider:"));
    console.log(`    ${a.dim("[1]")} Groq    -- Ultra-fast inference, open-source models`);
    console.log(`    ${a.dim("[2]")} Cerebras -- Wafer-scale AI, blazing speed`);
    const p = await ask("  Select (1-2): ");
    provider = p === "2" ? "cerebras" : "groq";
  }

  if (!config.apiKey) {
    console.log(a.dim(`\n  [2/4] Enter your ${PROVIDERS[provider].name} API Key:`));
    const key = await ask("  API Key: ");
    if (!key || key.length < 10) {
      console.error(a.red("  Invalid key. Exiting."));
      process.exit(1);
    }
    config.apiKey = key;
    config.provider = provider;
    saveConfig(config);
  } else {
    provider = config.provider || "groq";
  }

  const client = new OpenAI({
    baseURL: PROVIDERS[provider].baseURL,
    apiKey: config.apiKey,
  });

  let model = sanitizeModelId(
    config.defaultModel || PROVIDERS[provider].defaultModel
  );

  try {
    console.log(a.dim(`\n  [3/4] Fetching ${PROVIDERS[provider].name} models...`));
    const res = await client.models.list();
    const models = res.data
      .map((m) => m.id)
      .filter((id) => PROVIDERS[provider].modelFilter.test(id))
      .sort();
    if (models.length > 0) {
      console.log(a.bold("  Available Models:"));
      models.forEach((m, i) => console.log(`    ${a.dim(`[${i}]`)} ${m}`));
      const choice = await ask(`\n  Select model (0-${models.length - 1}, Enter for default): `);
      if (choice !== "") {
        const idx = parseInt(choice, 10);
        if (!isNaN(idx) && idx >= 0 && idx < models.length) {
          model = sanitizeModelId(models[idx]);
        }
      }
    }
  } catch {
    console.log(a.dim("  Could not fetch models. Using default."));
  }

  console.log(a.bold("\n  [4/4] Processing Mode:"));
  console.log(`    ${a.dim("[1]")} Low    -- Fast, single-file patches`);
  console.log(`    ${a.dim("[2]")} Medium -- Standard refactoring`);
  console.log(`    ${a.dim("[3]")} High   -- Deep scanning & DevOps`);

  const iMap: Record<string, Intensity> = { "1": "low", "2": "medium", "3": "high" };
  let intensity: Intensity = config.intensity || "medium";
  const choice = await ask("  Select mode (1-3): ");
  if (iMap[choice]) intensity = iMap[choice];

  config.defaultModel = model;
  config.intensity = intensity;
  config.provider = provider;
  saveConfig(config);

  console.log("");
  console.log(a.green("  Setup complete."));
  console.log(a.dim(`  ${PROVIDERS[provider].name} | ${model} | ${intensity.toUpperCase()}`));
  console.log("");

  // Offer to resume after setup
  if (hasSavedSession()) {
    const session = loadSession();
    if (session) {
      console.log(a.dim(`  Last session: ${timeAgo(session.timestamp)}`));
      const resume = await ask(a.bold("  Resume last conversation? (y/N): "));
      if (resume.toLowerCase() === "y" || resume.toLowerCase() === "yes") {
        // Will be handled in main after setup returns
        return { client, model, intensity, provider };
      }
    }
  }

  return { client, model, intensity, provider };
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

async function run(
  client: OpenAI,
  model: string,
  intensity: Intensity,
  provider: Provider,
  resume: boolean = false
): Promise<void> {
  const state: SessionState = { client, model, intensity, provider };
  const history: Message[] = [
    { role: "system", content: getSystemPrompt(state.intensity, process.cwd(), state.provider) },
  ];

  // Session-level usage tracking
  const usage = { totalTokens: 0, totalCost: 0 };

  const updateSystem = (): void => {
    history[0] = {
      role: "system",
      content: getSystemPrompt(state.intensity, process.cwd(), state.provider),
    };
  };

  // Restore session if resuming
  if (resume) {
    const result = restoreSession(history);
    if (result) {
      updateSystem();
      console.log(a.green(`  Resumed ${result.count} messages from ${timeAgo(result.timestamp)}.`));
      console.log(a.dim(`  Directory: ${result.cwd}\n`));
    }
  }

  while (true) {
    const input = await readInput();
    if (!input) continue;

    // ── Exit ──
    if (input === "exit" || input === "quit") {
      console.log(a.dim("\n  Goodbye.\n"));
      rl.close();
      process.exit(0);
    }

    // ── /help ──
    if (input === "/help" || input === "/cmds") {
      console.log(
        [
          "",
          a.bold(a.cyan("  Flow Code Commands")),
          a.dim("  " + "-".repeat(40)),
          "",
          `  ${a.cyan("cd <path>")}            Switch directory`,
          `  ${a.cyan("/search <query>")}      Search the web`,
          `  ${a.cyan("/fetch <url>")}         Fetch URL content`,
          `  ${a.cyan("/resume")}              Resume last conversation`,
          `  ${a.cyan("/clear")}               Reset conversation`,
          `  ${a.cyan("/compact")}             Trim context window`,
          `  ${a.cyan("/models")}              Re-select model`,
          `  ${a.cyan("/provider")}            Switch Groq / Cerebras`,
          `  ${a.cyan("/settings")}            Configure preferences`,
          `  ${a.cyan("/status")}              Show usage stats`,
          `  ${a.cyan("exit")}                 Quit`,
          "",
          a.dim("  The agent uses tools autonomously to complete tasks."),
          a.dim("  Multiline: end a line with \\ to continue."),
          "",
        ].join("\n")
      );
      continue;
    }

    // ── /clear ──
    if (input === "/clear") {
      history.length = 1;
      updateSystem();
      console.log(a.green("  Conversation cleared.\n"));
      continue;
    }

    // ── /resume ──
    if (input === "/resume") {
      const result = restoreSession(history);
      if (!result) {
        console.log(a.yellow("  No saved session.\n"));
        continue;
      }
      updateSystem();
      console.log(a.green(`  Resumed ${result.count} messages from ${timeAgo(result.timestamp)}.`));
      console.log(a.dim(`  Directory: ${result.cwd}\n`));
      continue;
    }

    // ── /compact ──
    if (input === "/compact") {
      const before = history.length;
      const compacted = trimHistory(history);
      history.length = 0;
      history.push(...compacted);
      console.log(a.green(`  Compacted: ${before} -> ${history.length} messages.\n`));
      continue;
    }

    // ── /status ──
    if (input === "/status") {
      const totalUsed = history.reduce(
        (acc, msg) => acc + estimateTokens(getMessageContent(msg)),
        0
      );
      console.log(
        [
          "",
          a.bold("  Session:"),
          `    Provider:   ${PROVIDERS[state.provider].name}`,
          `    Model:      ${state.model}`,
          `    Intensity:  ${state.intensity.toUpperCase()}`,
          `    Messages:   ${history.length}`,
          `    Tokens:     ${totalUsed.toLocaleString()} / ${CONTEXT_WINDOW.toLocaleString()}`,
          renderUsageBar(totalUsed, CONTEXT_WINDOW),
          "",
        ].join("\n")
      );
      continue;
    }

    // ── /models ──
    if (input === "/models") {
      try {
        const res = await state.client.models.list();
        const models = res.data
          .map((m) => m.id)
          .filter((id) => PROVIDERS[state.provider].modelFilter.test(id))
          .sort();
        console.log(a.bold(`\n  ${PROVIDERS[state.provider].name} Models:`));
        models.forEach((m, i) => console.log(`    ${a.dim(`[${i}]`)} ${m}`));
        const choice = await ask(`\n  Select (0-${models.length - 1}): `);
        const idx = parseInt(choice, 10);
        if (!isNaN(idx) && idx >= 0 && idx < models.length) {
          state.model = sanitizeModelId(models[idx]);
          console.log(a.green(`  Model: ${state.model}\n`));
          const cfg = loadConfig();
          cfg.defaultModel = state.model;
          saveConfig(cfg);
          updateSystem();
        }
      } catch {
        console.log(a.red("  Could not fetch models."));
      }
      continue;
    }

    // ── /provider ──
    if (input === "/provider") {
      console.log(a.bold("  Switch provider:"));
      console.log(`    ${a.dim("[1]")} Groq`);
      console.log(`    ${a.dim("[2]")} Cerebras`);
      const p = await ask("  Select (1-2): ");
      const np: Provider = p === "2" ? "cerebras" : "groq";
      if (np === state.provider) {
        console.log(a.yellow(`  Already using ${PROVIDERS[state.provider].name}.\n`));
        continue;
      }
      console.log(a.dim(`  Enter ${PROVIDERS[np].name} API Key:`));
      const key = await ask("  API Key: ");
      if (!key || key.length < MIN_API_KEY_LENGTH) {
        console.log(a.red("  Invalid key.\n"));
        continue;
      }
      const nc = new OpenAI({ baseURL: PROVIDERS[np].baseURL, apiKey: key });
      try {
        console.log(a.dim(`  Connecting to ${PROVIDERS[np].name}...`));
        await nc.models.list();
        state.client = nc;
        state.provider = np;
        state.model = PROVIDERS[np].defaultModel;
        const cfg = loadConfig();
        cfg.provider = np;
        cfg.apiKey = key;
        cfg.defaultModel = state.model;
        saveConfig(cfg);
        updateSystem();
        console.log(a.green(`  Switched to ${PROVIDERS[np].name} | ${state.model}\n`));
        logActivity(`Switched to ${PROVIDERS[np].name}`);
      } catch (err: unknown) {
        console.error(a.red(`  ${PROVIDERS[np].name}: ${err instanceof Error ? err.message : "Failed"}\n`));
      }
      continue;
    }

    // ── /settings ──
    if (input === "/settings") {
      const cfg = loadConfig();
      console.log(
        [
          "",
          a.bold("  Settings:"),
          `    ${a.dim("[1]")} Change API Key`,
          `    ${a.dim("[2]")} Switch Provider (${PROVIDERS[cfg.provider || "groq"].name})`,
          `    ${a.dim("[3]")} Change Model (${cfg.defaultModel || "default"})`,
          `    ${a.dim("[4]")} Change Intensity (${(cfg.intensity || "medium").toUpperCase()})`,
          `    ${a.dim("[5]")} View Config`,
          `    ${a.dim("[6]")} Reset All`,
          "",
        ].join("\n")
      );
      const sel = await ask("  Select (1-6): ");
      switch (sel) {
        case "1": {
          const key = await ask("  New API Key: ");
          if (key && key.length >= MIN_API_KEY_LENGTH) {
            cfg.apiKey = key;
            saveConfig(cfg);
            console.log(a.green("  API key updated.\n"));
          } else {
            console.log(a.yellow(`  Key too short (min ${MIN_API_KEY_LENGTH} chars).\n`));
          }
          break;
        }
        case "2": {
          console.log(`    ${a.dim("[1]")} Groq`);
          console.log(`    ${a.dim("[2]")} Cerebras`);
          const p = await ask("  Select: ");
          cfg.provider = p === "2" ? "cerebras" : "groq";
          cfg.apiKey = "";
          cfg.defaultModel = PROVIDERS[cfg.provider || "groq"].defaultModel;
          saveConfig(cfg);
          console.log(a.green(`  Provider: ${PROVIDERS[cfg.provider || "groq"].name}. Restart to apply.\n`));
          break;
        }
        case "3": {
          try {
            const cl = new OpenAI({
              baseURL: PROVIDERS[cfg.provider || "groq"].baseURL,
              apiKey: cfg.apiKey,
            });
            const res = await cl.models.list();
            const models = res.data
              .map((m) => m.id)
              .filter((id) => PROVIDERS[cfg.provider || "groq"].modelFilter.test(id))
              .sort();
            console.log(a.bold("\n  Models:"));
            models.forEach((m, i) => console.log(`    ${a.dim(`[${i}]`)} ${m}`));
            const choice = await ask(`\n  Select (0-${models.length - 1}): `);
            const idx = parseInt(choice, 10);
            if (!isNaN(idx) && idx >= 0 && idx < models.length) {
              cfg.defaultModel = models[idx];
              saveConfig(cfg);
              console.log(a.green(`  Model: ${models[idx]}\n`));
            }
          } catch {
            console.log(a.red("  Could not fetch models."));
          }
          break;
        }
        case "4": {
          console.log(`    ${a.dim("[1]")} Low`);
          console.log(`    ${a.dim("[2]")} Medium`);
          console.log(`    ${a.dim("[3]")} High`);
          const ic = await ask("  Select: ");
          const map: Record<string, Intensity> = { "1": "low", "2": "medium", "3": "high" };
          if (map[ic]) {
            cfg.intensity = map[ic];
            saveConfig(cfg);
            console.log(a.green(`  Intensity: ${cfg.intensity.toUpperCase()}\n`));
          }
          break;
        }
        case "5": {
          console.log("");
          console.log(a.bold("  Current config:"));
          console.log(`    Provider:   ${PROVIDERS[cfg.provider || "groq"].name}`);
          console.log(`    Model:      ${cfg.defaultModel || "default"}`);
          console.log(`    Intensity:  ${(cfg.intensity || "medium").toUpperCase()}`);
          console.log(`    API Key:    ${cfg.apiKey ? cfg.apiKey.slice(0, 3) + "..." + cfg.apiKey.slice(-2) : "not set"}`);
          console.log(`    Config:     ${CONFIG_PATH}`);
          console.log("");
          break;
        }
        case "6": {
          const confirm = await ask(a.yellow("  Are you sure? (yes/no): "));
          if (confirm.toLowerCase() === "yes") {
            try { fs.unlinkSync(CONFIG_PATH); } catch { /* ok */ }
            console.log(a.green("  Config reset. Restart to apply.\n"));
          }
          break;
        }
      }
      continue;
    }

    // ── /search (manual) ──
    if (input.startsWith("/search ")) {
      const query = input.slice(8).trim();
      if (!query) { console.log(a.yellow("  Usage: /search <query>")); continue; }
      console.log(a.dim(`  Searching: ${query}...`));
      const results = await searchWeb(query);
      if (results.length === 0) { console.log(a.dim("  No results.")); continue; }
      for (const [i, r] of results.entries()) {
        console.log(`  ${a.bold(a.green(`${i + 1}.`))} ${a.bold(r.title)}`);
        console.log(`     ${a.dim(r.snippet)}`);
        console.log(`     ${a.dim(r.url)}`);
        console.log("");
      }
      continue;
    }

    // ── /fetch (manual) ──
    if (input.startsWith("/fetch ")) {
      const url = input.slice(7).trim();
      if (!url || !url.startsWith("http")) { console.log(a.yellow("  Usage: /fetch <https://url>")); continue; }
      console.log(a.dim(`  Fetching: ${url}...`));
      const content = await fetchUrlContent(url);
      console.log(`\n${a.dim(content.slice(0, 3000))}`);
      if (content.length > 3000) console.log(a.dim(`\n  ... (${content.length} chars total)`));
      console.log("");
      continue;
    }

    // ── cd ──
    if (input === "cd" || input === "cd ~" || input === "cd $HOME") {
      const home = os.homedir();
      try {
        process.chdir(home);
        console.log(a.green(`  ${home}\n`));
        const tree = scanDirectory(home, 3);
        if (tree) { console.log(a.dim("  Directory:")); console.log(a.dim(tree) + "\n"); }
        updateSystem();
      } catch (err: unknown) {
        console.error(a.red(`  ${err instanceof Error ? err.message : String(err)}`));
      }
      continue;
    }

    if (input.startsWith("cd ")) {
      const raw = input.slice(3).replace(/^['"]|['"]$/g, "").trim();
      if (!raw || raw.includes("\0")) { console.error(a.red("  Invalid path.")); continue; }
      try {
        const target = path.resolve(raw);
        process.chdir(target);
        const cwd = process.cwd();
        console.log(a.green(`  ${cwd}\n`));
        const tree = scanDirectory(cwd, 3);
        if (tree) { console.log(a.dim("  Directory:")); console.log(a.dim(tree) + "\n"); }
        updateSystem();
      } catch (err: unknown) {
        console.error(a.red(`  ${err instanceof Error ? err.message : String(err)}`));
      }
      continue;
    }

    // ── Unknown command ──
    if (input.startsWith("/")) {
      console.log(a.yellow(`  Unknown command: ${input.split(" ")[0]}. Type /help for commands.\n`));
      continue;
    }

    // ── Send to agent (with tool loop) ──
    updateSystem();

    history.push({ role: "user", content: input });
    const trimmed = trimHistory(history);
    const historyLenBefore = history.length;
    const temp =
      state.intensity === "low" ? 0.0 : state.intensity === "medium" ? 0.1 : 0.2;

    try {
      const { content: reply, usage: roundUsage } = await streamWithTools(
        state.client,
        state.model,
        [...trimmed],
        temp,
        history
      );

      // Track session usage
      if (roundUsage.totalTokens > 0) {
        usage.totalTokens += roundUsage.totalTokens;
        usage.totalCost += roundUsage.totalTokens * COST_PER_TOKEN;
        printUsage(roundUsage, history);
      }

      if (reply) {
        history.push({ role: "assistant", content: reply });
      }

      saveSession(history, state.model, state.provider, state.intensity);
    } catch (err: unknown) {
      clearThinking();
      history.length = historyLenBefore;
      const msg = err instanceof Error ? err.message : String(err);
      if (msg.includes("401") || msg.toLowerCase().includes("invalid")) {
        console.error(a.red(`  Invalid API key for ${PROVIDERS[state.provider].name}. Run /settings to update.`));
      } else if (msg.includes("404") || msg.includes("does not exist")) {
        console.error(a.red(`  Model '${state.model}' not found.`));
        console.log(a.dim("  Type /models to re-select."));
        state.model = PROVIDERS[state.provider].defaultModel;
        const cfg = loadConfig();
        cfg.defaultModel = state.model;
        saveConfig(cfg);
      } else if (msg.includes("429")) {
        console.error(a.yellow("  Rate limited. Wait a moment."));
      } else if (msg.includes("503")) {
        console.error(a.yellow("  Model overloaded. Try again."));
      } else if (msg.includes("413") || msg.includes("too large")) {
        console.error(a.yellow("  Request too large. Try a shorter message."));
      } else {
        console.error(a.red(`  ${msg}`));
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

function main(): void {
  rl.on("close", () => process.exit(0));
  process.on("SIGINT", () => {
    clearThinking();
    console.log(a.dim("\n\n  Goodbye.\n"));
    rl.close();
    process.exit(0);
  });
  process.on("SIGTERM", () => { rl.close(); process.exit(0); });
  process.on("unhandledRejection", (err) => console.error(a.red(`\n  Unhandled: ${err}`)));
  process.on("uncaughtException", (err) => {
    console.error(a.red(`\n  Uncaught: ${err.message}`));
    process.exit(1);
  });

  printBanner();

  setup()
    .then(async ({ client, model, intensity, provider }) => {
      // Ask for project directory
      console.log(a.dim(`  Working directory: ${process.cwd()}`));
      const dirInput = await ask(a.bold("  Project directory (Enter to skip): "));
      if (dirInput.trim()) {
        const target = path.resolve(dirInput.trim().replace(/^['"]|['"]$/g, ""));
        try {
          process.chdir(target);
          console.log(a.green(`  Switched to: ${process.cwd()}\n`));
        } catch {
          console.log(a.yellow(`  Could not open '${target}'. Using current.\n`));
        }
      }

      // Check for saved session and offer resume
      let resumedSession = false;
      if (hasSavedSession()) {
        const session = loadSession();
        if (session) {
          const resume = await ask(
            a.bold(`  Resume ${timeAgo(session.timestamp)} session? (y/N): `)
          );
          if (resume.toLowerCase() === "y" || resume.toLowerCase() === "yes") {
            resumedSession = true;
          }
        }
      }

      const config = loadConfig();
      const usage = { totalTokens: 0, totalCost: 0 };
      printDashboard(config, usage);

      return run(client, model, intensity, provider, resumedSession);
    })
    .catch((err) => {
      console.error(a.red(`\n  Fatal: ${err.message || err}`));
      process.exit(1);
    });
}

main();
