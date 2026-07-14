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
const MAX_HISTORY_TOKENS = 8000;
const MAX_RESPONSE_TOKENS = 4096;
const SHELL_TIMEOUT_MS = 60_000;
const MAX_SHELL_OUTPUT = 4000;
const MAX_WEB_CONTENT = 12000;
const MAX_FILE_READ = 50_000;
const MAX_TOOL_ROUNDS = 15;

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
type Role = "system" | "user" | "assistant";
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
  history: Array<{ role: string; content: string }>;
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
    defaultModel: "llama-3.3-70b-versatile",
    modelFilter: /llama|mixtral|qwen|gemma|deepseek/i,
  },
  cerebras: {
    name: "Cerebras",
    baseURL: "https://api.cerebras.ai/v1",
    defaultModel: "llama-3.3-70b",
    modelFilter: /llama|qwen/i,
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
  return rel !== "" && !rel.startsWith("..");
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
      history: history.map((m) => ({
        role: m.role,
        content: getMessageContent(m),
      })),
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
        return data as SessionData;
      }
    }
  } catch {
    // corrupted
  }
  return null;
}

function hasSavedSession(): boolean {
  return loadSession() !== null;
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
      if (!isPathInside(fp, cwd) && !fp.startsWith(cwd)) {
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
        content = content.replace(oldText, newText);
        fs.writeFileSync(fp, content, "utf-8");
        const rel = path.relative(cwd, fp);
        logActivity(`Edited: ${rel}`);
        return { tool_call_id: toolCallId, output: `File edited: ${rel}` };
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
  const tree = scanDirectory(cwd, 1);

  const lines = [
    `You are Flow Code, an autonomous coding agent. You have tools to read, write, edit files, run commands, and search the web. Use them — do NOT just describe what to do.`,
    `Provider: ${pname} | CWD: ${cwd}`,
  ];

  if (tree) lines.push(`Current files:\n${tree}`);

  lines.push(
    "",
    "WORKFLOW:",
    "1. read_file to understand existing code before any change.",
    "2. write_file for new files. edit_file for surgical changes to existing files.",
    "3. run_command to install deps, build, test, start servers.",
    "4. list_directory to explore the project structure.",
    "5. search_web / fetch_url when you need current docs or solutions.",
    "",
    "TOOL USAGE RULES:",
    "- ALWAYS read a file before editing it. Never guess contents.",
    "- write_file: provide the FULL file content, not a snippet.",
    "- edit_file: provide the EXACT old_text (copy from read_file output).",
    "- run_command: use the exact command. Check package.json scripts first.",
    "- Do NOT output code blocks for files — use write_file tool instead.",
    "- Do NOT describe changes — make them with tools.",
    "",
    "CODE QUALITY:",
    "- TypeScript: no any, use interfaces, async/await, optional chaining.",
    "- React/Next.js: functional components, hooks, App Router, Tailwind.",
    "- HTML/CSS: semantic elements, responsive, flexbox/grid.",
    "- Python: type hints, PEP 8, f-strings, pathlib.",
    "- Bash: set -euo pipefail, quoted variables.",
    "- Always write complete, production-ready files.",
  );

  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// History management
// ---------------------------------------------------------------------------

function trimHistory(history: Message[]): Message[] {
  let total = 0;
  const result: Message[] = [];
  if (history.length > 0 && history[0].role === "system") {
    result.push(history[0]);
    total += estimateTokens(getMessageContent(history[0]));
  }
  for (let i = history.length - 1; i >= 1; i--) {
    const tokens = estimateTokens(getMessageContent(history[i]));
    if (total + tokens > MAX_HISTORY_TOKENS) break;
    total += tokens;
    result.splice(1, 0, history[i]);
  }
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
  return a.cyan(`flow-code [${path.basename(process.cwd())}] > `);
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
): Promise<string> {
  let rounds = 0;

  while (rounds < MAX_TOOL_ROUNDS) {
    rounds++;

    // Call API
    showThinking();
    let response: OpenAI.Chat.Completions.ChatCompletion;
    try {
      response = await client.chat.completions.create({
        model,
        messages,
        temperature,
        top_p: 0.95,
        max_tokens: MAX_RESPONSE_TOKENS,
        tools: TOOL_DEFINITIONS,
        tool_choice: "auto",
        stream: false,
      });
    } catch (err: unknown) {
      clearThinking();
      throw err;
    }
    clearThinking();

    const choice = response.choices[0];
    const msg = choice.message;

    // If no tool calls, we're done — stream the text
    if (!msg.tool_calls || msg.tool_calls.length === 0) {
      const content = msg.content || "";
      if (content) {
        // Re-stream for display
        process.stdout.write(formatResponse(content) + "\n");
      }
      return content;
    }

    // Process tool calls
    messages.push(msg as Message);
    history.push(msg as Message);

    for (const tc of msg.tool_calls) {
      const fnName = tc.function.name;
      let args: Record<string, unknown> = {};
      try {
        args = JSON.parse(tc.function.arguments);
      } catch {
        // malformed args
      }

      // Execute tool (showTool is called inside execToolAsync)
      const result = await execToolAsync(fnName, args, tc.id);

      // Add tool result to messages
      const toolMsg: Message = {
        role: "tool",
        tool_call_id: tc.id,
        content: result.error ? `ERROR: ${result.output}` : result.output,
      };
      messages.push(toolMsg);
      history.push(toolMsg);
    }

    // Loop again — model may want to call more tools
  }

  console.log(a.yellow("  (max tool rounds reached)"));
  return "";
}

// ---------------------------------------------------------------------------
// Banner + Dashboard
// ---------------------------------------------------------------------------

function printBanner(): void {
  console.clear();
  const ac = (t: string) => `${B}${CYN}${t}${R}`;
  const dm = (t: string) => `${D}${t}${R}`;
  const sp = (t: string) => `${B}${MAG}${t}${R}`;

  console.log("");
  console.log(`  ${sp("(*)")} ${ac("Flow Code")} ${dm("v" + VERSION)}`);
  console.log(`       ${dm("autonomous coding agent")}`);
  console.log(`       ${dm("-------------------------")}`);
  console.log("");
}

function printDashboard(config: Config): void {
  console.clear();
  const provider = PROVIDERS[config.provider || "groq"];
  const model = config.defaultModel || provider.defaultModel;
  const cwd = process.cwd();
  const activity = getRecentActivity();
  const dm = (t: string) => `${D}${t}${R}`;
  const bd = (t: string) => `${B}${t}${R}`;
  const yl = (t: string) => `${YEL}${t}${R}`;
  const w = 50;
  const line = dm("-".repeat(w));

  console.log("");
  console.log(
    `  ${dm("---")} ${bd(`${CYN}Flow Code${R}`)} ${dm(`v${VERSION} ${"-".repeat(w - 26)}`)}`
  );
  console.log(line);
  console.log(`  ${dm(`${provider.name} | ${model}`)}`);
  console.log(`  ${dm(cwd)}`);
  console.log(`  ${dm("Tools: read, write, edit, run, list, search, fetch")}`);

  if (activity.length > 0) {
    console.log("");
    console.log(`  ${bd(yl("Recent"))}`);
    for (const entry of activity) {
      const ago = timeAgo(entry.time);
      const action = truncate(entry.action, 38);
      console.log(`  ${dm(ago.padEnd(10))} ${action}`);
    }
  }

  console.log(line);
  console.log(
    `  ${dm("/cmds")} commands  ${dm("/help")} help  ${dm("exit")} quit`
  );
  console.log(line);
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

  return { client, model, intensity, provider };
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

async function run(
  client: OpenAI,
  model: string,
  intensity: Intensity,
  provider: Provider
): Promise<void> {
  const state: SessionState = { client, model, intensity, provider };
  const history: Message[] = [
    { role: "system", content: getSystemPrompt(state.intensity, process.cwd(), state.provider) },
  ];

  const updateSystem = (): void => {
    history[0] = {
      role: "system",
      content: getSystemPrompt(state.intensity, process.cwd(), state.provider),
    };
  };

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
    if (input === "/help") {
      console.log(
        [
          "",
          a.bold("  Commands:"),
          `    ${a.dim("cd <path>")}        Switch directory`,
          `    ${a.dim("/search <query>")}  Search the web`,
          `    ${a.dim("/fetch <url>")}     Fetch URL content`,
          `    ${a.dim("/settings")}        Configure preferences`,
          `    ${a.dim("/models")}          Re-select model`,
          `    ${a.dim("/provider")}        Switch Groq / Cerebras`,
          `    ${a.dim("/resume")}          Resume last conversation`,
          `    ${a.dim("/clear")}           Reset conversation`,
          `    ${a.dim("/compact")}         Trim context`,
          `    ${a.dim("/cmds")}            List all commands`,
          `    ${a.dim("/status")}          Show usage stats`,
          `    ${a.dim("exit")}             Quit`,
          "",
          a.dim("  The agent has tools: read, write, edit, run, list, search, fetch."),
          a.dim("  It will autonomously use them to complete your task."),
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
      const session = loadSession();
      if (!session) {
        console.log(a.yellow("  No saved session.\n"));
        continue;
      }
      history.length = 0;
      for (const msg of session.history) {
        history.push({ role: msg.role as Role, content: msg.content });
      }
      if (session.cwd && fs.existsSync(session.cwd)) {
        process.chdir(session.cwd);
        updateSystem();
      }
      const count = history.filter((m) => m.role === "user" || m.role === "assistant").length;
      console.log(a.green(`  Resumed ${count} messages from ${timeAgo(session.timestamp)}.\n`));
      continue;
    }

    // ── /cmds ──
    if (input === "/cmds") {
      console.log(
        [
          "",
          a.bold(a.cyan("  Flow Code Commands")),
          a.dim("  --------------------------------------------"),
          "",
          a.bold("  Navigation:"),
          `    ${a.cyan("cd <path>")}            Switch working directory`,
          `    ${a.cyan("cd ..")}                Go up one directory`,
          `    ${a.cyan("cd ~")}                 Go to home directory`,
          "",
          a.bold("  Web & Search:"),
          `    ${a.cyan("/search <query>")}      Search the web (also auto-detected)`,
          `    ${a.cyan("/fetch <url>")}         Fetch and read URL content`,
          "",
          a.bold("  Session:"),
          `    ${a.cyan("/resume")}              Resume last conversation`,
          `    ${a.cyan("/clear")}               Reset conversation history`,
          `    ${a.cyan("/compact")}             Trim history to fit context`,
          "",
          a.bold("  Configuration:"),
          `    ${a.cyan("/settings")}            Open settings menu`,
          `    ${a.cyan("/models")}              Re-select your model`,
          `    ${a.cyan("/provider")}            Switch Groq / Cerebras`,
          "",
          a.bold("  Info:"),
          `    ${a.cyan("/status")}              Show provider, model, tokens`,
          `    ${a.cyan("/cmds")}                Show this list`,
          `    ${a.cyan("/help")}                Condensed help`,
          `    ${a.cyan("exit")}                 Quit`,
          "",
          a.dim("  Multiline: end a line with \\ to continue."),
          "",
        ].join("\n")
      );
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
      if (!key || key.length < 10) {
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
          if (key && key.length > 10) {
            cfg.apiKey = key;
            saveConfig(cfg);
            console.log(a.green("  API key updated.\n"));
          } else {
            console.log(a.yellow("  Key too short.\n"));
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
          console.log(`    API Key:    ${cfg.apiKey ? cfg.apiKey.slice(0, 6) + "..." + cfg.apiKey.slice(-4) : "not set"}`);
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

    // ── Send to agent (with tool loop) ──
    updateSystem();

    history.push({ role: "user", content: input });
    const trimmed = trimHistory(history);
    const temp =
      state.intensity === "low" ? 0.0 : state.intensity === "medium" ? 0.2 : 0.4;

    try {
      const reply = await streamWithTools(
        state.client,
        state.model,
        [...trimmed], // don't mutate trimmed
        temp,
        history
      );

      if (reply) {
        history.push({ role: "assistant", content: reply });
      }

      saveSession(history, state.model, state.provider, state.intensity);
    } catch (err: unknown) {
      clearThinking();
      const msg = err instanceof Error ? err.message : String(err);
      if (msg.includes("401") || msg.toLowerCase().includes("invalid")) {
        console.error(a.red(`  Invalid API key for ${PROVIDERS[state.provider].name}. Delete ~/.flow-code-config and restart.`));
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
      } else {
        console.error(a.red(`  ${msg}`));
      }
      history.pop();
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

      const config = loadConfig();
      printDashboard(config);

      if (hasSavedSession()) {
        const session = loadSession();
        if (session) {
          console.log(a.dim(`  Last session: ${timeAgo(session.timestamp)} -- type /resume`));
        }
      }
      console.log(a.green("  Ready! Describe what you want to build.\n"));

      logActivity("Started session");
      return run(client, model, intensity, provider);
    })
    .catch((err) => {
      console.error(a.red(`\n  Fatal: ${err.message || err}`));
      process.exit(1);
    });
}

main();
