#!/usr/bin/env node

import OpenAI from "openai";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";
import * as readline from "readline";
import { spawn } from "child_process";
import { createInterface } from "readline";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const VERSION = "1.2.0";
const HOME = os.homedir();
const CONFIG_PATH = path.join(HOME, ".flow-code-config");
const SESSION_PATH = path.join(HOME, ".flow-code-session");
const ACTIVITY_PATH = path.join(HOME, ".flow-code-activity");
const CONTEXT_WINDOW = 128000;
const MAX_HISTORY_TOKENS = 8000;
const MAX_RESPONSE_TOKENS = 4096;
const SHELL_TIMEOUT_MS = 60_000;
const MAX_SHELL_OUTPUT = 2000;
const MAX_WEB_CONTENT = 12000;

// ANSI
const ESC = "\x1b[";
const RESET = `${ESC}0m`;
const BOLD = `${ESC}1m`;
const DIM = `${ESC}2m`;
const RED = `${ESC}31m`;
const GREEN = `${ESC}32m`;
const YELLOW = `${ESC}33m`;
const BLUE = `${ESC}34m`;
const MAGENTA = `${ESC}35m`;
const CYAN = `${ESC}36m`;

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

// ---------------------------------------------------------------------------
// Provider configuration
// ---------------------------------------------------------------------------

const PROVIDERS: Record<
  Provider,
  {
    name: string;
    baseURL: string;
    defaultModel: string;
    modelFilter: RegExp;
  }
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
// ANSI helpers
// ---------------------------------------------------------------------------

const ansi = {
  bold: (t: string) => `${BOLD}${t}${RESET}`,
  dim: (t: string) => `${DIM}${t}${RESET}`,
  red: (t: string) => `${RED}${t}${RESET}`,
  green: (t: string) => `${GREEN}${t}${RESET}`,
  yellow: (t: string) => `${YELLOW}${t}${RESET}`,
  blue: (t: string) => `${CYAN}${t}${RESET}`,
  cyan: (t: string) => `${CYAN}${t}${RESET}`,
  magenta: (t: string) => `${MAGENTA}${t}${RESET}`,
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
  // Resolve and normalize, strip null bytes
  return path.resolve(p.replace(/\0/g, ""));
}

function isPathInside(child: string, parent: string): boolean {
  const rel = path.relative(parent, child);
  return rel !== "" && !rel.startsWith("..");
}

// ---------------------------------------------------------------------------
// Context window display
// ---------------------------------------------------------------------------

function renderUsageBar(used: number, total: number): string {
  const pct = Math.min(used / total, 1);
  const filled = Math.round(pct * 20);
  const empty = 20 - filled;
  const bar =
    GREEN +
    "\u2588".repeat(filled) +
    RESET +
    DIM +
    "\u2591".repeat(empty) +
    RESET;
  const pctStr = Math.round(pct * 100) + "%";
  return `  ${ansi.dim("Context:")} ${bar} ${ansi.dim(`${used.toLocaleString()} / ${total.toLocaleString()} tokens (${pctStr})`)}`;
}

function printUsage(usage: UsageInfo, history: Message[]): void {
  const totalUsed = history.reduce(
    (acc, msg) => acc + estimateTokens(getMessageContent(msg)),
    0
  );
  console.log("");
  console.log(
    `  ${ansi.dim("Tokens:")} ${ansi.bold(String(usage.promptTokens))} prompt + ${ansi.bold(String(usage.completionTokens))} completion = ${ansi.bold(String(usage.totalTokens))} total`
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
    // non-critical
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
      // Validate structure
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
    // corrupted file, ignore
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
      apiKey:
        typeof parsed.apiKey === "string" ? parsed.apiKey : "",
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
    console.error(ansi.red("  Failed to save config."));
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
// Web search (DuckDuckGo HTML)
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

function formatSearchResults(results: SearchResult[], query: string): string {
  if (results.length === 0) return `No results for "${query}".`;

  const lines = [
    "",
    `  ${ansi.bold(ansi.blue("Search Results"))} ${ansi.dim(`"${query}"`)}`,
    "",
  ];

  for (const [i, r] of results.entries()) {
    lines.push(
      `  ${ansi.bold(ansi.green(`${i + 1}.`))} ${ansi.bold(r.title)}`
    );
    lines.push(`     ${ansi.dim(r.snippet)}`);
    lines.push(`     ${ansi.dim(r.url)}`);
    lines.push("");
  }

  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// Fetch URL
// ---------------------------------------------------------------------------

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
    return `Failed: ${err instanceof Error ? err.message : "unknown error"}`;
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
  const top = `${DIM}+-- ${label} ${"-".repeat(Math.max(0, w - label.length - 3))}+${RESET}`;
  const bottom = `${DIM}+${"-".repeat(w)}+${RESET}`;
  const body = lines.map((line) => {
    const t = line.length > w ? line.slice(0, w - 3) + "..." : line;
    return `${DIM}|${RESET} ${t}`;
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
    `FLOW CODE (${pname}). Write production-quality code.`,
    `CWD: ${cwd}`,
  ];

  if (tree) lines.push(`Files:\n${tree}`);

  lines.push(
    "Rules:",
    "- Always write complete, production-ready files.",
    "- When creating/updating files, put the filename on the first line inside the code block:",
    "  ```path/to/file.ts",
    "  <code here>",
    "  ```",
    "- This ensures files are auto-saved to disk.",
    "- Read existing files before modifying them.",
    "- TypeScript: no any, use interfaces, async/await, optional chaining.",
    "- React/Next.js: functional components, hooks, App Router, Tailwind CSS.",
    "- HTML/CSS: semantic elements, responsive, flexbox/grid.",
    "- Python: type hints, PEP 8, f-strings.",
    "- Bash: set -euo pipefail, quoted variables.",
    "- For small changes, show the diff. For new files, write the full file.",
  );

  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// History management
// ---------------------------------------------------------------------------

function trimHistory(history: Message[]): Message[] {
  let total = 0;
  const result: Message[] = [];

  // Always keep system message
  if (history.length > 0 && history[0].role === "system") {
    result.push(history[0]);
    total += estimateTokens(getMessageContent(history[0]));
  }

  // Keep most recent messages that fit
  for (let i = history.length - 1; i >= 1; i--) {
    const tokens = estimateTokens(getMessageContent(history[i]));
    if (total + tokens > MAX_HISTORY_TOKENS) break;
    total += tokens;
    result.splice(1, 0, history[i]);
  }

  return result;
}

// ---------------------------------------------------------------------------
// Bash block extraction
// ---------------------------------------------------------------------------

function extractBashBlocks(text: string): string[] {
  const blocks: string[] = [];
  const re = /```(?:bash|sh|shell|terminal)\n([\s\S]*?)```/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    const block = m[1].trim();
    if (block.length > 0 && block.length < 10000) blocks.push(block);
  }
  return blocks;
}

// ---------------------------------------------------------------------------
// File writing
// ---------------------------------------------------------------------------

const KNOWN_LANG_TAGS =
  /^(bash|sh|shell|js|ts|py|html|css|json|yaml|yml|md|sql|go|rs|java|c|cpp|rb|php|jsx|tsx|swift|kt|toml|xml|svg|dockerfile|makefile|prisma|graphql|lua|r|dart|scala|ex|hs|clj|text|txt)$/i;

const COMMON_FILENAMES = new Set([
  "index.html",
  "style.css",
  "app.js",
  "main.js",
  "App.tsx",
  "App.jsx",
  "index.tsx",
  "index.jsx",
  "page.tsx",
  "page.jsx",
  "layout.tsx",
  "layout.jsx",
  "globals.css",
  "tailwind.config.js",
  "tailwind.config.ts",
  "next.config.js",
  "next.config.ts",
  "vite.config.js",
  "vite.config.ts",
  "tsconfig.json",
  "package.json",
  ".env.example",
  "README.md",
  "app.py",
  "main.py",
  "server.py",
  "requirements.txt",
  "setup.py",
  "pyproject.toml",
  "Cargo.toml",
  "main.rs",
  "lib.rs",
  "go.mod",
  "main.go",
  "Dockerfile",
  "docker-compose.yml",
  "docker-compose.yaml",
  ".gitignore",
  "Makefile",
  "Procfile",
]);

function extractFilesFromResponse(text: string, cwd: string): FileToWrite[] {
  const files: FileToWrite[] = [];
  const seen = new Set<string>();

  function addFile(filepath: string, content: string): void {
    if (content.length === 0) return;
    const abs = safePath(path.resolve(cwd, filepath));
    // Security: don't write outside cwd
    if (!isPathInside(abs, cwd) && abs !== cwd) return;
    if (seen.has(abs)) return;
    // Must look like a real file
    if (!path.extname(abs)) return;
    seen.add(abs);
    files.push({ filepath: abs, content });
  }

  // Strategy 1: ```filename.ext\n...\n``` (filename as language tag)
  const re1 = /```([^\s`]+\.[^\s`]+)\n([\s\S]*?)```/g;
  let m: RegExpExecArray | null;
  while ((m = re1.exec(text)) !== null) {
    const fn = m[1].trim();
    const content = m[2].trim();
    if (KNOWN_LANG_TAGS.test(fn)) continue;
    if (!fn.includes(".")) continue;
    addFile(fn, content);
  }

  // Strategy 2: "create/write/save file <path>" + code block
  const re2 =
    /(?:create|write|save|make|generate)\s+(?:a\s+)?(?:new\s+)?(?:file\s+(?:called\s+|named\s+|at\s+)?)?["'`]?([^\s"'`]+\.[^\s"'`]+)["'`]?[\s\S]*?```(?:\w*)\n([\s\S]*?)```/gi;
  while ((m = re2.exec(text)) !== null) {
    const fn = m[1].trim();
    const content = m[2].trim();
    if (!fn.includes(".")) continue;
    addFile(fn, content);
  }

  // Strategy 3: standalone filename line followed by code block
  const lines = text.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    const fnMatch = line.match(
      /^(?:#+\s+|\*\*)?([a-zA-Z0-9_\-]+\.[a-zA-Z0-9]{1,10})(?:\*\*)?\s*$/
    );
    if (!fnMatch) continue;
    const fn = fnMatch[1];
    if (!COMMON_FILENAMES.has(fn)) continue;

    const remaining = lines.slice(i + 1).join("\n");
    const codeMatch = remaining.match(/```(\w*)\n([\s\S]*?)```/);
    if (codeMatch && codeMatch[2].trim().length > 0) {
      addFile(fn, codeMatch[2].trim());
    }
  }

  // Strategy 4: path/filename.ext\n```...\n```
  const re4 =
    /(?:^|\n)([a-zA-Z0-9_\-\/]+\.[a-zA-Z0-9]{1,10})\s*\n```(?:\w*)\n([\s\S]*?)```/g;
  while ((m = re4.exec(text)) !== null) {
    const fn = m[1].trim();
    const content = m[2].trim();
    if (content.length === 0) continue;
    if (KNOWN_LANG_TAGS.test(fn)) continue;
    addFile(fn, content);
  }

  return files;
}

function writeFiles(files: FileToWrite[]): boolean {
  let anyWritten = false;
  for (const file of files) {
    try {
      const dir = path.dirname(file.filepath);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      fs.writeFileSync(file.filepath, file.content, "utf-8");
      const rel = path.relative(process.cwd(), file.filepath);
      console.log(ansi.green(`  [CREATED] ${rel}`));
      logActivity(`Created: ${rel}`);
      anyWritten = true;
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Write failed";
      console.error(
        ansi.red(`  [FAILED] ${path.basename(file.filepath)}: ${msg}`)
      );
    }
  }
  return anyWritten;
}

// ---------------------------------------------------------------------------
// Auto-search detection
// ---------------------------------------------------------------------------

const SEARCH_TRIGGERS = [
  /\blatest\b/i,
  /\bcurrent(ly)?\b/i,
  /\brecent(ly)?\b/i,
  /\bwhat('s| is) the (best|new|latest|current)\b/i,
  /\bhow (do|to|does)\b/i,
  /\bwh?at (is|are|was|were)\b.*\b(version|release|update)\b/i,
  /\btoday\b/i,
  /\bthis (year|month|week)\b/i,
  /\b20\d{2}\b/,
  /\bvs\.?\b/i,
  /\bcompared? to\b/i,
  /\bnewest\b/i,
  /\bavailable\b/i,
  /\bsupport(ed|s|ing)?\b.*\bfor\b/i,
  /\bwhich .*(is best|should i|recommend)\b/i,
  /\bweather\b/i,
  /\bnews\b/i,
  /\bstock(s)?\b/i,
  /\bprice\b/i,
  /\brelease(d)?\b/i,
];

const CODE_TASK_RE =
  /^(create|write|build|fix|edit|refactor|implement|add|remove|delete|update|debug|test|check|run|start|stop|install|deploy)\s/i;

function needsWebSearch(input: string): boolean {
  if (CODE_TASK_RE.test(input)) return false;
  if (input.length < 15) return false;
  return SEARCH_TRIGGERS.some((re) => re.test(input));
}

// ---------------------------------------------------------------------------
// Shell execution
// ---------------------------------------------------------------------------

function execShell(
  cmd: string,
  cwd: string
): Promise<{ ok: boolean; output: string }> {
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

      const timer = setTimeout(() => {
        child.kill("SIGTERM");
      }, SHELL_TIMEOUT_MS);

      child.stdout?.on("data", (data: Buffer) => {
        const chunk = data.toString();
        stdout += chunk;
        process.stdout.write(ansi.dim("  | ") + chunk);
      });

      child.stderr?.on("data", (data: Buffer) => {
        const chunk = data.toString();
        stderr += chunk;
        process.stderr.write(ansi.red("  ! ") + chunk);
      });

      child.on("close", (code) => {
        clearTimeout(timer);
        resolve({
          ok: code === 0,
          output: (code === 0 ? stdout : stderr || stdout).slice(
            0,
            MAX_SHELL_OUTPUT
          ),
        });
      });

      child.on("error", (err) => {
        clearTimeout(timer);
        resolve({ ok: false, output: err.message });
      });
    } catch (err: unknown) {
      resolve({
        ok: false,
        output: err instanceof Error ? err.message : "spawn error",
      });
    }
  });
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
  return ansi.blue(`flow-code [${path.basename(process.cwd())}] > `);
}

async function readInput(): Promise<string> {
  const first = await ask(prompt());
  if (!first) return "";

  // Multiline: trailing backslash
  if (first.endsWith("\\")) {
    const lines: string[] = [first.replace(/\\$/, "")];
    while (true) {
      const line = await ask(ansi.dim("  ... "));
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

function showThinking(): void {
  process.stdout.write(ansi.dim("  Thinking...\r"));
}

function clearLine(): void {
  process.stdout.write(" ".repeat(40) + "\r");
}

// ---------------------------------------------------------------------------
// Streaming API
// ---------------------------------------------------------------------------

async function streamResponse(
  client: OpenAI,
  model: string,
  messages: Message[],
  temperature: number,
  retries: number = 2
): Promise<{ content: string; usage: UsageInfo }> {
  let lastError: unknown;

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const stream = await client.chat.completions.create({
        model,
        messages,
        temperature,
        top_p: 0.95,
        max_tokens: MAX_RESPONSE_TOKENS,
        stream: true,
        stream_options: { include_usage: true },
      });

      let full = "";
      let usage: UsageInfo = {
        promptTokens: 0,
        completionTokens: 0,
        totalTokens: 0,
      };
      let started = false;

      for await (const chunk of stream) {
        if (chunk.usage) {
          usage = {
            promptTokens: chunk.usage.prompt_tokens,
            completionTokens: chunk.usage.completion_tokens,
            totalTokens: chunk.usage.total_tokens,
          };
        }
        const delta = chunk.choices[0]?.delta?.content;
        if (delta) {
          if (!started) {
            clearLine();
            started = true;
          }
          process.stdout.write(delta);
          full += delta;
        }
      }

      process.stdout.write("\n");
      return { content: full, usage };
    } catch (err: unknown) {
      lastError = err;
      const msg = err instanceof Error ? err.message : String(err);

      if (msg.includes("429") && attempt < retries) {
        clearLine();
        const wait = (attempt + 1) * 5;
        console.log(
          ansi.yellow(`  Rate limited. Retrying in ${wait}s...`)
        );
        await new Promise((r) => setTimeout(r, wait * 1000));
        showThinking();
        continue;
      }

      if (msg.includes("413") || msg.includes("too large")) {
        clearLine();
        console.log(
          ansi.yellow(
            "  Request too large. Try /compact or a shorter message."
          )
        );
      }

      throw err;
    }
  }

  throw lastError;
}

// ---------------------------------------------------------------------------
// Banner + Dashboard
// ---------------------------------------------------------------------------

function printBanner(): void {
  console.clear();
  const a = (t: string) => `${BOLD}${CYAN}${t}${RESET}`;
  const d = (t: string) => `${DIM}${t}${RESET}`;
  const s = (t: string) => `${BOLD}${MAGENTA}${t}${RESET}`;

  console.log("");
  console.log(`  ${s("(*)")} ${a("Flow Code")} ${d("v" + VERSION)}`);
  console.log(`       ${d("terminal coding agent")}`);
  console.log(`       ${d("--------------------")}`);
  console.log("");
}

function printDashboard(config: Config): void {
  console.clear();
  const provider = PROVIDERS[config.provider || "groq"];
  const model = config.defaultModel || provider.defaultModel;
  const cwd = process.cwd();
  const activity = getRecentActivity();
  const d = (t: string) => `${DIM}${t}${RESET}`;
  const b = (t: string) => `${BOLD}${t}${RESET}`;
  const y = (t: string) => `${YELLOW}${t}${RESET}`;
  const w = 48;
  const line = d("-".repeat(w));

  console.log("");
  console.log(
    `  ${d("---")} ${b(`${CYAN}Flow Code${RESET}`)} ${d(`v${VERSION} ${"-".repeat(w - 26)}`)}`
  );
  console.log(line);
  console.log(`  ${d(`${provider.name} | ${model}`)}`);
  console.log(`  ${d(cwd)}`);

  if (activity.length > 0) {
    console.log("");
    console.log(`  ${b(y("Recent"))}`);
    for (const entry of activity) {
      const ago = timeAgo(entry.time);
      const action = truncate(entry.action, 38);
      console.log(`  ${d(ago.padEnd(10))} ${action}`);
    }
  }

  console.log(line);
  console.log(
    `  ${d("/cmds")} commands  ${d("/help")} help  ${d("exit")} quit`
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

  // 1. Provider
  let provider: Provider = config.provider || "groq";
  if (!config.apiKey) {
    console.log(ansi.bold("  [1/4] Select AI Provider:"));
    console.log(
      `    ${ansi.dim("[1]")} Groq    -- Ultra-fast inference, open-source models`
    );
    console.log(
      `    ${ansi.dim("[2]")} Cerebras -- Wafer-scale AI, blazing speed`
    );
    const p = await ask("  Select (1-2): ");
    provider = p === "2" ? "cerebras" : "groq";
  }

  // 2. API key
  if (!config.apiKey) {
    console.log(
      ansi.dim(`\n  [2/4] Enter your ${PROVIDERS[provider].name} API Key:`)
    );
    const key = await ask("  API Key: ");
    if (!key || key.length < 10) {
      console.error(ansi.red("  Invalid key. Exiting."));
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

  // 3. Model
  let model = sanitizeModelId(
    config.defaultModel || PROVIDERS[provider].defaultModel
  );

  try {
    console.log(
      ansi.dim(`\n  [3/4] Fetching ${PROVIDERS[provider].name} models...`)
    );
    const res = await client.models.list();
    const models = res.data
      .map((m) => m.id)
      .filter((id) => PROVIDERS[provider].modelFilter.test(id))
      .sort();

    if (models.length > 0) {
      console.log(ansi.bold("  Available Models:"));
      models.forEach((m, i) =>
        console.log(`    ${ansi.dim(`[${i}]`)} ${m}`)
      );
      const choice = await ask(
        `\n  Select model (0-${models.length - 1}, Enter for default): `
      );
      if (choice !== "") {
        const idx = parseInt(choice, 10);
        if (!isNaN(idx) && idx >= 0 && idx < models.length) {
          model = sanitizeModelId(models[idx]);
        }
      }
    }
  } catch {
    console.log(ansi.dim("  Could not fetch models. Using default."));
  }

  // 4. Intensity
  console.log(ansi.bold("\n  [4/4] Processing Mode:"));
  console.log(`    ${ansi.dim("[1]")} Low    -- Fast, single-file patches`);
  console.log(`    ${ansi.dim("[2]")} Medium -- Standard refactoring`);
  console.log(`    ${ansi.dim("[3]")} High   -- Deep scanning & DevOps`);

  const iMap: Record<string, Intensity> = {
    "1": "low",
    "2": "medium",
    "3": "high",
  };
  let intensity: Intensity = config.intensity || "medium";
  const choice = await ask("  Select mode (1-3): ");
  if (iMap[choice]) intensity = iMap[choice];

  config.defaultModel = model;
  config.intensity = intensity;
  config.provider = provider;
  saveConfig(config);

  console.log("");
  console.log(ansi.green("  Setup complete."));
  console.log(
    ansi.dim(
      `  ${PROVIDERS[provider].name} | ${model} | ${intensity.toUpperCase()}`
    )
  );
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
    {
      role: "system",
      content: getSystemPrompt(state.intensity, process.cwd(), state.provider),
    },
  ];

  const updateSystemPrompt = (): void => {
    history[0] = {
      role: "system",
      content: getSystemPrompt(
        state.intensity,
        process.cwd(),
        state.provider
      ),
    };
  };

  while (true) {
    const input = await readInput();
    if (!input) continue;

    // ── Exit ──
    if (input === "exit" || input === "quit") {
      console.log(ansi.dim("\n  Goodbye.\n"));
      rl.close();
      process.exit(0);
    }

    // ── /help ──
    if (input === "/help") {
      console.log(
        [
          "",
          ansi.bold("  Commands:"),
          `    ${ansi.dim("cd <path>")}        Switch directory`,
          `    ${ansi.dim("/search <query>")}  Search the web`,
          `    ${ansi.dim("/fetch <url>")}     Fetch URL content`,
          `    ${ansi.dim("/write <file>")}    Write a file manually`,
          `    ${ansi.dim("/settings")}        Configure preferences`,
          `    ${ansi.dim("/models")}          Re-select model`,
          `    ${ansi.dim("/provider")}        Switch Groq / Cerebras`,
          `    ${ansi.dim("/resume")}          Resume last conversation`,
          `    ${ansi.dim("/clear")}           Reset conversation`,
          `    ${ansi.dim("/compact")}         Trim context`,
          `    ${ansi.dim("/cmds")}            List all commands`,
          `    ${ansi.dim("/status")}          Show usage stats`,
          `    ${ansi.dim("exit")}             Quit`,
          "",
        ].join("\n")
      );
      continue;
    }

    // ── /clear ──
    if (input === "/clear") {
      history.length = 1;
      updateSystemPrompt();
      console.log(ansi.green("  Conversation cleared.\n"));
      continue;
    }

    // ── /resume ──
    if (input === "/resume") {
      const session = loadSession();
      if (!session) {
        console.log(
          ansi.yellow("  No saved session. Start a conversation first.\n")
        );
        continue;
      }
      history.length = 0;
      for (const msg of session.history) {
        const role = msg.role as Role;
        history.push({ role, content: msg.content });
      }
      if (session.cwd && fs.existsSync(session.cwd)) {
        process.chdir(session.cwd);
        updateSystemPrompt();
      }
      const count = history.filter(
        (m) => m.role === "user" || m.role === "assistant"
      ).length;
      console.log(
        ansi.green(
          `  Resumed ${count} messages from ${timeAgo(session.timestamp)}.\n`
        )
      );
      console.log(
        ansi.dim(
          `  Model: ${session.model} | Provider: ${PROVIDERS[session.provider || "groq"].name}\n`
        )
      );
      continue;
    }

    // ── /cmds ──
    if (input === "/cmds") {
      console.log(
        [
          "",
          ansi.bold(ansi.blue("  Flow Code Commands")),
          ansi.dim("  --------------------------------------------"),
          "",
          ansi.bold("  Navigation:"),
          `    ${ansi.cyan("cd <path>")}            Switch working directory`,
          `    ${ansi.cyan("cd ..")}                Go up one directory`,
          `    ${ansi.cyan("cd ~")}                 Go to home directory`,
          "",
          ansi.bold("  Web & Search:"),
          `    ${ansi.cyan("/search <query>")}      Search the web via DuckDuckGo`,
          `    ${ansi.cyan("/fetch <url>")}         Fetch and display URL content`,
          "",
          ansi.bold("  Files:"),
          `    ${ansi.cyan("/write <file>")}        Write a file manually (paste content)`,
          `    ${ansi.dim("Auto-write")}             Code blocks with filenames auto-save`,
          "",
          ansi.bold("  Session Management:"),
          `    ${ansi.cyan("/resume")}              Resume last conversation`,
          `    ${ansi.cyan("/clear")}               Reset conversation history`,
          `    ${ansi.cyan("/compact")}             Trim history to fit context`,
          "",
          ansi.bold("  Configuration:"),
          `    ${ansi.cyan("/settings")}            Open interactive settings menu`,
          `    ${ansi.cyan("/models")}              Re-select your model`,
          `    ${ansi.cyan("/provider")}            Switch between Groq / Cerebras`,
          "",
          ansi.bold("  Information:"),
          `    ${ansi.cyan("/status")}              Show provider, model, tokens`,
          `    ${ansi.cyan("/cmds")}                Show this command list`,
          `    ${ansi.cyan("/help")}                Show condensed help`,
          "",
          ansi.bold("  Exit:"),
          `    ${ansi.cyan("exit")}                 Quit Flow Code`,
          `    ${ansi.cyan("quit")}                 Quit Flow Code`,
          `    ${ansi.cyan("Ctrl+C")}               Quit Flow Code`,
          "",
          ansi.dim("  Multiline: end a line with \\ to continue."),
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
      console.log(
        ansi.green(
          `  Compacted: ${before} -> ${history.length} messages.\n`
        )
      );
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
          ansi.bold("  Session:"),
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
        console.log(
          ansi.bold(`\n  ${PROVIDERS[state.provider].name} Models:`)
        );
        models.forEach((m, i) =>
          console.log(`    ${ansi.dim(`[${i}]`)} ${m}`)
        );
        const choice = await ask(
          `\n  Select (0-${models.length - 1}): `
        );
        const idx = parseInt(choice, 10);
        if (!isNaN(idx) && idx >= 0 && idx < models.length) {
          state.model = sanitizeModelId(models[idx]);
          console.log(ansi.green(`  Model: ${state.model}\n`));
          const cfg = loadConfig();
          cfg.defaultModel = state.model;
          saveConfig(cfg);
          updateSystemPrompt();
        }
      } catch {
        console.log(ansi.red("  Could not fetch models."));
      }
      continue;
    }

    // ── /provider ──
    if (input === "/provider") {
      console.log(ansi.bold("  Switch provider:"));
      console.log(`    ${ansi.dim("[1]")} Groq`);
      console.log(`    ${ansi.dim("[2]")} Cerebras`);
      const p = await ask("  Select (1-2): ");
      const np: Provider = p === "2" ? "cerebras" : "groq";

      if (np === state.provider) {
        console.log(
          ansi.yellow(
            `  Already using ${PROVIDERS[state.provider].name}.\n`
          )
        );
        continue;
      }

      console.log(ansi.dim(`  Enter ${PROVIDERS[np].name} API Key:`));
      const key = await ask("  API Key: ");
      if (!key || key.length < 10) {
        console.log(ansi.red("  Invalid key. Provider not changed.\n"));
        continue;
      }

      const nc = new OpenAI({
        baseURL: PROVIDERS[np].baseURL,
        apiKey: key,
      });

      try {
        console.log(ansi.dim(`  Connecting to ${PROVIDERS[np].name}...`));
        await nc.models.list();

        state.client = nc;
        state.provider = np;
        state.model = PROVIDERS[np].defaultModel;

        const cfg = loadConfig();
        cfg.provider = np;
        cfg.apiKey = key;
        cfg.defaultModel = state.model;
        saveConfig(cfg);

        updateSystemPrompt();

        console.log(
          ansi.green(
            `  Switched to ${PROVIDERS[np].name} | ${state.model}\n`
          )
        );
        logActivity(`Switched to ${PROVIDERS[np].name}`);
      } catch (err: unknown) {
        const msg =
          err instanceof Error ? err.message : "Connection failed";
        console.error(
          ansi.red(`  ${PROVIDERS[np].name}: ${msg}\n`)
        );
      }
      continue;
    }

    // ── /settings ──
    if (input === "/settings") {
      const cfg = loadConfig();
      console.log(
        [
          "",
          ansi.bold("  Settings:"),
          `    ${ansi.dim("[1]")} Change API Key`,
          `    ${ansi.dim("[2]")} Switch Provider (${PROVIDERS[cfg.provider || "groq"].name})`,
          `    ${ansi.dim("[3]")} Change Model (${cfg.defaultModel || "default"})`,
          `    ${ansi.dim("[4]")} Change Intensity (${(cfg.intensity || "medium").toUpperCase()})`,
          `    ${ansi.dim("[5]")} View Config`,
          `    ${ansi.dim("[6]")} Reset All`,
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
            console.log(ansi.green("  API key updated.\n"));
          } else {
            console.log(ansi.yellow("  Key too short. Not saved.\n"));
          }
          break;
        }
        case "2": {
          console.log(`    ${ansi.dim("[1]")} Groq`);
          console.log(`    ${ansi.dim("[2]")} Cerebras`);
          const p = await ask("  Select: ");
          cfg.provider = p === "2" ? "cerebras" : "groq";
          cfg.apiKey = ""; // force re-entry on restart
          cfg.defaultModel =
            PROVIDERS[cfg.provider || "groq"].defaultModel;
          saveConfig(cfg);
          console.log(
            ansi.green(
              `  Provider: ${PROVIDERS[cfg.provider || "groq"].name}. Restart to apply.\n`
            )
          );
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
              .filter((id) =>
                PROVIDERS[cfg.provider || "groq"].modelFilter.test(id)
              )
              .sort();
            console.log(ansi.bold("\n  Models:"));
            models.forEach((m, i) =>
              console.log(`    ${ansi.dim(`[${i}]`)} ${m}`)
            );
            const choice = await ask(
              `\n  Select (0-${models.length - 1}): `
            );
            const idx = parseInt(choice, 10);
            if (!isNaN(idx) && idx >= 0 && idx < models.length) {
              cfg.defaultModel = models[idx];
              saveConfig(cfg);
              console.log(
                ansi.green(`  Model: ${models[idx]}\n`)
              );
            }
          } catch {
            console.log(ansi.red("  Could not fetch models."));
          }
          break;
        }
        case "4": {
          console.log(`    ${ansi.dim("[1]")} Low`);
          console.log(`    ${ansi.dim("[2]")} Medium`);
          console.log(`    ${ansi.dim("[3]")} High`);
          const ic = await ask("  Select: ");
          const map: Record<string, Intensity> = {
            "1": "low",
            "2": "medium",
            "3": "high",
          };
          if (map[ic]) {
            cfg.intensity = map[ic];
            saveConfig(cfg);
            console.log(
              ansi.green(
                `  Intensity: ${cfg.intensity.toUpperCase()}\n`
              )
            );
          }
          break;
        }
        case "5": {
          console.log("");
          console.log(ansi.bold("  Current config:"));
          console.log(
            `    Provider:   ${PROVIDERS[cfg.provider || "groq"].name}`
          );
          console.log(
            `    Model:      ${cfg.defaultModel || "default"}`
          );
          console.log(
            `    Intensity:  ${(cfg.intensity || "medium").toUpperCase()}`
          );
          console.log(
            `    API Key:    ${cfg.apiKey ? cfg.apiKey.slice(0, 6) + "..." + cfg.apiKey.slice(-4) : "not set"}`
          );
          console.log(`    Config:     ${CONFIG_PATH}`);
          console.log("");
          break;
        }
        case "6": {
          const confirm = await ask(
            ansi.yellow("  Are you sure? (yes/no): ")
          );
          if (confirm.toLowerCase() === "yes") {
            try {
              fs.unlinkSync(CONFIG_PATH);
            } catch {
              // ok
            }
            console.log(
              ansi.green("  Config reset. Restart to apply.\n")
            );
          }
          break;
        }
      }
      continue;
    }

    // ── /search ──
    if (input.startsWith("/search ")) {
      const query = input.slice(8).trim();
      if (!query) {
        console.log(ansi.yellow("  Usage: /search <query>"));
        continue;
      }
      console.log(ansi.dim(`  Searching: ${query}...`));
      const results = await searchWeb(query);
      console.log(formatSearchResults(results, query));

      if (results.length > 0) {
        const resultText = results
          .map(
            (r, i) =>
              `${i + 1}. ${r.title}\n   ${r.snippet}\n   ${r.url}`
          )
          .join("\n\n");
        history.push({
          role: "user",
          content: `Web search results for "${query}":\n\n${resultText}`,
        });
      }
      continue;
    }

    // ── /fetch ──
    if (input.startsWith("/fetch ")) {
      const url = input.slice(7).trim();
      if (!url || !url.startsWith("http")) {
        console.log(ansi.yellow("  Usage: /fetch <https://url>"));
        continue;
      }
      console.log(ansi.dim(`  Fetching: ${url}...`));
      const content = await fetchUrlContent(url);
      console.log(`\n${ansi.dim(content.slice(0, 3000))}`);
      if (content.length > 3000) {
        console.log(
          ansi.dim(`\n  ... (${content.length} chars total)`)
        );
      }
      console.log("");

      history.push({
        role: "user",
        content: `Content from ${url}:\n\n${content}`,
      });
      continue;
    }

    // ── /write ──
    if (input.startsWith("/write ")) {
      const rawPath = input
        .slice(7)
        .trim()
        .replace(/^['"]|['"]$/g, "");
      if (!rawPath) {
        console.log(ansi.yellow("  Usage: /write <filepath>"));
        console.log(
          ansi.dim(
            "  Paste content, end with a line containing only '---'"
          )
        );
        continue;
      }
      const filepath = safePath(path.resolve(process.cwd(), rawPath));
      // Security: only write inside cwd
      if (!isPathInside(filepath, process.cwd()) && filepath !== path.join(process.cwd(), path.basename(filepath))) {
        console.log(ansi.red("  Path must be inside current directory."));
        continue;
      }
      console.log(ansi.dim(`  Writing to: ${path.relative(process.cwd(), filepath)}`));
      console.log(
        ansi.dim("  Paste/type content. End with '---':")
      );
      const lines: string[] = [];
      while (true) {
        const line = await ask(ansi.dim("  | "));
        if (line.trim() === "---") break;
        lines.push(line);
      }
      const content = lines.join("\n");
      if (content.length === 0) {
        console.log(ansi.yellow("  Empty. File not written."));
        continue;
      }
      try {
        const dir = path.dirname(filepath);
        if (!fs.existsSync(dir)) {
          fs.mkdirSync(dir, { recursive: true });
        }
        fs.writeFileSync(filepath, content, "utf-8");
        const rel = path.relative(process.cwd(), filepath);
        console.log(
          ansi.green(`  Created: ${rel} (${content.length} chars)\n`)
        );
        logActivity(`Created: ${path.basename(filepath)}`);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Write failed";
        console.error(ansi.red(`  ${msg}\n`));
      }
      continue;
    }

    // ── cd ──
    if (input === "cd" || input === "cd ~" || input === "cd $HOME") {
      const home = os.homedir();
      try {
        process.chdir(home);
        console.log(ansi.green(`  ${home}\n`));
        const tree = scanDirectory(home, 3);
        if (tree) {
          console.log(ansi.dim("  Directory:"));
          console.log(ansi.dim(tree) + "\n");
        }
        updateSystemPrompt();
      } catch (err: unknown) {
        console.error(
          ansi.red(`  ${err instanceof Error ? err.message : String(err)}`)
        );
      }
      continue;
    }

    if (input.startsWith("cd ")) {
      const raw = input
        .slice(3)
        .replace(/^['"]|['"]$/g, "")
        .trim();
      if (!raw || raw.includes("\0")) {
        console.error(ansi.red("  Invalid path."));
        continue;
      }
      try {
        const target = path.resolve(raw);
        process.chdir(target);
        const cwd = process.cwd();
        console.log(ansi.green(`  ${cwd}\n`));
        const tree = scanDirectory(cwd, 3);
        if (tree) {
          console.log(ansi.dim("  Directory:"));
          console.log(ansi.dim(tree) + "\n");
        }
        updateSystemPrompt();
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error(ansi.red(`  ${msg}`));
        console.log(
          ansi.dim(
            "  Tip: use full paths like C:\\Users\\name\\project or /home/user/project"
          )
        );
      }
      continue;
    }

    // ── Send to LLM ──
    updateSystemPrompt();

    // Auto-search
    let finalInput = input;
    if (needsWebSearch(input)) {
      console.log(ansi.dim("  Searching the web..."));
      const sr = await searchWeb(input, 3);
      if (sr.length > 0) {
        const ctx = sr
          .map((r) => `- ${r.title}: ${r.snippet} (${r.url})`)
          .join("\n");
        finalInput = `${input}\n\n[Web search results:]\n${ctx}`;
        logActivity(`Searched: ${input.slice(0, 50)}`);
      }
    }

    history.push({ role: "user", content: finalInput });
    const trimmed = trimHistory(history);
    const temp =
      state.intensity === "low"
        ? 0.0
        : state.intensity === "medium"
          ? 0.2
          : 0.4;

    try {
      showThinking();
      const { content: reply, usage } = await streamResponse(
        state.client,
        state.model,
        trimmed,
        temp
      );

      if (!reply) {
        console.log(ansi.red("  No response."));
        history.pop();
        continue;
      }

      console.log(formatResponse(reply) + "\n");
      if (usage.totalTokens > 0) printUsage(usage, history);

      history.push({ role: "assistant", content: reply });

      // Auto-write files
      const files = extractFilesFromResponse(reply, process.cwd());
      if (files.length > 0) {
        console.log(ansi.bold("  Detected files:"));
        const wrote = writeFiles(files);
        if (wrote) logActivity(`Auto-wrote ${files.length} file(s)`);
        console.log("");
      }

      // Auto-execute bash
      const blocks = extractBashBlocks(reply);
      const cwd = process.cwd();
      for (const block of blocks) {
        console.log(ansi.dim(`  > ${block}`));
        const result = await execShell(block, cwd);
        if (result.ok) {
          console.log(ansi.green("  Done.\n"));
          logActivity(`Ran: ${block.slice(0, 60)}`);
        } else {
          console.log(ansi.red("  Failed.\n"));
          logActivity(`Failed: ${block.slice(0, 60)}`);
        }
        history.push({
          role: "user",
          content: `Terminal output:\n${result.output}`,
        });
      }

      saveSession(
        history,
        state.model,
        state.provider,
        state.intensity
      );
    } catch (err: unknown) {
      clearLine();
      const msg = err instanceof Error ? err.message : String(err);

      if (msg.includes("401") || msg.toLowerCase().includes("invalid")) {
        console.error(
          ansi.red(
            `  Invalid API key for ${PROVIDERS[state.provider].name}. Delete ~/.flow-code-config and restart.`
          )
        );
      } else if (msg.includes("404") || msg.includes("does not exist")) {
        console.error(
          ansi.red(`  Model '${state.model}' not found.`)
        );
        console.log(ansi.dim("  Type /models to re-select."));
        state.model = PROVIDERS[state.provider].defaultModel;
        const cfg = loadConfig();
        cfg.defaultModel = state.model;
        saveConfig(cfg);
        console.log(ansi.green(`  Default: ${state.model}\n`));
      } else if (msg.includes("429")) {
        console.error(ansi.yellow("  Rate limited. Wait a moment."));
      } else if (msg.includes("503")) {
        console.error(ansi.yellow("  Model overloaded. Try again."));
      } else {
        console.error(ansi.red(`  ${msg}`));
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
    console.log(ansi.dim("\n\n  Goodbye.\n"));
    rl.close();
    process.exit(0);
  });

  process.on("SIGTERM", () => {
    rl.close();
    process.exit(0);
  });

  process.on("unhandledRejection", (err) => {
    console.error(ansi.red(`\n  Unhandled: ${err}`));
  });

  process.on("uncaughtException", (err) => {
    console.error(ansi.red(`\n  Uncaught: ${err.message}`));
    process.exit(1);
  });

  printBanner();

  setup()
    .then(async ({ client, model, intensity, provider }) => {
      console.log(
        ansi.dim(`  Working directory: ${process.cwd()}`)
      );
      const dirInput = await ask(
        ansi.bold("  Project directory (Enter to skip): ")
      );
      if (dirInput.trim()) {
        const cleaned = dirInput
          .trim()
          .replace(/^['"]|['"]$/g, "");
        const target = path.resolve(cleaned);
        try {
          process.chdir(target);
          console.log(
            ansi.green(`  Switched to: ${process.cwd()}\n`)
          );
        } catch {
          console.log(
            ansi.yellow(
              `  Could not open '${target}'. Using current.\n`
            )
          );
        }
      }

      const config = loadConfig();
      printDashboard(config);

      if (hasSavedSession()) {
        const session = loadSession();
        if (session) {
          console.log(
            ansi.dim(
              `  Last session: ${timeAgo(session.timestamp)} -- type /resume to continue`
            )
          );
        }
      }
      console.log(
        ansi.green("  Ready! Type /cmds for commands.\n")
      );

      logActivity("Started session");
      return run(client, model, intensity, provider);
    })
    .catch((err) => {
      console.error(
        ansi.red(`\n  Fatal: ${err.message || err}`)
      );
      process.exit(1);
    });
}

main();
