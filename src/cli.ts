#!/usr/bin/env node

import OpenAI from "openai";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";
import * as readline from "readline";
import { spawn } from "child_process";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CONFIG_PATH = path.join(os.homedir(), ".flow-code-config");
const SESSION_PATH = path.join(os.homedir(), ".flow-code-session");
const ACTIVITY_PATH = path.join(os.homedir(), ".flow-code-activity");
const CONTEXT_WINDOW = 128000;
const MAX_HISTORY_TOKENS = 8000;
const MAX_RESPONSE_TOKENS = 4096;
const VERSION = "1.1.0";

const CYAN = "\x1b[36m";
const RESET = "\x1b[0m";
const BOLD = "\x1b[1m";
const DIM = "\x1b[2m";
const RED = "\x1b[31m";
const GREEN = "\x1b[32m";
const YELLOW = "\x1b[33m";
const MAGENTA = "\x1b[35m";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type Intensity = "low" | "medium" | "high";
type Provider = "groq" | "cerebras";
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
  path: string;
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

const PROVIDERS: Record<Provider, {
  name: string;
  baseURL: string;
  defaultModel: string;
  models: RegExp;
}> = {
  groq: {
    name: "Groq",
    baseURL: "https://api.groq.com/openai/v1",
    defaultModel: "llama-3.3-70b-versatile",
    models: /llama|mixtral|qwen|gemma|deepseek/i,
  },
  cerebras: {
    name: "Cerebras",
    baseURL: "https://api.cerebras.ai/v1",
    defaultModel: "llama-3.3-70b",
    models: /llama|qwen/i,
  },
};

// ---------------------------------------------------------------------------
// ANSI helpers
// ---------------------------------------------------------------------------

const c = {
  blue: (t: string) => `${CYAN}${t}${RESET}`,
  cyan: (t: string) => `${CYAN}${t}${RESET}`,
  green: (t: string) => `${GREEN}${t}${RESET}`,
  red: (t: string) => `${RED}${t}${RESET}`,
  dim: (t: string) => `${DIM}${t}${RESET}`,
  bold: (t: string) => `${BOLD}${t}${RESET}`,
  yellow: (t: string) => `${YELLOW}${t}${RESET}`,
  magenta: (t: string) => `${MAGENTA}${t}${RESET}`,
};

// ---------------------------------------------------------------------------
// Token estimation
// ---------------------------------------------------------------------------

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 3.5);
}

// ---------------------------------------------------------------------------
// Context window display
// ---------------------------------------------------------------------------

function renderUsageBar(used: number, total: number): string {
  const pct = Math.min(used / total, 1);
  const filled = Math.round(pct * 20);
  const empty = 20 - filled;
  const bar = GREEN + "\u2588".repeat(filled) + RESET + DIM + "\u2591".repeat(empty) + RESET;
  const pctStr = Math.round(pct * 100) + "%";
  return `  ${c.dim("Context:")} ${bar} ${c.dim(`${used.toLocaleString()} / ${total.toLocaleString()} tokens (${pctStr})`)}`;
}

function printUsage(usage: UsageInfo, history: Message[]): void {
  const totalUsed = history.reduce((acc, msg) => {
    const content = getMessageContent(msg);
    return acc + estimateTokens(content);
  }, 0);

  console.log("");
  console.log(
    `  ${c.dim("Tokens:")} ${c.bold(String(usage.promptTokens))} prompt + ${c.bold(String(usage.completionTokens))} completion = ${c.bold(String(usage.totalTokens))} total`
  );
  console.log(renderUsageBar(totalUsed, CONTEXT_WINDOW));
  console.log("");
}

// ---------------------------------------------------------------------------
// Message content extraction
// ---------------------------------------------------------------------------

function getMessageContent(msg: Message): string {
  if (typeof msg.content === "string") return msg.content;
  if (Array.isArray(msg.content)) {
    return msg.content.map((p) => ("text" in p ? p.text : "")).join("");
  }
  return "";
}

// ---------------------------------------------------------------------------
// Web search - DuckDuckGo, no API key needed
// ---------------------------------------------------------------------------

async function searchWeb(query: string, numResults: number = 5): Promise<SearchResult[]> {
  try {
    const url = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`;
    const res = await fetch(url, {
      headers: {
        "User-Agent":
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
      },
      signal: AbortSignal.timeout(10000),
    });

    const html = await res.text();
    const results: SearchResult[] = [];

    const resultRegex =
      /<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>([\s\S]*?)<\/a>[\s\S]*?<a[^>]*class="result__snippet"[^>]*>([\s\S]*?)<\/a>/g;
    let match: RegExpExecArray | null;

    while ((match = resultRegex.exec(html)) !== null && results.length < numResults) {
      const rawUrl = match[1];
      const title = match[2].replace(/<[^>]*>/g, "").trim();
      const snippet = match[3].replace(/<[^>]*>/g, "").trim();

      const urlMatch = rawUrl.match(/uddg=([^&]+)/);
      const finalUrl = urlMatch ? decodeURIComponent(urlMatch[1]) : rawUrl;

      if (title && snippet) {
        results.push({ title, snippet, url: finalUrl });
      }
    }

    return results;
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : "Search failed";
    console.error(c.red(`  Search error: ${msg}`));
    return [];
  }
}

function formatSearchResults(results: SearchResult[], query: string): string {
  if (results.length === 0) {
    return `No results found for "${query}".`;
  }

  const lines: string[] = [
    "",
    `  ${c.bold(c.blue("Search Results"))} ${c.dim(`for "${query}"`)}`,
    "",
  ];

  for (const [i, r] of results.entries()) {
    lines.push(`  ${c.bold(c.green(`${i + 1}.`))} ${c.bold(r.title)}`);
    lines.push(`     ${c.dim(r.snippet)}`);
    lines.push(`     ${c.dim(r.url)}`);
    lines.push("");
  }

  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// Fetch URL content
// ---------------------------------------------------------------------------

async function fetchUrlContent(url: string): Promise<string> {
  try {
    const res = await fetch(url, {
      headers: {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        Accept: "text/html,application/xhtml+xml",
      },
      signal: AbortSignal.timeout(15000),
    });

    const html = await res.text();

    const text = html
      .replace(/<script[\s\S]*?<\/script>/gi, "")
      .replace(/<style[\s\S]*?<\/style>/gi, "")
      .replace(/<[^>]*>/g, " ")
      .replace(/&nbsp;/g, " ")
      .replace(/&amp;/g, "&")
      .replace(/&lt;/g, "<")
      .replace(/&gt;/g, ">")
      .replace(/&#\d+;/g, "")
      .replace(/\s+/g, " ")
      .trim();

    return text.slice(0, 12000);
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : "Fetch failed";
    return `Failed to fetch URL: ${msg}`;
  }
}

// ---------------------------------------------------------------------------
// Activity log
// ---------------------------------------------------------------------------

function logActivity(action: string): void {
  try {
    let entries: ActivityEntry[] = [];
    if (fs.existsSync(ACTIVITY_PATH)) {
      entries = JSON.parse(fs.readFileSync(ACTIVITY_PATH, "utf-8"));
    }
    entries.unshift({ time: Date.now(), action });
    entries = entries.slice(0, 20);
    fs.writeFileSync(ACTIVITY_PATH, JSON.stringify(entries, null, 2), {
      encoding: "utf-8",
      mode: 0o600,
    });
  } catch {
    /* ignore */
  }
}

function getRecentActivity(): ActivityEntry[] {
  try {
    if (fs.existsSync(ACTIVITY_PATH)) {
      return JSON.parse(fs.readFileSync(ACTIVITY_PATH, "utf-8")).slice(0, 5);
    }
  } catch {
    /* ignore */
  }
  return [];
}

function timeAgo(ts: number): string {
  const diff = Date.now() - ts;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days}d ago`;
  return `${Math.floor(days / 7)}w ago`;
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
    /* ignore */
  }
}

function loadSession(): SessionData | null {
  try {
    if (fs.existsSync(SESSION_PATH)) {
      return JSON.parse(fs.readFileSync(SESSION_PATH, "utf-8")) as SessionData;
    }
  } catch {
    /* ignore */
  }
  return null;
}

function hasSavedSession(): boolean {
  return loadSession() !== null;
}

function formatSessionAge(ts: number): string {
  const diff = Date.now() - ts;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

// ---------------------------------------------------------------------------
// Banner + Dashboard
// ---------------------------------------------------------------------------

function printBanner(): void {
  console.clear();
  const accent = (t: string) => `${BOLD}${CYAN}${t}${RESET}`;
  const dim = (t: string) => `${DIM}${t}${RESET}`;
  const spark = (t: string) => `${BOLD}${MAGENTA}${t}${RESET}`;

  console.log("");
  console.log(`  ${spark("(*)")} ${accent("Flow Code")} ${dim("v" + VERSION)}`);
  console.log(`       ${dim("terminal coding agent")}`);
  console.log(`       ${dim("--------------------")}`);
  console.log("");
}

function printDashboard(config: Config): void {
  console.clear();
  const provider = PROVIDERS[config.provider || "groq"];
  const model = config.defaultModel || provider.defaultModel;
  const cwd = process.cwd();
  const activity = getRecentActivity();

  const dim = (t: string) => `${DIM}${t}${RESET}`;
  const bold = (t: string) => `${BOLD}${t}${RESET}`;
  const blue = (t: string) => `${CYAN}${t}${RESET}`;
  const yellow = (t: string) => `${YELLOW}${t}${RESET}`;
  const w = 48;
  const line = dim("-".repeat(w));

  console.log("");
  console.log(`  ${dim("---")} ${bold(blue("Flow Code"))} ${dim(`v${VERSION} ${"-".repeat(w - 26)}`)}`);
  console.log(line);
  console.log(`  ${dim(`${provider.name} | ${model}`)}`);
  console.log(`  ${dim(cwd)}`);

  if (activity.length > 0) {
    console.log("");
    console.log(`  ${bold(yellow("Recent"))}`);
    for (const entry of activity) {
      const ago = timeAgo(entry.time);
      const action =
        entry.action.length > 38 ? entry.action.slice(0, 35) + "..." : entry.action;
      console.log(`  ${dim(ago.padEnd(10))} ${action}`);
    }
  }

  console.log(line);
  console.log(`  ${dim("/cmds")} commands  ${dim("/help")} help  ${dim("exit")} quit`);
  console.log(line);
  console.log("");
}

// ---------------------------------------------------------------------------
// Readline
// ---------------------------------------------------------------------------

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

function ask(question: string): Promise<string> {
  return new Promise((resolve) => {
    rl.question(question, (answer) => resolve(answer.trim()));
  });
}

async function askMultiline(): Promise<string> {
  const first = await ask(c.blue(`flow-code [${path.basename(process.cwd())}] > `));
  if (!first) return "";

  if (first.endsWith("\\")) {
    const lines: string[] = [first.replace(/\\$/, "")];
    while (true) {
      const line = await ask(c.dim("  ... "));
      if (line === "") break;
      lines.push(line.replace(/\\$/, ""));
    }
    return lines.join("\n").trim();
  }

  return first;
}

// ---------------------------------------------------------------------------
// Shell execution
// ---------------------------------------------------------------------------

function execShellStreaming(
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

      const timeout = setTimeout(() => {
        child.kill("SIGTERM");
      }, 60000);

      child.stdout?.on("data", (data: Buffer) => {
        const chunk = data.toString();
        stdout += chunk;
        process.stdout.write(c.dim("  | ") + chunk);
      });

      child.stderr?.on("data", (data: Buffer) => {
        const chunk = data.toString();
        stderr += chunk;
        process.stderr.write(c.red("  ! ") + chunk);
      });

      child.on("close", (code) => {
        clearTimeout(timeout);
        resolve({
          ok: code === 0,
          output: code === 0 ? stdout : stderr || stdout,
        });
      });

      child.on("error", (err) => {
        clearTimeout(timeout);
        resolve({ ok: false, output: err.message });
      });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      resolve({ ok: false, output: msg });
    }
  });
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

function scanDirectory(dir: string, depth: number = 3, prefix: string = ""): string {
  const lines: string[] = [];
  if (depth < 0) return "";

  try {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    const dirs = entries
      .filter(
        (e) =>
          e.isDirectory() && !IGNORE_DIRS.has(e.name) && !e.name.startsWith(".")
      )
      .sort((a, b) => a.name.localeCompare(b.name));
    const files = entries
      .filter(
        (e) =>
          e.isFile() && !IGNORE_FILES.has(e.name) && !e.name.startsWith(".")
      )
      .sort((a, b) => a.name.localeCompare(b.name));

    const all = [...dirs, ...files];
    for (let i = 0; i < all.length; i++) {
      const entry = all[i];
      const isLast = i === all.length - 1;
      const connector = isLast ? "\\--- " : "|-- ";
      const childPrefix = isLast ? "    " : "|   ";

      if (entry.isDirectory()) {
        lines.push(`${prefix}${connector}${entry.name}/`);
        const sub = scanDirectory(
          path.join(dir, entry.name),
          depth - 1,
          prefix + childPrefix
        );
        if (sub) lines.push(sub);
      } else {
        lines.push(`${prefix}${connector}${entry.name}`);
      }
    }
  } catch {
    /* permission denied */
  }

  return lines.join("\n");
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
  const width = Math.min(maxLen + 2, 120);
  const top = `${DIM}+-- ${label} ${"-".repeat(Math.max(0, width - label.length - 3))}+${RESET}`;
  const bottom = `${DIM}+${"-".repeat(width)}+${RESET}`;
  const body = lines.map((line) => {
    const truncated =
      line.length > width ? line.slice(0, width - 3) + "..." : line;
    return `${DIM}|${RESET} ${truncated}`;
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
  const providerName = PROVIDERS[provider].name;
  const dirTree = scanDirectory(cwd, 1);

  const lines = [
    `FLOW CODE (${providerName}). Write production-quality code.`,
    `CWD: ${cwd}`,
  ];

  if (dirTree) {
    lines.push(`Files:\n${dirTree}`);
  }

  lines.push(
    "Rules:",
    "- Always write complete, production-ready files.",
    "- When creating/updating files, put the filename on the first line inside the code block, e.g.:",
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
    "- For small changes, just show the diff. For new files, write the full file.",
  );

  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

function loadConfig(): Config {
  try {
    if (fs.existsSync(CONFIG_PATH)) {
      const raw = fs.readFileSync(CONFIG_PATH, "utf-8");
      const parsed = JSON.parse(raw);
      if (typeof parsed === "object" && parsed !== null) {
        return {
          apiKey: typeof parsed.apiKey === "string" ? parsed.apiKey : "",
          provider: ["groq", "cerebras"].includes(parsed.provider)
            ? parsed.provider
            : "groq",
          defaultModel: typeof parsed.defaultModel === "string" ? parsed.defaultModel : undefined,
          intensity: ["low", "medium", "high"].includes(parsed.intensity)
            ? parsed.intensity
            : undefined,
        };
      }
    }
  } catch {
    /* start fresh */
  }
  return { apiKey: "", provider: "groq" };
}

function saveConfig(config: Config): void {
  fs.writeFileSync(CONFIG_PATH, JSON.stringify(config, null, 2), {
    encoding: "utf-8",
    mode: 0o600,
  });
}

function sanitizeModelId(id: string): string {
  return id.replace(/[^a-zA-Z0-9._/-]/g, "").slice(0, 128);
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

  // Provider selection
  let provider: Provider = config.provider || "groq";
  if (!config.apiKey) {
    console.log(c.bold("  [1/4] Select AI Provider:"));
    console.log(
      `    ${c.dim("[1]")} Groq    -- Ultra-fast inference, open-source models`
    );
    console.log(
      `    ${c.dim("[2]")} Cerebras -- Wafer-scale AI, blazing speed`
    );
    const pChoice = await ask("  Select (1-2): ");
    provider = pChoice === "2" ? "cerebras" : "groq";
  }

  const providerConfig = PROVIDERS[provider];

  // API key
  if (!config.apiKey) {
    console.log(c.dim(`\n  [2/4] Enter your ${providerConfig.name} API Key:`));
    const key = await ask("  API Key: ");
    if (!key || key.length < 10) {
      console.error(c.red("  Invalid key. Exiting."));
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

  // Model selection
  let model = sanitizeModelId(
    config.defaultModel || PROVIDERS[provider].defaultModel
  );

  try {
    console.log(c.dim(`\n  [3/4] Fetching ${PROVIDERS[provider].name} models...`));
    const res = await client.models.list();
    const models = res.data
      .map((m) => m.id)
      .filter((id) => PROVIDERS[provider].models.test(id))
      .sort();

    if (models.length > 0) {
      console.log(c.bold("  Available Models:"));
      models.forEach((m, i) =>
        console.log(`    ${c.dim(`[${i}]`)} ${m}`)
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
    console.log(c.dim("  Could not fetch models. Using default."));
  }

  // Intensity
  console.log(c.bold("\n  [4/4] Processing Mode:"));
  console.log(`    ${c.dim("[1]")} Low    -- Fast, single-file patches`);
  console.log(`    ${c.dim("[2]")} Medium -- Standard refactoring`);
  console.log(`    ${c.dim("[3]")} High   -- Deep scanning & DevOps`);

  const intensityMap: Record<string, Intensity> = {
    "1": "low",
    "2": "medium",
    "3": "high",
  };
  let intensity: Intensity = config.intensity || "medium";
  const choice = await ask("  Select mode (1-3): ");
  if (intensityMap[choice]) intensity = intensityMap[choice];

  config.defaultModel = model;
  config.intensity = intensity;
  config.provider = provider;
  saveConfig(config);

  console.log("");
  console.log(c.green(`  Setup complete.`));
  console.log(
    c.dim(`  ${PROVIDERS[provider].name} | ${model} | ${intensity.toUpperCase()}`)
  );
  console.log("");

  return { client, model, intensity, provider };
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
    const msg = history[i];
    const content = getMessageContent(msg);
    const tokens = estimateTokens(content);
    if (total + tokens > MAX_HISTORY_TOKENS) break;
    total += tokens;
    result.splice(1, 0, msg);
  }

  return result;
}

// ---------------------------------------------------------------------------
// Bash block extraction
// ---------------------------------------------------------------------------

function extractBashBlocks(text: string): string[] {
  const blocks: string[] = [];
  const regex = /```(?:bash|sh|shell)\n([\s\S]*?)```/g;
  let match: RegExpExecArray | null;
  while ((match = regex.exec(text)) !== null) {
    const block = match[1].trim();
    if (block.length > 0 && block.length < 10000) blocks.push(block);
  }
  return blocks;
}

// ---------------------------------------------------------------------------
// File writing - auto-detect and manual /write command
// ---------------------------------------------------------------------------

const COMMON_FILENAMES = new Set([
  // Web
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
  "webpack.config.js",
  "tsconfig.json",
  "package.json",
  ".env.example",
  "README.md",
  // Python
  "app.py",
  "main.py",
  "server.py",
  "requirements.txt",
  "setup.py",
  "pyproject.toml",
  // Rust
  "Cargo.toml",
  "main.rs",
  "lib.rs",
  // Go
  "go.mod",
  "main.go",
  // Config
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

  // Strategy 1: Code block where the first non-empty line is a filename
  // ```src/app.tsx\nimport ...\n```
  // This is the primary pattern the model should use
  const filenameBlockRegex = /```([^\s`]+\.[^\s`]+)\n([\s\S]*?)```/g;
  let match: RegExpExecArray | null;
  while ((match = filenameBlockRegex.exec(text)) !== null) {
    const filename = match[1].trim();
    const content = match[2].trim();

    // Skip if it's a known language tag (not a filename)
    const knownLangs =
      /^(bash|sh|shell|js|ts|py|html|css|json|yaml|yml|md|sql|go|rs|java|c|cpp|rb|php|jsx|tsx|swift|kt|toml|xml|svg|dockerfile|makefile|prisma|graphql|lua|r|dart|scala|ex|hs|clj|text|txt)$/i;
    if (knownLangs.test(filename)) continue;
    // Skip if it doesn't look like a real filename (must have a dot)
    if (!filename.includes(".")) continue;

    const absPath = path.resolve(cwd, filename);
    if (!seen.has(absPath) && content.length > 0) {
      seen.add(absPath);
      files.push({ path: absPath, content });
    }
  }

  // Strategy 2: "create/write/save file <path>" followed by a code block
  const createFileRegex =
    /(?:create|write|save|make|generate)\s+(?:a\s+)?(?:new\s+)?(?:file\s+(?:called\s+|named\s+|at\s+)?)?["'`]?([^\s"'`]+\.[^\s"'`]+)["'`]?[\s\S]*?```(?:\w*)\n([\s\S]*?)```/gi;
  while ((match = createFileRegex.exec(text)) !== null) {
    const filename = match[1].trim();
    const content = match[2].trim();

    if (content.length === 0) continue;
    if (!filename.includes(".")) continue;

    const absPath = path.resolve(cwd, filename);
    if (!seen.has(absPath)) {
      seen.add(absPath);
      files.push({ path: absPath, content });
    }
  }

  // Strategy 3: Standalone common filenames on their own line followed by a code block
  const lines = text.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    // Match lines like "index.html", "**index.html**", "## index.html", "### style.css"
    const filenameMatch = line.match(
      /^(?:#+\s+|\*\*)?([a-zA-Z0-9_\-]+\.[a-zA-Z0-9]{1,10})(?:\*\*)?\s*$/
    );
    if (!filenameMatch) continue;

    const filename = filenameMatch[1];
    if (!COMMON_FILENAMES.has(filename)) continue;

    // Look for the next code block after this line
    const remaining = lines.slice(i + 1).join("\n");
    const codeMatch = remaining.match(/```(\w*)\n([\s\S]*?)```/);
    if (codeMatch && codeMatch[2].trim().length > 0) {
      const absPath = path.resolve(cwd, filename);
      if (!seen.has(absPath)) {
        seen.add(absPath);
        files.push({ path: absPath, content: codeMatch[2].trim() });
      }
    }
  }

  // Strategy 4: File path followed by code block (src/components/Button.tsx etc.)
  const pathFileRegex =
    /(?:^|\n)([a-zA-Z0-9_\-\/]+\.[a-zA-Z0-9]{1,10})\s*\n```(?:\w*)\n([\s\S]*?)```/g;
  while ((match = pathFileRegex.exec(text)) !== null) {
    const filename = match[1].trim();
    const content = match[2].trim();

    if (content.length === 0) continue;
    if (!filename.includes("/") && !filename.includes(".")) continue;

    // Skip language tags
    const knownLangs =
      /^(bash|sh|shell|js|ts|py|html|css|json|yaml|yml|md|sql|go|rs|java|c|cpp|rb|php)$/i;
    if (knownLangs.test(filename)) continue;

    const absPath = path.resolve(cwd, filename);
    if (!seen.has(absPath)) {
      seen.add(absPath);
      files.push({ path: absPath, content });
    }
  }

  return files;
}

function writeFiles(files: FileToWrite[]): boolean {
  let anyWritten = false;
  for (const file of files) {
    try {
      const dir = path.dirname(file.path);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      fs.writeFileSync(file.path, file.content, "utf-8");
      const rel = path.relative(process.cwd(), file.path);
      console.log(c.green(`  [CREATED] ${rel}`));
      logActivity(`Created: ${rel}`);
      anyWritten = true;
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Write failed";
      console.error(
        c.red(`  [FAILED] ${path.basename(file.path)}: ${msg}`)
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

function needsWebSearch(input: string): boolean {
  if (
    /^(create|write|build|fix|edit|refactor|implement|add|remove|delete|update)\s/i.test(
      input
    )
  ) {
    return false;
  }
  if (input.length < 15) return false;
  return SEARCH_TRIGGERS.some((re) => re.test(input));
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

      let fullResponse = "";
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
            process.stdout.write(" ".repeat(30) + "\r");
            started = true;
          }
          process.stdout.write(delta);
          fullResponse += delta;
        }
      }

      process.stdout.write("\n");
      return { content: fullResponse, usage };
    } catch (err: unknown) {
      lastError = err;
      const msg = err instanceof Error ? err.message : String(err);

      if (msg.includes("429") && attempt < retries) {
        process.stdout.write(" ".repeat(30) + "\r");
        const wait = (attempt + 1) * 5;
        console.log(
          c.yellow(`  Rate limited. Retrying in ${wait}s...`)
        );
        await new Promise((r) => setTimeout(r, wait * 1000));
        process.stdout.write(c.dim("  Thinking...\r"));
        continue;
      }

      if (msg.includes("413") || msg.includes("too large")) {
        process.stdout.write(" ".repeat(30) + "\r");
        console.log(
          c.yellow(
            "  Request too large. Try a shorter message or /compact to clear history."
          )
        );
        throw err;
      }

      throw err;
    }
  }

  throw lastError;
}

// ---------------------------------------------------------------------------
// Main agent loop
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

  while (true) {
    const input = await askMultiline();
    if (!input) continue;

    // -- Exit --
    if (input === "exit" || input === "quit") {
      console.log(c.dim("\n  Goodbye.\n"));
      rl.close();
      process.exit(0);
    }

    // -- /help --
    if (input === "/help") {
      console.log(
        [
          "",
          c.bold("  Commands:"),
          `    ${c.dim("cd <path>")}        Switch directory`,
          `    ${c.dim("/search <query>")}  Search the web`,
          `    ${c.dim("/fetch <url>")}     Fetch URL content`,
          `    ${c.dim("/write <file>")}    Write a file manually`,
          `    ${c.dim("/settings")}        Configure preferences`,
          `    ${c.dim("/models")}          Re-select model`,
          `    ${c.dim("/provider")}        Switch Groq / Cerebras`,
          `    ${c.dim("/resume")}          Resume last conversation`,
          `    ${c.dim("/clear")}           Reset conversation`,
          `    ${c.dim("/compact")}         Trim context`,
          `    ${c.dim("/cmds")}            List all commands`,
          `    ${c.dim("/status")}          Show usage stats`,
          `    ${c.dim("exit")}             Quit`,
          "",
          c.dim(
            "  Auto-search: when a question needs current data,"
          ),
          c.dim("  the agent searches the web automatically."),
          "",
        ].join("\n")
      );
      continue;
    }

    // -- /clear --
    if (input === "/clear") {
      history.length = 1;
      history[0] = {
        role: "system",
        content: getSystemPrompt(
          state.intensity,
          process.cwd(),
          state.provider
        ),
      };
      console.log(c.green("  [OK] Conversation cleared.\n"));
      continue;
    }

    // -- /resume --
    if (input === "/resume") {
      const session = loadSession();
      if (!session) {
        console.log(
          c.yellow("  No saved session found. Start a conversation first.\n")
        );
        continue;
      }
      history.length = 0;
      for (const msg of session.history) {
        history.push({
          role: msg.role as "system" | "user" | "assistant",
          content: msg.content,
        });
      }
      if (session.cwd && fs.existsSync(session.cwd)) {
        process.chdir(session.cwd);
        history[0] = {
          role: "system",
          content: getSystemPrompt(
            state.intensity,
            process.cwd(),
            state.provider
          ),
        };
      }
      const msgCount = history.filter(
        (m) => m.role === "user" || m.role === "assistant"
      ).length;
      console.log(
        c.green(
          `  [OK] Resumed ${msgCount} messages from ${formatSessionAge(session.timestamp)}.\n`
        )
      );
      console.log(
        c.dim(
          `  Model: ${session.model} | Provider: ${PROVIDERS[session.provider || "groq"].name}\n`
        )
      );
      continue;
    }

    // -- /cmds --
    if (input === "/cmds") {
      console.log(
        [
          "",
          c.bold(c.blue("  Flow Code Commands")),
          c.dim("  --------------------------------------------"),
          "",
          c.bold("  Navigation:"),
          `    ${c.cyan("cd <path>")}            Switch working directory`,
          `    ${c.cyan("cd ..")}                Go up one directory`,
          `    ${c.cyan("cd ~")}                 Go to home directory`,
          "",
          c.bold("  Web & Search:"),
          `    ${c.cyan("/search <query>")}      Search the web via DuckDuckGo`,
          `    ${c.cyan("/fetch <url>")}         Fetch and display URL content`,
          "",
          c.bold("  Files:"),
          `    ${c.cyan("/write <file>")}        Write a file manually (paste content)`,
          `    ${c.dim("Auto-write")}             Code blocks with filenames auto-save`,
          "",
          c.bold("  Session Management:"),
          `    ${c.cyan("/resume")}              Resume last conversation`,
          `    ${c.cyan("/clear")}               Reset conversation history`,
          `    ${c.cyan("/compact")}             Trim history to fit context`,
          "",
          c.bold("  Configuration:"),
          `    ${c.cyan("/settings")}            Open interactive settings menu`,
          `    ${c.cyan("/models")}              Re-select your model`,
          `    ${c.cyan("/provider")}            Switch between Groq / Cerebras`,
          "",
          c.bold("  Information:"),
          `    ${c.cyan("/status")}              Show provider, model, tokens`,
          `    ${c.cyan("/cmds")}                Show this command list`,
          `    ${c.cyan("/help")}                Show condensed help`,
          "",
          c.bold("  Exit:"),
          `    ${c.cyan("exit")}                 Quit Flow Code`,
          `    ${c.cyan("quit")}                 Quit Flow Code`,
          `    ${c.cyan("Ctrl+C")}               Quit Flow Code`,
          "",
          c.dim("  Multiline: end a line with \\ to continue."),
          c.dim("  Auto-search: detects when queries need web data."),
          "",
        ].join("\n")
      );
      continue;
    }

    // -- /compact --
    if (input === "/compact") {
      const before = history.length;
      const compacted = trimHistory(history);
      history.length = 0;
      history.push(...compacted);
      console.log(
        c.green(
          `  [OK] Compacted: ${before} -> ${history.length} messages.\n`
        )
      );
      continue;
    }

    // -- /status --
    if (input === "/status") {
      const totalUsed = history.reduce((acc, msg) => {
        return acc + estimateTokens(getMessageContent(msg));
      }, 0);
      console.log(
        [
          "",
          c.bold("  Session:"),
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

    // -- /models --
    if (input === "/models") {
      try {
        const res = await state.client.models.list();
        const models = res.data
          .map((m) => m.id)
          .filter((id) => PROVIDERS[state.provider].models.test(id))
          .sort();
        console.log(
          c.bold(`\n  ${PROVIDERS[state.provider].name} Models:`)
        );
        models.forEach((m, i) =>
          console.log(`    ${c.dim(`[${i}]`)} ${m}`)
        );
        const choice = await ask(
          `\n  Select (0-${models.length - 1}): `
        );
        const idx = parseInt(choice, 10);
        if (!isNaN(idx) && idx >= 0 && idx < models.length) {
          const newModel = sanitizeModelId(models[idx]);
          state.model = newModel;
          console.log(c.green(`  [OK] ${newModel}\n`));
          const config = loadConfig();
          config.defaultModel = newModel;
          saveConfig(config);
          history[0] = {
            role: "system",
            content: getSystemPrompt(
              state.intensity,
              process.cwd(),
              state.provider
            ),
          };
        }
      } catch {
        console.log(c.red("  Could not fetch models."));
      }
      continue;
    }

    // -- /provider --
    if (input === "/provider") {
      console.log(c.bold("  Switch provider:"));
      console.log(`    ${c.dim("[1]")} Groq`);
      console.log(`    ${c.dim("[2]")} Cerebras`);
      const pChoice = await ask("  Select (1-2): ");
      const newProvider: Provider = pChoice === "2" ? "cerebras" : "groq";

      if (newProvider === state.provider) {
        console.log(
          c.yellow(
            `  Already using ${PROVIDERS[state.provider].name}.\n`
          )
        );
        continue;
      }

      console.log(
        c.dim(`  Enter ${PROVIDERS[newProvider].name} API Key:`)
      );
      const key = await ask("  API Key: ");
      if (!key || key.length < 10) {
        console.log(c.red("  Invalid key. Provider not changed.\n"));
        continue;
      }

      const newClient = new OpenAI({
        baseURL: PROVIDERS[newProvider].baseURL,
        apiKey: key,
      });

      try {
        console.log(
          c.dim(`  Connecting to ${PROVIDERS[newProvider].name}...`)
        );
        await newClient.models.list();

        state.client = newClient;
        state.provider = newProvider;
        state.model = PROVIDERS[newProvider].defaultModel;

        const config = loadConfig();
        config.provider = newProvider;
        config.apiKey = key;
        config.defaultModel = state.model;
        saveConfig(config);

        history[0] = {
          role: "system",
          content: getSystemPrompt(
            state.intensity,
            process.cwd(),
            state.provider
          ),
        };

        console.log(
          c.green(
            `  [OK] Switched to ${PROVIDERS[newProvider].name} | ${state.model}\n`
          )
        );
        logActivity(`Switched to ${PROVIDERS[newProvider].name}`);
      } catch (err: unknown) {
        const msg =
          err instanceof Error ? err.message : "Connection failed";
        console.error(
          c.red(`  [ERROR] ${PROVIDERS[newProvider].name}: ${msg}\n`)
        );
      }
      continue;
    }

    // -- /settings --
    if (input === "/settings") {
      const config = loadConfig();
      console.log(
        [
          "",
          c.bold("  Settings:"),
          `    ${c.dim("[1]")} Change API Key`,
          `    ${c.dim("[2]")} Switch Provider (${PROVIDERS[config.provider || "groq"].name})`,
          `    ${c.dim("[3]")} Change Model (${config.defaultModel || "default"})`,
          `    ${c.dim("[4]")} Change Intensity (${(config.intensity || "medium").toUpperCase()})`,
          `    ${c.dim("[5]")} View Config`,
          `    ${c.dim("[6]")} Reset All`,
          "",
        ].join("\n")
      );

      const setting = await ask("  Select (1-6): ");

      switch (setting) {
        case "1": {
          const key = await ask("  New API Key: ");
          if (key && key.length > 10) {
            config.apiKey = key;
            saveConfig(config);
            console.log(c.green("  [OK] API key updated.\n"));
          }
          break;
        }
        case "2": {
          console.log(`    ${c.dim("[1]")} Groq`);
          console.log(`    ${c.dim("[2]")} Cerebras`);
          const p = await ask("  Select: ");
          config.provider = p === "2" ? "cerebras" : "groq";
          config.apiKey = "";
          config.defaultModel =
            PROVIDERS[config.provider || "groq"].defaultModel;
          saveConfig(config);
          console.log(
            c.green(
              `  [OK] Provider: ${PROVIDERS[config.provider || "groq"].name}. Restart to apply.\n`
            )
          );
          break;
        }
        case "3": {
          try {
            const client = new OpenAI({
              baseURL: PROVIDERS[config.provider || "groq"].baseURL,
              apiKey: config.apiKey,
            });
            const res = await client.models.list();
            const models = res.data
              .map((m) => m.id)
              .filter((id) =>
                PROVIDERS[config.provider || "groq"].models.test(id)
              )
              .sort();
            console.log(c.bold("\n  Models:"));
            models.forEach((m, i) =>
              console.log(`    ${c.dim(`[${i}]`)} ${m}`)
            );
            const choice = await ask(
              `\n  Select (0-${models.length - 1}): `
            );
            const idx = parseInt(choice, 10);
            if (!isNaN(idx) && idx >= 0 && idx < models.length) {
              config.defaultModel = models[idx];
              saveConfig(config);
              console.log(
                c.green(`  [OK] Model: ${models[idx]}\n`)
              );
            }
          } catch {
            console.log(c.red("  Could not fetch models."));
          }
          break;
        }
        case "4": {
          console.log(`    ${c.dim("[1]")} Low`);
          console.log(`    ${c.dim("[2]")} Medium`);
          console.log(`    ${c.dim("[3]")} High`);
          const iChoice = await ask("  Select: ");
          const map: Record<string, Intensity> = {
            "1": "low",
            "2": "medium",
            "3": "high",
          };
          if (map[iChoice]) {
            config.intensity = map[iChoice];
            saveConfig(config);
            console.log(
              c.green(
                `  [OK] Intensity: ${config.intensity.toUpperCase()}\n`
              )
            );
          }
          break;
        }
        case "5": {
          console.log("");
          console.log(c.bold("  Current config:"));
          console.log(
            `    Provider:   ${PROVIDERS[config.provider || "groq"].name}`
          );
          console.log(
            `    Model:      ${config.defaultModel || "default"}`
          );
          console.log(
            `    Intensity:  ${(config.intensity || "medium").toUpperCase()}`
          );
          console.log(
            `    API Key:    ${config.apiKey ? config.apiKey.slice(0, 6) + "..." + config.apiKey.slice(-4) : "not set"}`
          );
          console.log(`    Config:     ${CONFIG_PATH}`);
          console.log("");
          break;
        }
        case "6": {
          const confirm = await ask(
            c.yellow("  Are you sure? (yes/no): ")
          );
          if (confirm.toLowerCase() === "yes") {
            try {
              fs.unlinkSync(CONFIG_PATH);
            } catch {
              /* ok */
            }
            console.log(
              c.green("  [OK] Config reset. Restart to apply.\n")
            );
          }
          break;
        }
      }
      continue;
    }

    // -- /search --
    if (input.startsWith("/search ")) {
      const query = input.slice(8).trim();
      if (!query) {
        console.log(c.yellow("  Usage: /search <query>"));
        continue;
      }
      console.log(c.dim(`  Searching: ${query}...`));
      const results = await searchWeb(query);
      console.log(formatSearchResults(results, query));

      const resultText = results
        .map(
          (r, i) => `${i + 1}. ${r.title}\n   ${r.snippet}\n   ${r.url}`
        )
        .join("\n\n");

      history.push({
        role: "user",
        content: `Web search results for "${query}":\n\n${resultText}`,
      });
      continue;
    }

    // -- /fetch --
    if (input.startsWith("/fetch ")) {
      const url = input.slice(7).trim();
      if (!url || !url.startsWith("http")) {
        console.log(c.yellow("  Usage: /fetch <https://url>"));
        continue;
      }
      console.log(c.dim(`  Fetching: ${url}...`));
      const content = await fetchUrlContent(url);
      console.log(`\n${c.dim(content.slice(0, 3000))}`);
      if (content.length > 3000) {
        console.log(
          c.dim(`\n  ... (${content.length} chars total)`)
        );
      }
      console.log("");

      history.push({
        role: "user",
        content: `Content from ${url}:\n\n${content}`,
      });
      continue;
    }

    // -- /write <filepath> --
    if (input.startsWith("/write ")) {
      const filepath = input
        .slice(7)
        .trim()
        .replace(/^['"]|['"]$/g, "");
      if (!filepath) {
        console.log(c.yellow("  Usage: /write <filepath>"));
        console.log(
          c.dim(
            "  Then paste or type content, end with a line containing only '---'"
          )
        );
        continue;
      }
      console.log(c.dim(`  Writing to: ${filepath}`));
      console.log(
        c.dim("  Paste/type content. End with a line containing only '---':")
      );
      const lines: string[] = [];
      while (true) {
        const line = await ask(c.dim("  | "));
        if (line.trim() === "---") break;
        lines.push(line);
      }
      const content = lines.join("\n");
      if (content.length === 0) {
        console.log(c.yellow("  Empty content. File not written."));
        continue;
      }
      const target = path.resolve(process.cwd(), filepath);
      try {
        const dir = path.dirname(target);
        if (!fs.existsSync(dir)) {
          fs.mkdirSync(dir, { recursive: true });
        }
        fs.writeFileSync(target, content, "utf-8");
        const rel = path.relative(process.cwd(), target);
        console.log(
          c.green(
            `  [OK] Created: ${rel} (${content.length} chars)\n`
          )
        );
        logActivity(`Created: ${path.basename(target)}`);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Write failed";
        console.error(c.red(`  [ERROR] ${msg}\n`));
      }
      continue;
    }

    // -- Native cd --
    if (input === "cd" || input === "cd ~" || input === "cd $HOME") {
      const home = os.homedir();
      try {
        process.chdir(home);
        console.log(c.green(`  [OK] ${home}\n`));
        const tree = scanDirectory(home, 3);
        if (tree) {
          console.log(c.dim("  Directory:"));
          console.log(c.dim(tree) + "\n");
        }
        history[0] = {
          role: "system",
          content: getSystemPrompt(state.intensity, home, state.provider),
        };
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error(c.red(`  [ERROR] ${msg}`));
      }
      continue;
    }

    if (input.startsWith("cd ")) {
      const raw = input
        .slice(3)
        .replace(/^['"]|['"]$/g, "")
        .trim();
      if (!raw || raw.includes("\0")) {
        console.error(c.red("  [ERROR] Invalid path."));
        continue;
      }
      try {
        const target = path.resolve(raw);
        process.chdir(target);
        const newCwd = process.cwd();
        console.log(c.green(`  [OK] ${newCwd}\n`));
        const tree = scanDirectory(newCwd, 3);
        if (tree) {
          console.log(c.dim("  Directory:"));
          console.log(c.dim(tree) + "\n");
        }
        history[0] = {
          role: "system",
          content: getSystemPrompt(
            state.intensity,
            newCwd,
            state.provider
          ),
        };
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error(c.red(`  [ERROR] ${msg}`));
        console.log(
          c.dim(
            "  Tip: use full paths like C:\\Users\\name\\project or /home/user/project"
          )
        );
      }
      continue;
    }

    // -- Update system prompt --
    history[0] = {
      role: "system",
      content: getSystemPrompt(
        state.intensity,
        process.cwd(),
        state.provider
      ),
    };

    // -- Auto-search --
    let finalInput = input;
    if (needsWebSearch(input)) {
      console.log(c.dim("  Auto-searching for current data..."));
      const searchResults = await searchWeb(input, 3);
      if (searchResults.length > 0) {
        const searchContext = searchResults
          .map((r) => `- ${r.title}: ${r.snippet} (${r.url})`)
          .join("\n");
        finalInput = `${input}\n\n[Web search results -- use this context to answer accurately:]\n${searchContext}`;
        logActivity(`Searched: ${input.slice(0, 50)}`);
      }
    }

    // Add user message
    history.push({ role: "user", content: finalInput });

    // Trim
    const trimmed = trimHistory(history);

    // Temperature
    const temp =
      state.intensity === "low"
        ? 0.0
        : state.intensity === "medium"
          ? 0.2
          : 0.4;

    // Stream response
    try {
      process.stdout.write(c.dim("  Thinking...\r"));
      const { content: reply, usage } = await streamResponse(
        state.client,
        state.model,
        trimmed,
        temp
      );

      if (!reply) {
        console.log(c.red("  No response."));
        history.pop();
        continue;
      }

      console.log(formatResponse(reply) + "\n");
      if (usage.totalTokens > 0) printUsage(usage, history);

      history.push({ role: "assistant", content: reply });

      // Auto-write files from code blocks
      const files = extractFilesFromResponse(reply, process.cwd());
      if (files.length > 0) {
        console.log(c.bold("  Auto-writing detected files:"));
        const wrote = writeFiles(files);
        if (wrote) {
          logActivity(`Auto-wrote ${files.length} file(s)`);
        }
        console.log("");
      }

      // Execute bash blocks
      const blocks = extractBashBlocks(reply);
      const cwd = process.cwd();

      for (const block of blocks) {
        console.log(c.dim(`  > ${block}`));
        const result = await execShellStreaming(block, cwd);
        if (result.ok) {
          console.log(c.green("  [OK] Done.\n"));
          logActivity(`Ran: ${block.slice(0, 60)}`);
        } else {
          console.log(c.red("  [FAIL] Failed.\n"));
          logActivity(`Failed: ${block.slice(0, 60)}`);
        }
        history.push({
          role: "user",
          content: `Terminal output:\n${result.output.slice(0, 2000)}`,
        });
      }

      // Auto-save session for /resume
      saveSession(history, state.model, state.provider, state.intensity);
    } catch (err: unknown) {
      process.stdout.write(" ".repeat(30) + "\r");
      const msg = err instanceof Error ? err.message : String(err);

      if (msg.includes("401") || msg.toLowerCase().includes("invalid")) {
        console.error(
          c.red(
            `  [ERROR] Invalid API key for ${PROVIDERS[state.provider].name}. Delete ~/.flow-code-config and restart.`
          )
        );
      } else if (msg.includes("404") || msg.includes("does not exist")) {
        console.error(
          c.red(`  [ERROR] Model '${state.model}' not found.`)
        );
        console.log(c.dim("  Type /models to re-select a valid model."));
        state.model = PROVIDERS[state.provider].defaultModel;
        const config = loadConfig();
        config.defaultModel = state.model;
        saveConfig(config);
        console.log(c.green(`  [OK] Default: ${state.model}\n`));
      } else if (msg.includes("429")) {
        console.error(
          c.yellow("  Rate limited. Wait a moment.")
        );
      } else if (msg.includes("503")) {
        console.error(
          c.yellow("  Model overloaded. Try again.")
        );
      } else {
        console.error(c.red(`  [ERROR] ${msg}`));
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
    console.log(c.dim("\n\n  Goodbye.\n"));
    rl.close();
    process.exit(0);
  });

  process.on("SIGTERM", () => {
    rl.close();
    process.exit(0);
  });

  process.on("unhandledRejection", (err) => {
    console.error(c.red(`\n  [UNHANDLED] ${err}`));
  });

  process.on("uncaughtException", (err) => {
    console.error(c.red(`\n  [UNCAUGHT] ${err.message}`));
    process.exit(1);
  });

  printBanner();

  setup()
    .then(async ({ client, model, intensity, provider }) => {
      console.log(c.dim(`  Working directory: ${process.cwd()}`));
      const dirInput = await ask(
        c.bold("  Project directory (Enter to skip): ")
      );
      if (dirInput.trim()) {
        const cleaned = dirInput
          .trim()
          .replace(/^['"]|['"]$/g, "");
        const target = path.resolve(cleaned);
        try {
          process.chdir(target);
          console.log(
            c.green(`  Switched to: ${process.cwd()}\n`)
          );
        } catch {
          console.log(
            c.yellow(
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
            c.dim(
              `  Last session: ${formatSessionAge(session.timestamp)} -- type /resume to continue`
            )
          );
        }
      }
      console.log(
        c.green(
          "  Ready! Type /cmds for commands, /help for help.\n"
        )
      );

      logActivity("Started session");
      return run(client, model, intensity, provider);
    })
    .catch((err) => {
      console.error(
        c.red(`\n  [FATAL] ${err.message || err}`)
      );
      process.exit(1);
    });
}

main();
