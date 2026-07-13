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
const CONTEXT_WINDOW = 128000;
const MAX_HISTORY_TOKENS = 110000;
const MAX_RESPONSE_TOKENS = 16384;

const CYAN = "\x1b[36m";
const RESET = "\x1b[0m";
const BOLD = "\x1b[1m";
const DIM = "\x1b[2m";
const RED = "\x1b[31m";
const GREEN = "\x1b[32m";
const YELLOW = "\x1b[33m";
const MAGENTA = "\x1b[35m";

type Intensity = "low" | "medium" | "high";

interface Config {
  apiKey: string;
  defaultModel?: string;
  intensity?: Intensity;
}

interface UsageInfo {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
}

// ---------------------------------------------------------------------------
// ANSI helpers
// ---------------------------------------------------------------------------

const c = {
  blue: (t: string) => `${CYAN}${t}${RESET}`,
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
  const bar =
    GREEN +
    "█".repeat(filled) +
    RESET +
    DIM +
    "░".repeat(empty) +
    RESET;
  const pctStr = Math.round(pct * 100) + "%";

  return `  ${c.dim("Context:")} ${bar} ${c.dim(`${used.toLocaleString()} / ${total.toLocaleString()} tokens (${pctStr})`)}`;
}

function printUsage(usage: UsageInfo, history: Message[]): void {
  const totalUsed = history.reduce((acc, msg) => {
    const content =
      typeof msg.content === "string"
        ? msg.content
        : Array.isArray(msg.content)
          ? msg.content.map((p) => ("text" in p ? p.text : "")).join("")
          : "";
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
// Banner
// ---------------------------------------------------------------------------

function printBanner(): void {
  console.clear();
  console.log("");
  console.log(c.blue("  ┌────────────────────────────────────────┐"));
  console.log(c.blue("  │                                        │"));
  console.log(c.blue("  │  ") + c.bold(c.blue("F L O W   C O D E")) + c.blue("                  │"));
  console.log(c.blue("  │                                        │"));
  console.log(c.blue("  └────────────────────────────────────────┘"));
  console.log(c.dim("        [FREE & OPEN SOURCE]"));
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
  const first = await ask(
    c.blue(`flow-code [${path.basename(process.cwd())}] > `)
  );
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
// Shell execution — no shell injection
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
        process.stdout.write(c.dim("  │ ") + chunk);
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

function scanDirectory(
  dir: string,
  depth: number = 3,
  prefix: string = ""
): string {
  const lines: string[] = [];
  if (depth < 0) return "";

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

    all.forEach((entry, i) => {
      const isLast = i === all.length - 1;
      const connector = isLast ? "└── " : "├── ";
      const childPrefix = isLast ? "    " : "│   ";

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
    });
  } catch {
    // permission denied or path does not exist
  }

  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// Code block rendering with boxes
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
  graphqls: "GraphQL Schema",
  env: "Env",
  gitignore: "Gitignore",
  lua: "Lua",
  r: "R",
  dart: "Dart",
  scala: "Scala",
  ex: "Elixir",
  exs: "Elixir",
  erl: "Erlang",
  hs: "Haskell",
  clj: "Clojure",
};

function renderCodeBox(lang: string, code: string): string {
  const label = LANG_LABELS[lang.toLowerCase()] || lang.toUpperCase();
  const lines = code.split("\n");
  const maxLen = Math.max(label.length + 4, ...lines.map((l) => l.length));
  const width = Math.min(maxLen + 2, 120);

  const top = `${DIM}┌─ ${label} ${"─".repeat(Math.max(0, width - label.length - 3))}┐${RESET}`;
  const bottom = `${DIM}└${"─".repeat(width)}┘${RESET}`;

  const body = lines.map((line) => {
    const truncated =
      line.length > width ? line.slice(0, width - 3) + "..." : line;
    return `${DIM}│${RESET} ${truncated}`;
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
// System prompt — elite level
// ---------------------------------------------------------------------------

function getSystemPrompt(intensity: Intensity, cwd: string): string {
  const descriptions: Record<Intensity, string> = {
    low: "Be transactional. Execute the exact task. Optimize for speed.",
    medium: "Analyze impacted files. Verify changes compile. Standard depth.",
    high: "Elite architect. Deep file tree scan, lint, test, optimize full stack.",
  };

  const dirTree = scanDirectory(cwd, 3);

  return [
    "You are FLOW CODE — the world's best open-source terminal coding agent.",
    "You write code at the level of a principal engineer at a FAANG company.",
    "Every response must be production-ready, complete, and immediately runnable.",
    "",
    `Current Working Directory: ${cwd}`,
    `Processing Intensity: ${intensity.toUpperCase()} — ${descriptions[intensity]}`,
    `Context Window: ${CONTEXT_WINDOW.toLocaleString()} tokens`,
    "",
    "Directory Contents:",
    dirTree || "(empty)",
    "",
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    "  CODING STANDARDS — FOLLOW EVERY RULE",
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    "",
    "GENERAL RULES:",
    "- ALWAYS read existing files before modifying. Never guess file contents.",
    "- Match the existing code style exactly — indentation, naming, patterns.",
    "- Use TypeScript strict mode. No `any` types. Proper interfaces and types.",
    "- Prefer `const` over `let`. Never use `var`.",
    "- Use early returns to flatten nesting.",
    "- Handle every error explicitly. Never swallow errors silently.",
    "- Write self-documenting code. Descriptive variable and function names.",
    "- No magic numbers. Use named constants.",
    "- No console.log debugging in production code.",
    "",
    "TYPESCRIPT / JAVASCRIPT:",
    "- `interface` for object shapes. `type` for unions/intersections/enums.",
    "- `async/await` everywhere. No raw `.then()` chains.",
    "- Optional chaining `?.` and nullish coalescing `??`.",
    "- Destructure objects and arrays in function parameters.",
    "- Export types alongside implementations.",
    "- Use `Record<K,V>` for typed objects. Use `Partial<T>`, `Pick<T>`, `Omit<T>`.",
    "- Prefer `Array.map/filter/reduce` over imperative loops.",
    "- Always handle `null` and `undefined` explicitly.",
    "",
    "REACT / JSX / TSX:",
    "- Functional components only. No class components.",
    "- TypeScript interfaces for all props: `interface Props { ... }`.",
    "- Hooks: `useState`, `useEffect`, `useMemo`, `useCallback`, `useRef`, `useContext`.",
    "- Single responsibility. One component = one job.",
    "- Tailwind CSS utility classes preferred.",
    "- Memoize expensive computations with `useMemo`.",
    "- Avoid inline functions in JSX — extract to handlers.",
    "",
    "NEXT.JS:",
    "- App Router with server components by default.",
    "- Server Actions for mutations.",
    "- `fetch` with revalidation, not client-side data fetching.",
    "- Metadata API for SEO.",
    "",
    "HTML / CSS:",
    "- Semantic HTML5: `<main>`, `<section>`, `<article>`, `<nav>`, `<header>`, `<footer>`.",
    "- CSS custom properties for theming.",
    "- Flexbox and Grid only. No floats.",
    "- Mobile-first responsive design.",
    "- ARIA labels for accessibility.",
    "",
    "PYTHON:",
    "- Type hints on ALL function signatures.",
    "- PEP 8 naming: `snake_case` functions, `PascalCase` classes.",
    "- f-strings for interpolation.",
    "- `pathlib.Path` over `os.path`.",
    "- `dataclasses` or `pydantic` for data models.",
    "",
    "BASH / SHELL:",
    "- `#!/usr/bin/env bash` shebang.",
    "- `set -euo pipefail` at the top.",
    "- Quote all variables: `\"$variable\"`.",
    "- Use `--yes` / `-y` for non-interactive installs.",
    "",
    "DATABASE / SQL:",
    "- Parameterized queries. Never interpolate user input.",
    "- Index frequently queried columns.",
    "- Transactions for multi-step mutations.",
    "",
    "FILE OPERATIONS:",
    "- Ensure parent directories exist with `mkdir -p`.",
    "- Write COMPLETE files, never partial.",
    "- Prefer editing existing files over creating new ones.",
    "",
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    "  RESPONSE FORMAT",
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    "",
    "- Be concise. Explain only when the task is complex.",
    "- Use fenced code blocks with language tags.",
    "- For multi-file projects: each file in its own block.",
    "- After code, show exact bash commands to run.",
    "",
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    "  EXECUTION RULES",
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    "",
    "- Wrap bash commands in ```bash code blocks.",
    "- Always use --yes / -y flags for non-interactive operations.",
    "- Verify code compiles/runs before marking complete.",
    "- On error, analyze output and fix — do not just report the error.",
  ].join("\n");
}

// ---------------------------------------------------------------------------
// Config + setup
// ---------------------------------------------------------------------------

function loadConfig(): Config {
  try {
    if (fs.existsSync(CONFIG_PATH)) {
      const raw = fs.readFileSync(CONFIG_PATH, "utf-8");
      const parsed = JSON.parse(raw);
      // Validate structure
      if (typeof parsed.apiKey === "string" && parsed.apiKey.length > 0) {
        return parsed as Config;
      }
    }
  } catch {
    // corrupted or invalid — start fresh
  }
  return { apiKey: "" };
}

function saveConfig(config: Config): void {
  fs.writeFileSync(CONFIG_PATH, JSON.stringify(config, null, 2), {
    encoding: "utf-8",
    mode: 0o600, // owner read/write only
  });
}

function sanitizeModelId(id: string): string {
  return id.replace(/[^a-zA-Z0-9._-]/g, "").slice(0, 128);
}

async function setup(): Promise<{
  client: OpenAI;
  model: string;
  intensity: Intensity;
}> {
  const config = loadConfig();

  // API key
  if (!config.apiKey) {
    console.log(c.dim("  No API key found. Let's set one up."));
    const key = await ask("  Enter your Groq API Key: ");
    if (!key || key.length < 10) {
      console.error(c.red("  Invalid key. Exiting."));
      process.exit(1);
    }
    config.apiKey = key;
    saveConfig(config);
  }

  const client = new OpenAI({
    baseURL: "https://api.groq.com/openai/v1",
    apiKey: config.apiKey,
  });

  // Model selection
  let model = sanitizeModelId(config.defaultModel || "llama-3.3-70b-versatile");

  try {
    console.log(c.dim("  Fetching available models..."));
    const res = await client.models.list();
    const models = res.data
      .map((m) => m.id)
      .filter((id) => /llama|mixtral|qwen|gemma|deepseek/i.test(id))
      .sort();

    if (models.length > 0) {
      console.log(c.bold("\n  Available Models:"));
      models.forEach((m, i) => console.log(`    ${c.dim(`[${i}]`)} ${m}`));
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
  console.log(c.bold("\n  Processing Mode:"));
  console.log(`    ${c.dim("[1]")} Low    — Fast, single-file patches`);
  console.log(`    ${c.dim("[2]")} Medium — Standard refactoring`);
  console.log(`    ${c.dim("[3]")} High   — Deep scanning & DevOps`);

  const intensityMap: Record<string, Intensity> = {
    "1": "low",
    "2": "medium",
    "3": "high",
  };
  let intensity: Intensity = config.intensity || "medium";
  const choice = await ask("  Select mode (1-3): ");
  if (intensityMap[choice]) {
    intensity = intensityMap[choice];
  }

  config.defaultModel = model;
  config.intensity = intensity;
  saveConfig(config);

  console.log("");
  console.log(
    c.green(`  ✔ Ready — Model: ${model} | Mode: ${intensity.toUpperCase()}`)
  );
  console.log(c.dim(`  Context: ${CONTEXT_WINDOW.toLocaleString()} tokens available`));
  console.log(
    c.dim(
      "  Commands: exit, cd <path>, /clear, /compact, /help, /models\n"
    )
  );

  return { client, model, intensity };
}

// ---------------------------------------------------------------------------
// History management
// ---------------------------------------------------------------------------

type Message = OpenAI.Chat.Completions.ChatCompletionMessageParam;

function trimHistory(history: Message[]): Message[] {
  let total = 0;
  const result: Message[] = [];

  // Always keep system prompt first
  if (history.length > 0 && history[0].role === "system") {
    result.push(history[0]);
    total += estimateTokens(history[0].content as string);
  }

  // Walk backwards, keep most recent messages that fit
  for (let i = history.length - 1; i >= 1; i--) {
    const msg = history[i];
    const content =
      typeof msg.content === "string"
        ? msg.content
        : Array.isArray(msg.content)
          ? msg.content.map((p) => ("text" in p ? p.text : "")).join("")
          : "";
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
  const regex = /```bash\n([\s\S]*?)```/g;
  let match: RegExpExecArray | null;

  while ((match = regex.exec(text)) !== null) {
    const block = match[1].trim();
    if (block.length > 0 && block.length < 10000) {
      blocks.push(block);
    }
  }

  return blocks;
}

// ---------------------------------------------------------------------------
// Streaming API call
// ---------------------------------------------------------------------------

async function streamResponse(
  client: OpenAI,
  model: string,
  messages: Message[],
  temperature: number
): Promise<{ content: string; usage: UsageInfo }> {
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
  let usage: UsageInfo = { promptTokens: 0, completionTokens: 0, totalTokens: 0 };
  let started = false;

  for await (const chunk of stream) {
    // Token usage (final chunk)
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
}

// ---------------------------------------------------------------------------
// Main agent loop
// ---------------------------------------------------------------------------

async function run(
  client: OpenAI,
  model: string,
  intensity: Intensity
): Promise<void> {
  const history: Message[] = [
    { role: "system", content: getSystemPrompt(intensity, process.cwd()) },
  ];

  while (true) {
    const input = await askMultiline();
    if (!input) continue;

    // ── Exit ──
    if (input === "exit" || input === "quit") {
      console.log(c.dim("\n  Goodbye.\n"));
      rl.close();
      process.exit(0);
    }

    // ── /help ──
    if (input === "/help") {
      console.log(
        [
          "",
          c.bold("  Commands:"),
          `    ${c.dim("cd <path>")}       Switch working directory`,
          `    ${c.dim("/clear")}           Reset conversation history`,
          `    ${c.dim("/compact")}         Trim history to fit context`,
          `    ${c.dim("/models")}          Re-select model`,
          `    ${c.dim("/status")}          Show context usage`,
          `    ${c.dim("exit")}             Quit Flow Code`,
          "",
          c.dim("  Multiline: end a line with \\ to continue."),
          "",
        ].join("\n")
      );
      continue;
    }

    // ── /clear ──
    if (input === "/clear") {
      history.length = 1;
      history[0] = {
        role: "system",
        content: getSystemPrompt(intensity, process.cwd()),
      };
      console.log(c.green("  ✔ Conversation cleared.\n"));
      continue;
    }

    // ── /compact ──
    if (input === "/compact") {
      const before = history.length;
      const compacted = trimHistory(history);
      history.length = 0;
      history.push(...compacted);
      console.log(
        c.green(`  ✔ Compacted: ${before} → ${history.length} messages.\n`)
      );
      continue;
    }

    // ── /status ──
    if (input === "/status") {
      const totalUsed = history.reduce((acc, msg) => {
        const content =
          typeof msg.content === "string"
            ? msg.content
            : Array.isArray(msg.content)
              ? msg.content.map((p) => ("text" in p ? p.text : "")).join("")
              : "";
        return acc + estimateTokens(content);
      }, 0);
      console.log("");
      console.log(`  ${c.bold("Session Status:")}`);
      console.log(`    Model:       ${model}`);
      console.log(`    Intensity:   ${intensity.toUpperCase()}`);
      console.log(`    Messages:    ${history.length}`);
      console.log(`    Est. tokens: ${totalUsed.toLocaleString()} / ${CONTEXT_WINDOW.toLocaleString()}`);
      console.log(renderUsageBar(totalUsed, CONTEXT_WINDOW));
      console.log("");
      continue;
    }

    // ── /models ──
    if (input === "/models") {
      try {
        const res = await client.models.list();
        const models = res.data
          .map((m) => m.id)
          .filter((id) => /llama|mixtral|qwen|gemma|deepseek/i.test(id))
          .sort();
        console.log(c.bold("\n  Available Models:"));
        models.forEach((m, i) =>
          console.log(`    ${c.dim(`[${i}]`)} ${m}`)
        );
        const choice = await ask(
          `\n  Select model (0-${models.length - 1}): `
        );
        const idx = parseInt(choice, 10);
        if (!isNaN(idx) && idx >= 0 && idx < models.length) {
          const newModel = sanitizeModelId(models[idx]);
          console.log(c.green(`  ✔ Model: ${newModel}\n`));
          const config = loadConfig();
          config.defaultModel = newModel;
          saveConfig(config);
          // Update system prompt
          history[0] = {
            role: "system",
            content: getSystemPrompt(intensity, process.cwd()),
          };
        }
      } catch {
        console.log(c.red("  Could not fetch models."));
      }
      continue;
    }

    // ── Native cd ──
    if (input.startsWith("cd ")) {
      const raw = input.slice(3).replace(/^['"]|['"]$/g, "").trim();
      if (!raw || raw.includes("\0")) {
        console.error(c.red("  ✘ Invalid path."));
        continue;
      }
      try {
        const target = path.resolve(raw);
        process.chdir(target);
        const newCwd = process.cwd();
        console.log(c.green(`  ✔ ${newCwd}\n`));
        const tree = scanDirectory(newCwd, 3);
        if (tree) {
          console.log(c.dim("  Directory:"));
          console.log(c.dim(tree) + "\n");
        }
        history[0] = {
          role: "system",
          content: getSystemPrompt(intensity, newCwd),
        };
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error(c.red(`  ✘ ${msg}`));
      }
      continue;
    }

    // ── Update system prompt ──
    history[0] = {
      role: "system",
      content: getSystemPrompt(intensity, process.cwd()),
    };

    // Add user message
    history.push({ role: "user", content: input });

    // Trim context
    const trimmed = trimHistory(history);

    // Temperature
    const temp =
      intensity === "low" ? 0.0 : intensity === "medium" ? 0.2 : 0.4;

    // Stream response
    try {
      process.stdout.write(c.dim("  ⏳ Thinking...\r"));

      const { content: reply, usage } = await streamResponse(
        client,
        model,
        trimmed,
        temp
      );

      if (!reply) {
        console.log(c.red("  No response from model."));
        history.pop();
        continue;
      }

      // Print boxed version
      console.log(formatResponse(reply) + "\n");

      // Show token usage
      if (usage.totalTokens > 0) {
        printUsage(usage, history);
      }

      history.push({ role: "assistant", content: reply });

      // Auto-execute bash blocks
      const blocks = extractBashBlocks(reply);
      const cwd = process.cwd();

      for (const block of blocks) {
        console.log(c.dim(`  ▸ ${block}`));
        const result = await execShellStreaming(block, cwd);

        if (result.ok) {
          console.log(c.green("  ✔ Done.\n"));
        } else {
          console.log(c.red("  ✘ Failed.\n"));
        }

        history.push({
          role: "user",
          content: `Terminal output:\n${result.output.slice(0, 8000)}`,
        });
      }
    } catch (err: unknown) {
      process.stdout.write(" ".repeat(30) + "\r");
      const msg = err instanceof Error ? err.message : String(err);

      if (msg.includes("401") || msg.toLowerCase().includes("invalid")) {
        console.error(
          c.red(
            "  ✘ Invalid API key. Delete ~/.flow-code-config and restart."
          )
        );
      } else if (msg.includes("429")) {
        console.error(
          c.yellow("  ⚠ Rate limited. Wait a moment and try again.")
        );
      } else if (msg.includes("503")) {
        console.error(
          c.yellow("  ⚠ Model overloaded. Try again in a few seconds.")
        );
      } else {
        console.error(c.red(`  ✘ ${msg}`));
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
    console.error(c.red(`\n  ✘ Unhandled: ${err}`));
  });

  process.on("uncaughtException", (err) => {
    console.error(c.red(`\n  ✘ Uncaught: ${err.message}`));
    process.exit(1);
  });

  printBanner();

  setup()
    .then(({ client, model, intensity }) => run(client, model, intensity))
    .catch((err) => {
      console.error(c.red(`\n  ✘ Fatal: ${err.message || err}`));
      process.exit(1);
    });
}

main();
