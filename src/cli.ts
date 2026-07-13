#!/usr/bin/env node

import OpenAI from "openai";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";
import * as readline from "readline";
import { execSync, spawn } from "child_process";

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const CONFIG_PATH = path.join(os.homedir(), ".flow-code-config");
const MAX_HISTORY_TOKENS = 12000;
const BANNER_COLOR = "\x1b[36m";
const RESET = "\x1b[0m";
const BOLD = "\x1b[1m";
const DIM = "\x1b[2m";
const RED = "\x1b[31m";
const GREEN = "\x1b[32m";
const YELLOW = "\x1b[33m";

type Intensity = "low" | "medium" | "high";

interface Config {
  apiKey: string;
  defaultModel?: string;
  intensity?: Intensity;
}

// ---------------------------------------------------------------------------
// ANSI helpers
// ---------------------------------------------------------------------------

function blue(text: string): string {
  return `${BANNER_COLOR}${text}${RESET}`;
}

function green(text: string): string {
  return `${GREEN}${text}${RESET}`;
}

function red(text: string): string {
  return `${RED}${text}${RESET}`;
}

function dim(text: string): string {
  return `${DIM}${text}${RESET}`;
}

function bold(text: string): string {
  return `${BOLD}${text}${RESET}`;
}

// ---------------------------------------------------------------------------
// Banner
// ---------------------------------------------------------------------------

function printBanner(): void {
  console.clear();
  console.log(blue(""));
  console.log(blue("  ┌────────────────────────────────────────┐"));
  console.log(blue("  │                                        │"));
  console.log(blue("  │  ") + bold(blue("F L O W   C O D E")) + blue("                  │"));
  console.log(blue("  │                                        │"));
  console.log(blue("  └────────────────────────────────────────┘"));
  console.log(dim("        [FREE & OPEN SOURCE]"));
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

// ---------------------------------------------------------------------------
// Shell execution
// ---------------------------------------------------------------------------

function execShell(cmd: string, cwd: string): { ok: boolean; output: string } {
  try {
    const output = execSync(cmd, {
      encoding: "utf-8",
      stdio: "pipe",
      cwd,
      timeout: 30000,
    });
    return { ok: true, output };
  } catch (err: unknown) {
    const e = err as { stderr?: string; message?: string };
    return { ok: false, output: e.stderr || e.message || "Unknown error" };
  }
}

function execShellStreaming(cmd: string, cwd: string): Promise<{ ok: boolean; output: string }> {
  return new Promise((resolve) => {
    const parts = cmd.split(/\s+/);
    const bin = parts[0];
    const args = parts.slice(1);

    let stdout = "";
    let stderr = "";

    try {
      const child = spawn(bin, args, { cwd, shell: true, stdio: "pipe" });

      child.stdout?.on("data", (data: Buffer) => {
        const chunk = data.toString();
        stdout += chunk;
        process.stdout.write(dim("  | ") + chunk);
      });

      child.stderr?.on("data", (data: Buffer) => {
        const chunk = data.toString();
        stderr += chunk;
        process.stderr.write(red("  ! ") + chunk);
      });

      child.on("close", (code) => {
        resolve({
          ok: code === 0,
          output: code === 0 ? stdout : stderr || stdout,
        });
      });

      child.on("error", () => {
        resolve({ ok: false, output: stderr || "Failed to spawn process" });
      });
    } catch {
      resolve({ ok: false, output: "Failed to execute command" });
    }
  });
}

// ---------------------------------------------------------------------------
// System prompt
// ---------------------------------------------------------------------------

function getSystemPrompt(intensity: Intensity): string {
  const descriptions: Record<Intensity, string> = {
    low: "Be transactional. Execute the exact task requested. Optimize for raw completion speed.",
    medium: "Analyze impacted files directly. Verify changes work before resolving.",
    high: "Act as an elite software architect. Deeply scan the file tree, run linting, tests, and optimize across integrations.",
  };

  return [
    "You are FLOW CODE, a free open-source terminal coding agent powered by Groq.",
    "",
    `Current Working Directory: ${process.cwd()}`,
    `Processing Intensity: ${intensity.toUpperCase()} — ${descriptions[intensity]}`,
    "",
    "Rules:",
    "- Always wrap executable bash commands in ```bash code blocks.",
    "- Always use non-interactive flags (e.g. --yes, -y) to prevent hanging.",
    "- Do NOT wrap multiple commands in a single code block unless they are sequential.",
    "- Be concise. Explain only when necessary.",
    "- Never fabricate file contents. Read files first.",
  ].join("\n");
}

// ---------------------------------------------------------------------------
// Config + setup
// ---------------------------------------------------------------------------

function loadConfig(): Config {
  try {
    if (fs.existsSync(CONFIG_PATH)) {
      return JSON.parse(fs.readFileSync(CONFIG_PATH, "utf-8")) as Config;
    }
  } catch {
    // corrupted config — start fresh
  }
  return { apiKey: "" };
}

function saveConfig(config: Config): void {
  fs.writeFileSync(CONFIG_PATH, JSON.stringify(config, null, 2), "utf-8");
}

async function setup(): Promise<{ client: OpenAI; model: string; intensity: Intensity }> {
  const config = loadConfig();

  // API key
  if (!config.apiKey) {
    console.log(dim("  No API key found. Let's set one up."));
    const key = await ask("  Enter your Groq API Key: ");
    if (!key) {
      console.error(red("  No key provided. Exiting."));
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
  let model = config.defaultModel || "llama-3.3-70b-versatile";

  try {
    console.log(dim("  Fetching available models..."));
    const res = await client.models.list();
    const models = res.data
      .map((m) => m.id)
      .filter((id) => /llama|mixtral|qwen/i.test(id))
      .sort();

    if (models.length > 0) {
      console.log(bold("\n  Available Models:"));
      models.forEach((m, i) => console.log(`    ${dim(`[${i}]`)} ${m}`));
      const choice = await ask(`\n  Select model (0-${models.length - 1}): `);
      const idx = parseInt(choice, 10);
      if (!isNaN(idx) && idx >= 0 && idx < models.length) {
        model = models[idx];
      }
    }
  } catch {
    console.log(dim("  Could not fetch models. Using default."));
  }

  // Intensity
  console.log(bold("\n  Processing Mode:"));
  console.log(`    ${dim("[1]")} Low    — Fast, single-file patches`);
  console.log(`    ${dim("[2]")} Medium — Standard refactoring`);
  console.log(`    ${dim("[3]")} High   — Deep scanning & DevOps`);

  const intensityMap: Record<string, Intensity> = { "1": "low", "2": "medium", "3": "high" };
  let intensity: Intensity = config.intensity || "medium";
  const choice = await ask("  Select mode (1-3): ");
  if (intensityMap[choice]) {
    intensity = intensityMap[choice];
  }

  // Persist
  config.defaultModel = model;
  config.intensity = intensity;
  saveConfig(config);

  console.log("");
  console.log(green(`  ✔ Ready — Model: ${model} | Mode: ${intensity.toUpperCase()}`));
  console.log(dim("  Type 'exit' to quit, 'cd <path>' to switch directories.\n"));

  return { client, model, intensity };
}

// ---------------------------------------------------------------------------
// History management
// ---------------------------------------------------------------------------

type Message = OpenAI.Chat.Completions.ChatCompletionMessageParam;

function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

function trimHistory(history: Message[]): Message[] {
  let total = 0;
  const trimmed: Message[] = [];

  // Always keep the system prompt
  if (history.length > 0 && history[0].role === "system") {
    trimmed.push(history[0]);
    total += estimateTokens(history[0].content as string);
  }

  // Walk backwards, keeping as many recent messages as fit
  for (let i = history.length - 1; i >= 1; i--) {
    const msg = history[i];
    const tokens = estimateTokens(msg.content as string);
    if (total + tokens > MAX_HISTORY_TOKENS) break;
    total += tokens;
    trimmed.splice(1, 0, msg); // insert after system prompt
  }

  return trimmed;
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
    if (block.length > 0) {
      blocks.push(block);
    }
  }

  return blocks;
}

// ---------------------------------------------------------------------------
// Main agent loop
// ---------------------------------------------------------------------------

async function run(client: OpenAI, model: string, intensity: Intensity): Promise<void> {
  const history: Message[] = [
    { role: "system", content: getSystemPrompt(intensity) },
  ];

  while (true) {
    const cwd = process.cwd();
    const folder = path.basename(cwd);
    const input = await ask(blue(`flow-code [${folder}] > `));

    if (!input) continue;

    // Exit
    if (input === "exit" || input === "quit") {
      console.log(dim("\n  Goodbye.\n"));
      rl.close();
      process.exit(0);
    }

    // Native cd
    if (input.startsWith("cd ")) {
      const target = path.resolve(input.slice(3).replace(/^['"]|['"]$/g, ""));
      try {
        process.chdir(target);
        console.log(green(`  ✔ ${process.cwd()}`));
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error(red(`  ✘ ${msg}`));
      }
      continue;
    }

    // Update system prompt with current cwd
    history[0] = { role: "system", content: getSystemPrompt(intensity) };

    // Add user message
    history.push({ role: "user", content: input });

    // Trim context window
    const trimmed = trimHistory(history);

    // Call Groq
    try {
      process.stdout.write(dim("  ⏳ Thinking...\r"));

      const res = await client.chat.completions.create({
        model,
        messages: trimmed,
        temperature: intensity === "low" ? 0.0 : intensity === "medium" ? 0.1 : 0.3,
        stream: false,
      });

      // Clear "thinking" line
      process.stdout.write(" ".repeat(20) + "\r");

      const reply = res.choices[0]?.message?.content ?? "";

      if (!reply) {
        console.log(red("  No response from model."));
        history.pop(); // remove failed user message
        continue;
      }

      // Print reply
      console.log(`\n  ${bold(blue("Flow"))} ${reply}\n`);

      history.push({ role: "assistant", content: reply });

      // Execute bash blocks
      const blocks = extractBashBlocks(reply);
      for (const block of blocks) {
        console.log(dim(`  ▸ ${block}`));
        const result = await execShellStreaming(block, cwd);
        if (result.ok) {
          console.log(green("  ✔ Done.\n"));
        } else {
          console.log(red("  ✘ Failed.\n"));
        }
        history.push({
          role: "user",
          content: `Terminal output:\n${result.output}`,
        });
      }
    } catch (err: unknown) {
      process.stdout.write(" ".repeat(20) + "\r");
      const msg = err instanceof Error ? err.message : String(err);

      // API key error — suggest reconfigure
      if (msg.includes("401") || msg.toLowerCase().includes("invalid")) {
        console.error(red("  ✘ Invalid API key. Delete ~/.flow-code-config and restart."));
      } else {
        console.error(red(`  ✘ ${msg}`));
      }

      history.pop(); // remove failed user message
    }
  }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

function main(): void {
  // Graceful shutdown
  rl.on("close", () => process.exit(0));

  process.on("SIGINT", () => {
    console.log(dim("\n\n  Goodbye.\n"));
    rl.close();
    process.exit(0);
  });

  process.on("SIGTERM", () => {
    rl.close();
    process.exit(0);
  });

  // Handle unhandled rejections
  process.on("unhandledRejection", (err) => {
    console.error(red(`\n  ✘ Unhandled error: ${err}`));
  });

  printBanner();

  setup()
    .then(({ client, model, intensity }) => run(client, model, intensity))
    .catch((err) => {
      console.error(red(`\n  ✘ Fatal: ${err}`));
      process.exit(1);
    });
}

main();
