#!/usr/bin/env node

import OpenAI from 'openai';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import * as readline from 'readline';
import { execSync } from 'child_process';

const CONFIG_PATH = path.join(os.homedir(), '.flow-code-config');

interface Config {
  apiKey: string;
  defaultModel?: string;
  intensity?: 'low' | 'medium' | 'high';
}

let activeConfig: Config = { apiKey: '' };
let currentModel = '';
let currentIntensity: 'low' | 'medium' | 'high' = 'medium';

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

const askQuestion = (query: string): Promise<string> => {
  return new Promise((resolve) => rl.question(query, resolve));
};

function runBashCommand(cmd: string): string {
  try {
    const output = execSync(cmd, { encoding: 'utf-8', stdio: 'pipe' });
    return `Success:\n${output}`;
  } catch (error: any) {
    return `Error:\n${error.stderr || error.message}`;
  }
}

function getSystemPrompt(intensity: 'low' | 'medium' | 'high'): string {
  const intensityRules = {
    low: "Be transactional. Execute the exact task requested without extra directory lookups. Optimize for raw completion speed.",
    medium: "Analyze local files directly impacted by the task. Verify code changes compile successfully before resolving.",
    high: "Act as an elite software architect. Deeply scan the file tree, run linting checks, build automated tests via bash, and optimize schemas across GitHub, Vercel, or Supabase integrations.",
  };

  return `You are FLOW CODE, a free open-source terminal coding agent powered by Groq.
Current Active Directory: ${process.cwd()}
Processing Intensity Level: ${intensity.toUpperCase()} (${intensityRules[intensity]})

Operational Instructions:
- Wrap executing bash or tool code steps inside standard \`\`\`bash markdown codeblocks.
- Append explicit non-interactive flags (e.g., '--yes', '-y') to prevent terminal hanging states.`;
}

async function initializeSetup() {
  if (fs.existsSync(CONFIG_PATH)) {
    activeConfig = JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf-8'));
  }

  if (!activeConfig.apiKey) {
    console.log("\x1b[34m\u{1F535} Welcome to Flow Code. Let's configure your Groq workspace.\x1b[0m");
    const key = await askQuestion("Enter your Groq API Key: ");
    activeConfig.apiKey = key.trim();
    fs.writeFileSync(CONFIG_PATH, JSON.stringify(activeConfig, null, 2));
  }

  const client = new OpenAI({
    baseURL: "https://api.groq.com/openai/v1",
    apiKey: activeConfig.apiKey,
  });

  try {
    console.log("\x1b[34m\u23F3 Syncing available open-source models...\x1b[0m");
    const modelsList = await client.models.list();
    const validModels = modelsList.data
      .map((m) => m.id)
      .filter((id) => id.includes('llama') || id.includes('mixtral') || id.includes('qwen'));

    console.log("\n\x1b[1mAvailable Production Models:\x1b[0m");
    validModels.forEach((mod, index) => console.log(`  [${index}] ${mod}`));

    if (validModels.length === 0) {
      console.log("No filtered models found. Using default: llama-3.3-70b-versatile");
      currentModel = 'llama-3.3-70b-versatile';
    } else {
      const choice = await askQuestion("\nSelect model index number: ");
      const parsed = parseInt(choice);
      currentModel = validModels[parsed] || 'llama-3.3-70b-versatile';
    }
  } catch (err) {
    console.log("Using default fallback model: llama-3.3-70b-versatile");
    currentModel = 'llama-3.3-70b-versatile';
  }

  console.log("\n\x1b[1mSelect Processing Mode:\x1b[0m");
  console.log("  [1] Low    (Fast, single-file patches)");
  console.log("  [2] Medium (Standard refactoring & execution verification)");
  console.log("  [3] High   (Deep structural context scanning & DevOps orchestration)");

  const intensityChoice = await askQuestion("Select intensity [1-3]: ");
  currentIntensity = intensityChoice === '1' ? 'low' : intensityChoice === '3' ? 'high' : 'medium';

  console.log(`\n\x1b[32m\u2714 Configuration Complete. Model: ${currentModel} | Mode: ${currentIntensity.toUpperCase()}\x1b[0m`);

  return client;
}

async function startAgentLoop(client: OpenAI, history: OpenAI.Chat.Completions.ChatCompletionMessageParam[]) {
  const userInput = await askQuestion(`\n\x1b[34mflow-code [${path.basename(process.cwd())}] > \x1b[0m`);

  const trimmedInput = userInput.trim();

  if (trimmedInput.toLowerCase() === 'exit' || trimmedInput.toLowerCase() === 'quit') {
    rl.close();
    process.exit(0);
  }

  if (trimmedInput.startsWith('cd ')) {
    const targetPath = path.resolve(trimmedInput.slice(3).replace(/['"]/g, ''));
    try {
      process.chdir(targetPath);
      console.log(`\x1b[32m\u2714 Directory shifted to: ${process.cwd()}\x1b[0m`);
    } catch (err: any) {
      console.error(`\x1b[31m\u274C Cannot navigate to path: ${err.message}\x1b[0m`);
    }
    return startAgentLoop(client, history);
  }

  if (history.length === 0) {
    history.push({ role: 'system', content: getSystemPrompt(currentIntensity) });
  } else {
    history[0] = { role: 'system', content: getSystemPrompt(currentIntensity) };
  }

  history.push({ role: 'user', content: trimmedInput });

  try {
    console.log("\x1b[2m\u23F3 Processing Groq LPU pipeline transaction...\x1b[0m");
    const response = await client.chat.completions.create({
      model: currentModel,
      messages: history,
      temperature: currentIntensity === 'low' ? 0.0 : currentIntensity === 'medium' ? 0.1 : 0.3,
    });

    const botMessage = response.choices[0]?.message?.content || "";
    console.log(`\n\u{1F916} \x1b[34m\x1b[1mFlow Agent:\x1b[0m\n${botMessage}`);
    history.push({ role: 'assistant', content: botMessage });

    const codeBlockRegex = /```bash\n([\s\S]*?)```/g;
    let match;
    while ((match = codeBlockRegex.exec(botMessage)) !== null) {
      const commandToRun = match[1].trim();
      console.log(`\n\u{1F527} Subprocess Run: \x1b[36m${commandToRun}\x1b[0m`);
      const executionResult = runBashCommand(commandToRun);
      console.log(executionResult);
      history.push({ role: 'user', content: `Terminal Action Result:\n${executionResult}` });
    }
  } catch (error) {
    console.error("\x1b[31m\u274C Execution Failure:\x1b[0m", error);
  }

  startAgentLoop(client, history);
}

async function main() {
  console.clear();
  console.log("\x1b[34m");
  console.log(" \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2552\u2588\u2588\u2554      \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2552 \u2588\u2588\u2554    \u2588\u2588\u2554     \u2588\u2588\u2588\u2588\u2588\u2588 \u2588\u2588\u2588\u2588\u2588\u2588\u2552 \u2588\u2588\u2588\u2588\u2588\u2588\u2552 \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2552 ");
  console.log(" \u2588\u2588\u2554\u2550\u2550\u2550\u2550\u255d\u2588\u2588\u2554     \u2588\u2588\u2554\u2550\u2550\u2550\u2588\u2588\u2554\u2588\u2588\u2554    \u2588\u2588\u2554    \u2588\u2588\u2554\u2550\u2550\u2550\u255d\u2588\u2588\u2554\u2550\u2550\u2550\u2588\u2588\u2554\u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2554\u2588\u2588\u2554\u2550\u2550\u2550\u2550\u255d ");
  console.log(" \u2588\u2588\u2588\u2588\u2588\u2557  \u2588\u2588\u2554     \u2588\u2588\u2554   \u2588\u2588\u2554\u2588\u2588\u2554 \u2588\u2557 \u2588\u2588\u2554    \u2588\u2588\u2554     \u2588\u2588\u2554   \u2588\u2588\u2554\u2588\u2588\u2554  \u2588\u2588\u2554\u2588\u2588\u2588\u2588\u2588\u2557  ");
  console.log(" \u2588\u2588\u2554\u2550\u2550\u2550\u255d  \u2588\u2588\u2554     \u2588\u2588\u2554   \u2588\u2588\u2554\u2588\u2588\u2554\u2588\u2588\u2588\u2557\u2588\u2588\u2554    \u2588\u2588\u2554     \u2588\u2588\u2554   \u2588\u2588\u2554\u2588\u2588\u2554  \u2588\u2588\u2554\u2588\u2588\u2554\u2550\u2550\u2550\u255d  ");
  console.log(" \u2588\u2588\u2554     \u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557\u255a\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2552\u255a\u2588\u2588\u2588\u2554\u2588\u2588\u2588\u2588\u2552    \u255a\u2588\u2588\u2588\u2588\u2588\u2588\u2552\u255a\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2552\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2552\u255a\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2552 ");
  console.log(" \u255a\u2550\u255d     \u255a\u2550\u2550\u2550\u2550\u2550\u2550\u255d \u255a\u2550\u2550\u2550\u2550\u2550\u2550\u255d \u255a\u2550\u2550\u255d\u255a\u2550\u2550\u2550\u255d      \u255a\u2550\u2550\u2550\u2550\u2550\u2550\u255d \u255a\u2550\u2550\u2550\u2550\u2550\u2550\u255d \u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d \u255a\u2550\u2550\u2550\u2550\u2550\u2550\u255d ");
  console.log("                                     [FREE & OPEN SOURCE]                      ");
  console.log("\x1b[0m");

  const client = await initializeSetup();
  startAgentLoop(client, []);
}

main();
