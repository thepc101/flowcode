# Flow Code

A lightning-fast, free, terminal-native autonomous coding agent powered by **Groq** and **Cerebras**. Open-source alternative to Claude Code.

## Features

- **Multi-Provider** — Groq and Cerebras API support
- **Auto-Search** — Detects when questions need current data and searches the web automatically
- **Code Box Rendering** — Generated code appears in bordered boxes with language labels
- **Auto Directory Scan** — Shows file tree when you `cd` into a folder
- **Streaming Responses** — Real-time token-by-token output
- **Context Usage Display** — Visual progress bar showing token consumption
- **Slash Commands** — `/search`, `/fetch`, `/settings`, `/models`, `/compact`, `/clear`, `/status`
- **128K Context Window** — Full leverage of Groq/Cerebras free API limits
- **Claude Code Dashboard** — Welcome screen with recent activity and quick tips

## Installation

```bash
git clone https://github.com/thepc101/flowcode.git
cd flowcode
npm install
npm run build
npm link
```

Then run from any directory:

```bash
flow-code
```

## First Launch

On first run you'll be prompted to:

1. **Select a provider** — Groq or Cerebras
2. **Enter your API key** — saved securely to `~/.flow-code-config` (mode `0o600`)
3. **Choose a model** — fetched live from the provider
4. **Select intensity** — Low, Medium, or High

## Slash Commands

| Command | Description |
|---------|-------------|
| `cd <path>` | Switch working directory (shows file tree) |
| `/search <query>` | Search the web via DuckDuckGo |
| `/fetch <url>` | Fetch and display URL content |
| `/settings` | Open interactive settings menu |
| `/models` | Re-select your model |
| `/provider` | Switch between Groq and Cerebras |
| `/compact` | Trim conversation history to fit context |
| `/clear` | Reset conversation history |
| `/status` | Show provider, model, intensity, and token usage |
| `/help` | List all commands |
| `exit` | Quit Flow Code |

## Auto-Search

Flow Code automatically detects when your prompt requires current or external information and searches the web before responding. Triggers include:

- Questions about latest versions, releases, or updates
- Comparisons ("vs", "compared to")
- "What is", "how to", "which is best"
- Year references (2024, 2025, 2026)
- Price, weather, news queries

Code-only tasks (create, write, build, fix) are never auto-searched.

## Intensity Modes

| Mode | Temperature | Behavior |
|------|-------------|----------|
| **Low** | 0.0 | Fast, single-file patches, transactional |
| **Medium** | 0.2 | Standard refactoring, verifies changes compile |
| **High** | 0.4 | Deep file tree scan, linting, tests, DevOps |

## Code Generation Standards

Flow Code enforces production-quality code output:

- **TypeScript** — strict mode, no `any`, interfaces, async/await, optional chaining
- **React / Next.js** — functional components, hooks, App Router, Tailwind CSS
- **HTML / CSS** — semantic elements, custom properties, flexbox/grid, mobile-first
- **Python** — type hints, PEP 8, f-strings, pathlib, dataclasses
- **Bash** — `set -euo pipefail`, quoted variables, non-interactive flags
- **Database** — parameterized queries, indexes, transactions

## Settings Menu

Run `/settings` to:

1. Change API Key
2. Switch Provider (Groq / Cerebras)
3. Change Model (live model list)
4. Change Intensity (Low / Medium / High)
5. View current config (API key is masked)
6. Reset all settings

## Configuration

Config is stored at `~/.flow-code-config`:

```json
{
  "apiKey": "gsk_...",
  "provider": "groq",
  "defaultModel": "llama-3.3-70b-versatile",
  "intensity": "medium"
}
```

Activity history is stored at `~/.flow-code-activity`.

## Context Window

Flow Code uses a **128,000 token** context window:

- System prompt + directory tree
- Conversation history (auto-trimmed to 110k tokens)
- Search results injected into context
- Terminal output from auto-executed bash blocks

Usage is displayed after every response:

```
Tokens: 3420 prompt + 1847 completion = 5267 total
Context: ██████████████░░░░░░ 8,241 / 128,000 tokens (6%)
```

## Security

- Config file saved with `0o600` permissions (owner read/write only)
- API key masked in `/settings` view output
- Model IDs sanitized (no injection possible)
- Null byte prevention in `cd` paths
- Bash output capped at 8,000 chars to prevent context overflow
- Shell commands timeout after 60 seconds
- Graceful `SIGINT`/`SIGTERM` handling

## License

MIT
