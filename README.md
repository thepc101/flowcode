# Flow Code

An autonomous coding agent that runs in your terminal. Powered by **Groq** and **Cerebras**. Open-source alternative to Claude Code.

## What It Does

Flow Code is not just a chatbot — it's an **agent** with tools. Describe what you want and it will:

1. **Read** your existing files to understand the codebase
2. **Plan** the implementation
3. **Write** new files and **edit** existing ones
4. **Run** commands (build, test, git, npm, etc.)
5. **Search** the web for docs, solutions, latest versions
6. **Verify** the changes work

All autonomously, with streaming output.

## Agent Tools

| Tool | Description |
|------|-------------|
| `read_file` | Read any file to understand its contents |
| `write_file` | Create or overwrite files (auto-creates directories) |
| `edit_file` | Surgical find-and-replace edits |
| `run_command` | Execute shell commands (build, test, git, npm) |
| `list_directory` | Browse the file tree |
| `search_web` | Search DuckDuckGo for current info |
| `fetch_url` | Fetch and read web pages (docs, GitHub, Stack Overflow) |

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

1. **Select a provider** — Groq or Cerebras
2. **Enter your API key** — saved to `~/.flow-code-config` (mode `0o600`)
3. **Choose a model** — fetched live from the provider
4. **Select intensity** — Low, Medium, or High

## Usage Examples

```
> Create a React todo app with TypeScript and Tailwind

The agent will:
- Create package.json, tsconfig.json, tailwind.config.js
- Create src/App.tsx, src/index.tsx, src/index.css
- Create public/index.html
- Run npm install and npm start to verify

> Fix the bug in auth/login.ts where JWT tokens expire too early

The agent will:
- Read the file
- Identify the issue
- Edit the file with the fix
- Run tests to verify

> What's the latest version of Next.js?

The agent will:
- Search the web
- Report the current version
- Offer to update your project
```

## Slash Commands

| Command | Description |
|---------|-------------|
| `cd <path>` | Switch working directory |
| `/search <query>` | Search the web manually |
| `/fetch <url>` | Fetch URL content |
| `/settings` | Configure preferences |
| `/models` | Re-select model |
| `/provider` | Switch Groq / Cerebras |
| `/resume` | Resume last conversation |
| `/clear` | Reset conversation |
| `/compact` | Trim context |
| `/status` | Show usage stats |
| `exit` | Quit |

## Intensity Modes

| Mode | Temperature | Behavior |
|------|-------------|----------|
| **Low** | 0.0 | Fast, single-file patches |
| **Medium** | 0.2 | Standard refactoring |
| **High** | 0.4 | Deep scanning, tests, DevOps |

## Security

- Config saved with `0o600` permissions (owner read/write only)
- API key masked in settings output
- File writes restricted to current working directory
- Null byte prevention in paths
- Shell commands timeout after 60 seconds
- Graceful SIGINT/SIGTERM handling

## License

MIT
