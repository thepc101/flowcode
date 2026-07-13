# Flow Code

Flow Code is a lightning-fast, free, terminal-native autonomous coding agent powered by Groq. It acts as an open-source alternative to Claude Code, letting you build apps, manipulate directories, and run DevOps pipelines directly from your terminal.

## Features
- **API Key Once**: Enter your Groq API key on first startup; it saves securely locally.
- **Dynamic Model Selector**: Automatically polls live Groq models.
- **Intensity Profiles**: Toggle between Low, Medium, and High processing depths.
- **Persistent Shell State**: Shift workspaces seamlessly using a native `cd` tracker.
- **Autonomous Tool Loops**: Automatically runs tests, scripts, and deployment code blocks.

## Installation

Clone the repository and install it globally on your machine:

```bash
git clone https://github.com/thepc101/flowcode.git
cd flowcode
npm install
npm run build
npm link
```

## Usage

Simply run the following command anywhere in your terminal:

```bash
flow-code
```
