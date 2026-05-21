// pm2 ecosystem for running a miner alongside the GEPA
// prompt-optimizer loop on the same host.
//
// Usage:
//   mkdir -p logs prompts
//   python miner/neuron.py optimize --seed --dry-run   # one-time: seed active.json
//   pm2 start ecosystem.config.js
//   pm2 logs
//   pm2 save && pm2 startup                            # survive reboot
//
// Both processes share env via `sharedEnv` so prompts/active.json is the
// single source of truth: `radar-gepa` rewrites it after each round's
// scores land, `radar-miner` reads it on the next round — no restart
// needed.
//
// Secrets: do NOT commit RADAR_MINER_API_KEY. Either keep this file out
// of git on the deploy host or load secrets from a sibling .env and
// reference process.env.RADAR_MINER_API_KEY here.

const path = require("path");

const sharedEnv = {
  RADAR_DB_URL: process.env.RADAR_DB_URL || "http://localhost:8090",
  RADAR_SERVICE_KEY: process.env.RADAR_SERVICE_KEY || "",
  RADAR_MINER_API_KEY: process.env.RADAR_MINER_API_KEY || "",
  MINER_PROMPTS_DIR: path.join(__dirname, "prompts"),

  // GEPA reflector LM — operator's shared LLM proxy. Swap to
  // MINER_REFLECTOR_API_{BASE,KEY,MODEL} for an explicit OpenAI /
  // Anthropic-style endpoint instead.
  MINER_LLM_URL: process.env.MINER_LLM_URL || "",
  MINER_LLM_API_KEY: process.env.MINER_LLM_API_KEY || "",
  MINER_LLM_MODEL: process.env.MINER_LLM_MODEL ||
    "deepseek-ai/DeepSeek-V3-0324",
};

// Default points at the autonomous example agent in a sibling
// radar-miner-examples checkout. Override with RADAR_AGENT_DIR to use a
// different example (openai_sdk_v2, claude_style_v2, patch_decoder) or
// your own agent directory.
const agentDir = process.env.RADAR_AGENT_DIR ||
  "../radar-miner-examples/agents/autonomous/";

module.exports = {
  apps: [
    {
      name: "radar-miner",
      script: "miner/neuron.py",
      interpreter: "python3",
      args: `--agent_dir ${agentDir}`,
      cwd: __dirname,
      env: sharedEnv,
      autorestart: true,
      max_restarts: 20,
      restart_delay: 5000,
      out_file: "./logs/miner.out.log",
      error_file: "./logs/miner.err.log",
      merge_logs: true,
      time: true,
    },
    {
      name: "radar-gepa",
      script: "miner/neuron.py",
      interpreter: "python3",
      args: [
        "optimize",
        "--optimizer", "gepa",
        "--population", "8",
        "--budget", "30",
        "--elite-k", "2",
        "--watch",
        "--every-seconds", "600",
      ].join(" "),
      cwd: __dirname,
      env: sharedEnv,
      autorestart: true,
      max_restarts: 20,
      restart_delay: 10000,
      out_file: "./logs/gepa.out.log",
      error_file: "./logs/gepa.err.log",
      merge_logs: true,
      time: true,
    },
  ],
};
