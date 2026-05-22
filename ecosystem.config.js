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
// Secrets: do NOT commit RADAR_MINER_API_KEY. Put it (and any other
// secret env vars) in a sibling `.env` file — this config loads it at
// startup so `pm2 restart` always picks up the current value without
// fighting a stale `pm2 save` dump. `.env` wins over the shell so a
// leftover pm2 dump can't silently override the file.

const path = require("path");
const fs = require("fs");

function loadDotenv(file) {
  if (!fs.existsSync(file)) return {};
  const out = {};
  for (const line of fs.readFileSync(file, "utf8").split("\n")) {
    const s = line.trim();
    if (!s || s.startsWith("#")) continue;
    const eq = s.indexOf("=");
    if (eq < 0) continue;
    let v = s.slice(eq + 1).trim();
    const quoted = (v.startsWith('"') && v.endsWith('"')) ||
                   (v.startsWith("'") && v.endsWith("'"));
    if (quoted) {
      v = v.slice(1, -1);
    } else {
      // Strip inline `# comment` on unquoted values — `KEY=val # note`
      // would otherwise pass the literal "val # note" into argv.
      const hash = v.indexOf(" #");
      if (hash >= 0) v = v.slice(0, hash).trim();
    }
    out[s.slice(0, eq).trim()] = v;
  }
  return out;
}

const env = { ...process.env, ...loadDotenv(path.join(__dirname, ".env")) };

const sharedEnv = {
  RADAR_DB_API_URL: env.RADAR_DB_API_URL || env.RADAR_DB_URL || "http://localhost:8090",
  RADAR_SERVICE_KEY: env.RADAR_SERVICE_KEY || "",
  RADAR_MINER_API_KEY: env.RADAR_MINER_API_KEY || "",
  MINER_PROMPTS_DIR: path.join(__dirname, "prompts"),

  // GEPA reflector LM — operator's shared LLM proxy. Swap to
  // MINER_REFLECTOR_API_{BASE,KEY,MODEL} for an explicit OpenAI /
  // Anthropic-style endpoint instead.
  MINER_LLM_URL: env.MINER_LLM_URL || "",
  MINER_LLM_API_KEY: env.MINER_LLM_API_KEY || "",
  MINER_LLM_MODEL: env.MINER_LLM_MODEL ||
    "deepseek-ai/DeepSeek-V3-0324",
};

// Default points at the autonomous example agent in a sibling
// radar-miner-examples checkout. Override with RADAR_AGENT_DIR to use a
// different example (openai_sdk_v2, claude_style_v2, patch_decoder) or
// your own agent directory.
const agentDir = env.RADAR_AGENT_DIR ||
  "../radar-miner-examples/agents/autonomous/";

// Listener — validators POST /prepare here to dispatch training. The
// URL the DB advertises is http://<externalIp>:<listenerPort>, so
// externalIp must be reachable from the validator (public IP or DNS
// name). Defaulting to 0.0.0.0 keeps the miner running but invisible
// to validators — set RADAR_MINER_EXTERNAL_IP in .env on real deploys.
const externalIp = env.RADAR_MINER_EXTERNAL_IP || "0.0.0.0";
const listenerPort = env.RADAR_MINER_LISTENER_PORT || "8090";

const minerArgs = [
  "--agent_dir", agentDir,
  "--external_ip", externalIp,
  "--listener_port", listenerPort,
].join(" ");

module.exports = {
  apps: [
    {
      name: "radar-miner",
      script: "miner/neuron.py",
      interpreter: "python3",
      args: minerArgs,
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
