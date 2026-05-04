#!/usr/bin/env node
// Convert /workdir/trace.jsonl into per-day JSONL session files in the
// format `lcm-tui backfill` expects, dropping them at
// `~/.openclaw/agents/<agent>/sessions/<session-id>.jsonl`.
//
// This replaces our old direct-SQL seed.mjs. Now lcm-tui actually owns
// ingestion: for each day's session file, `lcm-tui backfill ... --apply`
// imports messages AND runs lossless-claw's depth-aware compaction
// (leaf summaries, condensed roll-ups, DAG edges) using the configured
// LLM.
//
// JSONL row shape (per lcm-tui's parseBackfillSessionFile, see
// /tmp/lossless-claw/tui/data.go:152-166):
//   {"type":"message","id":"...","timestamp":"...","message":{"role":"user","content":"..."}}
//
// Roles must be in {system,user,assistant,tool} (anything else gets
// coerced to "assistant").
//
// Usage:
//   node convert_trace.mjs <trace_path> <agent_name> <session_prefix>
// Output (stdout):
//   {"sessions":[{"sessionId":"<id>","day":"YYYY-MM-DD","messages":N}, ...], "agentDir":"..."}

import fs from "node:fs";
import path from "node:path";
import os from "node:os";

const tracePath = process.argv[2];
const agentName = process.argv[3] || "main";
const sessionPrefix = process.argv[4] || "trace";

if (!tracePath) {
  console.error("usage: convert_trace.mjs <trace_path> <agent_name> <session_prefix>");
  process.exit(2);
}

if (!fs.existsSync(tracePath)) {
  console.log(JSON.stringify({ sessions: [], skipped: "no trace" }));
  process.exit(0);
}

const stateDir =
  process.env.OPENCLAW_STATE_DIR || path.join(os.homedir(), ".openclaw");
const agentDir = path.join(stateDir, "agents", agentName);
const sessionsDir = path.join(agentDir, "sessions");
fs.mkdirSync(sessionsDir, { recursive: true });

const groups = new Map();
const lines = fs.readFileSync(tracePath, "utf8").split("\n");
for (const raw of lines) {
  const line = raw.trim();
  if (!line) continue;
  let evt;
  try {
    evt = JSON.parse(line);
  } catch {
    continue;
  }
  const ts = evt.timestamp || "1970-01-01T00:00:00Z";
  let day;
  try {
    day = new Date(ts).toISOString().slice(0, 10);
  } catch {
    day = "undated";
  }
  if (!groups.has(day)) groups.set(day, []);
  groups.get(day).push(evt);
}

function normalizeRole(role) {
  const r = String(role || "").toLowerCase().trim();
  if (r === "system" || r === "user" || r === "assistant" || r === "tool") {
    return r;
  }
  return "assistant";
}

function eventToMessage(evt) {
  const data = evt.message_data || {};
  const t = data.type;
  if (t === "message") {
    return {
      role: normalizeRole(data.role || "user"),
      content:
        typeof data.content === "string"
          ? data.content
          : JSON.stringify(data.content || ""),
    };
  }
  if (t === "reasoning") {
    return {
      role: "assistant",
      content: "[reasoning] " + (data.summary || ""),
    };
  }
  if (t === "function_call") {
    return {
      role: "assistant",
      content: `[tool:${data.name || "?"}] ${data.arguments || "{}"}`,
    };
  }
  if (t === "function_call_output") {
    return { role: "tool", content: String(data.output || "") };
  }
  return null;
}

const written = [];
for (const [day, events] of [...groups.entries()].sort()) {
  if (day === "undated") continue;
  const sessionId = `${sessionPrefix}-${day}`;
  const filePath = path.join(sessionsDir, `${sessionId}.jsonl`);
  const out = [];
  let counter = 0;
  for (const evt of events) {
    const msg = eventToMessage(evt);
    if (!msg) continue;
    counter += 1;
    out.push(
      JSON.stringify({
        type: "message",
        id: `${sessionId}-msg-${counter}`,
        timestamp: evt.timestamp || `${day}T00:00:00Z`,
        message: { role: msg.role, content: msg.content },
      }),
    );
  }
  if (out.length === 0) continue;
  fs.writeFileSync(filePath, out.join("\n") + "\n");
  written.push({ sessionId, day, messages: out.length, path: filePath });
}

console.log(JSON.stringify({ sessions: written, agentDir, sessionsDir }));
