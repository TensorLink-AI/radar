"""Subagent implementations for the claude_style_v3 miner.

Four subagents: ``analyst`` (landscape-reflection digest over the
global DB), ``researcher`` (two-phase: brainstorm → brief),
``designer`` (code, validate, submit), ``critic`` (KEEP/CHANGE/DROP
between designer iterations).

Each subagent owns its own message list and tool subset. The
orchestrator instantiates them in sequence; state does not bleed
across subagents — context isolation is the whole point of the split.
"""
