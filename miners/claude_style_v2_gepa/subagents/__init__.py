"""Subagent implementations for the claude_style miner.

Each subagent owns its own message list and tool subset. The
orchestrator (``agents.claude_style.agent``) instantiates them in
sequence; their state does not bleed across subagents — context
isolation is the whole point of the split.
"""
