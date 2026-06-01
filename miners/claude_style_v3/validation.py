"""Thin re-export of ``core.validation`` so callers can do
``from agents.openai_sdk.validation import validate_code`` without
reaching into the autonomous-agent layout.

Importing ``agents.openai_sdk`` puts the autonomous agent on ``sys.path``;
this module simply forwards the symbol through.
"""

from core.validation import validate_code  # noqa: F401  (re-export)

__all__ = ["validate_code"]
