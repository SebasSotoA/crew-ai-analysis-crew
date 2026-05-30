"""Helpers for agent-generated HTML reports."""

import re


def strip_markdown_html_fences(text: str) -> str:
    """Remove ```html wrappers if the agent wrapped the document."""
    stripped = text.strip()
    fence = re.match(r"^```(?:html)?\s*([\s\S]*?)```\s*$", stripped, re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return stripped


def looks_like_html(text: str) -> bool:
    lower = text.lower()
    return "<!doctype html" in lower or "<html" in lower
