"""Plain text from a chat model reply, whatever shape its content has.

``AIMessage.content`` is a string for OpenAI-compatible models but a list of
content blocks for Anthropic models (``ChatAnthropic``) whenever the reply
carries more than one block: a thinking block before the text, a text block
next to a tool call, or text split across blocks. Code that treats
``.content`` as a string then breaks on Anthropic only, for example
``json.loads`` raising "the JSON object must be str, bytes or bytearray,
not list".

Only text blocks count. Thinking, redacted thinking and tool-use blocks are
not part of the answer and are dropped.
"""

from __future__ import annotations

from typing import Any

_TEXT_BLOCK_TYPES = ("text", "output_text")


def message_text(message: Any) -> str:
    """Return the text of a model reply or of its ``content`` value.

    Accepts a message object (anything with ``.content``), a raw content
    value, or ``None``. Strings are returned unchanged; block lists are
    joined in order without separators, because split text blocks are
    fragments of one answer.
    """
    content = getattr(message, "content", message)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    if isinstance(content, (list, tuple)):
        parts = []
        for block in content:
            if block is None:
                continue
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                block_type = block.get("type")
                if block_type is None or block_type in _TEXT_BLOCK_TYPES:
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            else:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)
