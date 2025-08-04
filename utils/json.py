from typing import Literal

def de_fence(text: str, type: Literal["json", "html"] = "json") -> str:
    """If response is fenced code block, remove fence"""
    stripped_text = text.strip()
    if stripped_text.startswith((f"```{type}\n", "```\n", f"``` {type}\n")):
        # Find the first newline after the opening fence
        first_newline = stripped_text.find('\n')
        if first_newline != -1:
            # Remove the opening fence
            stripped_text = stripped_text[first_newline + 1:]
        
        # Remove the closing fence if it exists
        if stripped_text.endswith("\n```"):
            stripped_text = stripped_text[:-4].rstrip()
    
    return stripped_text