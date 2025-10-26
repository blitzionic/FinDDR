import re
from pathlib import Path
from typing import List

HEADING_RE = re.compile(r'^(?P<indent>\s*)#{1,6}\s+(?P<title>.*?)(\s+#+\s*)?$')
SETEXT_RE = re.compile(r'^\s*(=+|-+)\s*$')  # underlines for setext h1/h2

def convert_setext_to_atx(lines: List[str]) -> List[str]:
    """
    Turn setext headings into ATX style so we can normalize them.
    Example:
      Title
      -----
    becomes:
      ## Title
    """
    out: List[str] = []
    i = 0
    while i < len(lines):
        if i + 1 < len(lines) and SETEXT_RE.match(lines[i + 1]) and not lines[i].strip().startswith("```"):
            title = lines[i].rstrip()
            underline = lines[i + 1].strip()
            if len(title) > 0:
                out.append(f"## {title}\n")
                i += 2
                continue
        out.append(lines[i])
        i += 1
    return out

def normalize_headings_to_h2(md_text: str) -> str:
    """
    - Converts setext headings to ATX.
    - Converts any ATX heading level (# .. ######) to exactly "## ".
    - Skips inside fenced code blocks.
    - Strips trailing closing hashes (e.g., "## Title ##").
    """
    lines = md_text.splitlines(keepends=True)
    lines = convert_setext_to_atx(lines)

    in_code_block = False
    normalized: List[str] = []

    for line in lines:
        stripped = line.lstrip()

        # Track fenced code blocks (``` or ~~~)
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code_block = not in_code_block
            normalized.append(line)
            continue

        if in_code_block:
            normalized.append(line)
            continue

        # Convert any ATX heading to level 2 (##)
        m = HEADING_RE.match(line)
        if m:
            indent = m.group("indent") if m.group("indent") is not None else ""
            title = m.group("title").strip()
            normalized.append(f"{indent}## {title}\n")
        else:
            normalized.append(line)

    return "".join(normalized)

def normalize_file(path: Path) -> None:
    original = path.read_text(encoding="utf-8")
    updated = normalize_headings_to_h2(original)
    path.write_text(updated, encoding="utf-8")

def normalize_folder(folder: Path) -> None:
    for md_path in folder.glob("*.md"):
        normalize_file(md_path)

# Matches: ![alt](None)  (allowing whitespace)
PAT_NONE = re.compile(r'^[ \t]*!\[[^\]]*\]\(\s*None\s*\)[ \t]*\r?\n?', re.MULTILINE)

# Matches: ![alt]() or ![alt](   )  (empty URL)
PAT_EMPTY = re.compile(r'^[ \t]*!\[[^\]]*\]\(\s*\)[ \t]*\r?\n?', re.MULTILINE)

# Collapse 3+ blank lines -> 2
PAT_BLANKS = re.compile(r'\n{3,}', re.MULTILINE)

def clean_markdown(text: str) -> str:
    # Remove invalid image placeholders
    text = PAT_NONE.sub('', text)
    text = PAT_EMPTY.sub('', text)
    # Normalize excessive blank lines
    text = PAT_BLANKS.sub('\n\n', text)
    return text

def clean_file(path: Path) -> None:
    original = path.read_text(encoding='utf-8')
    cleaned = clean_markdown(original)
    if cleaned != original:
        path.write_text(cleaned, encoding='utf-8')

def clean_folder(folder: Path) -> None:
    for md in folder.glob('*.md'):
        clean_file(md)


