"""
Chunking strategies for Python code and Markdown documentation.

Key concept: every chunk must store first_character_index and last_character_index
so we can map it back to its exact position in the original file.
This is required for the recall@k evaluation (5% overlap check).
"""
import ast
import re
from dataclasses import dataclass
from typing import List

# ---------------------------------------------------------------------------
# Data structure for a chunk
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A piece of a file with its exact position."""
    file_path: str
    content: str
    first_character_index: int
    last_character_index: int

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "content": self.content,
            "first_character_index": self.first_character_index,
            "last_character_index": self.last_character_index,
        }


# ---------------------------------------------------------------------------
# Utility: split a text block that is too large
# ---------------------------------------------------------------------------

def _split_by_size(
    content: str,
    file_path: str,
    offset: int,
    max_size: int,
) -> List[Chunk]:
    """
    Last-resort splitter: cuts a text every max_size characters.
    'offset' is the position of content[0] inside the original file.
    """
    chunks = []
    for i in range(0, len(content), max_size):
        piece = content[i : i + max_size]
        chunks.append(Chunk(
            file_path=file_path,
            content=piece,
            first_character_index=offset + i,
            last_character_index=offset + i + len(piece),
        ))
    return chunks


# ---------------------------------------------------------------------------
# Python chunker  (uses the AST to find function/class boundaries)
# ---------------------------------------------------------------------------

def _lines_to_char_index(lines: List[str], line_number: int) -> int:
    """
    Convert a 0-based line number to a character index in the full file.
    Each line contributes len(line) + 1 characters (the newline).
    """
    return sum(len(line) + 1 for line in lines[:line_number])


def chunk_python(
    content: str,
    file_path: str,
    max_chunk_size: int = 2000,
) -> List[Chunk]:
    """
    Chunk a Python file by top-level functions and classes.

    Strategy:
      1. Parse the file with `ast` to get the line ranges of each
         top-level node (FunctionDef, AsyncFunctionDef, ClassDef).
      2. Convert line numbers → character indices.
      3. If a node is larger than max_chunk_size, fall back to size splitting.
      4. Collect any code between top-level nodes (imports, constants…)
         as an extra chunk.
    """
    chunks: List[Chunk] = []

    # Try to parse; if the file has syntax errors fall back to size splitting
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return _split_by_size(content, file_path, 0, max_chunk_size)

    lines = content.split("\n")

    # Collect only TOP-LEVEL nodes (direct children of the Module node)
    # Sorting by line number ensures we process them in order
    top_level_nodes = sorted(
        (
            node for node in ast.iter_child_nodes(tree)
            if isinstance(node, (
                ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef
            ))
        ),
        key=lambda n: n.lineno,
    )

    covered_ranges: List[tuple] = []  # (start_char, end_char) of each node

    for node in top_level_nodes:
        # ast line numbers are 1-based; convert to 0-based for our helper
        start_char = _lines_to_char_index(lines, node.lineno - 1)
        end_char   = _lines_to_char_index(lines, node.end_lineno)

        covered_ranges.append((start_char, end_char))
        piece = content[start_char:end_char]

        if len(piece) <= max_chunk_size:
            chunks.append(Chunk(
                file_path=file_path,
                content=piece,
                first_character_index=start_char,
                last_character_index=end_char,
            ))
        else:
            # Node too large → size-split it, keeping the correct offset
            chunks.extend(_split_by_size(piece, file_path, start_char, max_chunk_size))

    # -----------------------------------------------------------------------
    # Handle code NOT covered by any top-level node
    # (imports, module-level variables, __all__ = …, etc.)
    # We build the "gaps" between covered ranges.
    # -----------------------------------------------------------------------
    gaps: List[tuple] = []
    prev_end = 0
    for start, end in sorted(covered_ranges):
        if start > prev_end:
            gaps.append((prev_end, start))
        prev_end = max(prev_end, end)
    if prev_end < len(content):
        gaps.append((prev_end, len(content)))

    for gap_start, gap_end in gaps:
        gap_text = content[gap_start:gap_end].strip()
        if not gap_text:
            continue
        # Re-align to the actual (non-stripped) start position
        real_start = gap_start + content[gap_start:gap_end].index(gap_text)
        if len(gap_text) <= max_chunk_size:
            chunks.append(Chunk(
                file_path=file_path,
                content=gap_text,
                first_character_index=real_start,
                last_character_index=real_start + len(gap_text),
            ))
        else:
            chunks.extend(_split_by_size(gap_text, file_path, real_start, max_chunk_size))

    return chunks


# ---------------------------------------------------------------------------
# Markdown chunker  (splits on headings #, ##, ###)
# ---------------------------------------------------------------------------

def chunk_markdown(
    content: str,
    file_path: str,
    max_chunk_size: int = 2000,
) -> List[Chunk]:
    """
    Chunk a Markdown file by headings (lines starting with #, ##, or ###).

    Strategy:
      1. Find all heading positions with a regex.
      2. Each section = from one heading to the next.
      3. If a section is larger than max_chunk_size, size-split it.
    """
    chunks: List[Chunk] = []

    # Find the start position of every heading
    heading_positions = [
        m.start()
        for m in re.finditer(r"^#{1,3} ", content, flags=re.MULTILINE)
    ]

    # If no headings found, treat the whole file as one section
    if not heading_positions:
        heading_positions = [0]

    # Add a sentinel at the end so the last section runs to EOF
    heading_positions.append(len(content))

    for i in range(len(heading_positions) - 1):
        start = heading_positions[i]
        end   = heading_positions[i + 1]
        section = content[start:end]

        if not section.strip():
            continue

        if len(section) <= max_chunk_size:
            chunks.append(Chunk(
                file_path=file_path,
                content=section,
                first_character_index=start,
                last_character_index=end,
            ))
        else:
            chunks.extend(_split_by_size(section, file_path, start, max_chunk_size))

    return chunks


# ---------------------------------------------------------------------------
# Dispatcher: choose the right strategy based on file extension
# ---------------------------------------------------------------------------

def chunk_file(
    content: str,
    file_path: str,
    max_chunk_size: int = 2000,
) -> List[Chunk]:
    """
    Route a file to the correct chunking strategy.
    Falls back to size splitting for unknown file types.
    """
    if file_path.endswith(".py"):
        return chunk_python(content, file_path, max_chunk_size)
    elif file_path.endswith((".md", ".rst", ".txt")):
        return chunk_markdown(content, file_path, max_chunk_size)
    else:
        # Generic fallback (e.g. .yaml, .toml, .cfg…)
        return _split_by_size(content, file_path, 0, max_chunk_size)
