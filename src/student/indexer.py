"""Indexer: walks the vLLM repository, chunks every file, builds a BM25 index."""
import json
import os
from pathlib import Path
from typing import List

from tqdm import tqdm

from student.chunker import Chunk, chunk_file

SUPPORTED_EXTENSIONS = {".py", ".md", ".rst", ".txt"}


def collect_files(repo_path: str) -> List[Path]:
    """Return all indexable files under repo_path."""
    skip_dirs = {"__pycache__", ".git", ".mypy_cache", "node_modules", ".venv", "venv"}
    files = []
    for root, dirs, filenames in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fname in filenames:
            if any(fname.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                files.append(Path(root) / fname)
    return sorted(files)


def build_chunks(
    repo_path: str,
    max_chunk_size: int = 2000,
    stored_prefix: str = "data/raw",
) -> List[dict]:
    """Chunk every file, storing paths as stored_prefix/file_path."""
    files = collect_files(repo_path)
    all_chunks: List[dict] = []

    for file_path in tqdm(files, desc="Chunking files"):
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            if not content.strip():
                continue
            stored_path = f"{stored_prefix}/{file_path}"
            chunks: List[Chunk] = chunk_file(
                content=content,
                file_path=stored_path,
                max_chunk_size=max_chunk_size,
            )
            all_chunks.extend(c.to_dict() for c in chunks)
        except Exception as e:
            print(f"  Warning: could not process {file_path}: {e}")

    return all_chunks


def save_index(chunks: List[dict], save_dir: str) -> None:
    """Build a BM25 index and persist it."""
    import bm25s

    if not chunks:
        print("ERROR: no chunks to index. Check your repo_path.")
        return

    chunks_dir = Path(save_dir) / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    with open(chunks_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    print(f"  Saved {len(chunks)} chunks to {chunks_dir}/chunks.json")

    corpus = [c["content"] for c in chunks]
    tokenized = bm25s.tokenize(corpus, show_progress=True)

    retriever = bm25s.BM25()
    retriever.index(tokenized)

    index_dir = Path(save_dir) / "bm25_index"
    index_dir.mkdir(parents=True, exist_ok=True)
    retriever.save(str(index_dir), corpus=corpus)
    print(f"  BM25 index saved to {index_dir}")


def run_indexing(
    repo_path: str,
    save_dir: str = "data/processed",
    max_chunk_size: int = 2000,
    stored_prefix: str = "data/raw",
) -> None:
    """Full pipeline: collect -> chunk -> index -> save."""
    print(f"Indexing repository: {repo_path}")
    print(f"Stored path prefix:  {stored_prefix}")
    chunks = build_chunks(repo_path, max_chunk_size=max_chunk_size, stored_prefix=stored_prefix)
    print(f"Total chunks created: {len(chunks)}")
    save_index(chunks, save_dir)
    print("Ingestion complete! Indices saved under", save_dir)


def run_indexing_split(
    repo_path: str,
    save_dir: str = "data/processed",
    stored_prefix: str = "data/raw",
) -> None:
    """Two separate indexes: one for .md (2000 chars), one for .py (1000 chars)."""
    from student.chunker import chunk_file, Chunk

    CODE_EXTS = {".py"}
    DOC_EXTS  = {".md", ".rst", ".txt"}

    files = collect_files(repo_path)
    code_chunks: List[dict] = []
    doc_chunks:  List[dict] = []

    for file_path in tqdm(files, desc="Chunking files"):
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            if not content.strip():
                continue
            stored_path = f"{stored_prefix}/{file_path}"
            ext = file_path.suffix

            if ext in CODE_EXTS:
                size = 1000
                bucket = code_chunks
            else:
                size = 2000
                bucket = doc_chunks

            chunks: List[Chunk] = chunk_file(content, stored_path, size)
            bucket.extend(c.to_dict() for c in chunks)
        except Exception as e:
            print(f"  Warning: could not process {file_path}: {e}")

    print(f"Code chunks: {len(code_chunks)}  |  Doc chunks: {len(doc_chunks)}")
    save_index(code_chunks, save_dir + "/code")
    save_index(doc_chunks,  save_dir + "/docs")
    print("Split indexing complete!")
