"""Retriever: loads the BM25 index and answers queries."""
import json
from functools import lru_cache
from pathlib import Path
from typing import List

from student.models import MinimalSource


class BM25Retriever:

    def __init__(self, processed_dir: str = "data/processed"):
        import bm25s
        processed = Path(processed_dir)
        with open(processed / "chunks" / "chunks.json", encoding="utf-8") as f:
            self._chunks: List[dict] = json.load(f)
        self._retriever = bm25s.BM25.load(
            str(processed / "bm25_index"), load_corpus=False)
        self._bm25s = bm25s

    def _get_chunk(self, item) -> dict:
        if isinstance(item, dict):
            return self._chunks[int(item["id"])]
        return self._chunks[int(item)]

    def search(self, query: str, k: int = 5) -> List[MinimalSource]:
        actual_k = min(max(k, 0), len(self._chunks))
        if actual_k == 0:
            return []
        tokenized = self._bm25s.tokenize([query], show_progress=False)
        results, _scores = self._retriever.retrieve(tokenized, k=actual_k)
        sources = []
        for item in results[0]:
            chunk = self._get_chunk(item)
            sources.append(MinimalSource(
                file_path=chunk["file_path"],
                first_character_index=chunk["first_character_index"],
                last_character_index=chunk["last_character_index"],
            ))
        return sources

    def search_with_content(self, query: str, k: int = 5) -> List[dict]:
        actual_k = min(max(k, 0), len(self._chunks))
        if actual_k == 0:
            return []
        tokenized = self._bm25s.tokenize([query], show_progress=False)
        results, _scores = self._retriever.retrieve(tokenized, k=actual_k)
        return [self._get_chunk(item) for item in results[0]]


@lru_cache(maxsize=1)
def get_retriever(processed_dir: str = "data/processed") -> BM25Retriever:
    return BM25Retriever(processed_dir)
