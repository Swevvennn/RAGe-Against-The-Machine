"""Pydantic models for the RAG pipeline."""
import uuid
from typing import List, Union

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Represents a source chunk with its location in the original file."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question without an answer yet."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its ground-truth answer and sources."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of RAG questions (answered or not)."""

    rag_questions: List[Union[AnsweredQuestion, UnansweredQuestion]]


class MinimalSearchResults(BaseModel):
    """Search results for a single question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Search results + generated answer for a single question."""

    answer: str


class StudentSearchResults(BaseModel):
    """All search results for a dataset."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    """All search results + answers for a dataset."""

    search_results: List[MinimalAnswer]  # type: ignore[assignment]
