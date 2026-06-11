"""
CLI entry point using Python Fire.
All commands map to the RAG pipeline steps described in the subject.

Usage:
    uv run python -m student index --max_chunk_size 2000
    uv run python -m student search "How to configure OpenAI server?" --k 5
    uv run python -m student answer "How to configure OpenAI server?" --k 10
    uv run python -m student search_dataset --dataset_path ... --k 10 --save_directory ...
    uv run python -m student answer_dataset --student_search_results_path ... --save_directory ...
    uv run python -m student evaluate --student_answer_path ... --dataset_path ... --k 10
"""
import json
from pathlib import Path

from tqdm import tqdm


class StudentCLI:
    """RAG pipeline CLI — all commands are methods of this class."""

    # ------------------------------------------------------------------
    # 1. INDEX
    # ------------------------------------------------------------------
    def index(
        self,
        repo_path: str = "vllm-0.10.1",
        save_dir: str = "data/processed",
        max_chunk_size: int = 2000,
        max_chunck_size: int = 0,
        stored_prefix: str = "data/raw",
    ) -> None:
        """Index the repository."""
        from student.indexer import run_indexing
        if max_chunck_size > 0:
            max_chunk_size = max_chunck_size
        run_indexing(repo_path, save_dir, max_chunk_size, stored_prefix)

    def index_split(
        self,
        repo_path: str = "vllm-0.10.1",
        save_dir: str = "data/processed",
        stored_prefix: str = "data/raw",
    ) -> None:
        """Two separate indexes: .md at 2000 chars, .py at 1000 chars."""
        from student.indexer import run_indexing_split
        run_indexing_split(repo_path, save_dir, stored_prefix)

    # ------------------------------------------------------------------
    # 2. SEARCH (single query)
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        k: int = 5,
        processed_dir: str = "data/processed",
    ) -> None:
        """Search the index for a single query and print results."""
        from student.retriever import get_retriever

        retriever = get_retriever(processed_dir)
        sources = retriever.search(query, k=k)

        print(f"\nTop-{k} results for: {query!r}\n")
        for i, src in enumerate(sources, 1):
            print(f"  {i}. {src.file_path}  [{src.first_character_index}:{src.last_character_index}]")

    # ------------------------------------------------------------------
    # 3. SEARCH DATASET (batch)
    # ------------------------------------------------------------------
    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
        processed_dir: str = "data/processed",
    ) -> None:
        """Process all questions in a dataset and save search results."""
        from student.models import (
            MinimalSearchResults,
            RagDataset,
            StudentSearchResults,
        )
        from student.retriever import get_retriever

        try:
            with open(dataset_path, encoding="utf-8") as f:
                raw = json.load(f)
        except FileNotFoundError as e:
            print(f"{e.__class__.__name__}: no file found at : {dataset_path}")

        dataset = RagDataset.model_validate(raw)

        retriever = get_retriever(processed_dir)
        results = []

        for q in tqdm(dataset.rag_questions, desc="Searching"):
            sources = retriever.search(q.question, k=k)
            results.append(MinimalSearchResults(
                question_id=q.question_id,
                question=q.question,
                retrieved_sources=sources,
            ))

        output = StudentSearchResults(search_results=results, k=k)

        Path(save_directory).mkdir(parents=True, exist_ok=True)
        out_path = Path(save_directory) / Path(dataset_path).name
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(output.model_dump_json(indent=2))

        print(f"Saved student_search_results to {out_path}")

    # ------------------------------------------------------------------
    # 4. ANSWER (single query)
    # ------------------------------------------------------------------
    def answer(
        self,
        query: str,
        k: int = 10,
        processed_dir: str = "data/processed",
    ) -> None:
        """Answer a single question using the RAG pipeline."""
        from student.generator import generate_answer
        from student.retriever import get_retriever
        try:
            if not query.strip():
                raise ValueError("Query cannot be empty")

            retriever = get_retriever(processed_dir)
            chunks = retriever.search_with_content(query, k=k)
            answer = generate_answer(query, chunks)
            print(f"\nQ: {query}\nA: {answer}\n")
        except ValueError as e:
            print(f"{e.__class__.__name__}: {e.args[0]}")

    # ------------------------------------------------------------------
    # 5. ANSWER DATASET (batch)
    # ------------------------------------------------------------------
    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
        processed_dir: str = "data/processed",
    ) -> None:
        """Generate answers for all questions in a search results file."""
        from student.generator import generate_answer
        from student.models import MinimalAnswer, StudentSearchResultsAndAnswer
        from student.retriever import get_retriever

        try:
            with open(student_search_results_path, encoding="utf-8") as f:
                raw = json.load(f)
        except FileNotFoundError as e:
            print(f"{e.__class__.__name__}: no file found at : {dataset_path}")

        k = raw["k"]
        questions = raw["search_results"]
        retriever = get_retriever(processed_dir)

        print(f"Loaded {len(questions)} questions from {student_search_results_path}")
        answered = []

        for i, q in enumerate(tqdm(questions, desc="Generating answers")):
            # Re-fetch content for the retrieved sources
            chunks = retriever.search_with_content(q["question"], k=k)
            ans = generate_answer(q["question"], chunks)
            answered.append(MinimalAnswer(
                question_id=q["question_id"],
                question=q["question"],
                retrieved_sources=q["retrieved_sources"],
                answer=ans,
            ))
            print(f"Processed {i + 1} of {len(questions)} questions", end="\r")

        output = StudentSearchResultsAndAnswer(search_results=answered, k=k)

        Path(save_directory).mkdir(parents=True, exist_ok=True)
        out_path = Path(save_directory) / Path(student_search_results_path).name
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(output.model_dump_json(indent=2))

        print(f"\nSaved student_search_results_and_answer to {out_path}")

    # ------------------------------------------------------------------
    # 6. EVALUATE
    # ------------------------------------------------------------------
    def evaluate(
        self,
        student_answer_path: str,
        dataset_path: str,
        k: int = 10,
        max_context_length: int = 2000,
        min_overlap: float = 0.05,
    ) -> None:
        """Evaluate recall@k against ground-truth annotations."""
        from student.evaluator import run_evaluation
        run_evaluation(
            student_answer_path=student_answer_path,
            dataset_path=dataset_path,
            k=k,
            max_context_length=max_context_length,
            min_overlap=min_overlap,
        )
