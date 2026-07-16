from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from personal_brain.semantic import cosine_similarity


FIXTURE = Path(__file__).with_name("fixtures") / "synthetic_memories.json"
DEFAULT_QUERY = "What notification policy did the greenhouse team choose?"
MIN_RELEVANCE = 0.05


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def vocabulary(query: str, memories: list[dict[str, object]]) -> list[str]:
    terms = tokens(query)
    for memory in memories:
        terms.update(tokens(" ".join([str(memory["title"]), str(memory["content"]), " ".join(memory["topics"])])))
    return sorted(terms)


def vector(text: str, terms: list[str]) -> list[float]:
    present = tokens(text)
    return [1.0 if term in present else 0.0 for term in terms]


def main() -> int:
    query = " ".join(sys.argv[1:]).strip() or DEFAULT_QUERY
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if payload.get("synthetic") is not True:
        raise RuntimeError("demo fixture must be explicitly synthetic")

    memories = payload["memories"]
    terms = vocabulary(query, memories)
    query_vector = vector(query, terms)
    ranked: list[tuple[float, dict[str, object]]] = []
    for memory in memories:
        text = " ".join([str(memory["title"]), str(memory["content"]), " ".join(memory["topics"])])
        ranked.append((cosine_similarity(query_vector, vector(text, terms)), memory))
    ranked.sort(key=lambda item: item[0], reverse=True)

    print(f"Question: {query}")
    relevant = [(score, memory) for score, memory in ranked if score >= MIN_RELEVANCE]
    if not relevant:
        print("\nNo matching synthetic evidence found.")
        print("\nThis deterministic demo refuses to present unrelated records as evidence.")
        return 0

    print("\nTop synthetic evidence:")
    for index, (score, memory) in enumerate(relevant[:2], start=1):
        print(f"[{index}] score={score:.3f} memory={memory['memory_id']} raw={memory['raw_message_id']}")
        print(f"    {memory['content']}")
    print("\nThis deterministic demo replaces production embeddings and model reranking with token vectors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
