from __future__ import annotations

from dataclasses import dataclass


INPUT_TYPES = {"reference", "identity", "fact", "concept", "todo"}


@dataclass(frozen=True)
class InputRoute:
    input_type: str
    trigger_reason: str
    original_input: str

    def as_debug_dict(self) -> dict[str, str]:
        return {
            "input_type": self.input_type,
            "trigger_reason": self.trigger_reason,
            "original_input": self.original_input,
        }


def route_input(text: str) -> InputRoute:
    """Lightweight pre-extraction label for debugging only.

    This router does not decide whether to remember, which category to use, or
    how memories should be written. Downstream extraction must remain the source
    of memory formation behavior.
    """

    original = text
    clean = " ".join(text.strip().split())
    lowered = clean.lower()

    if _contains_any(clean, ("第一个", "第二个", "第三个", "那个", "这个", "上面", "刚刚", "刚才", "前面")):
        return InputRoute("reference", "contains a reference/deictic trigger", original)

    if _contains_any(clean, ("我是谁", "我的性格", "你怎么看我", "我是一个", "我这个人", "我的特点", "我的优势", "我的弱点")):
        return InputRoute("identity", "contains self-identity or personality trigger", original)

    if _contains_any(clean, ("待办", "要做", "下一步", "计划", "别忘", "记得", "提醒", "准备", "联系")):
        return InputRoute("todo", "contains action, reminder, or planning trigger", original)

    if _looks_like_concept(clean, lowered):
        return InputRoute("concept", "contains definition, explanation, or X-is-Y trigger", original)

    return InputRoute("fact", "default factual record label", original)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _looks_like_concept(clean: str, lowered: str) -> bool:
    concept_terms = (
        "就是",
        "指的是",
        "定义",
        "区别",
        "类似于",
        "相当于",
        "可以理解为",
        "核心是",
        "本质是",
        "意思是",
        "等于",
    )
    if _contains_any(clean, concept_terms):
        return True

    technical_terms = (
        "memory",
        "recall",
        "embedding",
        "rag",
        "agent",
        "prompt",
        "workflow",
        "harness",
    )
    return "+" in clean and any(term in lowered for term in technical_terms)
