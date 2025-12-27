from __future__ import annotations
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

DEFAULT_GEN_MODEL = "google/flan-t5-small"

class RAGAnswerer:
    def __init__(self, model_name: str = DEFAULT_GEN_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.pipe = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def build_prompt(self, question: str, contexts: List[str]) -> str:
        # Keep context reasonably small to avoid token overflow
        context_block = "\n".join(contexts)[:4000]

        return (
            "Answer the question using ONLY this context.\n"
            "If the answer is not in the context, say: I don't know based on the provided document.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    def answer(self, question: str, contexts: List[str], max_new_tokens: int = 200) -> str:
        if not contexts:
            return "I don't know based on the provided document."

        prompt = self.build_prompt(question, contexts)
        out = self.pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)[0]["generated_text"].strip()

        # If model returns empty for any reason
        if not out:
            return "I don't know based on the provided document."
        return out
