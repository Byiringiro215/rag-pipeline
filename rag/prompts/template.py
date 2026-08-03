"""Prompt templates for the RAG pipeline."""
from dataclasses import dataclass

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly using the "
    "provided context. If the context does not contain enough information "
    "to answer, say so clearly instead of guessing. Cite sources using the "
    "[n] markers included in the context when relevant."
)

DEFAULT_USER_TEMPLATE = (
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer the question using only the context above."
)


@dataclass
class PromptTemplate:
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    user_template: str = DEFAULT_USER_TEMPLATE

    def build_messages(self, question: str, context: str):
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_template.format(context=context, question=question)},
        ]
