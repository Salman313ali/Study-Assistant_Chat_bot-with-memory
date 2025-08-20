from __future__ import annotations

from typing import Dict, Optional, List

from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import (
    InMemoryChatMessageHistory,
    BaseChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from groq import BadRequestError
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class StudyAssistantResponse(BaseModel):
    answer: str = Field(description="Direct, well-structured explanation to the user's question")
    key_points: List[str] = Field(default_factory=list, description="Bullet points summarizing main ideas")
    suggested_questions: List[str] = Field(
        default_factory=list, description="Follow-up questions the user could ask next, according to the user's question"
    )
    references: List[str] = Field(
        default_factory=list, description="Citations, URLs, or texts referenced if applicable"
    )


class StudyAssistant:
    def __init__(
        self,
        *,
        groq_api_key: str,
        model: str = "llama-3.3-70b-versatile",
        embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_path: str = "./storage/memory_db",
    ) -> None:
        self._message_store: Dict[str, BaseChatMessageHistory] = {}

        self._llm = ChatGroq(
            model=model,
            temperature=0.2,
            groq_api_key=groq_api_key,
        )

        self._structured_llm_json = self._llm.with_structured_output(
            StudyAssistantResponse, method="json_mode"
        )
        self._structured_llm_schema = self._llm.with_structured_output(
            StudyAssistantResponse, method="json_schema"
        )

        self._embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
        self._vectorstore = Chroma(
            collection_name="study_notes",
            embedding_function=self._embeddings,
            persist_directory=persist_path,
        )
        self._retriever = self._vectorstore.as_retriever(search_kwargs={"k": 4})

        self._prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a kind, rigorous study assistant. Be helpfull, cite facts, and help users learn. explain the answer in a way that is easy to understand and follow.\n"
                        "Use prior study notes when relevant. If you do not know, say so.\n"
                        "Never reveal chain-of-thought, analysis notes, hidden reasoning, or step-by-step traces.\n"
                        "Only provide final answers according to the style and brief bullet points.\n"
                        "always Make sure to add Follow-up questions on the topic.\n"
                        "Adjust verbosity based on the user's chosen style: {style}.\n"
                        "- If style is 'short': keep answers to 2-4 sentences and at most 3 key points.\n"
                        "- If style is 'detailed': provide thorough explanations, examples, and up to 7 key points.\n\n"
                        "Prior notes (may be empty):\n{notes}"
                    ),
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    (
                        "Follow the JSON schema strictly. Do not include extra text before or after the JSON.\n"
                        "Respond to the user's message and return a JSON object that matches the schema.\n"
                        "User message: {input}"
                    ),
                ),
            ]
        )

        def to_history_and_result(resp: StudyAssistantResponse):
            return {"ai_message": AIMessage(content=resp.answer), "result": resp}

        self._chain_json = self._prompt | self._structured_llm_json | RunnableLambda(to_history_and_result)
        self._with_history_json = RunnableWithMessageHistory(
            self._chain_json,
            get_session_history=self._get_or_create_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="ai_message",
        )
        self._chain_schema = self._prompt | self._structured_llm_schema | RunnableLambda(to_history_and_result)
        self._with_history_schema = RunnableWithMessageHistory(
            self._chain_schema,
            get_session_history=self._get_or_create_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="ai_message",
        )

        # Default to json_mode chain
        self._chain = self._chain_json
        self._with_history = self._with_history_json

    def _get_or_create_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self._message_store:
            self._message_store[session_id] = InMemoryChatMessageHistory()
        return self._message_store[session_id]

    def _retrieve_notes(self, query: str) -> str:
        docs = self._retriever.invoke(query)
        if not docs:
            return ""
        parts: List[str] = []
        for idx, d in enumerate(docs, start=1):
            source = d.metadata.get("source") if isinstance(d.metadata, dict) else None
            header = f"[Note {idx}]" + (f" source={source}" if source else "")
            parts.append(f"{header}\n{d.page_content}")
        return "\n\n".join(parts)

    def _save_note(self, text: str, *, source: Optional[str] = None) -> None:
        metadata = {"source": source} if source else None
        self._vectorstore.add_texts([text], metadatas=[metadata] if metadata else None)

    def ask(self, message: str, *, session_id: str = "default", style: str = "short") -> StudyAssistantResponse:
        notes = self._retrieve_notes(message)
        normalized_style = style if style in {"short", "detailed"} else "short"
        try:
            out = self._with_history.invoke(
                {"input": message, "notes": notes, "style": normalized_style},
                config={"configurable": {"session_id": session_id}},
            )
        except (BadRequestError, ValueError, KeyError):
            out = self._with_history_schema.invoke(
                {"input": message, "notes": notes, "style": normalized_style},
                config={"configurable": {"session_id": session_id}},
            )
        result: StudyAssistantResponse = out["result"]

        # Sanitize to avoid accidental model "thinking" disclosures in the answer
        result.answer = self._sanitize_answer(result.answer)

        note_text = (
            f"Question: {message}\n"
            f"Answer: {result.answer}\n"
            f"Key points: {', '.join(result.key_points) if result.key_points else 'None'}\n"
            f"Follow-ups: {', '.join(result.suggested_questions) if result.suggested_questions else 'None'}"
        )
        self._save_note(note_text, source="assistant_session_memory")
        return result

    @staticmethod
    def _sanitize_answer(text: str) -> str:
        # Remove common chain-of-thought markers if a model leaks them
        import re

        patterns = [
            r"(?i)^thoughts?:.*?(\n|$)",
            r"(?i)^reasoning:.*?(\n|$)",
            r"(?i)^analysis:.*?(\n|$)",
            r"(?is)```(?:thought|chain[- ]?of[- ]?thought|analysis)[\s\S]*?```",
        ]
        cleaned = text
        for pat in patterns:
            cleaned = re.sub(pat, "", cleaned).strip()
        return cleaned

    # Session management
    def reset_session(self, session_id: str) -> None:
        if session_id in self._message_store:
            del self._message_store[session_id]
