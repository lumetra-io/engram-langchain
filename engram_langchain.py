"""
engram_langchain — durable memory backend for LangChain agents.

Wraps the official lumetra-engram Python SDK as a LangChain `BaseChatMessageHistory`
so any LangChain chain / agent / runnable that takes a chat-history object can
get Engram's hybrid retrieval (BM25 + vector + knowledge graph) with one line.

Usage:

    from langchain_core.runnables.history import RunnableWithMessageHistory
    from engram_langchain import EngramChatMessageHistory

    chain_with_memory = RunnableWithMessageHistory(
        chain,
        lambda session_id: EngramChatMessageHistory(bucket=f"user-{session_id}"),
        input_messages_key="input",
        history_messages_key="history",
    )
"""

from __future__ import annotations

import os
from typing import List, Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from lumetra_engram import EngramClient


class EngramChatMessageHistory(BaseChatMessageHistory):
    """LangChain chat history backed by Engram.

    Every appended message is stored as a single memory in `bucket`. Reads
    return the most recent `read_limit` memories, newest-first, decoded back
    into LangChain message objects.

    For semantic recall (across the entire bucket's history, not just the
    most recent window), call `query(question)` directly.
    """

    def __init__(
        self,
        bucket: str,
        *,
        client: Optional[EngramClient] = None,
        read_limit: int = 20,
    ) -> None:
        self.bucket = bucket
        self.read_limit = read_limit
        self._client = client or EngramClient(
            api_key=os.environ.get("ENGRAM_API_KEY"),
        )

    @property
    def messages(self) -> List[BaseMessage]:
        result = self._client.list_memories(self.bucket, limit=self.read_limit)
        items = result.get("memories", []) if isinstance(result, dict) else []
        out: List[BaseMessage] = []
        # Engram returns newest first; LangChain expects chronological.
        for m in reversed(items):
            content = m.get("content", "")
            if content.startswith("USER: "):
                out.append(HumanMessage(content=content[len("USER: "):]))
            elif content.startswith("AI: "):
                out.append(AIMessage(content=content[len("AI: "):]))
            else:
                out.append(HumanMessage(content=content))
        return out

    def add_message(self, message: BaseMessage) -> None:
        role = "USER" if isinstance(message, HumanMessage) else "AI"
        body = message.content if isinstance(message.content, str) else str(message.content)
        self._client.store_memory(f"{role}: {body}", self.bucket)

    def clear(self) -> None:
        self._client.clear_memories(self.bucket)

    def query(self, question: str) -> dict:
        """Semantic retrieval over the entire bucket. Returns the raw Engram
        response (`answer`, `memories_found`, `explanation`, etc).
        """
        return self._client.query(question, buckets=[self.bucket])
