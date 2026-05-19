# engram-langchain

[LangChain](https://github.com/langchain-ai/langchain) integration for [Engram](https://lumetra.io) — drop-in durable memory backend for any chain, agent, or runnable that consumes a chat-history object.

Replaces the in-process `ConversationBufferMemory` / `ConversationBufferWindowMemory` / `VectorStoreRetrieverMemory` family with a single hosted backend that gives your agents hybrid retrieval (BM25 + vector + knowledge graph) and an explanation trace on every recall, across processes and machines.

## Install

```bash
pip install lumetra-engram langchain langchain-core
```

Drop `engram_langchain.py` from this repo into your project (it's ~30 LOC). A PyPI release of `engram-langchain` is coming; in the meantime the file is intentionally small so you can vendor it.

```bash
export ENGRAM_API_KEY="eng_live_..."
```

## Get an Engram API key

Sign up at <https://lumetra.io> — free tier, no card. You'll see an `eng_live_…` token in your dashboard.

**Don't forget BYOK** — Engram is bring-your-own-key end-to-end for the LLM that does extraction + synthesis. Configure a provider at <https://lumetra.io/models>. DeepSeek is what we recommend, cheap and fast. Without one, store/query returns HTTP 412.

## Usage

### As a `BaseChatMessageHistory`

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from engram_langchain import EngramChatMessageHistory

chain_with_memory = RunnableWithMessageHistory(
    chain,
    lambda session_id: EngramChatMessageHistory(bucket=f"user-{session_id}"),
    input_messages_key="input",
    history_messages_key="history",
)

result = chain_with_memory.invoke(
    {"input": "Remember that I prefer dark mode."},
    config={"configurable": {"session_id": "alex-42"}},
)
```

Every message — both human and AI — becomes one Engram memory in the bucket. Reads return the most recent N messages, newest-first, decoded back into LangChain message objects.

### For semantic recall across full history

`EngramChatMessageHistory.messages` returns a recent-window view of the bucket (good for prompt-stuffing). For semantic search across the entire bucket's history, call `query()` directly:

```python
history = EngramChatMessageHistory(bucket="user-alex")
result = history.query("What does this user prefer for UI themes?")
print(result["answer"])
```

That hits Engram's hybrid retrieval and returns a synthesized answer with citations from the explanation trace.

## Why this beats the built-ins

- **Persistent across processes.** LangChain's in-memory history dies when your worker restarts. Engram doesn't.
- **Hybrid retrieval.** Engram fuses BM25 + dense vector + a knowledge graph. The built-in `VectorStoreRetrieverMemory` is vector-only.
- **Bring-your-own-LLM** for extraction and synthesis (DeepSeek / OpenAI / Anthropic / etc., configured at <https://lumetra.io/models>).
- **Per-session buckets** — pass `bucket=f"user-{session_id}"` and isolation is automatic.

## Verified

Smoke-tested against live `api.lumetra.io`:

- A round of `add_message(HumanMessage(...))` + `add_message(AIMessage(...))` + `add_message(HumanMessage(...))` round-trips through Engram and `messages` returns all 3 entries with the right roles (`HumanMessage` / `AIMessage` reconstructed from the stored content).
- A subsequent `query(...)` over the same bucket returns Engram's synthesized answer citing the retrieved memories from the explanation trace.

## License

MIT — Lumetra
