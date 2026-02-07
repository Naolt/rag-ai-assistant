# Data

This directory contains LangChain documentation pages used as the knowledge base for the RAG assistant. Each `.txt` file is loaded, chunked, and embedded into ChromaDB on startup.

## Included Documents

| File | Topic |
|---|---|
| `overview.txt` | LangChain overview, core benefits, and getting started |
| `models.txt` | Chat models, providers, invocation, streaming, and structured output |
| `agents.txt` | Agent architecture, tools, system prompts, memory, and middleware |
| `tools.txt` | Creating tools, accessing context, ToolNode, and prebuilt tools |
| `messages.txt` | Message types, roles, and conversation history |
| `short-term-memory.txt` | Short-term memory and state management |
| `streaming-overview.txt` | Streaming overview and patterns |
| `streaming-frontend.txt` | Frontend streaming integration |

## Data Source

Content sourced from the [LangChain documentation](https://docs.langchain.com/) under their open-source license.

## Adding Your Own Data

To extend the knowledge base, place additional `.txt` files in this directory. The assistant will automatically load all `.txt` files on startup.
