## LLM Building — internal chat GUI

This is the Chainlit front end for the on-prem LLM adapter. Each session
opens a new conversation against the FastAPI backend
(`app/api/routes_conversations.py`); message history is kept server-side
via the LangGraph + SQLite checkpoint.
