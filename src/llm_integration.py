"""LLM setup and grounded answer generation for the RAG pipeline."""

import os


def create_openai_llm(
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
):
    """Create a LangChain ChatOpenAI instance using OPENAI_API_KEY."""
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set the OPENAI_API_KEY environment variable.")

    return ChatOpenAI(
        api_key=api_key,
        model=model or os.getenv("OPENAI_MODEL", "gpt-5.4"),
        temperature=temperature,
        max_tokens=max_tokens,
    )


def create_groq_llm(
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
):
    """Create a LangChain ChatGroq instance using GROQ_API_KEY."""
    from langchain_groq import ChatGroq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("Set the GROQ_API_KEY environment variable.")

    return ChatGroq(
        api_key=api_key,
        model=model or os.getenv("GROQ_MODEL", "qwen/qwen3-32b"),
        temperature=temperature,
        max_tokens=max_tokens,
    )


def generate_output(query: str, retriever, llm, top_k: int = 3):
    """Retrieve context and ask the selected LLM to answer from that context."""
    results = retriever.retrieve(query, top_k=top_k)
    context = "\n\n".join(item["document"] for item in results)

    if not context:
        return "I couldn't find relevant information in the indexed documents."

    prompt = f"""You are a helpful question-answering assistant.
Answer the question using only the context provided below.
If the context does not contain the answer, clearly say that you do not know.
Do not invent facts.

Context:
{context}

Question:
{query}

Answer:"""

    response = llm.invoke(prompt)
    return response.content
