# End-to-End RAG Pipeline using LangChain

An end-to-end **Retrieval-Augmented Generation (RAG)** implementation using LangChain, Sentence Transformers, ChromaDB, and Large Language Models. This project demonstrates how to build a modular pipeline that processes documents, generates embeddings, stores them in a vector database, retrieves relevant context, and generates context-aware responses using LLMs.

## Overview

Large Language Models can generate impressive responses, but they may produce inaccurate information or lack access to private and domain-specific knowledge.

**Retrieval-Augmented Generation (RAG)** addresses this limitation by retrieving relevant information from an external knowledge source and providing it as context to an LLM before generating a response.

This project implements a modular RAG architecture that enables users to query information from PDF documents using semantic search and LLM-powered question answering.

## Key Features

- **Document Ingestion:** Load and process PDF and text documents.
- **Text Chunking:** Split large documents into smaller, manageable chunks using Recursive Character Text Splitting.
- **Embedding Generation:** Generate semantic vector representations using Sentence Transformers.
- **Vector Storage:** Store document embeddings in ChromaDB for efficient similarity-based retrieval.
- **Semantic Retrieval:** Retrieve relevant document chunks based on user queries.
- **LLM Integration:** Integrate OpenAI and Groq models for context-aware response generation.
- **Modular Architecture:** Separate ingestion, embedding, storage, retrieval, and generation into reusable Python modules.
- **Persistent Vector Database:** Maintain document embeddings using persistent ChromaDB storage.

## RAG Architecture

The pipeline follows these stages:

1. **Document Loading:** Load PDF documents from the data directory.
2. **Text Splitting:** Divide documents into smaller chunks with configurable chunk size and overlap.
3. **Embedding Generation:** Convert document chunks into numerical vector representations.
4. **Vector Database:** Store embeddings and associated document metadata in ChromaDB.
5. **Query Processing:** Convert the user's question into a semantic search query.
6. **Context Retrieval:** Retrieve the most relevant document chunks using similarity search.
7. **LLM Generation:** Pass the retrieved context and user query to the selected LLM.
8. **Response Generation:** Generate a response grounded in the retrieved document context.

## Project Structure

```text
LangChain-RAG-Implementation/
│
├── data/
│   └── pdfs/
│       └── .gitkeep
│
├── notebooks/
│   └── rag_pipeline.ipynb
│
├── src/
│   ├── __init__.py
│   ├── ingestion.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   └── llm_integration.py
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| LangChain | Document processing and LLM integration |
| Sentence Transformers | Semantic embedding generation |
| ChromaDB | Vector database and similarity search |
| OpenAI API | LLM-powered response generation |
| Groq API | Fast LLM inference |
| PyPDF | PDF document extraction |
| NumPy | Numerical operations |
| Jupyter Notebook | Experimentation and pipeline development |

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/sbuttam18-pja/LangChain-RAG-Implementation.git
```

### 2. Navigate to the Project Directory

```bash
cd LangChain-RAG-Implementation
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file in the project root directory.

Add the API keys for the LLM providers you intend to use:

```env
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
```

**Important:** Never upload your `.env` file or expose API keys on GitHub.

### 6. Add Documents

Place your PDF documents inside:

```text
data/pdfs/
```

Use documents that you own or are permitted to process.

## Running the Project

### Jupyter Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
notebooks/rag_pipeline.ipynb
```

Run the notebook cells sequentially to explore the RAG workflow, from document ingestion to retrieval and response generation.

### Modular Implementation

The `src/` directory separates the pipeline into independent components:

| Module | Responsibility |
|---|---|
| `ingestion.py` | Document loading and text chunking |
| `embeddings.py` | Sentence Transformer embedding generation |
| `vector_store.py` | Persistent ChromaDB storage |
| `retriever.py` | Semantic document retrieval |
| `llm_integration.py` | OpenAI/Groq integration and response generation |

This separation makes the pipeline easier to maintain, test, extend, and reuse.

## How RAG Works

```text
             PDF Documents
                   |
                   v
            Document Loading
                   |
                   v
              Text Splitting
                   |
                   v
           Embedding Generation
                   |
                   v
              ChromaDB
                   |
                   |
User Query --------+
     |
     v
Query Embedding
     |
     v
Similarity Search
     |
     v
Relevant Context
     |
     v
   LLM
     |
     v
Generated Response
```

## Applications

This RAG implementation can serve as a foundation for building:

- Document Question Answering Systems
- PDF Chatbots
- Knowledge Base Assistants
- Research Paper Assistants
- Enterprise Knowledge Retrieval Systems
- Domain-Specific AI Assistants

## Future Improvements

- Implement a Streamlit or FastAPI-based user interface.
- Add support for multiple document formats.
- Implement conversation memory for multi-turn interactions.
- Explore hybrid search combining semantic and keyword retrieval.
- Implement reranking to improve retrieval relevance.
- Add evaluation metrics for retrieval and generation quality.
- Support multiple vector database backends.
- Add Docker-based deployment and MLOps integration.

## Learning Outcomes

Through this project, I explored and implemented key concepts in modern AI application development:

- Understanding the Retrieval-Augmented Generation architecture.
- Building document ingestion and preprocessing pipelines.
- Generating semantic embeddings using Sentence Transformers.
- Performing vector similarity search with ChromaDB.
- Integrating LLM providers into retrieval-based applications.
- Designing modular and reusable AI/ML components.

## Author

**Uttam SB**

Aspiring AI/ML Engineer | Generative AI | Agentic AI | RAG Systems | Deep Learning

- GitHub: https://github.com/sbuttam18-pja

---

⭐ If you find this project useful, consider giving it a star!

*Built as a hands-on implementation of Retrieval-Augmented Generation and modular LLM application development.*