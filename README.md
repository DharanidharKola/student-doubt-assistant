# 📚 Student Doubt-Solving Assistant

An AI-powered document question-answering application that allows students to upload a PDF learning document and ask questions based on its content.

The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from the uploaded document before generating an answer with an LLM.

---

## 🚀 Features

- Upload PDF learning material
- Extract text from PDF
- Split document into smaller chunks
- Generate semantic embeddings
- Store embeddings in Pinecone
- Retrieve relevant document chunks
- Generate answers using Groq LLM
- Display source chunks and page numbers
- Similarity threshold for weak-result filtering
- Handles questions that are not covered by the document
- Conversation history
- Simple Streamlit interface

---

## 🏗️ Architecture

```text
PDF Upload
    ↓
PDF Text Extraction
    ↓
Text Chunking
    ↓
Hugging Face Embeddings
    ↓
Pinecone Vector Database
    ↓
User Question
    ↓
Question Embedding
    ↓
Similarity Search
    ↓
Relevant Document Chunks
    ↓
Groq LLM
    ↓
Answer + Source References