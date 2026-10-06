import os
from dotenv import load_dotenv

from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from pinecone import Pinecone, ServerlessSpec


# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "student-doubt-assistant")


# --------------------------------------------------
# 1. PDF TEXT EXTRACTION
# --------------------------------------------------

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF and preserve page numbers.
    """

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text:
            pages.append({
                "page": page_number,
                "text": text
            })

    return pages


# --------------------------------------------------
# 2. CHUNKING
# --------------------------------------------------

def create_chunks(pages):
    """
    Split PDF text into smaller chunks for retrieval.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = []

    for page in pages:

        page_chunks = text_splitter.split_text(page["text"])

        for chunk in page_chunks:
            chunks.append({
                "text": chunk,
                "page": page["page"]
            })

    return chunks


# --------------------------------------------------
# 3. EMBEDDING MODEL
# --------------------------------------------------

def get_embeddings():
    """
    Load the Hugging Face embedding model.
    """

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# --------------------------------------------------
# 4. PINECONE CONNECTION
# --------------------------------------------------

def get_pinecone_index():
    """
    Create the Pinecone index if it doesn't exist
    and return the index object.
    """

    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [index.name for index in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:

        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    return pc.Index(INDEX_NAME)


# --------------------------------------------------
# 5. STORE DOCUMENT IN PINECONE
# --------------------------------------------------

def store_chunks(chunks):
    """
    Convert chunks into embeddings and store them in Pinecone.
    """

    embeddings = get_embeddings()
    index = get_pinecone_index()

    vectors = []

    for i, chunk in enumerate(chunks):

        vector = embeddings.embed_query(chunk["text"])

        vectors.append({
            "id": f"chunk-{i}",
            "values": vector,
            "metadata": {
                "text": chunk["text"],
                "page": chunk["page"]
            }
        })

    # Upload vectors in batches
    batch_size = 50

    for i in range(0, len(vectors), batch_size):

        batch = vectors[i:i + batch_size]

        index.upsert(
            vectors=batch
        )

    return len(vectors)


# --------------------------------------------------
# 6. RETRIEVE RELEVANT CHUNKS
# --------------------------------------------------

def retrieve_chunks(question, top_k=4):
    """
    Search Pinecone for the most relevant chunks.
    """

    embeddings = get_embeddings()
    index = get_pinecone_index()

    query_vector = embeddings.embed_query(question)

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    matches = []

    for match in results["matches"]:

    # Ignore weakly relevant results
        if match["score"] >= 0.35:

            matches.append({
                "text": match["metadata"]["text"],
                "page": match["metadata"]["page"],
                "score": match["score"]
        })

    return matches


# --------------------------------------------------
# 7. GENERATE ANSWER USING GROQ
# --------------------------------------------------

def generate_answer(question, retrieved_chunks):
    """
    Generate an answer using only the retrieved document context.
    """

    if not retrieved_chunks:
        return {
            "answer": "I could not find sufficient information in the uploaded document.",
            "sources": []
        }

    context_parts = []

    for chunk in retrieved_chunks:

        context_parts.append(
            f"Page {chunk['page']}:\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a helpful Student Doubt-Solving Assistant.

Answer the student's question using ONLY the information
provided in the document context below.

If the answer cannot be found in the provided context,
say:

"I could not find sufficient information in the uploaded document."

Do not use outside knowledge.
Do not invent information.

DOCUMENT CONTEXT:
{context}

STUDENT QUESTION:
{question}

Provide a clear and simple answer suitable for a student.
"""

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        groq_api_key=GROQ_API_KEY
    )

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": retrieved_chunks
    }


# --------------------------------------------------
# 8. COMPLETE QUESTION ANSWERING PIPELINE
# --------------------------------------------------

def ask_question(question):
    """
    Retrieve relevant information and generate an answer.
    """

    retrieved_chunks = retrieve_chunks(question)

    result = generate_answer(
        question,
        retrieved_chunks
    )

    return result