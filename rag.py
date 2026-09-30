import ollama
import os
from dotenv import load_dotenv
from pinecone import Pinecone

# ============================================================
# Load environment variables
# ============================================================

load_dotenv()

# ============================================================
# Get API key
# ============================================================

pinecone_api_key = os.getenv("PINECONE_API_KEY")

# ============================================================
# Create Pinecone Client
# ============================================================

pc = Pinecone(api_key=pinecone_api_key)

# ============================================================
# Connect to Pinecone Index
# ============================================================

index = pc.Index("company-documents")

# ============================================================
# Ask Question Function
# ============================================================

def ask_question(question):

    # --------------------------------
    # 1. Create embedding for question
    # --------------------------------
    response = ollama.embed(
        model="nomic-embed-text",
        input=question
    )

    question_embedding = response["embeddings"][0]

    # --------------------------------------------------------
    # 2. Search Pinecone
    # --------------------------------------------------------

    results = index.query(
        vector=question_embedding,
        top_k=3,
        include_metadata=True
    )

    print("Results:", results)

    # --------------------------------------------------------
    # Get Retrieved Documents
    # --------------------------------------------------------

    documents = []

    for match in results["matches"]:
        document = match["metadata"]["text"]

        documents.append(document)

    print("Documents:", documents)

    # --------------------------------------------------------
    # Combine Documents into Context
    # --------------------------------------------------------

    context = "\n\n".join(documents)

    print("Context:", context)

    # --------------------------------
    # 5. Create RAG prompt
    # --------------------------------
    prompt = f"""
Answer the question using only the context provided below.

If the answer cannot be found in the context, say:
"I don't know based on the provided documents."

Context:
{context}

Question:
{question}
"""

    # --------------------------------
    # 6. Generate answer using Llama
    # --------------------------------
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]
