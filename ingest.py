import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import ollama

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
# Pinecone Index Configuration
# ============================================================

index_name = "company-documents"

dimension = 768


# ============================================================
# Create Index if it doesn't exist
# ============================================================

if not pc.has_index(index_name):

    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
# ============================================================
# Connect to Index
# ============================================================

index = pc.Index(index_name)
# ============================================================
# Read Document
# ============================================================

with open(
    "documents/company.txt",
    "r",
    encoding="utf-8"
) as file:

    document = file.read()


# ============================================================
# Split Document into Chunks
# ============================================================

chunks = document.split("\n\n")

# ============================================================
# Generate Embeddings and Store in Pinecone
# ============================================================

vectors = []


for index_number, chunk in enumerate(chunks):

    # Generate embedding
    response = ollama.embed(
        model="nomic-embed-text",
        input=chunk
    )

    embedding = response["embeddings"][0]


    # Create Pinecone vector
    vectors.append({
        "id": f"chunk-{index_number}",
        "values": embedding,
        "metadata": {
            "text": chunk
        }
    })
# ============================================================
# Upload vectors to Pinecone
# ============================================================

index.upsert(
    vectors=vectors
)


print("Document successfully stored in Pinecone.")