from dotenv import load_dotenv
import os
import psycopg2
from chromadb import PersistentClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pgvector.psycopg2 import register_vector
from loaders.drive_loader import load_drive_folder_docs
from splitters.recursive_splitter import split_recursive_docs

client = PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection("google_drive")

load_dotenv()

folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
api_key = os.getenv("GEMINI_API_KEY")

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",  # or the correct Gemini embedding model name
    google_api_key=api_key,
)

# Load documents
try:
    docs = load_drive_folder_docs(folder_id)
except Exception as e:
    with (open(".logs/error.log", "a")) as f:
        f.write(f"Error loading documents: {e}\n")
    exit(1)

# Loop docs
for i, doc in enumerate(docs):
    try:
        # Split doc into chunks
        chunks = split_recursive_docs([doc])
    except Exception as e:
        with (open(".logs/doc_{i}.log", "a")) as f:
            f.write(f"Error splitting document: {e}\n")
        continue

    # Loop chunks
    for j, chunk in enumerate(chunks):
        embedding = embedding_model.embed_query(chunk.page_content)
        # Set initial data
        data = {
            "parent_id": folder_id,
            "type": "GoogleDrive",
            "document_id": None,
            "document_title": chunk.metadata.get("title", "unknown"),
            "chunk_index": j,
            "source": chunk.metadata.get("source"),
            "content": chunk.page_content,
            "embedding": embedding,
        }
        print(f"Adding chunk {j + 1} of {len(chunks)} for {data['document_title']}")
        # print(data)
        # Insert data into database
        collection.add(
            documents=[f"{data['document_title']}-{data['chunk_index']}"],
            # metadatas=[chunk.metadata],
            embeddings=[embedding],
            ids=[f"{data['document_title']}-{data['chunk_index']}"],
        )

print(collection.get(include=["embeddings", "documents", "ids"]))