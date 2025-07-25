from langchain.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import chromadb
import os
from dotenv import load_dotenv

load_dotenv()   

# Set up Gemini embedding model
api_key = os.getenv("GEMINI_API_KEY")
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",  # Use the correct Gemini embedding model name if different
    google_api_key=api_key,
)

# Set up ChromaDB client and collection
chroma_client = chromadb.PersistentClient(path="./chroma_data")
gdrive_collection = chroma_client.get_collection("google_drive")

@tool
def get_gdrive_context(query: str) -> str:
    """Returns relevant context from Google Drive documents for a given user query."""
    try:
        # Get embedding for the query
        query_embedding = embedding_model.embed_query(query)
        # Query ChromaDB for similar documents
        results = gdrive_collection.query(
            query_embeddings=[query_embedding],
            n_results=5  # Adjust as needed
        )
        # Format the results
        if not results["documents"] or not results["documents"][0]:
            return "No relevant context found in Google Drive documents."
        context = "\n---\n".join(doc for doc in results["documents"][0])
        return f"Relevant context from Google Drive:\n{context}"
    except Exception as e:
        return f"Error retrieving context: {e}" 