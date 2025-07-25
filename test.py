from chromadb import PersistentClient
from tools.google_drive import get_gdrive_context

client = PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection("google_drive")

print(collection.get())

# if __name__ == "__main__":
# query = "Summarize my travel plans from Google Drive."
# result = get_gdrive_context(query)
# print("Result from get_gdrive_context:")
# print(result)
