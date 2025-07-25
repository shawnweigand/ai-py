from chromadb import PersistentClient

client = PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection("my_collection")