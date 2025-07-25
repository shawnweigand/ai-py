from langchain.text_splitter import RecursiveCharacterTextSplitter

# https://www.haihai.ai/gpt-gdrive/

def split_recursive_docs(docs):
    """
    Split documents into smaller chunks.

    Args:
        docs (List[Document]): A list of documents to split.

    Returns:
        List[Chunk]: A list of split documents.
    """

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )
    
    # Split the documents
    texts = text_splitter.split_documents(docs)
    
    return texts