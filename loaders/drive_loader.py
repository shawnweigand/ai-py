from langchain.document_loaders import GoogleDriveLoader

# https://www.haihai.ai/gpt-gdrive/

def load_drive_folder_docs(folder_id: str, recursive: bool = True):
    """
    Load documents from a Google Drive folder.
    
    Args:
        folder_id (str): The ID of the Google Drive folder.
        recursive (bool): Whether to load files recursively from subfolders.
        
    Returns:
        List[Document]: A list of loaded documents.
    """
    # Initialize the Google Drive loader
    loader = GoogleDriveLoader(
        folder_id=folder_id,
        file_types=["document", "sheet"],
        recursive=recursive,
    )
    
    # Load the documents
    docs = loader.load()
    
    return docs