import os
import psycopg2
from psycopg2.extensions import register_adapter, AsIs
from dotenv import load_dotenv

load_dotenv()

# Register numpy array adapter if needed (for pgvector)
try:
    import numpy as np
    def adapt_numpy_array(arr):
        return AsIs(tuple(arr))
    register_adapter(np.ndarray, adapt_numpy_array)
except ImportError:
    pass

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")

conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASS
)

cur = conn.cursor()

# Enable pgvector extension
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

# Create table for embeddings
cur.execute('''
CREATE TABLE IF NOT EXISTS document_embeddings (
    id SERIAL PRIMARY KEY,
    parent_id TEXT,
    type TEXT,
    document_id TEXT,
    document_title TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    source TEXT,
    content TEXT,
    embedding VECTOR(1536) NOT NULL
);
''')

conn.commit()
cur.close()
conn.close()

print("✅ Table 'document_embeddings' is ready.")
