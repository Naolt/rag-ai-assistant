import os
import chromadb
from typing import List, Dict, Any
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        print(f"Vector database initialized with collection: {self.collection_name}")

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Simple text chunking by splitting on spaces and grouping into chunks.

        Args:
            text: Input text to chunk
            chunk_size: Approximate number of characters per chunk

        Returns:
            List of text chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,          # ~200 words per chunk
            chunk_overlap=200,        # Overlap to preserve context
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_text(text)
    
        return chunks

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of documents
        """
        print(f"Processing {len(documents)} documents...")
        
        for idx, doc in enumerate(documents):
            # Compatibility check: standard LangChain docs use .page_content
            content = doc.page_content if hasattr(doc, 'page_content') else doc['content']
            metadata = doc.metadata if hasattr(doc, 'metadata') else doc.get('metadata', {})
            chunks = self.chunk_text(content)
           # 1. Create unique IDs for every chunk in this document
            ids = [f"doc_{idx}_chunk_{i}" for i in range(len(chunks))]

            # 2. Create a metadata list (Chroma needs one dict per chunk)
            # We copy the original metadata and add the chunk index to it
            metadatas = []
            for i in range(len(chunks)):
                chunk_meta = metadata.copy()
                chunk_meta["chunk_index"] = i
                metadatas.append(chunk_meta)

            # 3. Generate embeddings (PASS THE CHUNKS LIST, NOT DICTS)
            # Convert to list because Chroma prefers it over numpy arrays
            embeddings = self.embedding_model.encode(chunks).tolist()
            

            # 4. Add to the collection
            self.collection.add(
                embeddings=embeddings,
                ids=ids,
                documents=chunks, # List of strings
                metadatas=metadatas # List of dictionaries
            )

            print(f"Processed Document {idx+1}. Total chunks: {len(chunks)}")


    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary containing search results with keys: 'documents', 'metadatas', 'distances', 'ids'
        """
        # Convert question to vector
        query_vector = self.embedding_model.encode([query]).tolist()
    
        # Search for similar content
        results = self.collection.query(
            query_embeddings=query_vector,
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        return results
