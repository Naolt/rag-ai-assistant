import sys
import os
import pytest

# Add src to path so we can import vectordb
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vectordb import VectorDB


@pytest.fixture(scope="module")
def vdb(tmp_path_factory):
    """Create a VectorDB instance with a temporary ChromaDB path."""
    tmp_dir = tmp_path_factory.mktemp("chroma_test")
    original_init = VectorDB.__init__

    def patched_init(self, collection_name=None, embedding_model=None):
        self.collection_name = collection_name or "test_collection"
        self.embedding_model_name = embedding_model or "sentence-transformers/all-MiniLM-L6-v2"

        import chromadb
        self.client = chromadb.PersistentClient(path=str(tmp_dir))

        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Test collection"},
        )

    VectorDB.__init__ = patched_init
    db = VectorDB()
    VectorDB.__init__ = original_init
    return db


class TestChunkText:
    def test_returns_list(self, vdb):
        chunks = vdb.chunk_text("Hello world. This is a test.")
        assert isinstance(chunks, list)

    def test_chunks_nonempty(self, vdb):
        chunks = vdb.chunk_text("Hello world. This is a test.")
        assert len(chunks) > 0
        assert all(len(c) > 0 for c in chunks)

    def test_long_text_produces_multiple_chunks(self, vdb):
        long_text = "This is a sentence about AI. " * 100
        chunks = vdb.chunk_text(long_text, chunk_size=200)
        assert len(chunks) > 1

    def test_respects_chunk_size(self, vdb):
        long_text = "Word " * 500
        chunk_size = 200
        chunks = vdb.chunk_text(long_text, chunk_size=chunk_size)
        # Chunks should be roughly within the chunk_size (with some tolerance for overlap)
        for chunk in chunks:
            assert len(chunk) <= chunk_size + 100


class TestAddAndSearch:
    def test_add_documents_dict_format(self, vdb):
        docs = [
            {"content": "Python is a programming language used in AI.", "metadata": {"source": "test1.txt"}},
            {"content": "Machine learning is a subset of artificial intelligence.", "metadata": {"source": "test2.txt"}},
        ]
        vdb.add_documents(docs)
        assert vdb.collection.count() >= 2

    def test_search_returns_expected_keys(self, vdb):
        results = vdb.search("What is Python?", n_results=2)
        assert "documents" in results
        assert "metadatas" in results
        assert "distances" in results

    def test_search_returns_results(self, vdb):
        results = vdb.search("programming language", n_results=2)
        assert len(results["documents"]) > 0
        assert len(results["documents"][0]) > 0

    def test_search_results_are_relevant(self, vdb):
        results = vdb.search("programming language", n_results=1)
        top_result = results["documents"][0][0].lower()
        assert "python" in top_result or "programming" in top_result or "language" in top_result
