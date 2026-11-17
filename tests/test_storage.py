import os
import sys
import tempfile
import unittest
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import storage


class TestStorageLegacySessions(unittest.TestCase):
    def test_legacy_session_load_without_chunk_count_or_model(self):
        """Verify sessions saved before chunk_count and embedding_model existed still load cleanly."""
        sid = "deadbeefcafebabe"
        embed_model = FakeEmbeddings(size=8)
        chunks = [Document(page_content="Legacy document chunk", metadata={"chunk_id": "legacy#1"})]

        original_dir = storage.BASE_STORAGE_DIR
        with tempfile.TemporaryDirectory() as tmp_dir:
            storage.BASE_STORAGE_DIR = tmp_dir
            try:
                faiss_db = FAISS.from_documents(chunks, embed_model)
                storage.save_persisted_state(sid, faiss_db, chunks, ["legacy.pdf"], 0.5)

                # Overwrite metadata.json without chunk_count and embedding_model
                storage._atomic_write_json(
                    os.path.join(storage.get_session_dir(sid), "metadata.json"),
                    {"indexed_files": ["legacy.pdf"], "dense_weight": 0.5},
                )

                loaded_db, loaded_chunks, loaded_files, d_weight, _, warn = storage.load_persisted_state(
                    sid, embed_model, expected_embedding_model="all-MiniLM-L6-v2"
                )
                self.assertIsNotNone(loaded_db)
                self.assertEqual(len(loaded_chunks), 1)
                self.assertEqual(loaded_files, ["legacy.pdf"])
                self.assertEqual(d_weight, 0.5)
                self.assertIsNone(warn)
            finally:
                storage.BASE_STORAGE_DIR = original_dir


if __name__ == "__main__":
    unittest.main()
