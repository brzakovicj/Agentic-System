from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
import logging
import uuid
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from dotenv import load_dotenv
from datetime import datetime
import sqlite3 
from pathlib import Path

# FastMCP imports
from fastmcp import FastMCP

# ChromaDB imports
import chromadb

# Importing our document partitioner from unstructured.io
from unstructured.partition.auto import partition

# Importing various text splitters for different chunking strategies
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)

from sentence_transformers import SentenceTransformer, CrossEncoder

from transformers import logging as hf_logging

# ── Logging ──────────────────────────────────────────────────────────────────

# Replace your entire logging block with this:
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("mcp_debug.log", mode="w")
    ]
)

# Use root logger — all child loggers propagate to it automatically
logger = logging.getLogger()  # ← no name argument, gets root logger

# Silence noisy libraries
for noisy in ("httpx", "sentence_transformers", "transformers", "huggingface_hub"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

hf_logging.set_verbosity_error()

logger.info("Logging initialized")

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".html", ".txt"}
COLLECTION_NAME = "study_materials"
DEFAULT_CHUNK_SIZE = 300
DEFAULT_CHUNK_OVERLAP = 50
CHROMA_UPSERT_BATCH = 512     # stay well under ChromaDB's per-call limit
MAX_PARSE_WORKERS = 4         # parallel file-parsing threads

# Initialize FastMCP server with ChromaDB capabilities
mcp = FastMCP("ChromaDB Server")

# ─────────────────────────────────────────────────────────────────────────────
# Data classes (plain dicts keep it dependency-free, typed aliases for clarity)
# ─────────────────────────────────────────────────────────────────────────────
PageDoc = Dict[str, Any]   # one page from a file
Chunk = Dict[str, Any]     # one vector-store record

class ChromaDB_Server:

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    
    def __init__(self) -> None:
        """Initializes the ChromaDB server, setting up the database connection."""
        self.chroma_client = None
        self.collection = None
        self.registry_db = None

        # Model loading is the slowest startup step — do it once.
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')

        # In-memory hash cache: populated on first use, invalidated on write.
        self._hash_cache: Optional[Set[str]] = None

        self._init_registry()
        self._initialize_chromadb()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _initialize_chromadb(self) -> None:
        """Initialize ChromaDB client and collection, then auto-ingest files from data directory"""
        try:            
            self.chroma_client = chromadb.HttpClient(
                host=os.getenv("CHROMA_HOST", "localhost"),
                port=int(os.getenv("CHROMA_PORT", 8000)),
            )
            
            self.collection = self.chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "Collection for document storage"}
            )
            
            count = self.collection.count()
            logger.info(f"ChromaDB initialized successfully. Vector database has {count} documents.")

            if count == 0:
                self._auto_ingest_files()
        except Exception as e:
            error_msg = f"Failed to initialize ChromaDB: {str(e)}"
            logger.error(error_msg)
            return error_msg
        
    def _init_registry(self):
        db_path = Path(__file__).resolve().parent / "sqlite" / "file_registry.db"
    
        # ensure folder exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.registry_db = sqlite3.connect(
            db_path,
            check_same_thread=False
        )
        self.registry_db.execute("""
            CREATE TABLE IF NOT EXISTS files (
                file_hash TEXT PRIMARY KEY,
                file_name TEXT,
                file_path TEXT UNIQUE,
                chunk_count INTEGER,
                created_at TEXT
            )
        """)
        self.registry_db.commit()

    # ── Directory helpers ─────────────────────────────────────────────────────

    def _get_data_directory(self) -> Path:
        """Get the data directory path with flexible resolution strategy."""
        # 1. Check environment variable first
        env = os.environ.get('DATA_DIR')
        if env:
            path = Path(env).expanduser().resolve()
            logger.info(f"Using data directory from DATA_DIR: {path}")
            return path
        
        # 2. No environment variable and no existing data directory - raise error
        error_msg = (
            "No data directory found. Please either:\n"
            "1. Set the DATA_DIR environment variable to specify a data directory, or\n"
            "2. Create a 'data' directory in the current working directory\n\n"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    def _check_data_directory_configured(self) -> tuple[bool, str]:
        """Check if data directory is properly configured."""
        try:
            path = self._get_data_directory()
            return True, f"Data directory configured: {path}"
        except ValueError:
            message = (
                "No data directory is configured for this ChromaDB system. "
                "The system cannot access any documents without a data directory.\n\n"
                "To set up a data directory, you can:\n"
                "1. Set the DATA_DIR environment variable:\n"
                "   export DATA_DIR=/path/to/your/documents\n"
                "2. Create a 'data' directory in the current working directory:\n"
                "   mkdir data\n\n"
                "After setting up the data directory, add your documents to it and restart the server "
                "or use the reingest_data_directory tool to load them."
            )
            return False, message

    # ── Hash cache ────────────────────────────────────────────────────────────

    def _build_hash_cache(self) -> Set[str]:
        """Fetch all stored hashes in one round-trip and cache them."""
        all_docs = self.collection.get(include=["metadatas"])
        cache: Set[str] = set()

        for meta in all_docs.get("metadatas") or []:
            if meta and "file_hash" in meta:
                cache.add(meta["file_hash"])
        
        logger.info("Hash cache built: %d unique file hashes.", len(cache))
        return cache
 
    def _get_hash_cache(self) -> Set[str]:
        if self._hash_cache is None:
            self._hash_cache = self._build_hash_cache()
        return self._hash_cache
 
    def _invalidate_hash_cache(self) -> None:
        self._hash_cache = None

    # ── File utilities ────────────────────────────────────────────────────────

    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        stat = file_path.stat()
        fingerprint = f"{stat.st_mtime}:{stat.st_size}"
        return hashlib.md5(fingerprint.encode()).hexdigest()

    @staticmethod
    def _sanitize_metadata(metadata: dict) -> dict:
        """Sanitize metadata to ensure all values are JSON serializable and non-null."""
        sanitized: dict = {}
        for key, value in metadata.items():
            if value is None:
                sanitized[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, (list, dict, tuple)):
                sanitized[key] = json.dumps(value)
            else:
                sanitized[key] = str(value)
        return sanitized
    
    # ── Parsing (parallel) ────────────────────────────────────────────────────

    def _parse_one_file(self, file_path: Path) -> List[PageDoc]:
        """
        Parse a single file into per-page dicts.
        Safe to run in a thread — unstructured's partition is CPU-bound.
        """
        try:
            elements = partition(str(file_path))
        except Exception as exc:
            logger.warning("Could not parse %s: %s", file_path.name, exc)
            return []
 
        if not elements:
            return []
 
        file_hash = self._compute_file_hash(file_path)
        doc_id = file_hash
        meta_dict = elements[0].metadata.to_dict()
        last_modified = meta_dict.get("last_modified")
 
        # Group element text by page number
        page_texts: Dict[int, List[str]] = {}
        for el in elements:
            pn = getattr(el.metadata, "page_number", None)
            pn = int(pn) if pn is not None else 1
            page_texts.setdefault(pn, []).append(str(el))
 
        pages: List[PageDoc] = []
        for page_number, texts in page_texts.items():
            pages.append(
                {
                    "file_id": doc_id,
                    "file_hash": file_hash,
                    "file_content": " ".join(texts).strip(),
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "file_type": file_path.suffix,
                    "page_number": page_number,
                    "last_modified": last_modified,
                }
            )

        return pages
    
    def _process_documents_pages(self, source_dir: Path, skip_hashes: Optional[Set[str]] = None) -> List[PageDoc]:
        """
        Discover all supported files under source_dir.
 
        When skip_hashes is provided:
          - Files whose hash is already in the set are skipped entirely
            (we compute the hash before parsing to avoid unnecessary I/O).
        Parsing runs in parallel via ThreadPoolExecutor.
        """
        all_files = [
            f
            for f in Path(source_dir).rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
 
        if not all_files:
            logger.info("No supported files found in %s.", source_dir)
            return []
 
        # ── Pre-filter by hash (cheap) before parsing (expensive) ────────────
        if skip_hashes:
            new_files = [
                f for f in all_files
                if self._compute_file_hash(f) not in skip_hashes
            ]
            skipped = len(all_files) - len(new_files)
            if skipped:
                logger.info("Skipping %d unchanged file(s). Parsing %d new.", skipped, len(new_files))
            all_files = new_files
 
        if not all_files:
            return []
 
        logger.info("Parsing %d file(s) with %d worker(s)…", len(all_files), MAX_PARSE_WORKERS)
 
        pages: List[PageDoc] = []
        with ThreadPoolExecutor(max_workers=MAX_PARSE_WORKERS) as pool:
            futures = {pool.submit(self._parse_one_file, f): f for f in all_files}
            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    result = future.result()
                    pages.extend(result)
                    logger.info("Parsed %s → %d page(s).", file_path.name, len(result))
                except Exception as exc:
                    logger.error("Error parsing %s: %s", file_path.name, exc)
 
        return pages
    
    # ── Chunking ──────────────────────────────────────────────────────────────

    def _chunk_document(self, text: str, method: str = 'recursive', encoding_name: str = "cl100k_base", chunk_size: int = 300, chunk_overlap: int = 50) -> List[str]:
        """
        Chunk a document's text using the selected strategy.

        Args:
            text (str): The document's text to chunk.

        Returns:
            List[str]: A list of text chunks.
        """
        if method == 'fixed':
            splitter = CharacterTextSplitter.from_tiktoken_encoder(
                encoding_name=encoding_name, 
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
        elif method == 'recursive':
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name=encoding_name, 
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
        else:
            raise ValueError("Unknown chunking method: choose 'fixed' or 'recursive'.")
        
        return splitter.split_text(text)

    def _pages_to_chunks(self, pages: List[PageDoc]) -> List[Chunk]:
        """Convert page dicts into flat chunk dicts (no embeddings yet)."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        all_chunks: List[Chunk] = []

        for page in pages:
            for text in self._chunk_document(page["file_content"]):
                all_chunks.append(
                    {
                        "id": str(uuid.uuid4()),
                        "text": text,
                        "metadata": {
                            "file_id": page["file_id"],
                            "file_hash": page["file_hash"],
                            "file_path": page["file_path"],
                            "file_name": page["file_name"],
                            "file_type": page["file_type"],
                            "page_number": page["page_number"],
                            "last_modified_date": page["last_modified"] or "",
                            "creation_date": now,
                            "chunk_method": "recursive_character",
                        },
                    }
                )
        return all_chunks
    
    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        texts = [c["text"] for c in chunks]
        embeddings = self.embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

        for chunk, vec in zip(chunks, embeddings):
            chunk["embedding"] = vec.tolist()
        
        return chunks

    # ── ChromaDB write ────────────────────────────────────────────────────────
 
    def _upsert_chunks(self, chunks: List[Chunk]) -> None:
        """
        Write chunks to ChromaDB in fixed-size batches.
        Uses upsert so the pipeline is idempotent (safe to re-run).
        Invalidates the hash cache after every successful write.
        """
        total = len(chunks)

        for start in range(0, total, CHROMA_UPSERT_BATCH):
            batch = chunks[start : start + CHROMA_UPSERT_BATCH]
            
            self.collection.upsert(
                ids=[c["id"] for c in batch],
                embeddings=[c["embedding"] for c in batch],
                metadatas=[c["metadata"] for c in batch],
                documents=[c["text"] for c in batch],
            )

            logger.info(
                "Upserted %d / %d chunks to ChromaDB.",
                min(start + CHROMA_UPSERT_BATCH, total),
                total,
            )
        
        self._invalidate_hash_cache()

    # ── Registry ─────────────────────────────────────────────────────────────

    def _register_file(self, file_hash, file_name, file_path, chunk_count):
        self.registry_db.execute("""
            INSERT OR REPLACE INTO files
            (file_hash, file_name, file_path, chunk_count, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            file_hash,
            file_name,
            file_path,
            chunk_count,
            datetime.now().isoformat()
        ))
        self.registry_db.commit()

    # ── High-level ingestion entry points ─────────────────────────────────────
 
    def _run_ingestion(self, source_dir: Path, skip_existing: bool = True) -> str:
        """
        Core pipeline: discover → (optionally filter) → parse → chunk → embed → store.
 
        Args:
            source_dir:     Directory to scan for documents.
            skip_existing:  When True, files already in the DB are skipped.
        """
        skip_hashes = self._get_hash_cache() if skip_existing else None
 
        pages = self._process_documents_pages(source_dir, skip_hashes = skip_hashes)
        if not pages:
            return "No new files to ingest."
 
        logger.info("Building %d pages into chunks…", len(pages))
        chunks = self._pages_to_chunks(pages)

        logger.info("%d chunks created. Generating embeddings…", len(chunks))
        chunks = self._embed_chunks(chunks)

        try:
            logger.info("Embedding done. Writing to ChromaDB…")
            self._upsert_chunks(chunks)

            # ── NEW: register files in SQLite ─────────────────────────────
            file_stats = {}

            for page in pages:
                file_hash = page["file_hash"]

                if file_hash not in file_stats:
                    file_stats[file_hash] = {
                        "file_name": page["file_name"],
                        "file_path": page["file_path"],
                        "chunk_count": 0
                    }

                # estimate chunk count later more accurately
                file_stats[file_hash]["chunk_count"] += 1

            # overwrite with correct chunk count from Chroma grouping
            for file_hash in file_stats:
                actual_chunk_count = len([
                    c for c in chunks
                    if c["metadata"]["file_hash"] == file_hash
                ])
                file_stats[file_hash]["chunk_count"] = actual_chunk_count

            # write registry
            for file_hash, info in file_stats.items():
                self._register_file(
                    file_hash=file_hash,
                    file_name=info["file_name"],
                    file_path=info["file_path"],
                    chunk_count=info["chunk_count"]
                )

    
            final_count = self.collection.count()
            return (
                f"Ingestion complete. "
                f"Added {len(chunks)} chunk(s). "
                f"Collection now has {final_count} chunk(s)."
            )
        except Exception as e:
            # Chroma has no transactions, but at least roll back SQLite
            self.registry_db.rollback()
            self._invalidate_hash_cache()
            raise ################################################################################################
    
    def _auto_ingest_files(self) -> str:
        """Called on startup when the collection is empty."""
        try:
            data_path = self._get_data_directory()
            os.makedirs(data_path, exist_ok=True)
            # Fresh DB — no need to check existing hashes
            return self._run_ingestion(data_path, skip_existing=False)
        except ValueError as exc:
            logger.warning("Skipping auto-ingestion: %s", exc)
            return str(exc)
        except Exception as exc:
            logger.error("Auto-ingestion failed: %s", exc)
            return str(exc)

    # ── MCP tool implementations ───────────────────────────────────────────────

    async def _ingest_new_documents(self) -> str:
        """Ingest only files not already in the DB (delta ingestion)."""
        try:
            data_path = self._get_data_directory()
            return self._run_ingestion(data_path, skip_existing = True)
        except ValueError as exc:
            return f"Skipping ingestion: {exc}"
        except Exception as exc:
            logger.error("Incremental ingestion failed: %s", exc)
            return f"Ingestion error: {exc}"
        
    async def _reingest_data_directory(self) -> str:
        """Drop the collection and re-ingest everything from scratch."""
        ok, msg = self._check_data_directory_configured()

        if not ok:
            return msg
        if not self.chroma_client:
            return "Error: ChromaDB client not initialised."
 
        logger.info("Dropping collection '%s' for full re-ingestion.", COLLECTION_NAME)
        
        try:
            self.chroma_client.delete_collection(name = COLLECTION_NAME)
            self.registry_db.execute("DELETE FROM files")
            self.registry_db.commit()
        except Exception:
            logger.warning("Collection did not exist or could not be deleted.")
 
        self.collection = self.chroma_client.get_or_create_collection(
            name = COLLECTION_NAME,
            metadata = {"description": "Collection for document storage"},
        )

        self._invalidate_hash_cache()
 
        return self._auto_ingest_files()
    
    async def _list_ingested_files(self) -> str:
        ok, msg = self._check_data_directory_configured()
        if not ok:
            return msg
        
        # Cross-check: get hashes from Chroma
        chroma_hashes = self._get_hash_cache()

        cursor = self.registry_db.execute("""
            SELECT file_name, file_path, chunk_count, created_at, file_hash
            FROM files
            ORDER BY created_at DESC
        """)

        rows = cursor.fetchall()

        if not rows:
            return "No files have been ingested yet."

        lines = [f"Ingested files ({len(rows)} total):\n"]

        for name, path, chunks, created, fhash in rows:
            in_chroma = "✓" if fhash in chroma_hashes else "⚠ missing from Chroma"
            lines.append(f"{name} [{in_chroma}] — {chunks} chunks")

        # for i, (name, path, chunks, created) in enumerate(rows, 1):
        #     lines.append(
        #         f"{i}. {name}\n"
        #         f"   Path: {path}\n"
        #         f"   Chunks: {chunks}\n"
        #         f"   Created: {created}\n"
        #     )

        return "\n".join(lines)
        # ok, msg = self._check_data_directory_configured()
        # if not ok:
        #     return msg
 
        # all_docs = self.collection.get(include = ["metadatas"])
        # if not all_docs["metadatas"]:
        #     return "No files have been ingested yet."
 
        # file_info: Dict[str, dict] = {}
        # for meta in all_docs["metadatas"]:
        #     if not meta or "file_name" not in meta:
        #         continue

        #     key = f"{meta['file_name']} ({meta.get('file_path', '?')})"
            
        #     if key not in file_info:
        #         file_info[key] = {
        #             "file_name": meta["file_name"],
        #             "file_path": meta.get("file_path", "?"),
        #             "file_type": meta.get("file_type", "?"),
        #             "creation_date": meta.get("creation_date", "?"),
        #             "last_modified_date": meta.get("last_modified_date", "?"),
        #             "chunks": 0,
        #         }
            
        #     file_info[key]["chunks"] += 1
 
        # lines = [f"Ingested files ({len(file_info)} total):\n"]

        # for i, info in enumerate(file_info.values(), 1):
        #     lines.append(
        #         f"{i}. {info['file_name']}\n"
        #         f"   Path:     {info['file_path']}\n"
        #         f"   Type:     {info['file_type']}\n"
        #         f"   Created:  {info['creation_date']}\n"
        #         f"   Modified: {info['last_modified_date']}\n"
        #         f"   Chunks:   {info['chunks']}\n"
        #     )

        # total_chunks = sum(v["chunks"] for v in file_info.values())
        # lines.append(f"Total chunks in DB: {total_chunks}")
        
        # return "\n".join(lines)
        
    async def _delete_files_from_db(self, file_names: List[str]) -> str:
        if not self.collection:
            return "Error: collection not initialised."

        if not file_names:
            return "No file names provided."

        try:
            # 1. Get file hashes from registry
            placeholders = ",".join("?" for _ in file_names)

            cursor = self.registry_db.execute(f"""
                SELECT file_hash FROM files
                WHERE file_name IN ({placeholders})
            """, file_names)

            file_hashes = [row[0] for row in cursor.fetchall()]

            if not file_hashes:
                return f"No files found in registry: {file_names}"

            # 2. Fetch the actual chunk IDs from Chroma for those hashes
            #    Do it one hash at a time — $in filter is the unreliable part
            ids_to_delete = []
            for fhash in file_hashes:
                result = self.collection.get(
                    where={"file_hash": {"$eq": fhash}},  # $eq is more reliable than $in
                    include=[]  # we only need IDs
                )
                ids_to_delete.extend(result["ids"])

            if not ids_to_delete:
                return f"No chunks found in Chroma for: {file_names}. Registry may be out of sync."

            # 3. Delete by explicit IDs — this is always reliable
            self.collection.delete(ids=ids_to_delete)

            # 4. Verify deletion actually worked
            verify = self.collection.get(ids=ids_to_delete, include=[])
            if verify["ids"]:
                return f"Warning: deletion may have partially failed — {len(verify['ids'])} chunks remain."

            # 5. Remove from SQLite only after confirmed Chroma delete
            self.registry_db.execute(
                f"DELETE FROM files WHERE file_name IN ({placeholders})",
                file_names
            )
            self.registry_db.commit()

            self._invalidate_hash_cache()

            return f"Successfully deleted {len(ids_to_delete)} chunks across {len(file_hashes)} file(s)."

        except Exception as e:
            logger.exception("Delete failed")
            return f"Error deleting files: {e}"
        
server = ChromaDB_Server()

@mcp.tool()
async def list_ingested_files() -> str:
    """
    List all files currently stored in the vector database.

    This tool returns information about ingested documents, such as:
    - file name or path
    - number of chunks
    - basic metadata

    Returns:
        str: A summary of all ingested files.

    Usage:
        Call this tool when the user wants to see which documents
        are available in the knowledge base.
    """
    return await server._list_ingested_files()

@mcp.tool()
async def ingest_new_documents() -> str:
    """
    Ingest new documents from the data directory into the vector database.

    This tool scans the configured data directory and adds ONLY documents
    that are not already stored in the database (based on file hash).

    It does NOT modify or delete existing documents.

    Returns:
        str: A message indicating whether new documents were ingested
        or if no new files were found.

    Usage:
        Call this tool when the user wants to add new documents to the system.
    """
    return await server._ingest_new_documents()

@mcp.tool()
async def reingest_data_directory() -> str:
    """
    Reset and rebuild the entire vector database from the data directory.

    This tool deletes all existing embeddings and reprocesses all documents
    from the configured data directory.

    WARNING:
        This is a destructive operation. All existing data will be permanently deleted.

    Returns:
        str: A message indicating success or failure of the reingestion process.

    Usage:
        Call this tool when the user wants to fully rebuild the knowledge base
        or fix inconsistent/corrupted data.
    """
    return await server._reingest_data_directory()

@mcp.tool()
async def delete_files_from_db(file_names: List[str]) -> str:
    """
    Delete documents from the vector database by file name.

    This tool removes all vector chunks associated with the given file names.
    It does NOT delete the original files from disk.

    Arguments:
        file_names (List[str]): A list of file names to delete from the database.

    Returns:
        str: A message indicating the result of the deletion.

    Usage:
        Call this tool when the user wants to remove specific documents
        from the database.

        Example:
        {
            "file_names": ["file1.pdf", "notes.txt"]
        }
    """
    return await server._delete_files_from_db(file_names=file_names)


if __name__ == "__main__":
    try:
        logger.info("Starting ChromaDB MCP Server...")
        mcp.run(transport="stdio")
    except Exception as e:
        logger.exception("SERVER CRASHED")
        raise