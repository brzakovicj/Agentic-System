import os
import json
import logging
import uuid
import numpy as np
import hashlib
from pathlib import Path
from typing import Dict, Any, List
from textwrap import dedent
from dotenv import load_dotenv
from datetime import datetime

# FastMCP imports
from fastmcp import FastMCP
from fastmcp.prompts import Message

# ChromaDB imports
import chromadb
from chromadb.config import Settings

# Importing our document partitioner from unstructured.io
from unstructured.partition.auto import partition

# Importing various text splitters for different chunking strategies
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)

from sentence_transformers import SentenceTransformer, CrossEncoder

from transformers import logging as hf_logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Silence noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# Transformers verbosity
hf_logging.set_verbosity_error()

load_dotenv()

# Initialize FastMCP server with RAG capabilities
mcp = FastMCP("RAG Server")

class RAG_Server:
    def __init__(self): # ok
        """Initializes the RAG server, setting up the database connection."""
        self.chroma_client = None
        self.collection = None

        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')

        self._initialize_chromadb()

    ##########################################################################################################################
    #################################################### HELPER FUNCTIONS ####################################################
    ##########################################################################################################################

    def _initialize_chromadb(self): # ok
        """Initialize ChromaDB client and collection, then auto-ingest files from data directory"""
        try:
            # Get database directory using flexible resolution
            persist_directory = self._get_database_directory()
            
            os.makedirs(persist_directory, exist_ok=True)
            
            self.chroma_client = chromadb.PersistentClient(
                path=str(persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create fresh collection for RAG documents
            self.collection = self.chroma_client.get_or_create_collection(
                name="study_materials",
                metadata={"description": "Collection for RAG document storage"}
            )
            
            logger.info(f"ChromaDB initialized successfully. Vector database has {self.collection.count()} documents.")

            if self.collection.count() > 0:
                logger.info("Database already has data. Skipping auto-ingestion.")
                return
            
            # Auto-ingest files from data directory
            self._auto_ingest_files()
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def _get_database_directory(self): # ok
        """Get the ChromaDB persistent directory path with flexible resolution strategy."""
        # 1. Check environment variable first
        env_db_dir = os.environ.get("RAG_DB_DIR", None)
        if env_db_dir:
            db_path = Path(env_db_dir).expanduser().resolve()
            logger.info(f"Using database directory from RAG_DB_DIR: {db_path}")
            return db_path
    
        # 2. No environment variable - raise error
        error_msg = (
            "No database directory found. Please either:\n"
            "1. Set the RAG_DB_DIR environment variable to specify a database directory, or\n"
            "2. Ensure a valid database directory is configured\n\n"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    def _get_data_directory(self): # ok
        """Get the data directory path with flexible resolution strategy."""
        # 1. Check environment variable first
        env_data_dir = os.environ.get('RAG_DATA_DIR')
        if env_data_dir:
            data_path = Path(env_data_dir).expanduser().resolve()
            logger.info(f"Using data directory from RAG_DATA_DIR: {data_path}")
            return data_path
        
        # 2. No environment variable and no existing data directory - raise error
        error_msg = (
            "No data directory found. Please either:\n"
            "1. Set the RAG_DATA_DIR environment variable to specify a data directory, or\n"
            "2. Create a 'data' directory in the current working directory\n\n"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    def _check_data_directory_configured(self): # ok
        """Check if data directory is properly configured."""
        try:
            data_path = self._get_data_directory()
            return True, f"Data directory configured: {data_path}"
        except ValueError:
            message = (
                "No data directory is configured for this RAG system. "
                "The system cannot access any documents without a data directory.\n\n"
                "To set up a data directory, you can:\n"
                "1. Set the RAG_DATA_DIR environment variable:\n"
                "   export RAG_DATA_DIR=/path/to/your/documents\n"
                "2. Create a 'data' directory in the current working directory:\n"
                "   mkdir data\n\n"
                "After setting up the data directory, add your documents to it and restart the server "
                "or use the reingest_data_directory tool to load them."
            )
            return False, message

    def _sanitize_metadata(self, metadata: dict) -> dict:
        """Sanitize metadata to ensure all values are JSON serializable and non-null."""
        sanitized = {}
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

    def _auto_ingest_files(self):
        """Automatically ingest all files from the data directory"""
        try:
            ################ LOAD DATA ################

            data_path = self._get_data_directory()
            os.makedirs(data_path, exist_ok=True)

            pages = self._process_documents_pages(data_path)

            if not pages or len(pages) == 0:
                logger.info("No files found in data directory. Skipping auto-ingestion.")
                return

            self._ingest_files(pages = pages)
            
        except ValueError as e:
            logger.warning(f"Skipping auto-ingestion: {e}")
            return
        except Exception as e:
            logger.error(f"Failed during auto-ingestion: {e}")

    def _process_documents_pages(self, source_dir: str) -> List[Dict]:
        """
        Load documents from a given directory using Unstructured IO and group text by page number.

        Args:
            source_dir (str): Directory containing documents.

        Returns:
            List[Dict]: A list of dictionaries where each dictionary contains:
                        - 'text': combined text from a page,
                        - 'source': file path,
                        - 'file_name': file name with extension,
                        - 'file_type': file extension,
                        - 'page_number': page number,
                        - 'doc_id': a unique id for the entire file.
        """
        pages = []
        
        all_files = [f for f in Path(source_dir).rglob('*') if f.is_file()]

        if not all_files:
            #logger.info("No documents found in directory.")
            return []  # or raise an exception if you prefer
        
        logger.info(f"Found {len(all_files)} files in data directory. Starting ingestion...")
            
        for file_path in all_files:
            if file_path.suffix.lower() in ('.pdf', '.docx', '.pptx', '.html', '.txt'):
                logger.info(f"Processing file: {file_path}")
                elements = partition(str(file_path))

                # DEBUG
                # if elements:
                #     # Ispisuje sve dostupne atribute metapodataka prvog elementa
                
                for e in elements:
                    logger.info(f"Dostupni metapodaci za {file_path.name}: {e.metadata.to_dict()}")

                # Single doc_id per file
                doc_id = str(uuid.uuid4())

                # Group text by page number
                page_texts = {}
                for el in elements:
                    # Get the page number; default to 1 if not provided
                    page_number = getattr(el.metadata, "page_number", None)
                    page_number = int(page_number) if page_number is not None else 1
                    
                    # Append element text to the corresponding page's list
                    page_texts.setdefault(page_number, []).append(str(el))
                
                # Added metadata
                meta_dict = elements[0].metadata.to_dict() if elements else {}
                last_modified = meta_dict.get("last_modified", None) 

                # Compute hash
                file_hash = self._compute_file_hash(file_path)
                
                # Create a document entry for each page
                for page_number, texts in page_texts.items():
                    combined_text = " ".join(texts).strip()
                    pages.append({
                        "file_id": doc_id,
                        "file_hash": file_hash,
                        "file_content": combined_text,
                        "file_path": str(file_path),
                        "file_name": file_path.name,
                        "file_type": file_path.suffix,
                        "page_number": page_number,
                        "last_modified": last_modified
                    })
        return pages
    
    def _ingest_files(self, pages: List[Dict]):
        """Chunks, embeds and inserts document pages into ChromaDB collection."""
        # DEBUG
        for page in pages:
            logger.info(f"File Name: {page['file_name']}")
            logger.info(f"Page Number: {page['page_number']}")
            logger.info(f"Text: {page['file_content'][:200]}...")
            logger.info("-" * 80)
            #logger.info(page)
            #logger.info("-" * 80)  # Separator for readability
        
        ################ CHUNKING ################
        
        all_chunks = []

        for page in pages:
            chunks = self._chunk_document(text = page["file_content"], chunk_size = 300, chunk_overlap = 50)
            
            for chunk in chunks:
                all_chunks.append({
                    "id": str(uuid.uuid4()),
                    "text": chunk,
                    "metadata": {
                        "file_id": page["file_id"],
                        "file_hash": page["file_hash"],
                        "file_path": page["file_path"],
                        "file_name": page["file_name"],
                        "file_type": page["file_type"],
                        "page_number": page["page_number"],
                        "last_modified_date": page["last_modified"],
                        "creation_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "chunk_method": "recursive_character",
                    }
                })

        logger.info(f"Created {len(all_chunks)} text chunks.")

        ################ EMBEDDING ################

        # Gather all chunk texts for embedding generation
        chunk_texts = [chunk["text"] for chunk in all_chunks]

        # Compute embeddings (using batch encoding for efficiency)
        embeddings = self.embed_model.encode(chunk_texts, convert_to_numpy=True, show_progress_bar=True)

        # Attach each embedding (converted to a list) to the corresponding chunk
        for i, chunk in enumerate(all_chunks):
            chunk["embedding"] = embeddings[i].tolist()

        logger.info("Embeddings generated and attached to each chunk.")
        
        ################ CHROMADB ################

        self.collection.add(
            ids=[chunk["id"] for chunk in all_chunks],
            embeddings=embeddings.tolist(),
            metadatas=[chunk["metadata"] for chunk in all_chunks],
            documents=chunk_texts
        )

        final_count = self.collection.count()
        logger.info(f"Auto-ingestion completed. Collection now has {final_count} documents.")

    # DEBUG
    def _reformat(self, chroma_results: dict) -> list:
        """
        Reformat chroma db results to a list of search items containing:
        - chunk_id
        - chunk_index
        - doc_id
        - page_number
        - source
        - text (from documents)
        - distance
        - score

        Parameters:
            chroma_results (dict): The raw results from the Chroma DB query.

        Returns:
            list: A list of dictionaries with the desired keys.
        """
        reformatted = []
        
        # Get the lists from the results. They are expected to be lists of lists.
        metadatas = chroma_results.get("metadatas", [])
        documents = chroma_results.get("documents", [])
        distances = chroma_results.get("distances", [])
        
        # Loop over each group (each inner list represents one set of matches)
        chunk_index = 1
        for meta_group, doc_group, distance_group in zip(metadatas, documents, distances):
            # Iterate over each item in the inner lists
            for meta, text, distance in zip(meta_group, doc_group, distance_group):
                item = {
                    "chunk_index": chunk_index,
                    "chunk_id": meta.get("chunk_id"),
                    "doc_id": meta.get("doc_id"),
                    "page_number": meta.get("page_number"),
                    "source": meta.get("source"),
                    "text": text,
                    "distance": distance,
                    "score": 1 - distance
                }
                reformatted.append(item)
                chunk_index += 1
        
        return reformatted

    def _compute_file_hash(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def _get_existing_file_hashes(self) -> set:
        all_docs = self.collection.get(include=["metadatas"])
        
        hashes = set()
        for meta in all_docs.get("metadatas", []):
            if meta and "file_hash" in meta:
                hashes.add(meta["file_hash"])
        
        return hashes

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
            return splitter.split_text(text)
        elif method == 'recursive':
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name=encoding_name, 
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
            return splitter.split_text(text)
        else:
            raise ValueError("Unknown chunking method: choose 'fixed' or 'recursive'.")

    ###################################################################################################################
    #################################################### MCP TOOLS ####################################################
    ###################################################################################################################
    
    # @mcp.tool()
    def query_documents(self, query: str, n_results: int = 5, include_metadata: bool = True, top_n: int = 3) -> str: # Ogranicenje za top_n?
        """
        Performs semantic search over the vector database to retrieve and rerank the most relevant document chunks for a given query.

        This tool executes a two-stage retrieval process:

        1. INITIAL RETRIEVAL (Vector Search):
        - Uses ChromaDB to perform semantic similarity search over stored document embeddings
        - Retrieves the top N candidate chunks based on vector similarity (controlled by `n_results`)

        2. RERANKING (Cross-Encoder):
        - Applies a cross-encoder model to rerank retrieved chunks for improved relevance accuracy
        - Produces a refined ranking of the most semantically relevant results
        - Final output is limited to `top_n` results

        This hybrid approach improves accuracy by combining fast vector search with deeper semantic scoring.

        Use cases:
        - Finding precise answers in large document collections
        - Extracting contextually relevant passages across multiple files
        - Improving search accuracy beyond embedding similarity
        - Supporting question answering and RAG-based reasoning

        Args:
            query (str):
                The natural language search query used to retrieve relevant document chunks.

            n_results (int, default=5):
                Number of candidate chunks retrieved from the vector database before reranking.
                Higher values improve recall but increase latency.

            include_metadata (bool, default=True):
                Whether to include document metadata (file name, source path, page number) in the output.

            top_n (int, default=3):
                Number of final results returned after reranking.
                Must be ≤ n_results for meaningful ranking behavior.

        Returns:
            A formatted string containing:
            - The original query
            - Top-ranked document chunks after reranking
            - Optional metadata (source file, page number)
            - Relevance scores and similarity scores

        Scoring:
            - Similarity Score: cosine distance-based similarity from vector search (pre-rerank)
            - Relevance Score: cross-encoder score (final ranking signal)

        Notes:
        - This tool prioritizes precision over recall in the final output due to reranking
        - Increasing `n_results` improves candidate diversity for reranking
        - `top_n` controls final output size after reranking
        """
        try:
            ################ RETRIEVE ################

            is_configured, config_message = self._check_data_directory_configured()
            if not is_configured:
                return config_message
            
            if not query.strip():
                return "Error: Query cannot be empty."
            
            if n_results <= 0:
                n_results = 5
            elif n_results > 20:
                n_results = 20
            
            top_n = min(top_n, n_results)

            if top_n <= 0:
                top_n = n_results

            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            if not results["documents"] or not results["documents"][0]:
                return "No relevant documents found for your query."
            
            ################ RERANK ################

            documents = results["documents"][0]
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(documents)
            distances = results["distances"][0] if results["distances"] else [0] * len(documents)

            # Rerank
            pairs = [(query, doc) for doc in documents]
            scores = self.cross_encoder.predict(pairs)

            # Sort by score descending
            top_indices = np.argsort(scores)[-top_n:][::-1]

            # Apply same ordering to everything
            documents = [documents[i] for i in top_indices]
            metadatas = [metadatas[i] for i in top_indices]
            distances = [distances[i] for i in top_indices]
            scores = [scores[i] for i in top_indices]

            ################ FORMAT ################

            formatted_results = []
            
            for i, (doc, metadata, distance, score) in enumerate(zip(documents, metadatas, distances, scores)):
                result_text = f"\n--- Result {i+1} ---\n"
                result_text += f"Content: {doc}\n"
                
                if include_metadata and metadata:
                    result_text += f"File Name: {metadata.get('file_name', 'Unknown')}\n"
                    result_text += f"Source: {metadata.get('file_path', 'Unknown')}\n"
                    result_text += f"Page Number: {metadata.get('page_number', 'Unknown')}\n"
                    #result_text += f"Chunk: {metadata.get('chunk_index', 'Unknown')} of {metadata.get('total_chunks', 'Unknown')}\n"
                    result_text += f"Similarity Score: {1 - distance:.3f}\n" # Sta ce nam ovo?
                    result_text += f"Relevance Score: {score:.3f}\n" # Sta ce nam ovo?
                
                formatted_results.append(result_text)
            
            response = f"Found {len(documents)} relevant documents for query: '{query}'\n"
            response += "\n".join(formatted_results)
            
            logger.info(f"Query '{query}' returned {len(documents)} results")
            return response
            
        except Exception as e:
            error_msg = f"Error querying documents: {str(e)}"
            logger.error(error_msg)
            return error_msg

    # OK treba verovatno izmeniti metadata koje se izlistavaju i generalno koje se cuvaju u vectorstore
    # @mcp.tool()
    def list_ingested_files(self) -> str:
        """
        Provides a comprehensive list of all files that have been successfully ingested into the vector database.

        This tool is useful for:
        - Verifying which files are included in the knowledge base.
        - Checking the metadata of each file, such as file type, size, and ingestion date.
        - Understanding how documents are chunked and stored in the database.

        The output includes:
        - A summary of each ingested file with its path, type, size, and modification dates.
        - The number of chunks each file has been divided into.
        - The total size of the content stored in the database.
        """

        try:
            is_configured, config_message = self._check_data_directory_configured()
            if not is_configured:
                return config_message
            
            all_docs = self.collection.get(include=["metadatas"])
            if not all_docs["metadatas"]:
                return "No files have been ingested yet."
            
            file_info = {}
            for metadata in all_docs["metadatas"]:
                if metadata and "file_name" in metadata:
                    file_name = metadata["file_name"]
                    file_path = metadata.get("file_path", "Unknown path")
                    file_key = f"{file_name} ({file_path})"
                    
                    if file_key not in file_info:
                        file_info[file_key] = {
                            "file_name": file_name,
                            "file_path": file_path,
                            "file_type": metadata.get("file_type", "Unknown"),
                            "file_size": metadata.get("file_size", 0),
                            "creation_date": metadata.get("creation_date", "Unknown"),
                            "last_modified_date": metadata.get("last_modified_date", "Unknown"),
                            "ingestion_method": metadata.get("ingestion_method", "Unknown"),
                            "chunks_found": 0,
                            "total_chunk_size": 0
                        }

                    file_info[file_key]["chunks_found"] += 1
                    file_info[file_key]["total_chunk_size"] += metadata.get("chunk_size", 0)
            
            if not file_info:
                return "No files have been ingested yet."
            
            response = f"Ingested Files ({len(file_info)} total):\n\n"
            for i, (file_key, info) in enumerate(file_info.items(), 1):
                response += f"{i}. {info['file_name']}\n"
                response += f"   Path: {info['file_path']}\n"
                response += f"   Type: {info['file_type']}\n"
                response += f"   Size: {info['file_size']:,} bytes\n"
                response += f"   Created: {info['creation_date']}\n"
                response += f"   Modified: {info['last_modified_date']}\n"
                response += f"   Chunks: {info['chunks_found']}\n"
                response += f"   Total chunk size: {info['total_chunk_size']:,} characters\n"
                response += f"   Ingestion method: {info['ingestion_method']}\n\n"
            
            total_chunks = sum(info["chunks_found"] for info in file_info.values())
            total_chunk_size = sum(info["total_chunk_size"] for info in file_info.values())
            response += f"Total chunks in database: {total_chunks}\n"
            response += f"Total content size: {total_chunk_size:,} characters"

            return response
            
        except Exception as e:
            error_msg = f"Error listing ingested files: {str(e)}"
            logger.error(error_msg)
            return error_msg
        
    # @mcp.tool()
    def ingest_new_documents(self) -> str:
        """
        Incrementally ingests only new documents from the configured data directory into the vector database.

        This tool scans the local data directory and compares files against existing entries in the ChromaDB
        using file-level hashing. Only documents that have NOT been previously ingested are processed.

        Workflow:
        1. Retrieves existing file hashes from the vector database
        2. Scans the data directory for all supported documents
        3. Filters out files that already exist in the database
        4. Processes remaining new files (chunking + embedding)
        5. Stores new vector embeddings in ChromaDB

        IMPORTANT:
        - This is a non-destructive, incremental ingestion operation
        - Existing documents in the database are NOT modified or deleted
        - Only completely new files are added
        - Files are identified using file_hash (not file name)

        Use cases:
        - Adding new study materials without reprocessing the entire dataset
        - Periodic syncing of a growing document folder
        - Efficient updates when only a subset of files changed

        Returns:
        - A status message indicating success or that no new files were found

        Behavior notes:
        - Skips ingestion if no new files are detected
        - Logs detailed ingestion progress for debugging
        """
        try:
            existing_hashes = self._get_existing_file_hashes()

            data_path = self._get_data_directory()
            pages = self._process_documents_pages(data_path)

            # Filter pages belonging to new files only
            new_pages = [p for p in pages if p["file_hash"] not in existing_hashes]

            if not new_pages:
                logger.info("\nNo new files to ingest.\n")
                return "No new files to ingest."
            
            logger.info("\nFound new files to ingest!\n")

            self._ingest_files(pages = new_pages)
            
        except ValueError as e:
            logger.warning(f"Skipping auto-ingestion: {e}")
            return
        except Exception as e:
            logger.error(f"Failed during auto-ingestion: {e}")

    # Ok ne treba nam bas, ali neka
    # @mcp.tool()
    def get_rag_status(self) -> Dict[str, Any]:
        """
        Returns a comprehensive diagnostic snapshot of the current RAG system state, configuration, and runtime environment.

        This tool is intended for system inspection, debugging, and verification of correct setup.

        It aggregates information across all major components of the RAG pipeline:

        1. SYSTEM STATUS
        - Whether the MCP server is running
        - Whether the ChromaDB client and collection are initialized
        - Total number of documents (chunks) currently stored
        - Whether auto-ingestion is functionally enabled

        2. DATABASE CONFIGURATION
        - Vector database type (ChromaDB)
        - Storage directory location and existence status
        - Collection name and source configuration
        - Whether configuration originates from environment variables or defaults

        3. DATA DIRECTORY STATUS
        - Path to the document ingestion directory
        - Whether the directory exists and is accessible
        - Whether it is properly configured via environment variables or workspace defaults

        4. ENVIRONMENT VARIABLES
        - Status of all relevant runtime configuration variables (e.g., RAG_DATA_DIR, RAG_DB_DIR)
        - Whether each variable is set and its current value

        5. CONFIGURATION PRIORITY
        - Order in which the system resolves:
            • Data directory location
            • Database storage location

        Use cases:
        - Debugging ingestion or retrieval issues
        - Verifying correct system initialization
        - Checking whether documents were successfully indexed
        - Diagnosing missing or misconfigured environment variables
        - Auditing current system state before reingestion or reset operations

        Returns:
            A structured dictionary containing:
            - system status
            - database configuration
            - data directory information
            - environment variable state
            - configuration resolution rules

        Error behavior:
        - If an internal failure occurs, returns a minimal diagnostic object with:
        - error message
        - partial system status (safe fallback state)

        Notes:
        - This tool is read-only and does not modify system state
        - Safe to call at any time for debugging or monitoring purposes
        """
        try:
            doc_count = self.collection.count() if self.collection else 0
            
            db_directory = self._get_database_directory()
            
            data_directory_status = {}
            try:
                data_directory = self._get_data_directory()
                data_directory_status = {
                    "path": str(data_directory),
                    "exists": data_directory.exists(),
                    "configured": True,
                    "source": "environment" if os.getenv('RAG_DATA_DIR') else "workspace"
                }
            except ValueError as e:
                data_directory_status = {
                    "path": None,
                    "exists": False,
                    "configured": False,
                    "error": str(e),
                    "source": "none"
                }
            
            env_vars = {
                "RAG_DATA_DIR": {
                    "set": bool(os.getenv('RAG_DATA_DIR')),
                    "value": os.getenv('RAG_DATA_DIR')
                },
                "RAG_DB_DIR": {
                    "set": bool(os.getenv('RAG_DB_DIR')),
                    "value": os.getenv('RAG_DB_DIR')
                }
            }
            
            db_config = {
                "type": "ChromaDB",
                "directory": str(db_directory),
                "exists": db_directory.exists(),
                "collection_name": "study_materials",
                "source": "environment" if os.getenv('RAG_DB_DIR') else "standard"
            }
            
            system_status = {
                "server_active": True,
                "database_initialized": self.chroma_client is not None,
                "collection_ready": self.collection is not None,
                "total_documents": doc_count,
                "auto_ingestion_enabled": data_directory_status["configured"]
            }
            
            return {
                "status": "active",
                "system": system_status,
                "database": db_config,
                "data_directory": data_directory_status,
                "environment_variables": env_vars,
                "configuration": {
                    "data_dir_priority": [
                        "RAG_DATA_DIR environment variable",
                        "Error if not found"
                    ],
                    "db_dir_priority": [
                        "RAG_DB_DIR environment variable",
                        "./chromadb in current working directory"
                    ]
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "system": {
                    "server_active": True,
                    "database_initialized": False,
                    "collection_ready": False,
                    "total_documents": 0,
                    "auto_ingestion_enabled": False
                }
            }
        
    # @mcp.tool()
    def reingest_data_directory(self) -> str:
        """
        Performs a full reset and re-ingestion of the entire document knowledge base from the configured data directory.

        This tool completely rebuilds the vector database from scratch.

        Process overview:
        1. Deletes the existing ChromaDB collection (removes all stored embeddings and metadata)
        2. Recreates a fresh empty collection
        3. Scans the configured data directory for all supported documents
        4. Processes documents (loading → chunking → embedding)
        5. Stores all embeddings into the vector database

        IMPORTANT (DESTRUCTIVE OPERATION):
        - This operation permanently deletes all previously stored embeddings
        - Any incremental updates, deletions, or partial ingestion states are lost
        - The database state is fully replaced after execution

        Use cases:
        - Major dataset updates where many files have changed
        - Fixing inconsistencies or corruption in the vector database
        - Updating ingestion pipeline logic (chunking, embedding model, metadata schema)
        - Rebuilding the system after configuration or model changes

        Guarantees:
        - Final database state reflects exactly the contents of the data directory at execution time
        - No duplicate or stale embeddings remain after completion

        Returns:
            A status message indicating:
            - success or failure of the reingestion process
            - total number of documents/chunks stored in the database

        Side effects:
        - Fully resets vector database state
        - Recomputes all embeddings from scratch
        - May be time-consuming for large datasets
        """
        try:
            is_configured, config_message = self._check_data_directory_configured()
            if not is_configured:
                return config_message

            if not self.chroma_client:
                return "Error: ChromaDB client is not initialized."

            collection_name = "study_materials"

            logger.info("Starting full re-ingestion process...")

            # 1. Delete existing collection
            try:
                self.chroma_client.delete_collection(name=collection_name)
                logger.info(f"Deleted existing collection '{collection_name}'.")
            except Exception:
                logger.warning("Collection did not exist or could not be deleted.")

            # 2. Recreate collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Collection for RAG document storage"}
            )

            logger.info("Created fresh collection.")

            # 3. Re-ingest all files
            self._auto_ingest_files()

            final_count = self.collection.count()

            return f"Re-ingestion complete. Database now contains {final_count} chunks."

        except Exception as e:
            logger.error(f"Re-ingestion failed: {e}")
            return f"Error during re-ingestion: {str(e)}"

    # @mcp.tool()
    def delete_files_from_db(self, file_names: List[str]) -> str:
        """
        Deletes all vector chunks associated with one or more files from the database.

        This tool performs a metadata-filtered deletion in ChromaDB using file names.

        Behavior:
        - Finds all chunks where metadata.file_name matches any of the provided names
        - Deletes all matching vector entries from the collection
        - Does NOT affect other files or the collection structure

        Args:
            file_names (List[str]):
                List of file names to remove from the vector database

        IMPORTANT:
        - This operation is irreversible for the selected files
        - Only affects embeddings stored in ChromaDB, not the actual files on disk
        - File names must match exactly (case-sensitive unless normalized earlier)

        Use cases:
        - Removing outdated documents
        - Fixing incorrect ingestion
        - Cleaning specific datasets without full reset

        Returns:
        - Confirmation message with deletion result or error information
        """
        try:
            if not self.collection:
                return "Error: Collection not initialized."

            if not file_names:
                return "No file names provided."

            # Multiple files → use $in operator
            result = self.collection.delete(where={"file_name": {"$in": file_names}})

            return f"Delete result {result}"

        except Exception as e:
            return f"Error deleting files: {str(e)}"
        
    #####################################################################################################################
    #################################################### MCP PROMPTS ####################################################
    #####################################################################################################################

    # @mcp.prompt
    def rag_analysis_prompt(self, topic: str) -> Message:
        """
        Generates a structured research prompt that instructs the AI to perform an in-depth, multi-step analysis over the RAG knowledge base for a given topic.

        This tool does not execute retrieval or analysis directly. Instead, it creates a guided instruction prompt that triggers a full RAG reasoning workflow in a downstream LLM.

        The generated prompt instructs the AI to:

        1. Query the vector database for documents related to the specified topic
        2. Extract and synthesize relevant information from retrieved chunks
        3. Produce a structured summary of key findings
        4. Identify patterns, insights, and relationships across documents
        5. Suggest directions for further investigation
        6. Cite sources based on retrieved document metadata

        Use cases:
        - Initiating deep research workflows over the document corpus
        - Generating structured analytical reports from stored knowledge
        - Exploring complex topics that require multi-document reasoning
        - Producing study notes or synthesized summaries from raw sources

        Args:
            topic (str):
                The subject or concept to analyze across the document collection

        Returns:
            Message:
                A formatted user message containing a structured instruction prompt
                that guides the LLM to perform retrieval + synthesis using the RAG system

        Behavior notes:
        - This tool does NOT perform retrieval itself
        - It depends on downstream tool use (e.g., query_documents)
        - Designed to bootstrap higher-level reasoning workflows over the RAG system
        """
        text = dedent(f"""
          Please analyze the documents in the RAG database related to '{topic}'. 

          First, query the database for relevant information about this topic, then provide:
          1. A comprehensive summary of the key points
          2. Any important insights or patterns you notice
          3. Potential areas for further investigation
          4. Sources and references from the retrieved documents

          Use the query_documents tool to search for information about '{topic}' and base your analysis on the retrieved content.
          """)
        
        return Message(
            role="user",
            content=text
        )


if __name__ == "__main__":
    # Create an instance of our service, which will register the tools.
    rag = RAG_Server()

    print(rag.list_ingested_files())

    #print(rag.query_documents("Sta je to aritmeticka sredina?"))
    
    rag.ingest_new_documents()

    # print(rag.get_rag_status())

    #print(rag.rag_analysis_prompt(topic = "SLAYYYY"))

    #print(rag.reingest_data_directory())

    #print(rag.delete_files_from_db(["new document test.txt", "Predavanje 3.pdf"]))

    # Run the MCP server
    logger.info("Starting RAG MCP Server...")

    # mcp.run("stdio")