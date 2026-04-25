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

# Prompt manager
from agents.prompts.prompt_manager import PromptManager

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

        self.prompt_manager = PromptManager()

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
            error_msg = f"Failed to initialize ChromaDB: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
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

    def _auto_ingest_files(self) -> str:
        """Automatically ingest all files from the data directory"""
        try:
            ################ LOAD DATA ################

            data_path = self._get_data_directory()
            os.makedirs(data_path, exist_ok=True)

            pages = self._process_documents_pages(data_path)

            if not pages or len(pages) == 0:
                msg = f"No files found in data directory. Skipping auto-ingestion."
                logger.info(msg)
                return msg

            return self._ingest_files(pages = pages)
            
        except ValueError as e:
            error_msg = f"Data directory not configured. Skipping auto-ingestion: {str(e)}"
            logger.warning(error_msg)
            raise
        except Exception as e:
            error_msg = f"Failed during auto-ingestion: {str(e)}"
            logger.error(error_msg)
            raise

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
    
    def _ingest_files(self, pages: List[Dict]) -> str:
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
        
        msg = f"Auto-ingestion completed. Collection now has {final_count} documents."
        logger.info(msg)
        return msg

    def _compute_file_hash(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            return hashlib.file_digest(f, "md5").hexdigest()

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
    
    async def _query_documents(self, query: str, n_results: int = 5, include_metadata: bool = True, top_n: int = 3) -> str: # Ogranicenje za top_n?        
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
    async def _list_ingested_files(self) -> str:
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
        
    async def _ingest_new_documents(self) -> str:
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

            response = self._ingest_files(pages = new_pages)
            return response
            
        except ValueError as e:
            error_msg = f"Skipping auto-ingestion: {str(e)}"
            logger.warning(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Failed during auto-ingestion: {str(e)}"
            logger.error(error_msg)
            return error_msg

    # Ok ne treba nam bas, ali neka
    async def _get_rag_status(self) -> Dict[str, Any]:
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
        
    async def _reingest_data_directory(self) -> str:
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
                error_msg = "Collection did not exist or could not be deleted."
                logger.warning(error_msg)

            # 2. Recreate collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Collection for RAG document storage"}
            )

            logger.info("Created fresh collection.")

            # 3. Re-ingest all files
            response = self._auto_ingest_files()

            final_count = self.collection.count()

            return f"Re-ingestion complete. Database now contains {final_count} chunks."

        except Exception as e:
            error_msg = f"Error during re-ingestion: {str(e)}"
            logger.error(error_msg)
            return error_msg

    async def _delete_files_from_db(self, file_names: List[str]) -> str:
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

    async def _rag_analysis_prompt(self, topic: str) -> Message:
        text = self.prompt_manager.get("rag_analysis_prompt", topic=topic)
        return Message(role="user", content=text)
        
rag = RAG_Server()

@mcp.tool()
async def query_documents(query: str) -> str:
    """
    Search the knowledge base.

    Input:
    - query: string

    IMPORTANT:
    - Always pass a plain string
    - Do NOT wrap it in an object
    """
    n_results = 5
    include_metadata = True
    top_n = 3
    return await rag._query_documents(query=query, n_results=n_results, include_metadata=include_metadata, top_n=top_n)

@mcp.tool()
async def list_ingested_files(dummy: str = "") -> str:
    """
    List all files currently stored in the vector database.

    This tool returns information about ingested documents, such as:
    - file name or path
    - number of chunks
    - basic metadata

    Argument:
        dummy (str): This argument is required by FastMCP but is not used. 
                     Do not provide any value for this argument.

    Returns:
        str: A summary of all ingested files.

    Usage:
        Call this tool when the user wants to see which documents
        are available in the knowledge base.
    """
    return await rag._list_ingested_files()

@mcp.tool()
async def ingest_new_documents(dummy: str = "") -> str:
    """
    Ingest new documents from the data directory into the vector database.

    This tool scans the configured data directory and adds ONLY documents
    that are not already stored in the database (based on file hash).

    It does NOT modify or delete existing documents.

    Argument:
        dummy (str): This argument is required by FastMCP but is not used. 
                     Do not provide any value for this argument.

    Returns:
        str: A message indicating whether new documents were ingested
        or if no new files were found.

    Usage:
        Call this tool when the user wants to add new documents to the system.
    """
    return await rag._ingest_new_documents()

@mcp.tool()
async def get_rag_status(dummy: str = "") -> Dict[str, Any]:
    """
    Get the current status and configuration of the RAG system.

    This tool returns diagnostic information about:
    - system state
    - vector database (ChromaDB)
    - document storage
    - data directory
    - environment variables

    It is useful for debugging, verification, and monitoring.

    Argument:
        dummy (str): This argument is required by FastMCP but is not used. 
                     Do not provide any value for this argument.

    Returns:
        Dict[str, Any]: A structured object containing the current RAG system status.

    Usage:
        Call this tool when the user asks about system status, configuration,
        indexing state, or debugging information.
    """
    return await rag._get_rag_status()

@mcp.tool()
async def reingest_data_directory(dummy: str = "") -> str:
    """
    Reset and rebuild the entire vector database from the data directory.

    This tool deletes all existing embeddings and reprocesses all documents
    from the configured data directory.

    WARNING:
        This is a destructive operation. All existing data will be permanently deleted.

    Argument:
        dummy (str): This argument is required by FastMCP but is not used. 
                     Do not provide any value for this argument.

    Returns:
        str: A message indicating success or failure of the reingestion process.

    Usage:
        Call this tool when the user wants to fully rebuild the knowledge base
        or fix inconsistent/corrupted data.
    """
    return await rag._reingest_data_directory()

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
    return await rag._delete_files_from_db(file_names=file_names)

@mcp.prompt()
async def rag_analysis_prompt(topic: str) -> Message:
    """
    Generate a structured research prompt for analyzing a topic using the RAG system.

    This tool creates a prompt that guides the AI to:
    - retrieve relevant documents
    - extract key information
    - summarize findings
    - identify insights and relationships
    - suggest further exploration

    It does NOT perform retrieval or analysis directly.

    Arguments:
        topic (str): The topic to analyze.

    Returns:
        Message: A formatted prompt for downstream LLM analysis.

    Usage:
        Call this tool when the user wants to perform deep analysis
        or research on a specific topic.

        Example:
        {
            "topic": "neural networks"
        }
    """
    return await rag._rag_analysis_prompt(topic=topic)

if __name__ == "__main__":
    try:
        logger.info("Starting RAG MCP Server...")
        mcp.run(transport="stdio")
    except Exception as e:
        logger.exception("SERVER CRASHED")
        raise