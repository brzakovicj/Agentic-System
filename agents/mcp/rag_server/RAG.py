import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from textwrap import dedent
from dotenv import load_dotenv
from datetime import datetime
import uuid

# FastMCP imports
from fastmcp import FastMCP
from fastmcp.prompts import Message#, TextContent

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

from sentence_transformers import SentenceTransformer

# Debugging formatter
from pprint import pprint

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize FastMCP server with RAG capabilities
mcp = FastMCP("RAG Server")

class RAG_Server:
    def __init__(self): # ok
        """Initializes the RAG server, setting up the database connection."""
        self.chroma_client = None
        self.collection = None
        self._initialize_chromadb()


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

    # def safe_reset_chromadb(self, chroma_path: str = "./chroma_db") -> bool:
    #     """Safely reset ChromaDB by deleting the entire directory.
        
    #     This is the most reliable way to reset ChromaDB and avoid corruption issues.
        
    #     Args:
    #         chroma_path: Path to the ChromaDB directory
            
    #     Returns:
    #         True if reset was successful, False otherwise
    #     """
    #     try:
    #         if os.path.exists(chroma_path):
    #             # Add a small delay to ensure any file handles are released
    #             # time.sleep(0.1)
    #             shutil.rmtree(chroma_path)
    #             print(f"Successfully deleted ChromaDB directory: {chroma_path}")
    #         else:
    #             print(f"ChromaDB directory does not exist: {chroma_path}")
            
    #         return True
            
    #     except Exception as e:
    #         print(f"Error deleting ChromaDB directory: {e}")
    #         return False
    
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

            # DEBUG
            for page in pages:
                logger.info(f"File Name: {page['file_name']}")
                logger.info(f"Page Number: {page['page_number']}")
                logger.info(f"Text: {page['text'][:200]}...")  # Truncate text to 100 characters
                logger.info("-" * 80)  # Separator for readability
                logger.info(page)
                logger.info("-" * 80)  # Separator for readability

            ################ CHUNKING TEST ################

            # Create a ChunkingStrategy instance with the desired settings.
            chunker = ChunkingStrategy(chunk_size=300, chunk_overlap=50)

            # Retrieve page information from the pages list (using the 11th page as an example).
            page = pages[11]
            file_name = page["file_name"]
            file_type = page["file_type"]
            page_number = page["page_number"]
            text_content = page["text"]

            # Debug: Print the text content of the selected page.
            logger.info(f"Debug - Page Text: {text_content}")

            ################ CHUNKING ################
            # Chunk the document text using the chunk_document method.
            chunks = chunker.chunk_document(text_content)

            # Display file and page details along with the number of chunks generated.
            logger.info(f"File: {file_name} (Page Number: {page_number})")
            logger.info(f"Number of chunks: {len(chunks)}")
            logger.info("-" * 80)

            # Iterate through each chunk and print its contents.
            for idx, chunk in enumerate(chunks):
                logger.info(f"Chunk {idx + 1}:")
                logger.info(chunk)
                logger.info("-" * 80)

            all_chunks = []

            for page in pages:
                chunks = chunker.chunk_document(page["text"])
                
                for chunk in chunks:
                    all_chunks.append({
                        "id": str(uuid.uuid4()),  # Generate a unique ID for each chunk
                        "text": chunk,
                        "metadata": {
                            "source": page["source"],
                            "doc_id": page["doc_id"],
                            "file_name": page["file_name"],
                            "file_type": page["file_type"],
                            "page_number": page["page_number"],
                            "chunk_method": chunker.method
                        }
                    })

            logger.info(f"Created {len(all_chunks)} text chunks.")

            ################ EMBEDDING ################
            # Initialize a SentenceTransformer model (choose one appropriate for your use case)
            embed_model = SentenceTransformer('all-MiniLM-L6-v2')

            # Gather all chunk texts for embedding generation
            chunk_texts = [chunk["text"] for chunk in all_chunks]

            # Compute embeddings (using batch encoding for efficiency)
            embeddings = embed_model.encode(chunk_texts, convert_to_numpy=True, show_progress_bar=True)

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

            ################ QUERY ################ ONLY FOR DEBUGGING
            # Define your text query
            query_text = "Šta je testiranje softvera?"

            # Generate the embedding for the query text
            query_embedding = embed_model.encode(query_text).tolist()

            # Query the collection for the top 3 most similar documents.
            # The 'include' parameter lets you retrieve documents, metadatas, and distances.
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=3,
                include=["documents", "metadatas", "distances"]
            )

            # Print the retrieval results
            logger.info("Query Results:")
            pprint(results)
            
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
        
        logger.info(f"Found {len(all_files)} files in data directory. Starting auto-ingestion...")
            
        for file_path in all_files:
            if file_path.suffix.lower() in ('.pdf', '.docx', '.pptx', '.html', '.txt'):
                logger.info(f"processing file: {file_path}")
                elements = partition(str(file_path))
                doc_id = str(uuid.uuid4())  # Single doc_id per file

                # Group text by page number
                page_texts = {}
                for el in elements:
                    # Get the page number; default to 1 if not provided
                    page_number = getattr(el.metadata, "page_number", None)
                    page_number = int(page_number) if page_number is not None else 1
                    
                    # Append element text to the corresponding page's list
                    page_texts.setdefault(page_number, []).append(str(el))
                
                # Create a document entry for each page
                for page_number, texts in page_texts.items():
                    combined_text = " ".join(texts).strip()
                    pages.append({
                        "text": combined_text,
                        "source": str(file_path),
                        "file_name": file_path.name,
                        "file_type": file_path.suffix,
                        "page_number": page_number,
                        "doc_id": doc_id
                    })
        return pages
    
    ########### ONLY FOR DEBUGGING
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

    @mcp.tool()
    def query_documents(self, query: str, n_results: int = 5, include_metadata: bool = True) -> str:
        """
        Query the local document knowledge base (vector database) to retrieve relevant information based on your search query.
        
        This tool allows you to search through all documents that have been ingested into the system's vector database.
        It uses semantic search to find the most relevant text passages that match your query, going beyond simple
        keyword matching to understand the meaning and context of your search.
        
        The tool is particularly useful for:
        - Finding specific information across multiple documents
        - Discovering connections between different pieces of content
        - Getting quick access to relevant passages without reading entire documents
        - Verifying facts or finding supporting evidence in your document collection
        
        Each result includes:
        - The relevant text passage
        - Source document information
        - Similarity score showing how well it matches your query
        - Additional metadata about the document chunk
        
        You can control the number of results returned and whether to include detailed metadata
        in the response.
        
        Args:
            query: The search query
            n_results: Number of results to return (default: 5, min: 5, max: 20)
            include_metadata: Whether to include metadata in results (default: True)
        
        Returns:
            Formatted string with relevant documents and their metadata
        """
        try:
            is_configured, config_message = self._check_data_directory_configured()
            if not is_configured:
                return config_message
            
            if not query.strip():
                return "Error: Query cannot be empty."
            
            if n_results <= 0:
                n_results = 5
            elif n_results > 20:
                n_results = 20
            
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            if not results["documents"] or not results["documents"][0]:
                return "No relevant documents found for your query."
            
            formatted_results = []
            documents = results["documents"][0]
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(documents)
            distances = results["distances"][0] if results["distances"] else [0] * len(documents)
            
            for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
                result_text = f"\n--- Result {i+1} ---\n"
                result_text += f"Content: {doc}\n"
                
                if include_metadata and metadata:
                    result_text += f"Source: {metadata.get('source_file', 'Unknown')}\n"
                    result_text += f"Chunk: {metadata.get('chunk_index', 'Unknown')} of {metadata.get('total_chunks', 'Unknown')}\n"
                    result_text += f"Similarity Score: {1 - distance:.3f}\n"
                
                formatted_results.append(result_text)
            
            response = f"Found {len(documents)} relevant documents for query: '{query}'\n"
            response += "\n".join(formatted_results)
            
            logger.info(f"Query '{query}' returned {len(documents)} results")
            return response
            
        except Exception as e:
            error_msg = f"Error querying documents: {str(e)}"
            logger.error(error_msg)
            return error_msg

    @mcp.tool()
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

    @mcp.tool()
    def reingest_data_directory(self) -> str:
        """
        Performs a complete re-ingestion of all files from the configured data directory into the vector database.

        This tool first clears the existing database of all documents and then processes all files in the data directory from scratch.
        It is particularly useful when:
        - New files have been added to the data directory and need to be included in the knowledge base.
        - Existing files have been updated and their contents need to be re-indexed.
        - You want to ensure the vector database is in a clean, consistent state with the latest file versions.

        The process is atomic; it completely replaces the old database with a new one.

        Returns:
            A status message indicating the success or failure of the re-ingestion process, including the final document count.
        """
        try:
            is_configured, config_message = self._check_data_directory_configured()
            if not is_configured:
                return config_message
            
            if not self.collection:
                return "Error: Database is not initialized."
            
            logger.info("Clearing existing database before reingestion...")
            self.chroma_client.delete_collection(name="rag_documents")
            self.collection = self.chroma_client.create_collection(
                name="rag_documents",
                metadata={"description": "Collection for RAG document storage"}
            )
            
            logger.info("Starting reingestion of data directory...")
            self._auto_ingest_files()
            
            final_count = self.collection.count()
            
            success_msg = f"Successfully reingested data directory. Database now contains {final_count} documents."
            logger.info(success_msg)
            return success_msg
            
        except Exception as e:
            error_msg = f"Error during reingestion: {str(e)}"
            logger.error(error_msg)
            return error_msg

    @mcp.tool()
    def get_rag_status(self) -> Dict[str, Any]:
        """
        Provides a comprehensive overview of the RAG system's status and configuration.

        This tool is essential for diagnosing issues and understanding the current state of the system.
        It returns a detailed report including:
        - System Status: Whether the server is active, the database is initialized, and the total number of documents.
        - Database Configuration: The type of database, its storage directory, and collection details.
        - Data Directory: The path to the data directory, whether it exists, and how it's configured.
        - Environment Variables: The status of relevant environment variables (e.g., API keys, custom paths).
        - Configuration Priority: The order in which the system looks for configuration settings.

        This tool helps you to:
        - Verify that the system is running correctly.
        - Debug configuration problems related to data and database directories.
        - Check which environment variables are being used.
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
                    "source": "environment" if os.getenv('LLAMA_RAG_DATA_DIR') else "workspace"
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
                "LLAMA_RAG_DATA_DIR": {
                    "set": bool(os.getenv('LLAMA_RAG_DATA_DIR')),
                    "value": os.getenv('LLAMA_RAG_DATA_DIR')
                },
                "LLAMA_RAG_DB_DIR": {
                    "set": bool(os.getenv('LLAMA_RAG_DB_DIR')),
                    "value": os.getenv('LLAMA_RAG_DB_DIR')
                },
                "LLAMA_CLOUD_API_KEY": {
                    "set": bool(os.getenv('LLAMA_CLOUD_API_KEY')),
                    "value": "[REDACTED]" if os.getenv('LLAMA_CLOUD_API_KEY') else None
                }
            }
            
            db_config = {
                "type": "ChromaDB",
                "directory": str(db_directory),
                "exists": db_directory.exists(),
                "collection_name": "rag_documents",
                "source": "environment" if os.getenv('LLAMA_RAG_DB_DIR') else "standard"
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
                        "LLAMA_RAG_DATA_DIR environment variable",
                        "./data in current working directory",
                        "Error if neither found"
                    ],
                    "db_dir_priority": [
                        "LLAMA_RAG_DB_DIR environment variable",
                        "~/.local/share/rag-server (XDG standard)",
                        "./chroma in current working directory"
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

    @mcp.prompt
    def rag_analysis_prompt(self, topic: str) -> Message:
        """
        Generates a sophisticated prompt to guide the AI in performing an in-depth analysis of documents related to a specific topic.

        This tool is designed to kickstart a detailed investigation into a subject by leveraging the RAG system's capabilities.
        It constructs a prompt that instructs the AI to:
        1. Query the knowledge base for information on the given topic.
        2. Synthesize the findings into a comprehensive summary.
        3. Identify key insights, patterns, and connections within the documents.
        4. Suggest areas for further exploration.
        5. Cite the sources from the retrieved documents to support its analysis.

        Use this tool to:
        - Initiate a research task on a particular topic.
        - Generate a structured analysis of your documents.
        - Uncover deeper insights that may not be apparent from a simple query.
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
            content=TextContent(type="text", text=text)
        )
    
class ChunkingStrategy:
    def __init__(self, method: str = 'recursive', encoding_name: str = "cl100k_base", chunk_size: int = 300, chunk_overlap: int = 50):
        """
        Initialize a chunking strategy.

        Args:
            method (str): The chunking method, e.g. 'fixed' or 'recursive'.
            encoding_name (str): The name of the encoding to use.
            chunk_size (int): The size of each chunk.
            chunk_overlap (int): The overlap between chunks.
        """
        self.method = method
        self.encoding_name = encoding_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, text: str) -> List[str]:
        """
        Chunk a document's text using the selected strategy.

        Args:
            text (str): The document's text to chunk.

        Returns:
            List[str]: A list of text chunks.
        """
        if self.method == 'fixed':
            splitter = CharacterTextSplitter.from_tiktoken_encoder(
                encoding_name=self.encoding_name, 
                chunk_size=self.chunk_size, 
                chunk_overlap=self.chunk_overlap
            )
            return splitter.split_text(text)
        elif self.method == 'recursive':
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name=self.encoding_name, 
                chunk_size=self.chunk_size, 
                chunk_overlap=self.chunk_overlap
            )
            return splitter.split_text(text)
        else:
            raise ValueError("Unknown chunking method: choose 'fixed' or 'recursive'.")


if __name__ == "__main__":
    # Create an instance of our service, which will register the tools.
    RAG_Server()
    
    # Run the MCP server
    logger.info("Starting RAG MCP Server...")

    # mcp.run("stdio")