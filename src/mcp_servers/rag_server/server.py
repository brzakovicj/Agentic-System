import os
import json
import logging
import numpy as np
from pathlib import Path
from textwrap import dedent
from dotenv import load_dotenv

# FastMCP imports
from fastmcp import FastMCP
from fastmcp.prompts import Message

# Prompt manager
from src.prompts.prompt_manager import PromptManager

# ChromaDB imports
import chromadb
from chromadb.config import Settings

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

            self.chroma_client = chromadb.HttpClient(
                host=os.getenv("CHROMA_HOST", "localhost"),
                port=int(os.getenv("CHROMA_PORT", 8000)),
            )
            
            # Create fresh collection for RAG documents
            self.collection = self.chroma_client.get_or_create_collection(
                name="study_materials",
                metadata={"description": "Collection for RAG document storage"}
            )
            
            logger.info(f"ChromaDB initialized successfully. Vector database has {self.collection.count()} documents.")
            
        except Exception as e:
            error_msg = f"Failed to initialize ChromaDB: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def _get_data_directory(self): # ok
        """Get the data directory path with flexible resolution strategy."""
        # 1. Check environment variable first
        env_data_dir = os.environ.get('DATA_DIR')
        if env_data_dir:
            data_path = Path(env_data_dir).expanduser().resolve()
            logger.info(f"Using data directory from DATA_DIR: {data_path}")
            return data_path
        
        # 2. No environment variable and no existing data directory - raise error
        error_msg = (
            "No data directory found. Please either:\n"
            "1. Set the DATA_DIR environment variable to specify a data directory, or\n"
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

    ###################################################################################################################
    #################################################### MCP TOOLS ####################################################
    ###################################################################################################################
    
    async def _query_documents(self, query: str, n_results: int = 5, include_metadata: bool = True, top_n: int = 3) -> dict: # Ogranicenje za top_n?        
        try:
            ################ RETRIEVE ################

            is_configured, config_message = self._check_data_directory_configured()
            if not is_configured:
                return config_message
            
            if not query.strip():
                return {
                    "query": query,
                    "error": "Query cannot be empty.",
                    "results": []
                }
            
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
                return {
                    "query": query,
                    "error": "No relevant documents found for the query.",
                    "results": []
                }
            
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

            # formatted_results = []
            
            # for i, (doc, metadata, distance, score) in enumerate(zip(documents, metadatas, distances, scores)):
            #     result_text = f"\n--- Result {i+1} ---\n"
            #     result_text += f"Content: {doc}\n"
                
            #     if include_metadata and metadata:
            #         result_text += f"File Name: {metadata.get('file_name', 'Unknown')}\n"
            #         result_text += f"Source: {metadata.get('file_path', 'Unknown')}\n"
            #         result_text += f"Page Number: {metadata.get('page_number', 'Unknown')}\n"
            #         #result_text += f"Chunk: {metadata.get('chunk_index', 'Unknown')} of {metadata.get('total_chunks', 'Unknown')}\n"
            #         result_text += f"Similarity Score: {1 - distance:.3f}\n" # Sta ce nam ovo?
            #         result_text += f"Relevance Score: {score:.3f}\n" # Sta ce nam ovo?
                
            #     formatted_results.append(result_text)
            
            # response = f"Found {len(documents)} relevant documents for query: '{query}'\n"
            # response += "\n".join(formatted_results)
            
            logger.info(f"Query '{query}' returned {len(documents)} results")

            return {
                "query": query,
                "error": None,
                "results": [
                    {
                        "content": doc,
                        "file_name": metadata.get("file_name", 'Unknown'),
                        "source": metadata.get("file_path", 'Unknown'),
                        "page_number": metadata.get("page_number", 'Unknown'),
                        "similarity_score": f"{1 - distance:.3f}",
                        "relevance_score": f"{score:.3f}",
                    }
                    for doc, metadata, distance, score in zip(documents, metadatas, distances, scores)
                ]
            }
            
        except Exception as e:
            error_msg = f"Error querying documents: {str(e)}"
            logger.error(error_msg)
            return {
                "query": query,
                "error": error_msg,
                "results": []
            }

    #####################################################################################################################
    #################################################### MCP PROMPTS ####################################################
    #####################################################################################################################

    async def _rag_analysis_prompt(self, topic: str) -> Message:
        text = self.prompt_manager.get("rag_analysis_prompt", topic=topic)
        return Message(role="user", content=text)
        
rag = RAG_Server()

@mcp.tool()
async def query_documents(query: str, n_results: int = 5, top_n: int = 3) -> dict:
    """
    Search the INTERNAL knowledge base using semantic search with cross-encoder reranking.

    This tool provides access to the user's PRIVATE, INTERNAL document repository.
    It should be preferred over web search whenever the query relates to internal processes,
    proprietary data, domain-specific documentation, or any topic that may be covered by
    the internal knowledge base.

    Retrieves the most relevant document chunks for a given natural language query.
    Internally fetches up to n_results candidates via vector similarity, then reranks them
    using a cross-encoder model and returns the top_n most relevant results.

    Inputs:
        query (str): A plain natural language string describing what you are looking for.
                     Must be non-empty. Do NOT wrap it in an object or JSON.
        n_results (int): Number of candidates to retrieve before reranking. Default 5, max 20.
                         Increase this when the topic is broad or you expect many relevant chunks.
        top_n (int): Number of reranked results to return. Default 3, cannot exceed n_results.
                     Increase this when you need broader coverage of a topic.

    Returns a dict with:
        - query (str): The original query string.
        - error (str | None): Error message if something went wrong, otherwise null.
        - results (list): Up to top_n reranked document chunks, each containing:
            - content (str): The text of the document chunk.
            - file_name (str): Name of the source file.
            - source (str): Full file path of the source document.
            - page_number (int | str): Page number within the source file.
            - similarity_score (str): Vector similarity score (0–1), from initial retrieval.
            - relevance_score (str): Cross-encoder reranking score, more reliable than similarity.

    Usage notes:
        - Prefer relevance_score over similarity_score when judging result quality.
        - Always check the error field before using results.
        - If results is empty and error is set, the query returned no matches or failed.
        - Use specific, descriptive queries — vague queries produce lower quality results.
        - If the first query yields poor results, try rephrasing before concluding
          the information is not in the knowledge base.
        - Consider increasing n_results and top_n for broad topics requiring wider coverage.
    """
    #n_results = 5
    include_metadata = True
    #top_n = 3
    return await rag._query_documents(query=query, n_results=n_results, include_metadata=include_metadata, top_n=top_n)

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