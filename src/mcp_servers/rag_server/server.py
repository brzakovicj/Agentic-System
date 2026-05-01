import os
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