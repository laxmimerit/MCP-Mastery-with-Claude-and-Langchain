# server.py - Research Assistant MCP Server with ChromaDB
# uv add chromadb langchain-chroma langchain-ollama

from mcp.server.fastmcp import FastMCP
from pathlib import Path
from typing import List, Set, Dict, Any
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import os
import shutil
import hashlib
import json

# Initialize MCP Server
mcp = FastMCP("Research Assistant")

# Constants
current_dir = Path(__file__).parent.absolute()
CHROMA_DB_ROOT = current_dir / "research_chroma_dbs"
EMBED_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"

# Initialize embeddings once
embeddings = OllamaEmbeddings(
    model=EMBED_MODEL,
    base_url=OLLAMA_BASE_URL
)

def get_content_hash(content: str) -> str:
    """Generate a hash for content to check for duplicates."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def load_content_hashes(topic_path: Path) -> Set[str]:
    """Load existing content hashes from metadata file."""
    metadata_file = topic_path / "content_hashes.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                return set(json.load(f))
        except:
            return set()
    return set()

def save_content_hashes(topic_path: Path, hashes: Set[str]):
    """Save content hashes to metadata file."""
    metadata_file = topic_path / "content_hashes.json"
    with open(metadata_file, 'w') as f:
        json.dump(list(hashes), f)

def get_vectorstore(topic: str) -> Chroma:
    """Get or create a ChromaDB vectorstore for a topic."""
    topic_path = CHROMA_DB_ROOT / topic
    topic_path.mkdir(parents=True, exist_ok=True)
    
    return Chroma(
        persist_directory=str(topic_path),
        embedding_function=embeddings,
        collection_name=f"research_{topic}"
    )

# === Tools ===

@mcp.tool()
def save_research_data(content: List[str], topic: str = "default") -> str:
    """
    Save research content to vector database for future retrieval.
    Args:
        content: List of text content to save
        topic: Topic name for organizing the data (creates separate DB)
    """
    try:
        topic_path = CHROMA_DB_ROOT / topic
        topic_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing content hashes
        existing_hashes = load_content_hashes(topic_path)
        
        # Filter out duplicate content
        new_content = []
        new_hashes = set(existing_hashes)
        
        for text in content:
            content_hash = get_content_hash(text)
            if content_hash not in existing_hashes:
                new_content.append(text)
                new_hashes.add(content_hash)
        
        if not new_content:
            return f"No new content to save - all {len(content)} documents already exist in topic: {topic}"
        
        # Get vectorstore for this topic
        vectorstore = get_vectorstore(topic)
        
        # Create documents with metadata
        documents = []
        doc_ids = []
        for i, text in enumerate(new_content):
            content_hash = get_content_hash(text)
            doc = Document(
                page_content=text,
                metadata={
                    "topic": topic,
                    "content_hash": content_hash,
                    "doc_index": len(existing_hashes) + i
                }
            )
            documents.append(doc)
            doc_ids.append(f"{topic}_{content_hash}")
        
        # Add documents to vectorstore
        vectorstore.add_documents(documents=documents, ids=doc_ids)
        
        # Save updated content hashes
        save_content_hashes(topic_path, new_hashes)
        
        return f"Successfully saved {len(new_content)} new documents to topic: {topic} (skipped {len(content) - len(new_content)} duplicates)"
        
    except Exception as e:
        return f"Error saving research data: {str(e)}"

@mcp.tool()
def search_research_data(query: str, topic: str = "default", max_results: int = 5) -> str:
    """
    Search through saved research data using semantic similarity.
    Args:
        query: Search query
        topic: Topic database to search in
        max_results: Maximum number of results to return
    """
    try:
        topic_path = CHROMA_DB_ROOT / topic
        
        if not topic_path.exists():
            return f"No research data found for topic: {topic}"
        
        # Get vectorstore for this topic
        vectorstore = get_vectorstore(topic)
        
        # Check if collection has any documents
        try:
            collection = vectorstore._collection
            count = collection.count()
            if count == 0:
                return f"No documents found in topic: {topic}"
        except:
            return f"No research data found for topic: {topic}"
        
        # Search for similar documents
        results = vectorstore.similarity_search_with_score(query, k=max_results)
        
        if not results:
            return f"No relevant results found for query: '{query}' in topic: {topic}"
        
        # Format results
        formatted_results = []
        for i, (doc, score) in enumerate(results):
            similarity = 1 - score  # Convert distance to similarity
            result_text = f"Result {i+1} (Similarity: {similarity:.3f}):\n{doc.page_content}\n"
            formatted_results.append(result_text)
        
        return "\n" + "="*50 + "\n".join(formatted_results) + "="*50
        
    except Exception as e:
        return f"Error searching research data: {str(e)}"

@mcp.tool()
def list_research_topics() -> str:
    """
    List all available research topics (vector databases).
    """
    try:
        if not CHROMA_DB_ROOT.exists():
            return "No research topics found"
        
        topics = []
        for path in CHROMA_DB_ROOT.iterdir():
            if path.is_dir():
                # Try to get document count from ChromaDB
                try:
                    vectorstore = get_vectorstore(path.name)
                    collection = vectorstore._collection
                    doc_count = collection.count()
                    topics.append(f"Topic: {path.name} ({doc_count} documents)")
                except Exception as e:
                    # Fallback to hash count if ChromaDB fails
                    try:
                        hashes = load_content_hashes(path)
                        doc_count = len(hashes)
                        topics.append(f"Topic: {path.name} ({doc_count} documents)")
                    except:
                        topics.append(f"Topic: {path.name}")
        
        if not topics:
            return "No research topics found"
        
        return "\n".join(topics)
        
    except Exception as e:
        return f"Error listing topics: {str(e)}"

@mcp.tool()
def delete_research_topic(topic: str) -> str:
    """
    Delete a research topic and all its data.
    Args:
        topic: Topic name to delete
    """
    try:
        topic_path = CHROMA_DB_ROOT / topic
        
        if not topic_path.exists():
            return f"Topic '{topic}' does not exist"
        
        # Try to delete the ChromaDB collection first
        try:
            vectorstore = get_vectorstore(topic)
            vectorstore.delete_collection()
        except:
            pass  # Continue even if collection deletion fails
        
        # Remove the entire directory
        shutil.rmtree(topic_path)
        
        return f"Successfully deleted topic: {topic}"
        
    except Exception as e:
        return f"Error deleting topic: {str(e)}"

@mcp.tool()
def get_topic_info(topic: str) -> str:
    """
    Get detailed information about a research topic.
    Args:
        topic: Topic name to get info for
    """
    try:
        topic_path = CHROMA_DB_ROOT / topic
        
        if not topic_path.exists():
            return f"Topic '{topic}' does not exist"
        
        # Get vectorstore info
        vectorstore = get_vectorstore(topic)
        collection = vectorstore._collection
        doc_count = collection.count()
        
        # Get content hashes info
        hashes = load_content_hashes(topic_path)
        hash_count = len(hashes)
        
        info = f"""Topic Information: {topic}
                - ChromaDB Collection: research_{topic}
                - Document Count: {doc_count}
                - Hash Records: {hash_count}
                - Database Path: {topic_path}
                - Embedding Model: {EMBED_MODEL}
                - Ollama URL: {OLLAMA_BASE_URL}"""
        
        return info
        
    except Exception as e:
        return f"Error getting topic info: {str(e)}"

if __name__ == "__main__":

    mcp.run(transport="stdio")