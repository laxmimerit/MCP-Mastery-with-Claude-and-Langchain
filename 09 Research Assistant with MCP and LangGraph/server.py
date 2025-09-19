# server.py -  Research Assistant MCP Server
# uv add faiss-cpu langchain_community

from mcp.server.fastmcp import FastMCP
from pathlib import Path
from typing import List, Set
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
import os
import shutil
import hashlib
import json

# Initialize MCP Server
mcp = FastMCP("Research Assistant")

# Constants - Use Path objects consistently
current_dir = Path(__file__).parent.absolute()
VECTOR_DB_ROOT = current_dir / "research_vector_dbs"

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
        target_path = VECTOR_DB_ROOT / topic
        target_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing content hashes
        existing_hashes = load_content_hashes(target_path)
        
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
        
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        
        documents = [Document(page_content=text) for text in new_content]
        index_file = target_path / "index.faiss"
        
        if index_file.exists():
            # Load existing database and add new documents
            vectorstore = FAISS.load_local(
                str(target_path), 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            vectorstore.add_documents(documents)
            vectorstore.save_local(str(target_path))
        else:
            # Create new database
            vectorstore = FAISS.from_documents(documents, embeddings)
            vectorstore.save_local(str(target_path))
        
        # Save updated content hashes
        save_content_hashes(target_path, new_hashes)
        
        return f"Successfully saved {len(new_content)} new documents to topic: {topic} (skipped {len(content) - len(new_content)} duplicates)"
        
    except Exception as e:
        return f"Error saving research data: {str(e)}"

@mcp.tool()
def search_research_data(query: str, topic: str = "default", max_results: int = 5) -> List[str]:
    """
    Search through saved research data using semantic similarity.
    Args:
        query: Search query
        topic: Topic database to search in
        max_results: Maximum number of results to return
    """
    try:
        target_path = VECTOR_DB_ROOT / topic
        index_file = target_path / "index.faiss"
        
        if not index_file.exists():
            return [f"No research data found for topic: {topic}"]
        
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        
        vectorstore = FAISS.load_local(
            str(target_path), 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        results = vectorstore.similarity_search(query, k=max_results)
        
        # Remove any potential duplicates from search results
        unique_results = []
        seen_content = set()
        
        for result in results:
            content_hash = get_content_hash(result.page_content)
            if content_hash not in seen_content:
                unique_results.append(result.page_content)
                seen_content.add(content_hash)
        
        return unique_results
        
    except Exception as e:
        return [f"Error searching research data: {str(e)}"]

@mcp.tool()
def list_research_topics() -> List[str]:
    """
    List all available research topics (vector databases).
    """
    try:
        if not VECTOR_DB_ROOT.exists():
            return ["No research topics found"]
        
        topics = []
        for path in VECTOR_DB_ROOT.iterdir():
            if path.is_dir() and (path / "index.faiss").exists():
                # Count documents in the topic
                try:
                    hashes = load_content_hashes(path)
                    doc_count = len(hashes)
                    topics.append(f"Topic: {path.name} ({doc_count} documents)")
                except:
                    topics.append(f"Topic: {path.name}")
        
        return topics if topics else ["No research topics found"]
        
    except Exception as e:
        return [f"Error listing topics: {str(e)}"]

@mcp.tool()
def delete_research_topic(topic: str) -> str:
    """
    Delete a research topic and all its data.
    Args:
        topic: Topic name to delete
    """
    try:
        target_path = VECTOR_DB_ROOT / topic
        
        if not target_path.exists():
            return f"Topic '{topic}' does not exist"
        
        # Remove the entire directory
        shutil.rmtree(target_path)
        
        return f"Successfully deleted topic: {topic}"
        
    except Exception as e:
        return f"Error deleting topic: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")