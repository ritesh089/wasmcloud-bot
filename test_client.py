#!/usr/bin/env python3
"""Simple test client for wasmCloud RAG bot."""

import requests
import json
import time
from typing import Dict, Any


class WasmCloudRAGClient:
    """Client for interacting with wasmCloud RAG bot API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the server is healthy."""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def ask_question(self, question: str, include_sources: bool = True) -> Dict[str, Any]:
        """Ask a question to the RAG bot."""
        payload = {
            "question": question,
            "include_sources": include_sources
        }
        
        response = requests.post(
            f"{self.base_url}/query",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": f"HTTP {response.status_code}",
                "detail": response.text
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        response = requests.get(f"{self.base_url}/stats")
        return response.json()
    
    def list_documents(self, limit: int = 10, offset: int = 0) -> Dict[str, Any]:
        """List documents in the database."""
        response = requests.get(
            f"{self.base_url}/documents",
            params={"limit": limit, "offset": offset}
        )
        return response.json()
    
    def trigger_ingestion(self) -> Dict[str, Any]:
        """Manually trigger documentation ingestion."""
        response = requests.post(f"{self.base_url}/ingest")
        return response.json()


def main():
    """Demo the RAG bot client."""
    print("wasmCloud RAG Bot Test Client")
    print("=" * 40)
    
    client = WasmCloudRAGClient()
    
    # Health check
    print("1. Health Check:")
    health = client.health_check()
    print(json.dumps(health, indent=2))
    
    if health.get('status') != 'healthy':
        print("❌ Server is not healthy. Please check the server and database.")
        return
    
    print("✅ Server is healthy!")
    print()
    
    # Get stats
    print("2. Database Statistics:")
    stats = client.get_stats()
    print(json.dumps(stats, indent=2))
    print()
    
    # Sample questions
    questions = [
        "What is wasmCloud?",
        "How do I install wasmCloud?",
        "What are wasmCloud capabilities?",
        "How does wasmCloud handle scaling?",
        "What is the difference between components and providers in wasmCloud?"
    ]
    
    print("3. Sample Questions:")
    for i, question in enumerate(questions, 1):
        print(f"\n--- Question {i}: {question} ---")
        
        start_time = time.time()
        result = client.ask_question(question)
        end_time = time.time()
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            print(f"Details: {result['detail']}")
        else:
            print(f"⏱️  Response time: {result['response_time']:.2f}s")
            print(f"📊 Chunks used: {result['chunks_used']}")
            print(f"💬 Answer:\n{result['answer']}")
            
            if result['sources']:
                print(f"\n📚 Sources:")
                for j, source in enumerate(result['sources'], 1):
                    print(f"  {j}. {source['title']} (similarity: {source['similarity']:.3f})")
                    print(f"     {source['url']}")
        
        print("-" * 60)
    
    # List documents
    print("\n4. Sample Documents:")
    docs = client.list_documents(limit=5)
    if isinstance(docs, list):
        for doc in docs:
            print(f"📄 {doc['title']}")
            print(f"   URL: {doc['url']}")
            print(f"   Chunks: {doc['chunk_count']}")
            print(f"   Scraped: {doc['scraped_at']}")
            print()
    
    print("🎉 Demo completed!")


if __name__ == "__main__":
    main() 