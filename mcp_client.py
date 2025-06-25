#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Client for wasmCloud RAG Bot

This client demonstrates how to interact with the wasmCloud RAG bot using the MCP protocol.
It provides tools and resources that can be used by AI assistants to query wasmCloud documentation.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Sequence
from dataclasses import dataclass
import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import (
    CallToolRequest,
    GetPromptRequest,
    ListPromptsRequest,
    ListResourcesRequest,
    ListToolsRequest,
    ReadResourceRequest,
    Tool,
    Prompt,
    Resource,
    TextContent,
    ImageContent,
    EmbeddedResource,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WasmCloudRAGConfig:
    """Configuration for wasmCloud RAG bot connection."""
    base_url: str = "http://localhost:8000"
    timeout: int = 30


class WasmCloudRAGMCPClient:
    """MCP Client for wasmCloud RAG Bot."""
    
    def __init__(self, config: WasmCloudRAGConfig = None):
        self.config = config or WasmCloudRAGConfig()
        self.http_client = httpx.AsyncClient(timeout=self.config.timeout)
        
    async def __aenter__(self):
        """Async context manager entry."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.http_client.aclose()

    async def list_tools(self) -> List[Tool]:
        """List available tools for the MCP client."""
        return [
            Tool(
                name="query_wasmcloud_docs",
                description="Query wasmCloud documentation using RAG (Retrieval-Augmented Generation). "
                           "This tool searches through wasmCloud documentation and provides accurate answers "
                           "with source citations.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question to ask about wasmCloud"
                        },
                        "include_sources": {
                            "type": "boolean",
                            "description": "Whether to include source citations in the response",
                            "default": True
                        }
                    },
                    "required": ["question"]
                }
            ),
            Tool(
                name="get_wasmcloud_stats",
                description="Get statistics about the wasmCloud documentation database, "
                           "including number of documents and chunks indexed.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            ),
            Tool(
                name="list_wasmcloud_documents",
                description="List documents available in the wasmCloud documentation database.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of documents to return",
                            "default": 10
                        },
                        "offset": {
                            "type": "integer", 
                            "description": "Number of documents to skip",
                            "default": 0
                        }
                    },
                    "required": []
                }
            ),
            Tool(
                name="check_wasmcloud_bot_health",
                description="Check the health status of the wasmCloud RAG bot server.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            )
        ]

    async def list_prompts(self) -> List[Prompt]:
        """List available prompts for the MCP client."""
        return [
            Prompt(
                name="wasmcloud_expert",
                description="Act as a wasmCloud expert assistant that can answer questions about wasmCloud "
                           "using the RAG bot. This prompt helps you provide comprehensive answers about "
                           "wasmCloud concepts, installation, usage, and best practices.",
                arguments=[
                    {
                        "name": "topic",
                        "description": "Specific wasmCloud topic to focus on (optional)",
                        "required": False
                    }
                ]
            ),
            Prompt(
                name="wasmcloud_troubleshooter",
                description="Act as a wasmCloud troubleshooting assistant that helps diagnose and solve "
                           "wasmCloud-related issues using the documentation knowledge base.",
                arguments=[
                    {
                        "name": "issue_type",
                        "description": "Type of issue (installation, deployment, configuration, etc.)",
                        "required": False
                    }
                ]
            )
        ]

    async def list_resources(self) -> List[Resource]:
        """List available resources for the MCP client."""
        return [
            Resource(
                uri="wasmcloud://docs/overview",
                name="wasmCloud Documentation Overview",
                description="Overview of available wasmCloud documentation",
                mimeType="application/json"
            ),
            Resource(
                uri="wasmcloud://stats",
                name="wasmCloud RAG Bot Statistics", 
                description="Current statistics of the wasmCloud documentation database",
                mimeType="application/json"
            )
        ]

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Call a tool and return the results."""
        try:
            if name == "query_wasmcloud_docs":
                return await self._query_wasmcloud_docs(arguments)
            elif name == "get_wasmcloud_stats":
                return await self._get_wasmcloud_stats()
            elif name == "list_wasmcloud_documents":
                return await self._list_wasmcloud_documents(arguments)
            elif name == "check_wasmcloud_bot_health":
                return await self._check_wasmcloud_bot_health()
            else:
                return [TextContent(
                    type="text",
                    text=f"Unknown tool: {name}"
                )]
        except Exception as e:
            logger.error(f"Error calling tool {name}: {e}")
            return [TextContent(
                type="text",
                text=f"Error calling tool {name}: {str(e)}"
            )]

    async def get_prompt(self, name: str, arguments: Dict[str, Any] = None) -> str:
        """Get a prompt by name."""
        arguments = arguments or {}
        
        if name == "wasmcloud_expert":
            topic = arguments.get("topic", "")
            topic_focus = f" with a focus on {topic}" if topic else ""
            
            return f"""You are a wasmCloud expert assistant{topic_focus}. You have access to the wasmCloud RAG bot 
that contains comprehensive knowledge about wasmCloud from the official documentation.

Use the query_wasmcloud_docs tool to answer questions about:
- wasmCloud concepts and architecture
- Installation and setup procedures  
- Development workflows and best practices
- Deployment and scaling strategies
- Troubleshooting common issues
- Integration with other technologies

Always provide accurate, helpful answers based on the official documentation. When using the RAG bot,
include source citations when available to help users find more detailed information.

How can I help you with wasmCloud today?"""

        elif name == "wasmcloud_troubleshooter":
            issue_type = arguments.get("issue_type", "")
            issue_focus = f" particularly {issue_type} issues" if issue_type else ""
            
            return f"""You are a wasmCloud troubleshooting specialist{issue_focus}. You help users diagnose 
and resolve wasmCloud-related problems using the comprehensive documentation knowledge base.

Use the query_wasmcloud_docs tool to help with:
- Installation and setup problems
- Configuration issues
- Deployment failures
- Runtime errors and debugging
- Performance optimization
- Integration challenges

Follow this troubleshooting approach:
1. Understand the specific problem
2. Query the documentation for relevant solutions
3. Provide step-by-step resolution steps
4. Include relevant documentation links
5. Suggest preventive measures

What wasmCloud issue can I help you troubleshoot?"""

        else:
            return f"Unknown prompt: {name}"

    async def read_resource(self, uri: str) -> str:
        """Read a resource by URI."""
        try:
            if uri == "wasmcloud://docs/overview":
                docs = await self._list_wasmcloud_documents({"limit": 50})
                return docs[0].text if docs else "No documents available"
            
            elif uri == "wasmcloud://stats":
                stats = await self._get_wasmcloud_stats()
                return stats[0].text if stats else "No statistics available"
            
            else:
                return f"Unknown resource URI: {uri}"
                
        except Exception as e:
            logger.error(f"Error reading resource {uri}: {e}")
            return f"Error reading resource {uri}: {str(e)}"

    async def _query_wasmcloud_docs(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Query wasmCloud documentation using RAG."""
        question = arguments.get("question", "")
        include_sources = arguments.get("include_sources", True)
        
        if not question:
            return [TextContent(type="text", text="Question is required")]
        
        try:
            response = await self.http_client.post(
                f"{self.config.base_url}/query",
                json={
                    "question": question,
                    "include_sources": include_sources
                }
            )
            response.raise_for_status()
            data = response.json()
            
            # Format the response
            result = f"**Answer:** {data['answer']}\n\n"
            result += f"**Response Time:** {data['response_time']:.2f}s\n"
            result += f"**Chunks Used:** {data['chunks_used']}\n"
            
            if data.get('sources') and include_sources:
                result += "\n**Sources:**\n"
                for i, source in enumerate(data['sources'], 1):
                    result += f"{i}. [{source['title']}]({source['url']}) "
                    result += f"(similarity: {source['similarity']:.1%})\n"
            
            return [TextContent(type="text", text=result)]
            
        except httpx.HTTPError as e:
            return [TextContent(type="text", text=f"HTTP error querying wasmCloud docs: {e}")]

    async def _get_wasmcloud_stats(self) -> List[TextContent]:
        """Get wasmCloud RAG bot statistics."""
        try:
            response = await self.http_client.get(f"{self.config.base_url}/stats")
            response.raise_for_status()
            data = response.json()
            
            result = "**wasmCloud RAG Bot Statistics:**\n\n"
            result += f"- **Total Documents:** {data['total_documents']}\n"
            result += f"- **Total Chunks:** {data['total_chunks']}\n"
            
            if data.get('database_stats'):
                result += "\n**Database Statistics:**\n"
                for stat in data['database_stats']:
                    result += f"- **{stat['tablename']}:** {stat['live_tuples']} records\n"
            
            return [TextContent(type="text", text=result)]
            
        except httpx.HTTPError as e:
            return [TextContent(type="text", text=f"HTTP error getting stats: {e}")]

    async def _list_wasmcloud_documents(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """List wasmCloud documents."""
        limit = arguments.get("limit", 10)
        offset = arguments.get("offset", 0)
        
        try:
            response = await self.http_client.get(
                f"{self.config.base_url}/documents",
                params={"limit": limit, "offset": offset}
            )
            response.raise_for_status()
            documents = response.json()
            
            if not documents:
                return [TextContent(type="text", text="No documents found")]
            
            result = f"**wasmCloud Documents ({len(documents)} of {limit} requested):**\n\n"
            
            for doc in documents:
                result += f"**{doc['title']}**\n"
                result += f"- URL: {doc['url']}\n"
                result += f"- Chunks: {doc['chunk_count']}\n"
                result += f"- Scraped: {doc['scraped_at']}\n\n"
            
            return [TextContent(type="text", text=result)]
            
        except httpx.HTTPError as e:
            return [TextContent(type="text", text=f"HTTP error listing documents: {e}")]

    async def _check_wasmcloud_bot_health(self) -> List[TextContent]:
        """Check wasmCloud RAG bot health."""
        try:
            response = await self.http_client.get(f"{self.config.base_url}/health")
            response.raise_for_status()
            data = response.json()
            
            status_emoji = "✅" if data['status'] == 'healthy' else "❌"
            db_emoji = "✅" if data['database_connected'] else "❌"
            
            result = f"**wasmCloud RAG Bot Health Status:**\n\n"
            result += f"{status_emoji} **Status:** {data['status']}\n"
            result += f"{db_emoji} **Database:** {'Connected' if data['database_connected'] else 'Disconnected'}\n"
            result += f"**Message:** {data['message']}\n"
            
            return [TextContent(type="text", text=result)]
            
        except httpx.HTTPError as e:
            return [TextContent(type="text", text=f"HTTP error checking health: {e}")]


# Example usage and testing
async def demo_mcp_client():
    """Demonstrate the MCP client functionality."""
    print("🚀 wasmCloud RAG Bot MCP Client Demo")
    print("=" * 50)
    
    async with WasmCloudRAGMCPClient() as client:
        # Test health check
        print("1. Health Check:")
        health_result = await client.call_tool("check_wasmcloud_bot_health", {})
        print(health_result[0].text)
        print()
        
        # Test statistics
        print("2. Statistics:")
        stats_result = await client.call_tool("get_wasmcloud_stats", {})
        print(stats_result[0].text)
        print()
        
        # Test documentation query
        print("3. Sample Query:")
        query_result = await client.call_tool("query_wasmcloud_docs", {
            "question": "What is wasmCloud and what are its main benefits?",
            "include_sources": True
        })
        print(query_result[0].text)
        print()
        
        # Test document listing
        print("4. Document List (first 3):")
        docs_result = await client.call_tool("list_wasmcloud_documents", {
            "limit": 3
        })
        print(docs_result[0].text)
        print()
        
        # Test prompts
        print("5. Expert Prompt:")
        expert_prompt = await client.get_prompt("wasmcloud_expert", {"topic": "installation"})
        print(expert_prompt[:200] + "..." if len(expert_prompt) > 200 else expert_prompt)


if __name__ == "__main__":
    asyncio.run(demo_mcp_client()) 