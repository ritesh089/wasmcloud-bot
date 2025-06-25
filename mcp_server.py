#!/usr/bin/env python3
"""
MCP Server for wasmCloud RAG Bot

This server exposes the wasmCloud RAG bot functionality through the Model Context Protocol (MCP),
allowing AI assistants to access wasmCloud documentation knowledge.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Sequence
import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    Prompt,
    Resource,
    TextContent,
    CallToolRequest,
    GetPromptRequest,
    ListPromptsRequest,
    ListResourcesRequest,
    ListToolsRequest,
    ReadResourceRequest,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Server configuration
WASMCLOUD_RAG_URL = "http://localhost:8000"
HTTP_TIMEOUT = 30

# Create MCP server
app = Server("wasmcloud-rag-bot")
http_client = httpx.AsyncClient(timeout=HTTP_TIMEOUT)


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools."""
    return [
        Tool(
            name="query_wasmcloud_docs",
            description="Query wasmCloud documentation using RAG. Provides accurate answers with source citations from official wasmCloud docs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Question about wasmCloud to search for in the documentation"
                    },
                    "include_sources": {
                        "type": "boolean",
                        "description": "Whether to include source document citations",
                        "default": True
                    }
                },
                "required": ["question"]
            }
        ),
        Tool(
            name="get_wasmcloud_stats",
            description="Get statistics about the wasmCloud documentation database",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="list_wasmcloud_documents", 
            description="List available wasmCloud documentation pages in the database",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of documents to return",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 100
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="check_wasmcloud_health",
            description="Check the health status of the wasmCloud RAG bot service",
            inputSchema={
                "type": "object", 
                "properties": {},
                "required": []
            }
        )
    ]


@app.list_prompts()
async def list_prompts() -> List[Prompt]:
    """List available prompts."""
    return [
        Prompt(
            name="wasmcloud_expert",
            description="Become a wasmCloud expert assistant with access to comprehensive documentation",
            arguments=[
                {
                    "name": "specialization",
                    "description": "Area of wasmCloud expertise to focus on (e.g., 'installation', 'development', 'deployment')",
                    "required": False
                }
            ]
        ),
        Prompt(
            name="wasmcloud_troubleshooter",
            description="Become a wasmCloud troubleshooting specialist to help solve problems",
            arguments=[
                {
                    "name": "problem_area",
                    "description": "Type of problem to focus on (e.g., 'installation', 'runtime', 'configuration')",
                    "required": False
                }
            ]
        )
    ]


@app.list_resources()
async def list_resources() -> List[Resource]:
    """List available resources."""
    return [
        Resource(
            uri="wasmcloud://documentation/overview",
            name="wasmCloud Documentation Overview",
            description="Overview of all available wasmCloud documentation",
            mimeType="text/plain"
        ),
        Resource(
            uri="wasmcloud://bot/statistics",
            name="RAG Bot Statistics",
            description="Current statistics and status of the wasmCloud RAG bot",
            mimeType="application/json"
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""
    try:
        if name == "query_wasmcloud_docs":
            return await _query_wasmcloud_docs(arguments)
        elif name == "get_wasmcloud_stats":
            return await _get_wasmcloud_stats()
        elif name == "list_wasmcloud_documents":
            return await _list_wasmcloud_documents(arguments)
        elif name == "check_wasmcloud_health":
            return await _check_wasmcloud_health()
        else:
            return [TextContent(
                type="text",
                text=f"Unknown tool: {name}"
            )]
    except Exception as e:
        logger.error(f"Error in tool {name}: {e}")
        return [TextContent(
            type="text",
            text=f"Error executing {name}: {str(e)}"
        )]


@app.get_prompt()
async def get_prompt(name: str, arguments: Dict[str, Any] = None) -> str:
    """Handle prompt requests."""
    arguments = arguments or {}
    
    if name == "wasmcloud_expert":
        specialization = arguments.get("specialization", "")
        focus = f" specializing in {specialization}" if specialization else ""
        
        return f"""You are a wasmCloud expert{focus} with access to comprehensive wasmCloud documentation through a RAG system.

**Your capabilities:**
- Answer questions about wasmCloud concepts, architecture, and features
- Provide installation and setup guidance
- Help with development workflows and best practices  
- Explain deployment and scaling strategies
- Troubleshoot common issues
- Compare wasmCloud with other platforms

**Available tools:**
- `query_wasmcloud_docs`: Search wasmCloud documentation for specific information
- `get_wasmcloud_stats`: Get current database statistics
- `list_wasmcloud_documents`: Browse available documentation
- `check_wasmcloud_health`: Verify the RAG system is working

**Guidelines:**
1. Always use the RAG tools to get accurate, up-to-date information
2. Provide source citations when available
3. Give practical, actionable advice
4. Explain complex concepts clearly
5. Suggest next steps or related topics

How can I help you with wasmCloud today?"""

    elif name == "wasmcloud_troubleshooter":
        problem_area = arguments.get("problem_area", "")
        focus = f" focusing on {problem_area} issues" if problem_area else ""
        
        return f"""You are a wasmCloud troubleshooting specialist{focus} with access to comprehensive documentation.

**Troubleshooting approach:**
1. **Understand the problem** - Ask clarifying questions about symptoms, environment, and context
2. **Research solutions** - Use query_wasmcloud_docs to find relevant troubleshooting information
3. **Provide step-by-step guidance** - Give clear, actionable resolution steps
4. **Verify understanding** - Ensure the user can follow the instructions
5. **Suggest prevention** - Recommend ways to avoid similar issues

**Common problem areas:**
- Installation and setup issues
- Configuration problems  
- Runtime errors and crashes
- Performance and scaling issues
- Integration challenges
- Development workflow problems

**Available tools:**
- `query_wasmcloud_docs`: Search for troubleshooting information
- `check_wasmcloud_health`: Verify the RAG system is working
- `get_wasmcloud_stats`: Check system status
- `list_wasmcloud_documents`: Browse available documentation

What wasmCloud issue can I help you troubleshoot?"""

    else:
        return f"Unknown prompt: {name}"


@app.read_resource()
async def read_resource(uri: str) -> str:
    """Handle resource reading."""
    try:
        if uri == "wasmcloud://documentation/overview":
            # Get document overview
            response = await http_client.get(f"{WASMCLOUD_RAG_URL}/documents", params={"limit": 50})
            response.raise_for_status()
            documents = response.json()
            
            overview = "# wasmCloud Documentation Overview\n\n"
            overview += f"Total documents available: {len(documents)}\n\n"
            
            for doc in documents:
                overview += f"## {doc['title']}\n"
                overview += f"- **URL**: {doc['url']}\n"
                overview += f"- **Chunks**: {doc['chunk_count']}\n"
                overview += f"- **Last Updated**: {doc.get('updated_at', 'N/A')}\n\n"
            
            return overview
            
        elif uri == "wasmcloud://bot/statistics":
            # Get bot statistics
            response = await http_client.get(f"{WASMCLOUD_RAG_URL}/stats")
            response.raise_for_status()
            return response.text
            
        else:
            return f"Unknown resource URI: {uri}"
            
    except Exception as e:
        logger.error(f"Error reading resource {uri}: {e}")
        return f"Error reading resource: {str(e)}"


# Tool implementation functions
async def _query_wasmcloud_docs(arguments: Dict[str, Any]) -> List[TextContent]:
    """Query wasmCloud documentation."""
    question = arguments.get("question", "")
    include_sources = arguments.get("include_sources", True)
    
    if not question:
        return [TextContent(type="text", text="Error: Question is required")]
    
    try:
        response = await http_client.post(
            f"{WASMCLOUD_RAG_URL}/query",
            json={
                "question": question,
                "include_sources": include_sources
            }
        )
        response.raise_for_status()
        data = response.json()
        
        # Format response
        result = f"{data['answer']}\n\n"
        
        if include_sources and data.get('sources'):
            result += "**Sources:**\n"
            for i, source in enumerate(data['sources'], 1):
                result += f"{i}. [{source['title']}]({source['url']}) (relevance: {source['similarity']:.1%})\n"
        
        result += f"\n*Query processed in {data['response_time']:.2f}s using {data['chunks_used']} knowledge chunks*"
        
        return [TextContent(type="text", text=result)]
        
    except httpx.HTTPError as e:
        return [TextContent(type="text", text=f"Error querying documentation: {e}")]


async def _get_wasmcloud_stats() -> List[TextContent]:
    """Get wasmCloud RAG bot statistics."""
    try:
        response = await http_client.get(f"{WASMCLOUD_RAG_URL}/stats")
        response.raise_for_status()
        data = response.json()
        
        result = "**wasmCloud RAG Bot Statistics:**\n\n"
        result += f"📄 **Documents**: {data['total_documents']}\n"
        result += f"🧩 **Knowledge Chunks**: {data['total_chunks']}\n"
        
        if data.get('database_stats'):
            result += "\n**Database Details:**\n"
            for stat in data['database_stats']:
                result += f"- {stat['tablename']}: {stat['live_tuples']} records\n"
        
        return [TextContent(type="text", text=result)]
        
    except httpx.HTTPError as e:
        return [TextContent(type="text", text=f"Error getting statistics: {e}")]


async def _list_wasmcloud_documents(arguments: Dict[str, Any]) -> List[TextContent]:
    """List wasmCloud documents."""
    limit = arguments.get("limit", 10)
    
    try:
        response = await http_client.get(
            f"{WASMCLOUD_RAG_URL}/documents",
            params={"limit": limit}
        )
        response.raise_for_status()
        documents = response.json()
        
        if not documents:
            return [TextContent(type="text", text="No documents found in the database")]
        
        result = f"**wasmCloud Documentation ({len(documents)} documents):**\n\n"
        
        for i, doc in enumerate(documents, 1):
            result += f"{i}. **{doc['title']}**\n"
            result += f"   📍 {doc['url']}\n"
            result += f"   🧩 {doc['chunk_count']} chunks\n"
            if doc.get('scraped_at'):
                result += f"   📅 Scraped: {doc['scraped_at'][:10]}\n"
            result += "\n"
        
        return [TextContent(type="text", text=result)]
        
    except httpx.HTTPError as e:
        return [TextContent(type="text", text=f"Error listing documents: {e}")]


async def _check_wasmcloud_health() -> List[TextContent]:
    """Check wasmCloud RAG bot health."""
    try:
        response = await http_client.get(f"{WASMCLOUD_RAG_URL}/health")
        response.raise_for_status()
        data = response.json()
        
        status_icon = "✅" if data['status'] == 'healthy' else "❌"
        db_icon = "✅" if data['database_connected'] else "❌"
        
        result = f"**wasmCloud RAG Bot Health Check:**\n\n"
        result += f"{status_icon} **Overall Status**: {data['status'].title()}\n"
        result += f"{db_icon} **Database**: {'Connected' if data['database_connected'] else 'Disconnected'}\n"
        result += f"💬 **Message**: {data['message']}\n"
        
        return [TextContent(type="text", text=result)]
        
    except httpx.HTTPError as e:
        return [TextContent(type="text", text=f"Error checking health: {e}")]


async def main():
    """Run the MCP server."""
    logger.info("Starting wasmCloud RAG Bot MCP Server...")
    
    # Test connection to RAG bot
    try:
        response = await http_client.get(f"{WASMCLOUD_RAG_URL}/health")
        if response.status_code == 200:
            logger.info("✅ Connected to wasmCloud RAG Bot")
        else:
            logger.warning(f"⚠️ RAG Bot responded with status {response.status_code}")
    except Exception as e:
        logger.error(f"❌ Cannot connect to RAG Bot at {WASMCLOUD_RAG_URL}: {e}")
        logger.error("Make sure the RAG bot server is running!")
    
    # Start MCP server
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main()) 