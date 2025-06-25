# wasmCloud RAG Bot MCP Usage Guide

This guide explains how to use the Model Context Protocol (MCP) client to integrate the wasmCloud RAG bot with AI assistants like Claude Desktop, ChatGPT, and other MCP-compatible tools.

## What is MCP?

The Model Context Protocol (MCP) is an open standard that enables AI assistants to securely access external data sources and tools. Our wasmCloud RAG bot provides MCP integration so AI assistants can:

- Query wasmCloud documentation using RAG
- Get real-time statistics about the knowledge base
- Access comprehensive wasmCloud expertise
- Provide troubleshooting assistance

## Quick Start

### 1. Start the RAG Bot Server

First, make sure your wasmCloud RAG bot is running:

```bash
# Start the database
make start-db

# Initialize and ingest documentation (if not done already)
make init-db
make ingest

# Start the RAG bot server
make run
```

The server should be running at `http://localhost:8000`.

### 2. Test the MCP Client

Test the MCP client functionality:

```bash
python3 mcp_client.py
```

This will run a demo showing all available tools and capabilities.

### 3. Start the MCP Server

Run the MCP server that exposes wasmCloud RAG bot functionality:

```bash
python3 mcp_server.py
```

## Integration with AI Assistants

### Claude Desktop Integration

1. **Install Claude Desktop** from Anthropic
2. **Configure MCP Server** by adding to Claude's configuration:

```json
{
  "mcpServers": {
    "wasmcloud-rag-bot": {
      "command": "python3",
      "args": ["/path/to/your/wasmcloud-bot/mcp_server.py"],
      "cwd": "/path/to/your/wasmcloud-bot",
      "env": {
        "WASMCLOUD_RAG_URL": "http://localhost:8000"
      }
    }
  }
}
```

3. **Restart Claude Desktop** to load the MCP server
4. **Start using wasmCloud tools** in your conversations!

### Other MCP-Compatible Clients

The MCP server works with any MCP-compatible client. You can integrate it with:

- Custom AI applications
- Development tools
- IDE extensions
- Chatbots and assistants

## Available Tools

### 1. `query_wasmcloud_docs`

Query wasmCloud documentation using RAG.

**Parameters:**
- `question` (required): Your question about wasmCloud
- `include_sources` (optional): Whether to include source citations (default: true)

**Example:**
```
Ask the wasmCloud expert: "How do I install wasmCloud on Kubernetes?"
```

### 2. `get_wasmcloud_stats`

Get statistics about the wasmCloud documentation database.

**Parameters:** None

**Example:**
```
Check the current statistics of the wasmCloud knowledge base
```

### 3. `list_wasmcloud_documents`

List available wasmCloud documentation pages.

**Parameters:**
- `limit` (optional): Maximum number of documents to return (default: 10)

**Example:**
```
Show me the first 20 wasmCloud documentation pages
```

### 4. `check_wasmcloud_health`

Check the health status of the wasmCloud RAG bot.

**Parameters:** None

**Example:**
```
Check if the wasmCloud RAG bot is working properly
```

## Available Prompts

### 1. `wasmcloud_expert`

Activates wasmCloud expert mode with comprehensive knowledge.

**Arguments:**
- `specialization` (optional): Focus area like "installation", "development", "deployment"

**Usage:**
```
Use the wasmcloud_expert prompt with specialization "kubernetes"
```

### 2. `wasmcloud_troubleshooter`

Activates troubleshooting specialist mode.

**Arguments:**
- `problem_area` (optional): Type of problem like "installation", "runtime", "configuration"

**Usage:**
```
Use the wasmcloud_troubleshooter prompt for installation issues
```

## Available Resources

### 1. `wasmcloud://documentation/overview`

Provides an overview of all available wasmCloud documentation.

### 2. `wasmcloud://bot/statistics`

Returns current RAG bot statistics in JSON format.

## Example Usage Scenarios

### Scenario 1: Learning wasmCloud

**User:** "I'm new to wasmCloud. Can you explain what it is and how to get started?"

**AI Assistant with MCP:**
1. Uses `query_wasmcloud_docs` to get comprehensive information about wasmCloud
2. Provides detailed explanation with source citations
3. Offers next steps and related topics

### Scenario 2: Troubleshooting

**User:** "My wasmCloud application won't start and I'm getting connection errors"

**AI Assistant with MCP:**
1. Uses `wasmcloud_troubleshooter` prompt
2. Asks clarifying questions about the environment
3. Uses `query_wasmcloud_docs` to find relevant troubleshooting information
4. Provides step-by-step resolution guidance

### Scenario 3: Development Guidance

**User:** "How do I create a custom capability provider for wasmCloud?"

**AI Assistant with MCP:**
1. Uses `query_wasmcloud_docs` to find development documentation
2. Provides detailed implementation guidance
3. Includes code examples and best practices from the docs
4. Suggests related documentation for deeper learning

## Configuration Options

### Environment Variables

- `WASMCLOUD_RAG_URL`: URL of the RAG bot server (default: `http://localhost:8000`)
- `MCP_LOG_LEVEL`: Logging level for MCP server (default: `INFO`)

### Custom Configuration

You can customize the MCP server by modifying `mcp_server.py`:

```python
# Change the RAG bot URL
WASMCLOUD_RAG_URL = "http://your-server:8000"

# Adjust timeout settings
HTTP_TIMEOUT = 60

# Modify tool descriptions or add new tools
```

## Troubleshooting

### Common Issues

1. **MCP Server Won't Start**
   - Check that the RAG bot server is running at `http://localhost:8000`
   - Verify Python dependencies are installed: `pip3 install -r requirements.txt`
   - Check logs for specific error messages

2. **AI Assistant Can't Find Tools**
   - Ensure MCP server configuration is correct
   - Restart the AI assistant after configuration changes
   - Check that the MCP server process is running

3. **Queries Return Errors**
   - Verify the RAG bot database is properly initialized and populated
   - Check that your OpenAI API key is configured
   - Test the RAG bot directly at `http://localhost:8000`

### Debug Commands

```bash
# Test RAG bot health
curl http://localhost:8000/health

# Test MCP client
python3 mcp_client.py

# Check MCP server logs
python3 mcp_server.py  # Look for connection messages

# Test direct API call
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is wasmCloud?"}'
```

## Advanced Usage

### Custom Tools

You can extend the MCP server with custom tools by adding them to `mcp_server.py`:

```python
@app.list_tools()
async def list_tools() -> List[Tool]:
    tools = [
        # ... existing tools ...
        Tool(
            name="my_custom_tool",
            description="My custom wasmCloud tool",
            inputSchema={
                "type": "object",
                "properties": {
                    "param": {"type": "string"}
                },
                "required": ["param"]
            }
        )
    ]
    return tools

@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    if name == "my_custom_tool":
        return await _my_custom_implementation(arguments)
    # ... handle other tools ...
```

### Integration with IDEs

You can integrate the MCP server with development environments:

1. **VS Code Extension**: Create a VS Code extension that uses the MCP client
2. **IntelliJ Plugin**: Build a plugin that connects to the MCP server
3. **Command Line Tool**: Create CLI commands that use the MCP client

### Batch Processing

For processing multiple queries, use the MCP client programmatically:

```python
import asyncio
from mcp_client import WasmCloudRAGMCPClient

async def batch_queries():
    questions = [
        "What is wasmCloud?",
        "How do I install wasmCloud?",
        "What are wasmCloud capabilities?"
    ]
    
    async with WasmCloudRAGMCPClient() as client:
        for question in questions:
            result = await client.call_tool("query_wasmcloud_docs", {
                "question": question
            })
            print(f"Q: {question}")
            print(f"A: {result[0].text}\n")

asyncio.run(batch_queries())
```

## Support and Contributing

- **Issues**: Report bugs or request features in the project repository
- **Documentation**: Contribute to improving this guide
- **Code**: Submit pull requests for new features or improvements

## License

This MCP integration is part of the wasmCloud RAG Bot project and follows the same licensing terms. 