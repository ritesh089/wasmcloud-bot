# wasmCloud RAG Bot - Comprehensive System Design Document

## 📋 **Table of Contents**

1. [System Overview](#system-overview)
2. [Architecture Components](#architecture-components)
3. [Knowledge Graph Enhanced RAG](#knowledge-graph-enhanced-rag)
4. [Client-Side Implementation](#client-side-implementation)
5. [Server-Side Implementation](#server-side-implementation)
6. [RAG Pipeline Deep Dive](#rag-pipeline-deep-dive)
7. [Knowledge Graph Pipeline](#knowledge-graph-pipeline)
8. [OpenAI API Integration](#openai-api-integration)
9. [Data Flow Analysis](#data-flow-analysis)
10. [Database Design](#database-design)
11. [Optimization Features](#optimization-features)
12. [API Endpoints](#api-endpoints)
13. [Security & Performance](#security--performance)
14. [Deployment & Configuration](#deployment--configuration)

---

## 🔍 **System Overview**

The wasmCloud RAG (Retrieval-Augmented Generation) Bot is an intelligent question-answering system that provides accurate, contextual responses about wasmCloud technology by combining modern AI techniques with a comprehensive knowledge base and **Knowledge Graph Enhanced Reasoning**.

### **Core Functionality**
- **Intelligent Document Processing**: AI-powered scraping and chunking of wasmCloud documentation
- **Knowledge Graph Construction**: Automated extraction of entities and relationships using GPT-4
- **Multi-Strategy Search System**: Vector similarity, keyword search, concept expansion, and graph traversal
- **Advanced RAG Pipeline**: Three-tier system (Basic → Advanced → Knowledge Graph Enhanced)
- **Real-time Query Processing**: Provides comprehensive answers with relationship-aware reasoning
- **Optimization Controls**: Configurable AI features with cost management and graceful fallbacks

### **Technology Stack**
```
Frontend:  HTML5, CSS3, JavaScript (Dark Mode UI)
Backend:   Python 3.13, FastAPI, SQLAlchemy
Database:  PostgreSQL with pgvector extension
AI/ML:     OpenAI GPT-4, OpenAI Embeddings, tiktoken
Search:    Vector similarity, PostgreSQL full-text search, Knowledge Graph traversal
KG:        Entity extraction, Relationship mapping, Triplet reasoning
Tools:     Docker, uvicorn, BeautifulSoup, requests
```

### **Three-Tier RAG Architecture**

```mermaid
graph LR
    A[Query] --> B{Auto-Route}
    B -->|Simple| C[Basic RAG]
    B -->|Complex| D[Advanced RAG]
    B -->|Relationship| E[KG Enhanced RAG]
    
    C --> F[Vector Search Only]
    D --> G[Hybrid Search + AI Reranking]
    E --> H[Graph Reasoning + Multi-source Integration]
    
    F --> I[Response]
    G --> I
    H --> I
```

---

## 🏗️ **Architecture Components**

The system follows a modern microservices-inspired architecture with knowledge graph enhancement:

```
┌─────────────────────────────────────────────────────────────┐
│                     CLIENT LAYER                           │
├─────────────────────────────────────────────────────────────┤
│  Web Interface (HTML/CSS/JS)  │  API Clients (REST/HTTP)   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                     API GATEWAY                            │
├─────────────────────────────────────────────────────────────┤
│     FastAPI Server │ CORS Middleware │ Request Validation  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                  BUSINESS LOGIC LAYER                      │
├─────────────────────────────────────────────────────────────┤
│ KG Enhanced RAG │ Advanced RAG │ Basic RAG │ Optimization  │
├─────────────────────────────────────────────────────────────┤
│ Knowledge Graph │ AI Chunker │ Hybrid Search │ Auto-Router │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    DATA LAYER                              │
├─────────────────────────────────────────────────────────────┤
│ PostgreSQL DB │ pgvector │ KG Tables │ Embedding Cache     │
├─────────────────────────────────────────────────────────────┤
│ Entities │ Relations │ Triplets │ Chunks │ Documents       │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                 EXTERNAL SERVICES                          │
├─────────────────────────────────────────────────────────────┤
│      OpenAI API        │     wasmCloud Documentation       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🕸️ **Knowledge Graph Enhanced RAG**

### **Knowledge Graph Architecture**

```mermaid
graph TD
    A[Document Chunks] --> B[GPT-4 Entity Extraction]
    B --> C[Entity Normalization]
    C --> D[Triplet Storage]
    
    E[User Query] --> F[Query Analysis]
    F --> G[Hybrid Retrieval]
    F --> H[Graph Context Retrieval]
    
    G --> I[Text Chunks]
    H --> J[Related Entities & Relationships]
    
    I --> K[GPT-4 Multi-source Reasoning]
    J --> K
    K --> L[Enhanced Response]
    
    subgraph "Knowledge Graph Components"
        M[Entities Table]
        N[Relations Table]
        O[Triplets Table]
        
        M --> O
        N --> O
    end
```

### **Knowledge Graph Features**

#### **1. Entity Extraction & Management**
- **wasmCloud-Specific Entities**: Actors, providers, interfaces, capabilities, applications
- **AI-Powered Extraction**: GPT-4 identifies and categorizes entities from documentation
- **Embedding Storage**: Vector representations for semantic similarity
- **Deduplication**: Intelligent merging of similar entities

#### **2. Relationship Discovery**
- **Relationship Types**: implements, provides, requires, uses, contains, extends, depends_on
- **Frequency Tracking**: Relationship importance based on occurrence
- **Bidirectional Relations**: Automatic inverse relationship creation
- **Confidence Scoring**: AI-generated confidence levels for relationships

#### **3. Graph-Enhanced Query Processing**
```python
# Multi-step KG Enhanced Query Process
1. Traditional Retrieval (Vector + Keyword)
2. Entity Identification in Query
3. Graph Traversal for Related Concepts
4. Context Integration 
5. Multi-source AI Reasoning
6. Enhanced Response Generation
```

### **Expected Performance Improvements**

| Metric | Basic RAG | Advanced RAG | KG Enhanced RAG |
|--------|-----------|--------------|-----------------|
| Technical Accuracy | 60-70% | 75-85% | 85-95% |
| Relationship Understanding | 30-40% | 50-60% | 80-90% |
| Context Completeness | 50-60% | 70-80% | 85-95% |
| Complex Query Handling | 40-50% | 65-75% | 90-95% |

### **Cost Analysis**

| Operation | Time | Cost per Query | API Calls |
|-----------|------|---------------|-----------|
| Traditional Search | 2-4s | $0.02-0.05 | 3-4 |
| KG Context Retrieval | 1-2s | $0.01 | 1 |
| Multi-source Reasoning | 3-6s | $0.02-0.06 | 1 |
| **Total KG Enhanced** | **6-12s** | **$0.05-0.12** | **5-6** |

---

## 🔄 **Knowledge Graph Pipeline**

### **Phase 1: Knowledge Extraction**

```python
# Automated Triplet Extraction Process
async def extract_triplets_from_chunk(chunk_content: str) -> List[Triplet]:
    """
    Extract semantic relationships using GPT-4
    """
    prompt = f"""
    Extract semantic triplets from this wasmCloud documentation:
    
    Content: {chunk_content}
    
    Focus on wasmCloud entities: actors, providers, interfaces, capabilities
    Relationship types: implements, provides, requires, uses, contains
    
    Return JSON format: [{{
        "subject": "entity1",
        "predicate": "relationship", 
        "object": "entity2",
        "confidence": 0.9
    }}]
    """
    
    # GPT-4 extraction + entity normalization + embedding generation
    return normalized_triplets
```

### **Phase 2: Graph Construction**

```sql
-- Knowledge Graph Schema
CREATE TABLE entities (
    id SERIAL PRIMARY KEY,
    name VARCHAR NOT NULL,
    entity_type VARCHAR NOT NULL, -- actor, provider, interface, etc.
    description TEXT,
    embedding VECTOR(1536), -- Semantic representation
    metadata JSONB,
    first_seen TIMESTAMP DEFAULT NOW(),
    last_updated TIMESTAMP DEFAULT NOW()
);

CREATE TABLE relations (
    id SERIAL PRIMARY KEY,
    name VARCHAR NOT NULL, -- implements, provides, requires, etc.
    description TEXT,
    frequency INTEGER DEFAULT 1, -- How often this relation appears
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE triplets (
    id SERIAL PRIMARY KEY,
    subject_id INTEGER REFERENCES entities(id),
    predicate_id INTEGER REFERENCES relations(id),
    object_id INTEGER REFERENCES entities(id),
    confidence FLOAT NOT NULL, -- AI confidence in this relationship
    source_chunk_id INTEGER REFERENCES chunks(id),
    created_at TIMESTAMP DEFAULT NOW()
);
```

### **Phase 3: Graph-Enhanced Retrieval**

```python
async def kg_enhanced_search(query: str, k: int = 5) -> Dict:
    """
    Multi-strategy search with knowledge graph reasoning
    """
    # Step 1: Traditional hybrid search
    text_chunks = await hybrid_search(query, k=10)
    
    # Step 2: Extract entities from query
    query_entities = await extract_entities_from_query(query)
    
    # Step 3: Graph traversal for related concepts
    related_entities = await find_related_entities(query_entities)
    
    # Step 4: Retrieve relationship context
    kg_context = await get_relationship_context(related_entities)
    
    # Step 5: Multi-source reasoning
    enhanced_response = await multi_source_reasoning(
        text_chunks=text_chunks,
        kg_context=kg_context,
        query=query
    )
    
    return enhanced_response
```

---

## 💻 **Client-Side Implementation**

### **Web Interface (`static/index.html`)**

The client provides a modern, ChatGPT-inspired interface with dark mode theming.

#### **Key Features:**
- **Responsive Design**: Mobile-first responsive layout
- **Dark Mode UI**: Professional black/gray color scheme  
- **Real-time Interaction**: Instant query submission and response display
- **Source Citations**: Clickable links to original documentation
- **Message History**: Persistent chat-like conversation view
- **Typing Indicators**: Visual feedback during processing

#### **User Interface Components:**

```html
<!-- Main Layout Structure -->
<div class="app-container">
    <aside class="sidebar">
        <!-- Navigation and settings -->
        <div class="sidebar-header">
            <h2>wasmCloud Bot</h2>
        </div>
        <div class="conversation-list">
            <!-- Chat threads -->
        </div>
    </aside>
    
    <main class="chat-container">
        <div class="messages-container">
            <!-- Message display area -->
        </div>
        <div class="input-container">
            <textarea placeholder="Ask about wasmCloud..."></textarea>
            <button class="send-button">Send</button>
        </div>
    </main>
</div>
```

#### **JavaScript Functionality:**

```javascript
// Core client-side features
class WasmCloudBot {
    constructor() {
        this.apiBase = '';
        this.currentConversation = [];
        this.isProcessing = false;
    }
    
    async sendQuery(question) {
        // 1. Validate input
        // 2. Show typing indicator
        // 3. Send POST request to /query
        // 4. Display response with source citations
        // 5. Update conversation history
    }
    
    displayResponse(data) {
        // Format answer with markdown support
        // Render source citations as clickable links
        // Add to conversation history
    }
}
```

---

## 🖥️ **Server-Side Implementation**

### **FastAPI Application (`server/main.py`)**

The server is built with FastAPI, providing automatic API documentation, type validation, and high performance.

#### **Application Structure:**

```python
# Core Application Setup
app = FastAPI(
    title="wasmCloud RAG Bot",
    description="AI-powered documentation assistant",
    version="1.0.0"
)

# Middleware Configuration
app.add_middleware(CORSMiddleware, allow_origins=["*"])
app.mount("/static", StaticFiles(directory="static"))

# Dependency Injection
def get_db() -> Session:
    """Database session dependency"""
    
def get_optimization_config() -> OptimizationConfig:
    """Configuration dependency"""
```

#### **Request/Response Models:**

```python
class QueryRequest(BaseModel):
    question: str
    include_sources: bool = True

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    chunks_used: int
    response_time: float
    search_method: str = "hybrid_advanced"
```

### **Database Layer (`server/database.py`)**

#### **Connection Management:**
```python
# PostgreSQL with pgvector support
DATABASE_URL = "postgresql://user:pass@localhost:5432/wasmcloud_rag"

engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

#### **Models (`server/models.py`):**

```python
class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True)
    url = Column(String, unique=True, nullable=False)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    content_hash = Column(String, nullable=False)
    scraped_at = Column(DateTime, default=datetime.utcnow)
    
    chunks = relationship("Chunk", back_populates="document")

class Chunk(Base):
    __tablename__ = "chunks"
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    token_count = Column(Integer, nullable=False)
    
    # Vector embedding (pgvector or JSON fallback)
    embedding = Column(Vector(1536) if PGVECTOR_AVAILABLE else JSON)
    
    document = relationship("Document", back_populates="chunks")
```

---

## 🧠 **RAG Pipeline Deep Dive**

### **Basic RAG Pipeline (`server/rag.py`)**

#### **Core Components:**

```python
class RAGPipeline:
    def __init__(self, top_k: int = 5, similarity_threshold: float = 0.7):
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.client = OpenAI(api_key=OPENAI_API_KEY)
    
    def retrieve_relevant_chunks(self, query: str, db: Session) -> List[Dict]:
        """Vector similarity search for relevant content"""
        
    def generate_response(self, query: str, context: str) -> str:
        """Generate GPT-4 response with context"""
        
    def answer_question(self, query: str, db: Session) -> Dict:
        """Complete RAG pipeline execution"""
```

#### **Vector Search Implementation:**

```python
def retrieve_relevant_chunks(self, query: str, db: Session) -> List[Dict]:
    # 1. Generate query embedding
    query_embedding = generate_query_embedding(query)
    
    if PGVECTOR_AVAILABLE:
        # Use pgvector for fast similarity search
        sql = """
            SELECT content, title, url, 
                   (1 - (embedding <=> %s)) as similarity
            FROM chunks c JOIN documents d ON c.document_id = d.id
            WHERE (1 - (embedding <=> %s)) > %s
            ORDER BY embedding <=> %s
            LIMIT %s
        """
    else:
        # Fallback to manual cosine similarity
        chunks = db.query(Chunk).all()
        similarities = [
            cosine_similarity(query_embedding, chunk.embedding)
            for chunk in chunks
        ]
```

### **Advanced RAG Pipeline (`server/advanced_rag.py`)**

#### **Hybrid Search Strategy:**

```python
class HybridSearchRAG(RAGPipeline):
    def hybrid_retrieve(self, query: str, db: Session) -> List[Dict]:
        # Multi-strategy retrieval
        vector_results = self.retrieve_relevant_chunks(query, db)
        keyword_results = self._keyword_search(query, db)
        concept_results = self._concept_search(query, db)
        
        # Combine and deduplicate
        combined = self._combine_search_results(
            vector_results, keyword_results, concept_results
        )
        
        # AI reranking for optimal relevance
        if len(combined) > self.final_k:
            return self._ai_rerank(query, combined)
        
        return combined
```

#### **Search Strategies:**

1. **Vector Similarity Search**
   ```python
   def _vector_similarity_search(self, query_embedding, db):
       # Uses OpenAI embeddings for semantic similarity
       # Leverages pgvector for efficient nearest neighbor search
   ```

2. **Keyword Search**
   ```python
   def _keyword_search(self, query: str, db: Session):
       # PostgreSQL full-text search
       search_terms = self._extract_keywords(query)
       sql = """
           SELECT *, ts_rank(to_tsvector('english', content), 
                            plainto_tsquery('english', %s)) as rank
           FROM chunks WHERE to_tsvector('english', content) 
           @@ plainto_tsquery('english', %s)
           ORDER BY rank DESC
       """
   ```

3. **Concept Search**
   ```python
   def _concept_search(self, query: str, db: Session):
       # AI-powered query expansion
       expanded_concepts = self._expand_query_concepts(query)
       # Search for each concept using embeddings
   ```

4. **AI Reranking**
   ```python
   def _ai_rerank(self, query: str, chunks: List[Dict]) -> List[Dict]:
       # Use GPT-3.5 to evaluate relevance
       prompt = f"""
       Rank these chunks by relevance to: {query}
       Consider: directness, completeness, accuracy, practical value
       """
       # Returns optimally ordered results
   ```

---

## 🤖 **OpenAI API Integration**

### **API Usage Breakdown**

#### **1. Embedding Generation (`server/embeddings.py`)**

```python
class EmbeddingGenerator:
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.client = OpenAI(api_key=OPENAI_API_KEY)
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate 1536-dimensional embedding vector"""
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch processing for efficiency"""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts  # Up to 2048 texts per request
        )
        return [data.embedding for data in response.data]
```

**Cost Optimization:**
- **Model**: `text-embedding-3-small` (Lower cost than `text-embedding-3-large`)
- **Batch Processing**: Process multiple texts per API call
- **Caching**: Store embeddings to avoid regeneration
- **Smart Chunking**: Optimal chunk sizes to maximize content per embedding

#### **2. Text Chunking AI (`server/ai_chunking.py`)**

```python
class AIChunker:
    def analyze_document_structure(self, content: str, title: str) -> Dict:
        """Use GPT-3.5 to identify optimal chunk boundaries"""
        prompt = f"""
        Analyze this wasmCloud documentation and identify 3-8 main sections:
        
        Document: {title}
        Content: {content[:3000]}...
        
        Return JSON with:
        - sections: [{{topic, start_phrase, end_phrase, rationale}}]
        - strategy: chunking approach description
        - key_concepts: main wasmCloud concepts covered
        """
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1  # Low temperature for consistent results
        )
```

**Benefits:**
- **Semantic Coherence**: Keeps related concepts together
- **Context Preservation**: Maintains logical document structure
- **Better Embeddings**: Coherent text creates more meaningful vectors

#### **3. Response Generation**

```python
def generate_response(self, query: str, context: str) -> str:
    """Generate comprehensive answer using GPT-4"""
    prompt = f"""
    You are a wasmCloud expert assistant. Answer based on official documentation.
    
    CONTEXT:
    {context}
    
    QUESTION: {query}
    
    GUIDELINES:
    - Provide accurate, specific information
    - Include code examples when available
    - Structure response clearly
    - Cite sources for specific claims
    """
    
    response = self.client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=2000
    )
```

#### **4. Query Concept Expansion**

```python
def _expand_query_concepts(self, query: str) -> List[str]:
    """AI-powered query expansion for better search coverage"""
    prompt = f"""
    Given this wasmCloud question, identify 2-3 related concepts:
    
    Question: {query}
    
    Return related terms like:
    - Technical terms (actors, capabilities, providers, WASM, OCI, NATS)
    - Operations (deployment, scaling, configuration)
    - Platforms (Kubernetes, cloud, edge)
    """
    
    # Returns expanded search terms for concept search
```

### **Cost Management**

```python
@dataclass
class OptimizationConfig:
    max_ai_calls_per_query: int = 3
    use_cheaper_models_for_analysis: bool = True
    embedding_batch_size: int = 10
    
    # Cost per operation (approximate)
    # Embedding: $0.00002 per 1K tokens
    # GPT-3.5: $0.0015 per 1K tokens  
    # GPT-4: $0.03 per 1K tokens
```

---

## 📊 **Data Flow Analysis**

### **Enhanced System Data Flow with Knowledge Graph**

The following flowchart shows the complete data flow including the new Knowledge Graph Enhanced RAG pipeline:

```mermaid
flowchart TD
    A["🌐 wasmCloud Documentation<br/>https://wasmcloud.com/docs/"] --> B["📥 Web Scraper<br/>(scraper.py)"]
    
    B --> C["📄 Raw HTML Content"]
    C --> D["🔧 Content Processing<br/>- Extract text<br/>- Clean HTML<br/>- Parse structure"]
    
    D --> E["✂️ Text Chunking<br/>(embeddings.py)<br/>- Configurable size<br/>- Overlap handling"]
    
    E --> F["🤖 OpenAI Embeddings<br/>text-embedding-3-small<br/>1536 dimensions"]
    E --> KG1["🕸️ Knowledge Graph Extraction<br/>(knowledge_graph.py)<br/>- GPT-4 triplet extraction<br/>- Entity normalization"]
    
    F --> G["🗄️ PostgreSQL + pgvector<br/>- Documents table<br/>- Chunks table<br/>- Vector storage"]
    
    KG1 --> KG2["🧠 Knowledge Graph Storage<br/>- Entities table<br/>- Relations table<br/>- Triplets table"]
    
    H["👤 User Query"] --> I["🔍 Query Router<br/>(main.py)"]
    I --> I1{{"🎯 Strategy Selection"}}
    
    I1 -->|Simple| J1["🔍 Basic RAG<br/>(rag.py)"]
    I1 -->|Complex| J2["🔍 Advanced RAG<br/>(advanced_rag.py)"]
    I1 -->|Relationships| J3["🕸️ KG Enhanced RAG<br/>(kg_enhanced_rag.py)"]
    
    J1 --> K1["🎯 Vector Search Only"]
    J2 --> K2["🎯 Hybrid Search<br/>+ AI Reranking"]
    J3 --> K3["🎯 Graph Traversal<br/>+ Multi-source Reasoning"]
    
    G --> K1
    G --> K2
    G --> K3
    KG2 --> K3
    
    K1 --> L1["📋 Vector Results"]
    K2 --> L2["📋 Reranked Results"]
    K3 --> L3["📋 Graph-Enhanced Results<br/>+ Relationship Context"]
    
    L1 --> M1["🧠 Basic Context Assembly"]
    L2 --> M2["🧠 Advanced Context Assembly"]
    L3 --> M3["🧠 Multi-source Context<br/>+ Knowledge Graph"]
    
    M1 --> N1["🤖 GPT-3.5 Response"]
    M2 --> N2["🤖 GPT-4 Response"]
    M3 --> N3["🤖 GPT-4 Multi-source<br/>Reasoning"]
    
    N1 --> O["📤 Response Delivery"]
    N2 --> O
    N3 --> O
    
    subgraph "🖥️ User Interfaces"
        P["🌐 Web Interface<br/>(index.html)"]
        Q["🔌 REST API<br/>(FastAPI)"]
        R["🤝 MCP Integration<br/>(mcp_server.py)"]
    end
    
    O --> P
    O --> Q
    O --> R
    
    subgraph "📊 Monitoring & Logging"
        S["📈 Query Statistics<br/>+ KG Metrics"]
        T["🗃️ Query Logs<br/>+ Strategy Used"]
        U["⚡ Health Checks"]
    end
    
    I --> S
    I --> T
    Q --> U
    
    subgraph "🔧 Development Tools"
        V["🚀 Start Script<br/>(start.zsh)"]
        W["🛠️ Dev Environment<br/>(dev.zsh)"]
        X["⚙️ Setup Script<br/>(setup.zsh)"]
    end
    
    classDef input fill:#e1f5fe
    classDef processing fill:#f3e5f5
    classDef storage fill:#e8f5e8
    classDef output fill:#fff3e0
    classDef interface fill:#fce4ec
    classDef monitoring fill:#f1f8e9
    classDef tools fill:#fff8e1
    classDef kg fill:#e8eaf6
    
    class A,H input
    class B,C,D,E,F,I,I1,J1,J2,J3,K1,K2,K3,L1,L2,L3,M1,M2,M3,N1,N2,N3 processing
    class G storage
    class KG1,KG2 kg
    class O output
    class P,Q,R interface
    class S,T,U monitoring
    class V,W,X tools
```

### **Three-Tier Query Processing Flow**

#### **Query Example: "How do capability providers work with actors?"**

```
🎯 INTELLIGENT ROUTING (0-100ms)
├── Query analysis: Contains relationship keywords
├── Check KG availability: ✅ Knowledge Graph enabled
└── Route decision: → Knowledge Graph Enhanced RAG

📊 BASIC RAG FLOW (2-4 seconds)
└── Simple queries like "what is wash?"
    ├── Vector search only (500ms)
    ├── GPT-3.5 response (2-3s)
    └── Cost: $0.02-0.03

📊 ADVANCED RAG FLOW (4-8 seconds)  
└── Complex queries like "how to deploy multiple actors?"
    ├── Hybrid search + AI reranking (3-5s)
    ├── GPT-4 response (2-3s)
    └── Cost: $0.03-0.06

🕸️ KNOWLEDGE GRAPH ENHANCED RAG FLOW (6-12 seconds)
└── Relationship queries like "how do providers work with actors?"
    ├── 1. HYBRID RETRIEVAL (1-3s)
    │   ├── Vector similarity search
    │   ├── Keyword search  
    │   └── AI concept expansion
    │
    ├── 2. GRAPH CONTEXT RETRIEVAL (1-2s)
    │   ├── Extract entities from query ("providers", "actors")
    │   ├── Graph traversal (find: implements, provides, requires)
    │   └── Retrieve relationship context
    │
    ├── 3. MULTI-SOURCE REASONING (3-6s)
    │   ├── Combine text chunks + graph relationships
    │   ├── GPT-4 enhanced reasoning
    │   └── Generate comprehensive response
    │
    └── 4. ENHANCED RESPONSE (100-500ms)
        ├── Include relationship insights
        ├── Multiple source types
        └── Cost: $0.05-0.12
```

### **Knowledge Graph Construction Pipeline**

```
🔄 BACKGROUND EXTRACTION PROCESS (Continuous)

📄 Document Chunks (234 total)
├── Unprocessed chunks detected
└── Batch processing (10 chunks at a time)

🤖 GPT-4 TRIPLET EXTRACTION (Per chunk: 15-30s)
├── Input: wasmCloud documentation chunk
├── AI Analysis: Extract entities and relationships  
├── Output: JSON triplets with confidence scores
└── Example: {"subject": "actors", "predicate": "implements", "object": "interfaces", "confidence": 0.9}

🧠 ENTITY NORMALIZATION & STORAGE
├── Normalize entity names ("Actor" → "actor")
├── Generate embeddings for semantic similarity
├── Store in entities table with metadata
└── Update frequency counts for relations

🕸️ KNOWLEDGE GRAPH CONSTRUCTION
├── Create bidirectional relationships
├── Calculate confidence aggregations
├── Build graph indexes for fast traversal
└── Status: 80/234 chunks processed, 682 triplets extracted

⚡ REAL-TIME GRAPH QUERIES (Per user query)
├── Entity extraction from user query
├── Graph traversal (BFS, depth=2)
├── Relationship context retrieval
└── Integration with traditional search results
```

### **Detailed Flow Steps**

#### **Step 1: Query Reception (0-10ms)**
```python
@app.post("/query")
def query_bot(request: QueryRequest, db: Session = Depends(get_db)):
    # Input validation
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    
    # Configuration check
    config = get_optimization_config()
    strategy = config.get_rag_strategy()  # "basic" or "advanced"
```

#### **Step 2: RAG Pipeline Selection (10-20ms)**
```python
if strategy == "advanced":
    result = ask_question_advanced(request.question, db)
else:
    result = ask_question(request.question, db)
```

#### **Step 3: Hybrid Search Execution (200-2000ms)**

**3a. Vector Search (100-500ms)**
```python
# Generate query embedding
query_embedding = generate_query_embedding(query)  # OpenAI API call

# Vector similarity search
if PGVECTOR_AVAILABLE:
    # Fast pgvector search
    results = db.execute("""
        SELECT *, (1 - (embedding <=> %s)) as similarity
        FROM chunks WHERE (1 - (embedding <=> %s)) > 0.6
        ORDER BY embedding <=> %s LIMIT 10
    """, [query_embedding])
else:
    # Manual cosine similarity computation
    similarities = [cosine_similarity(query_embedding, chunk.embedding) 
                   for chunk in all_chunks]
```

**3b. Keyword Search (50-200ms)**
```python
keywords = extract_keywords(query)  # "scaling", "actors", "replicas"
search_terms = " | ".join(keywords)

results = db.execute("""
    SELECT *, ts_rank(to_tsvector('english', content), 
                     plainto_tsquery('english', %s)) as rank
    FROM chunks 
    WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
    ORDER BY rank DESC LIMIT 15
""", [search_terms])
```

**3c. Concept Search (500-1500ms)**
```python
# AI concept expansion
expanded_concepts = expand_query_concepts(query)  # OpenAI API call
# Example: "scaling actors" → ["horizontal scaling", "load balancing", "deployment patterns"]

# Search for each concept
for concept in expanded_concepts:
    concept_embedding = generate_query_embedding(concept)  # OpenAI API call
    concept_results = vector_search(concept_embedding)
```

#### **Step 4: Result Combination & Deduplication (20-50ms)**
```python
def combine_search_results(*result_lists):
    seen_ids = set()
    combined = []
    
    for results in result_lists:
        for result in results:
            if result['id'] not in seen_ids:
                seen_ids.add(result['id'])
                combined.append(result)
    
    return sorted(combined, key=lambda x: x.get('similarity', 0), reverse=True)
```

#### **Step 5: AI Reranking (1000-3000ms)**
```python
def ai_rerank(query: str, chunks: List[Dict]) -> List[Dict]:
    chunk_summaries = [
        {
            'index': i,
            'title': chunk['title'],
            'content_preview': chunk['content'][:200],
            'search_type': chunk['search_type'],
            'similarity': chunk['similarity']
        }
        for i, chunk in enumerate(chunks)
    ]
    
    prompt = f"""
    Rank these wasmCloud documentation chunks by relevance to: {query}
    
    Chunks: {json.dumps(chunk_summaries, indent=2)}
    
    Consider:
    1. Direct relevance to the question
    2. Completeness of information  
    3. Accuracy and specificity
    4. Practical value for the user
    
    Return JSON: {{"ranking": [2, 0, 5, 1, 3]}}
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    # Reorder chunks based on AI ranking
```

#### **Step 6: Context Building (10-30ms)**
```python
def build_enhanced_context(chunks: List[Dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        search_info = f"[Found via {chunk['search_type']} search"
        if chunk.get('concept'):
            search_info += f", concept: {chunk['concept']}"
        search_info += "]"
        
        context_parts.append(
            f"Source {i} {search_info}: {chunk['title']} - {chunk['url']}\n"
            f"{chunk['content']}\n"
        )
    
    return "\n---\n".join(context_parts)
```

#### **Step 7: Response Generation (2000-8000ms)**
```python
def generate_enhanced_response(query: str, context: str, chunks: List[Dict]) -> str:
    search_methods = list(set(chunk.get('search_type', 'vector') for chunk in chunks))
    
    prompt = f"""
    You are an expert wasmCloud assistant with access to official documentation.
    
    SEARCH CONTEXT: Information gathered using {', '.join(search_methods)} search methods.
    
    GUIDELINES:
    1. Provide accurate, specific information based on the context
    2. Use examples and code snippets when available
    3. Structure your response clearly with sections if needed
    4. Mention specific wasmCloud concepts, tools, and best practices
    5. Always cite sources when making specific claims
    
    CONTEXT:
    {context}
    
    QUESTION: {query}
    
    ANSWER:
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=2000
    )
    
    return response.choices[0].message.content.strip()
```

#### **Step 8: Response Assembly & Logging (10-50ms)**
```python
# Prepare response with metadata
response_data = {
    'answer': answer,
    'sources': [
        {
            'title': chunk['title'],
            'url': chunk['url'],
            'similarity': chunk.get('similarity', 0),
            'search_type': chunk.get('search_type', 'vector'),
            'rerank_position': chunk.get('rerank_position'),
            'concept': chunk.get('concept')
        }
        for chunk in relevant_chunks
    ],
    'chunks_used': len(relevant_chunks),
    'response_time': time.time() - start_time,
    'search_method': 'hybrid_advanced'
}

# Log query for analytics
log_entry = QueryLog(
    query=query,
    response=answer,
    chunks_used=len(relevant_chunks),
    response_time=response_time
)
db.add(log_entry)
db.commit()
```

### **Performance Metrics**

| Operation | Time Range | Cost | API Calls |
|-----------|------------|------|-----------|
| Query Embedding | 100-300ms | $0.00002 | 1 |
| Vector Search | 50-200ms | Free | 0 |
| Keyword Search | 20-100ms | Free | 0 |
| Concept Expansion | 200-800ms | $0.0005 | 1 |
| AI Reranking | 800-2000ms | $0.002 | 1 |
| Response Generation | 2000-6000ms | $0.015 | 1 |
| **Total** | **3-10 seconds** | **~$0.02-0.05** | **4-5** |

---

## 🗄️ **Database Design**

### **Enhanced Schema with Knowledge Graph**

#### **Core Document Tables**
```sql
-- Documents table
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    url VARCHAR UNIQUE NOT NULL,
    title VARCHAR NOT NULL,
    content TEXT NOT NULL,
    content_hash VARCHAR NOT NULL,
    scraped_at TIMESTAMP DEFAULT NOW()
);

-- Chunks table with vector support and KG processing status
CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    embedding VECTOR(1536), -- pgvector extension
    kg_processed BOOLEAN DEFAULT FALSE, -- Knowledge Graph processing status
    kg_processed_at TIMESTAMP
);
```

#### **Knowledge Graph Tables**
```sql
-- Entities table - wasmCloud concepts (actors, providers, interfaces, etc.)
CREATE TABLE entities (
    id SERIAL PRIMARY KEY,
    name VARCHAR NOT NULL,
    entity_type VARCHAR NOT NULL, -- actor, provider, interface, capability, application
    description TEXT,
    embedding VECTOR(1536), -- Semantic representation for similarity
    metadata JSONB, -- Additional properties
    first_seen TIMESTAMP DEFAULT NOW(),
    last_updated TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(name, entity_type) -- Prevent duplicates
);

-- Relations table - relationship types with frequency tracking
CREATE TABLE relations (
    id SERIAL PRIMARY KEY,
    name VARCHAR NOT NULL UNIQUE, -- implements, provides, requires, uses, contains, extends
    description TEXT,
    frequency INTEGER DEFAULT 1, -- How often this relation appears
    created_at TIMESTAMP DEFAULT NOW()
);

-- Triplets table - stores (subject, predicate, object) relationships
CREATE TABLE triplets (
    id SERIAL PRIMARY KEY,
    subject_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
    predicate_id INTEGER REFERENCES relations(id) ON DELETE CASCADE,
    object_id INTEGER REFERENCES entities(id) ON DELETE CASCADE,
    confidence FLOAT NOT NULL, -- AI confidence in this relationship (0.0-1.0)
    source_chunk_id INTEGER REFERENCES chunks(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(subject_id, predicate_id, object_id) -- Prevent duplicate relationships
);
```

#### **Enhanced Query Logs**
```sql
-- Query logs with Knowledge Graph tracking
CREATE TABLE query_logs (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    chunks_used INTEGER,
    triplets_used INTEGER DEFAULT 0, -- Number of KG relationships used
    response_time FLOAT,
    search_strategy VARCHAR DEFAULT 'basic', -- basic, advanced, knowledge_graph
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### **Performance Indexes**
```sql
-- Core document indexes
CREATE INDEX idx_chunks_document_id ON chunks(document_id);
CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_chunks_kg_processed ON chunks(kg_processed) WHERE kg_processed = FALSE;
CREATE INDEX idx_documents_url ON documents(url);
CREATE INDEX idx_documents_content_hash ON documents(content_hash);

-- Knowledge Graph indexes
CREATE INDEX idx_entities_name ON entities(name);
CREATE INDEX idx_entities_type ON entities(entity_type);
CREATE INDEX idx_entities_embedding ON entities USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_relations_name ON relations(name);
CREATE INDEX idx_relations_frequency ON relations(frequency DESC);

-- Triplet traversal indexes
CREATE INDEX idx_triplets_subject ON triplets(subject_id);
CREATE INDEX idx_triplets_object ON triplets(object_id);
CREATE INDEX idx_triplets_predicate ON triplets(predicate_id);
CREATE INDEX idx_triplets_confidence ON triplets(confidence DESC);
CREATE INDEX idx_triplets_source_chunk ON triplets(source_chunk_id);

-- Composite indexes for graph traversal
CREATE INDEX idx_triplets_subject_confidence ON triplets(subject_id, confidence DESC);
CREATE INDEX idx_triplets_object_confidence ON triplets(object_id, confidence DESC);

-- Query analytics indexes
CREATE INDEX idx_query_logs_strategy ON query_logs(search_strategy);
CREATE INDEX idx_query_logs_created_at ON query_logs(created_at DESC);
```

### **Vector Search Optimization**

```sql
-- pgvector configuration for optimal performance
SET ivfflat.probes = 10;

-- Vector similarity search query
SELECT 
    c.content,
    d.title,
    d.url,
    (1 - (c.embedding <=> $1)) as similarity
FROM chunks c
JOIN documents d ON c.document_id = d.id
WHERE (1 - (c.embedding <=> $1)) > 0.6
ORDER BY c.embedding <=> $1
LIMIT 10;
```

### **Full-Text Search Setup**

```sql
-- Enable PostgreSQL full-text search
CREATE INDEX idx_chunks_content_fts ON chunks 
USING gin(to_tsvector('english', content));

-- Full-text search query
SELECT 
    c.content,
    d.title,
    d.url,
    ts_rank(to_tsvector('english', c.content), 
            plainto_tsquery('english', $1)) as rank
FROM chunks c
JOIN documents d ON c.document_id = d.id
WHERE to_tsvector('english', c.content) @@ plainto_tsquery('english', $1)
ORDER BY rank DESC
LIMIT 15;
```

---

## ⚙️ **Optimization Features**

### **Configuration System (`server/optimization_config.py`)**

```python
@dataclass
class OptimizationConfig:
    # AI Chunking
    enable_ai_chunking: bool = True
    ai_chunking_threshold: int = 1000
    ai_chunking_model: str = "gpt-3.5-turbo"
    
    # Advanced RAG
    enable_hybrid_search: bool = False
    enable_keyword_search: bool = True
    enable_concept_search: bool = True
    enable_ai_reranking: bool = True
    
    # Performance Settings
    vector_search_k: int = 10
    final_results_k: int = 5
    similarity_threshold: float = 0.6
    max_ai_calls_per_query: int = 3
    
    def get_chunking_strategy(self) -> str:
        return "hybrid" if self.enable_ai_chunking else "traditional"
    
    def get_rag_strategy(self) -> str:
        return "advanced" if self.enable_hybrid_search else "basic"
```

### **Feature Toggle Endpoints**

```python
@app.post("/optimization/enable/{feature}")
def enable_optimization_feature(feature: str):
    valid_features = {
        "ai_chunking": "enable_ai_chunking",
        "hybrid_search": "enable_hybrid_search",
        "keyword_search": "enable_keyword_search",
        "concept_search": "enable_concept_search",
        "ai_reranking": "enable_ai_reranking"
    }
    
    config_key = valid_features[feature]
    update_config(**{config_key: True})
```

---

## 🔌 **API Endpoints**

### **Core Endpoints**

| Method | Endpoint | Description | Parameters |
|--------|----------|-------------|------------|
| GET | `/` | Web interface | - |
| GET | `/health` | Health check | - |
| POST | `/query` | Basic/Advanced query | `{question, include_sources}` |
| POST | `/query/advanced` | Force advanced RAG | `{question, include_sources}` |
| POST | `/ingest` | Trigger doc ingestion | - |
| GET | `/stats` | System statistics | - |
| GET | `/documents` | List documents | `limit, offset` |

### **Optimization Endpoints**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/optimization/config` | Get current config |
| POST | `/optimization/enable/{feature}` | Enable feature |
| POST | `/optimization/disable/{feature}` | Disable feature |

### **Response Examples**

#### **Query Response**
```json
{
    "answer": "To scale wasmCloud actors, you can use...",
    "sources": [
        {
            "title": "wash scale",
            "url": "https://wasmcloud.com/docs/0.82/cli/scale/",
            "similarity": 0.654517501000631,
            "search_type": "vector",
            "rerank_position": 1,
            "concept": null
        }
    ],
    "chunks_used": 5,
    "response_time": 4.23,
    "search_method": "hybrid_advanced"
}
```

#### **Configuration Response**
```json
{
    "status": "success",
    "configuration": {
        "chunking_strategy": "hybrid",
        "rag_strategy": "advanced",
        "ai_chunking_enabled": true,
        "hybrid_search_enabled": true,
        "vector_search_k": 10,
        "final_results_k": 5
    }
}
```

---

## 🔒 **Security & Performance**

### **Security Measures**

```python
# Environment variable protection
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Input validation
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    include_sources: bool = True

# SQL injection prevention
# Using SQLAlchemy ORM with parameterized queries
result = db.execute(text("SELECT * FROM chunks WHERE id = :id"), {"id": chunk_id})
```

### **Performance Optimizations**

#### **Database Performance**
```sql
-- Vector index for fast similarity search
CREATE INDEX idx_chunks_embedding ON chunks 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Connection pooling
engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20)
```

#### **Caching Strategy**
```python
# Embedding caching (future enhancement)
class EmbeddingCache:
    def __init__(self):
        self.cache = {}
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return self.cache.get(text_hash)
    
    def set_embedding(self, text: str, embedding: List[float]):
        text_hash = hashlib.md5(text.encode()).hexdigest()
        self.cache[text_hash] = embedding
```

---

## 🚀 **Deployment & Configuration**

### **Environment Configuration**

```bash
# Core Configuration
OPENAI_API_KEY=your_openai_api_key_here
DATABASE_URL=postgresql://user:pass@localhost:5432/wasmcloud_rag
PORT=8000

# AI Model Configuration
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4-1106-preview
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Optimization Features
ENABLE_AI_CHUNKING=true
ENABLE_HYBRID_SEARCH=false
ENABLE_KEYWORD_SEARCH=true
ENABLE_CONCEPT_SEARCH=true
ENABLE_AI_RERANKING=true

# Performance Settings
VECTOR_SEARCH_K=10
FINAL_RESULTS_K=5
SIMILARITY_THRESHOLD=0.6
MAX_AI_CALLS_PER_QUERY=3
```

### **Docker Deployment**

```yaml
# docker-compose.yml
version: '3.8'
services:
  postgres:
    image: pgvector/pgvector:pg15
    environment:
      POSTGRES_DB: wasmcloud_rag
      POSTGRES_USER: wasmcloud_user
      POSTGRES_PASSWORD: wasmcloud_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://wasmcloud_user:wasmcloud_password@postgres:5432/wasmcloud_rag
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - postgres

volumes:
  postgres_data:
```

---

## 🎯 **Summary**

The wasmCloud RAG Bot represents a sophisticated AI-powered documentation assistant that combines:

- **Modern Web Interface**: Dark mode, responsive design with real-time interaction
- **Intelligent Backend**: FastAPI with PostgreSQL and vector search capabilities  
- **Advanced AI Pipeline**: Multi-strategy search with AI-powered optimization
- **Flexible Configuration**: Runtime feature toggling and cost management
- **Production Ready**: Comprehensive logging, monitoring, and deployment support

The system intelligently adapts its search strategy based on query type, uses multiple AI models for different tasks, and provides comprehensive answers with source attribution, making it an effective tool for wasmCloud documentation assistance.

### **Key Innovations**

1. **Hybrid Search Intelligence**: Automatically selects optimal search strategy
2. **AI-Enhanced Chunking**: Creates semantically coherent document sections
3. **Cost-Aware Design**: Configurable features with transparent cost tracking
4. **Real-time Optimization**: Runtime configuration changes without restart
5. **Comprehensive Observability**: Detailed logging and performance metrics

This design ensures both excellent user experience and operational efficiency while maintaining the flexibility to adapt to changing requirements and costs. 