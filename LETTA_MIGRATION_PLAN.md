# Hetairos → Letta Migration: Comprehensive Implementation Plan

## Executive Summary

**Objective**: Migrate hetairos from proprietary Praxos memory system to open-source Letta (formerly MemGPT) with LangGraph orchestration for sophisticated, stateful AI agent memory.

**Timeline**: 8-12 weeks
**Estimated Cost Savings**: $500-2,000/month
**Key Benefits**:
- True agent learning and memory evolution
- No vendor lock-in
- Full control over infrastructure
- Superior personalization capabilities

---

## Phase 1: Foundation & Setup (Weeks 1-2)

### 1.1 Infrastructure Setup

**Letta Server Deployment**
```bash
# Docker-based deployment (recommended)
docker run -d \
  --name letta-server \
  -p 8283:8283 \
  -v letta-data:/root/.letta \
  -e OPENAI_API_KEY=${OPENAI_API_KEY} \
  -e POSTGRES_URI=postgresql://user:pass@localhost:5432/letta \
  letta/letta:latest
```

**Database Setup**
- PostgreSQL 14+ for agent state and recall memory
- ChromaDB for archival memory (vector store)
- MongoDB for short-term conversation cache (existing)

**Dependencies**
```txt
# Core Letta integration
letta-client==0.4.0
letta-server==0.4.0

# LangGraph orchestration
langgraph==0.2.0
langchain==0.2.0
langchain-openai==0.1.0

# Vector stores
chromadb==0.5.0
pgvector==0.2.0

# Email integration
google-auth-oauthlib==1.2.0
google-api-python-client==2.100.0
msal==1.24.0
requests==2.31.0

# Existing hetairos stack
fastapi==0.104.0
pydantic==2.4.0
motor==3.3.0  # Async MongoDB
redis==5.0.0
azure-servicebus==7.11.0
```

### 1.2 Architecture Design

**System Components**

```
┌─────────────────────────────────────────────────────────────┐
│                    HETAIROS APPLICATION                      │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │           Ingress Layer (FastAPI)                      │ │
│  │  • WhatsApp/Telegram/Email webhooks                    │ │
│  │  • Message normalization                               │ │
│  │  • Authentication & routing                            │ │
│  └────────────────────────────────────────────────────────┘ │
│                            ↓                                 │
│  ┌────────────────────────────────────────────────────────┐ │
│  │        LangGraph Agent Orchestration                   │ │
│  │                                                         │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │ │
│  │  │   Reasoning  │  │ Tool Execute │  │   Memory    │ │ │
│  │  │     Node     │→ │    Node      │→ │    Node     │ │ │
│  │  └──────────────┘  └──────────────┘  └─────────────┘ │ │
│  │         ↓                                      ↓       │ │
│  └─────────┼──────────────────────────────────────┼───────┘ │
│            │                                      │          │
│            ↓                                      ↓          │
│  ┌─────────────────┐                  ┌──────────────────┐ │
│  │  Tools Layer    │                  │  Letta SDK       │ │
│  │  • Email ops    │                  │  • Agent API     │ │
│  │  • Calendar     │                  │  • Memory ops    │ │
│  │  • Search       │                  └──────────────────┘ │
│  └─────────────────┘                           │            │
│                                                 ↓            │
│  ┌────────────────────────────────────────────────────────┐ │
│  │           Egress Layer                                 │ │
│  │  • Multi-channel response formatting                  │ │
│  │  • Platform-specific delivery                         │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│                    LETTA SERVER                              │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              User-Specific Agents                      │ │
│  │                                                         │ │
│  │  Agent_User_1  Agent_User_2  ...  Agent_User_N        │ │
│  │                                                         │ │
│  │  Each agent has:                                       │ │
│  │  • Core Memory (persona, human context)                │ │
│  │  • Recall Memory (conversation history)                │ │
│  │  • Archival Memory (emails, documents)                 │ │
│  │  • Custom tools (email search, calendar)               │ │
│  └────────────────────────────────────────────────────────┘ │
│                            ↓                                 │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              Memory Management Engine                   │ │
│  │  • Context window paging                               │ │
│  │  • Memory consolidation                                │ │
│  │  • Self-editing operations                             │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│                  STORAGE LAYER                               │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ PostgreSQL   │  │  ChromaDB    │  │    MongoDB       │  │
│  │              │  │              │  │                  │  │
│  │ • Agent      │  │ • Embeddings │  │ • Short-term     │  │
│  │   state      │  │ • Archival   │  │   cache          │  │
│  │ • Recall     │  │   memory     │  │ • Session data   │  │
│  │   memory     │  │              │  │                  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Core Abstractions

**Memory Client Interface**
```python
from typing import Protocol, List, Dict, Optional

class MemoryClient(Protocol):
    """Abstract memory interface - supports both Letta and fallback implementations"""

    def add_conversation(
        self,
        messages: List[Dict],
        metadata: Dict
    ) -> str: ...

    def add_email_conversation(
        self,
        email_data: Dict
    ) -> str: ...

    def search_memory(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict]: ...

    def add_file(
        self,
        file_path: str,
        metadata: Dict
    ) -> str: ...

    def get_memory_state(self) -> Dict: ...
```

---

## Phase 2: Core Implementation (Weeks 3-5)

### 2.1 Letta Memory Client

**File**: `src/core/letta_memory_client.py`

Key responsibilities:
- User-specific agent management
- Memory operations (add, search, update)
- Email data source integration
- Conversation archival
- Memory state introspection

### 2.2 LangGraph Integration

**File**: `src/core/langgraph_agent_runner.py`

Graph structure:
```
Entry → Retrieve Memory → Reasoning → Should Continue?
                             ↓             ↙        ↘
                        Tool Execute    Yes        No
                             ↓           ↓          ↓
                        Update Memory ←─┘      Finalize
                                                   ↓
                                                  End
```

Nodes:
1. **Retrieve Memory Node**: Query Letta for relevant context
2. **Reasoning Node**: LLM decides action with memory context
3. **Tool Execute Node**: Execute email/calendar/search tools
4. **Update Memory Node**: Let Letta agent learn from interaction
5. **Finalize Node**: Format response for delivery

### 2.3 Email Integration

**Gmail**:
- Use existing Gmail API integration
- Load emails into Letta data sources
- Index incrementally as new emails arrive

**Outlook**:
- Custom Microsoft Graph API reader
- Similar data source pattern
- Webhook-based incremental indexing

### 2.4 Worker Infrastructure

**Conversation Consolidator**:
- Runs on schedule (e.g., every 5 minutes)
- Transfers MongoDB conversations to Letta agents
- Allows agents to process and learn

**Email Indexer**:
- Background worker for bulk email indexing
- Batches emails for efficiency
- Updates Letta data sources

---

## Phase 3: Testing & Validation (Weeks 6-7)

### 3.1 Unit Testing

Test coverage:
- Letta client operations
- LangGraph node functions
- Email data source creation
- Memory search accuracy
- Tool execution

### 3.2 Integration Testing

Scenarios:
1. **New user onboarding**: Create agent, load initial emails
2. **Conversation flow**: Multi-turn conversation with memory
3. **Email query**: "What did John say about the budget?"
4. **Learning**: Agent updates core memory from interaction
5. **Multi-channel**: Same agent across WhatsApp, Telegram, email

### 3.3 Performance Benchmarking

Metrics:
- Query latency (target: <3s P95)
- Memory retrieval accuracy
- Agent creation time
- Bulk email indexing throughput

### 3.4 Memory Quality Evaluation

Comparisons:
- vs. baseline RAG (LlamaIndex)
- vs. Praxos (if available for comparison)

Criteria:
- Relevance of retrieved memories
- Learning effectiveness
- Personalization quality

---

## Phase 4: Migration Strategy (Weeks 8-10)

### 4.1 Parallel Deployment

**Configuration-based routing**:
```python
# settings.py
MEMORY_BACKEND = "letta"  # or "praxos" for fallback

# Factory pattern
def get_memory_client(user_id: str) -> MemoryClient:
    if settings.MEMORY_BACKEND == "letta":
        return LettaMemoryClient(user_id)
    elif settings.MEMORY_BACKEND == "praxos":
        return PraxosClient(user_id)
    else:
        raise ValueError(f"Unknown backend: {settings.MEMORY_BACKEND}")
```

**A/B Testing**:
- 10% of users → Letta
- 90% of users → Praxos (or existing system)
- Compare metrics for 2 weeks

### 4.2 Data Migration

**For users with existing Praxos data**:

```python
async def migrate_user_from_praxos(user_id: str):
    """One-time migration from Praxos to Letta"""

    # Export from Praxos
    praxos = PraxosClient(user_id)
    conversations = praxos.export_conversations()
    emails = praxos.export_emails()
    files = praxos.export_files()

    # Create Letta agent
    letta = LettaMemoryClient(user_id)

    # Seed agent memory
    for conv in conversations:
        letta.add_conversation(
            messages=conv['messages'],
            metadata=conv['metadata']
        )

    for email in emails:
        letta.add_email_conversation(email)

    for file in files:
        letta.add_file(
            file_path=file['path'],
            metadata=file['metadata']
        )

    # Mark user as migrated
    await mark_user_migrated(user_id, backend="letta")
```

### 4.3 Gradual Rollout

**Week 8**:
- 10% of users on Letta
- Monitor errors, performance, user feedback

**Week 9**:
- 50% of users on Letta
- Compare retention, engagement metrics

**Week 10**:
- 100% of users on Letta
- Keep Praxos on hot standby for 1 week

**Week 11**:
- Decommission Praxos
- Archive data for compliance

---

## Phase 5: Optimization & Scaling (Weeks 11-12)

### 5.1 Performance Tuning

**Embedding model selection**:
- Development: `text-embedding-3-small` (fast, cheap)
- Production: `text-embedding-3-large` (better quality)
- Local option: `BAAI/bge-large-en-v1.5` (privacy-focused)

**Chunking optimization**:
```python
# Email-specific chunking
from llama_index.core.node_parser import SentenceSplitter

email_splitter = SentenceSplitter(
    chunk_size=512,      # Emails are naturally concise
    chunk_overlap=128,   # 25% overlap
    paragraph_separator="\n\n"
)
```

**Caching strategy**:
- LLM response caching for common queries
- Embedding caching for repeated documents
- Agent state caching in Redis

### 5.2 Monitoring & Observability

**Metrics to track**:
- Agent query latency (P50, P95, P99)
- Memory operation success rate
- Tool execution time
- User satisfaction (via feedback)
- Cost per query

**Logging**:
```python
import structlog

logger = structlog.get_logger()

logger.info(
    "memory_search_completed",
    user_id=user_id,
    query=query,
    results_count=len(results),
    latency_ms=latency,
    memories_accessed=memories_accessed
)
```

**Alerting**:
- Query latency > 5s
- Memory operation failure rate > 1%
- Agent creation failures
- Database connection issues

### 5.3 Cost Optimization

**LLM usage**:
- Use GPT-3.5-turbo for simple queries
- Reserve GPT-4 for complex reasoning
- Consider local LLM (Ollama + Mistral) for sensitive data

**Embedding optimization**:
- Batch embedding generation
- Incremental indexing (only new emails)
- Embedding dimension reduction for speed

**Infrastructure**:
- Right-size PostgreSQL and ChromaDB
- Use connection pooling
- Implement query result caching

---

## Implementation Priorities

### Must-Have (MVP)
1. ✅ Letta server deployment
2. ✅ LettaMemoryClient with core operations
3. ✅ LangGraph basic workflow (retrieve → reason → respond)
4. ✅ Email indexing for Gmail
5. ✅ Conversation consolidation worker
6. ✅ Basic testing

### Should-Have (V1)
1. Outlook email integration
2. Advanced LangGraph workflows (tool chaining)
3. Memory consolidation ("sleep time" compute)
4. Comprehensive testing suite
5. Monitoring and alerting
6. A/B testing infrastructure

### Nice-to-Have (V2)
1. Multi-agent coordination
2. Advanced personalization
3. Local LLM support
4. Embedding fine-tuning
5. Advanced analytics dashboard
6. Agent behavior customization UI

---

## Risk Mitigation

### Technical Risks

**Risk**: Letta server instability
- **Mitigation**: Keep fallback to simple RAG (LlamaIndex)
- **Mitigation**: Run redundant Letta instances

**Risk**: Migration data loss
- **Mitigation**: Parallel run with verification
- **Mitigation**: Maintain Praxos backup for 30 days

**Risk**: Performance degradation
- **Mitigation**: Extensive load testing before rollout
- **Mitigation**: Progressive rollout with rollback plan

### Operational Risks

**Risk**: Team unfamiliarity with Letta
- **Mitigation**: Training sessions and documentation
- **Mitigation**: Start with small pilot group

**Risk**: Cost overruns (LLM API usage)
- **Mitigation**: Set usage budgets and alerts
- **Mitigation**: Consider local LLM fallback

---

## Success Metrics

### Technical KPIs
- Query latency < 3s (P95)
- Memory retrieval accuracy > 85%
- System uptime > 99.5%
- Zero data loss during migration

### Business KPIs
- Cost savings: $500-2,000/month
- User satisfaction: > 4.5/5 stars
- Agent learning rate: Measurable improvement in personalization
- Development velocity: Faster feature development post-migration

### User Experience KPIs
- Response relevance: > 90% (via user feedback)
- Context retention: > 95% (agent remembers key facts)
- Personalization quality: Qualitative improvement

---

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| 1. Foundation | 2 weeks | Infrastructure, architecture, dependencies |
| 2. Core Implementation | 3 weeks | Letta client, LangGraph integration, email indexing |
| 3. Testing | 2 weeks | Unit tests, integration tests, benchmarks |
| 4. Migration | 3 weeks | A/B testing, data migration, rollout |
| 5. Optimization | 2 weeks | Performance tuning, monitoring, cost optimization |
| **Total** | **12 weeks** | Production-ready Letta-based system |

---

## Next Steps

1. **Week 1 Day 1-2**: Deploy Letta server and validate connectivity
2. **Week 1 Day 3-5**: Implement LettaMemoryClient skeleton
3. **Week 2**: Integrate with LangGraph basic workflow
4. **Week 3**: Email indexing implementation
5. **Week 4**: End-to-end testing with sample user

---

## Appendix: Decision Log

**Why Letta over LlamaIndex?**
- Need for stateful agents that learn
- Superior context window management
- Self-editing memory capabilities
- Better suited for conversational AI

**Why LangGraph + Letta hybrid?**
- LangGraph: Excellent workflow orchestration
- Letta: Excellent memory management
- Combined: Best of both worlds

**Why self-hosted Letta vs. cloud?**
- Full control over data and infrastructure
- Cost predictability
- Customization flexibility
- No vendor lock-in

**Why PostgreSQL + ChromaDB?**
- PostgreSQL: Mature, reliable, excellent for structured data
- ChromaDB: Simple, effective, easy to self-host
- Both open-source with strong communities

---

## Resources

- Letta Documentation: https://docs.letta.com
- LangGraph Documentation: https://python.langchain.com/docs/langgraph
- MemGPT Paper: https://arxiv.org/abs/2310.08560
- Hetairos Repository: (this repo)

---

**Document Version**: 1.0
**Last Updated**: 2025-10-25
**Author**: Claude Code
**Status**: READY FOR IMPLEMENTATION
