# Hetairos Praxos → Letta Migration Guide

## Overview

This guide provides step-by-step instructions for migrating your Hetairos deployment from Praxos to Letta for long-term agent memory.

**Benefits of Migration:**
- 💰 Cost savings: $500-2,000/month → $0-500/month
- 🔓 No vendor lock-in
- 🧠 Superior agent learning and memory capabilities
- 🔧 Full control over infrastructure
- 📈 Better personalization over time

**Timeline:** 8-12 weeks for complete migration
**Risk Level:** Low (parallel deployment with rollback capability)

---

## Prerequisites

### Required

- Python 3.10+
- Docker and Docker Compose
- PostgreSQL 14+ (for Letta state storage)
- Existing Hetairos deployment
- OpenAI API key (for embeddings and LLM)

### Optional

- Kubernetes cluster (for production deployment)
- Local LLM setup (Ollama + Mistral for privacy)

---

## Phase 1: Infrastructure Setup (Week 1-2)

### Step 1: Deploy Letta Server

#### Option A: Docker Compose (Recommended for Development)

```bash
# Clone this repository
git clone https://github.com/your-org/agents-jetson-oss.git
cd agents-jetson-oss

# Copy environment template
cp .env.example .env

# Edit .env with your configuration
nano .env

# Start Letta stack
docker-compose up -d

# Verify services are running
docker-compose ps
```

Expected output:
```
NAME                     STATUS     PORTS
hetairos-letta-server    Up         0.0.0.0:8283->8283/tcp
hetairos-postgres        Up         0.0.0.0:5432->5432/tcp
hetairos-chroma          Up         0.0.0.0:8000->8000/tcp
hetairos-mongodb         Up         0.0.0.0:27017->27017/tcp
hetairos-redis           Up         0.0.0.0:6379->6379/tcp
```

#### Option B: Kubernetes (Production)

```bash
# Apply Kubernetes manifests (create these based on your k8s setup)
kubectl apply -f k8s/letta-deployment.yaml
kubectl apply -f k8s/postgres-deployment.yaml
kubectl apply -f k8s/chroma-deployment.yaml
```

#### Option C: Letta Cloud (Managed Service)

```bash
# Sign up at https://www.letta.com/cloud
# Get API key and endpoint
# Update .env:
LETTA_SERVER_URL=https://api.letta.com
LETTA_API_KEY=your_cloud_api_key
```

### Step 2: Verify Letta Installation

```bash
# Test Letta server
curl http://localhost:8283/health

# Should return:
# {"status": "ok"}

# Test with Python
python -c "
from pymemgpt import create_client
client = create_client(base_url='http://localhost:8283')
agents = client.list_agents()
print(f'Letta server connected. Agents: {len(agents)}')
"
```

### Step 3: Update Hetairos Configuration

```python
# In hetairos/src/config/settings.py, add:

class Settings:
    # ... existing settings ...

    # Memory backend selection
    MEMORY_BACKEND = os.getenv("MEMORY_BACKEND", "praxos")  # Start with praxos

    # Letta configuration
    LETTA_SERVER_URL = os.getenv("LETTA_SERVER_URL", "http://localhost:8283")
    LETTA_API_KEY = os.getenv("LETTA_API_KEY")  # Can be None for local

    # ... rest of settings ...
```

Update `.env`:
```bash
# Add these lines
MEMORY_BACKEND=praxos  # Keep praxos for now, will switch later
LETTA_SERVER_URL=http://localhost:8283
LETTA_API_KEY=  # Empty for local server
```

---

## Phase 2: Code Integration (Week 3-4)

### Step 1: Install Dependencies

```bash
# In your hetairos project
cd hetairos

# Add to requirements.txt:
echo "letta>=0.4.0" >> requirements.txt
echo "pymemgpt>=0.3.0" >> requirements.txt
echo "structlog>=24.0.0" >> requirements.txt

# Install
pip install -r requirements.txt
```

### Step 2: Add Letta Memory Client

```bash
# Copy Letta client from this repo
cp ../agents-jetson-oss/src/core/letta_memory_client_v2.py \
   hetairos/src/core/letta_memory_client.py

# Or create symlink for easy updates
ln -s $(pwd)/../agents-jetson-oss/src/core/letta_memory_client_v2.py \
      hetairos/src/core/letta_memory_client.py
```

### Step 3: Update LangGraph Agent Runner

Edit `hetairos/src/core/agent_runner_langgraph.py`:

```python
# Add import at top
from src.core.letta_memory_client import LettaMemoryClient
from src.config.settings import settings

# Replace _get_long_term_memory method (around line 69-77)
async def _get_long_term_memory(self, user_context, input_text: str) -> str:
    """Fetch long-term memory using configured backend."""

    # Get user email for environment name
    user_email = user_context.user_record.get('email', f"user_{user_context.user_id}")
    env_name = f"env_for_{user_email}"

    # Check backend configuration
    if settings.MEMORY_BACKEND == "letta":
        # Use Letta
        letta_api_key = (
            user_context.user_record.get("letta_api_key") or
            settings.LETTA_API_KEY
        )

        letta_client = LettaMemoryClient(
            environment_name=env_name,
            api_key=letta_api_key
        )

        letta_history = await letta_client.search_memory(input_text, 10)
        sentences = letta_history.get('sentences', [])

    else:
        # Use Praxos (existing code)
        from src.core.praxos_client import PraxosClient

        if settings.OPERATING_MODE == "local":
            praxos_api_key = settings.PRAXOS_API_KEY
        else:
            praxos_api_key = user_context.user_record.get("praxos_api_key")

        praxos_client = PraxosClient(env_name, api_key=praxos_api_key)
        praxos_history = await praxos_client.search_memory(input_text, 10)
        sentences = praxos_history.get('sentences', [])

    # Format memory context (same for both backends)
    long_term_memory_context = ''
    for i, itm in enumerate(sentences):
        long_term_memory_context += f"Context Info{i+1}: {itm}\n"

    if long_term_memory_context:
        long_term_memory_context = (
            "\n\nThe following relevant information is known about "
            "this user from their long-term memory:\n" +
            long_term_memory_context
        )

    return long_term_memory_context
```

### Step 4: Update Conversation Consolidator

Edit `hetairos/src/workers/conversation_consolidator.py`:

```python
# Add imports at top
from src.core.letta_memory_client import LettaMemoryClient
from src.config.settings import settings

# Update consolidate_conversation method (around line 77-92)
async def consolidate_conversation(self, conversation_id: int) -> bool:
    # ... existing code to get conversation and messages ...

    # Check backend configuration
    if settings.MEMORY_BACKEND == "letta":
        # Use Letta
        letta_api_key = user_record.get("letta_api_key") or settings.LETTA_API_KEY

        letta_client = LettaMemoryClient(
            environment_name=env_name,
            api_key=letta_api_key
        )

        source_data = await letta_client.add_conversation(
            user_id=conversation_user_id,
            source='conversation_summary',
            messages=messages,
            metadata={
                'conversation_id': conversation_id,
                'message_count': len(messages),
                'platform': conversation['platform'],
                'start_time': conversation['start_time'],
                'end_time': conversation['last_activity']
            },
            user_record=user_record,
            conversation_id=conversation_id
        )

    else:
        # Use Praxos (existing code)
        praxos_client = PraxosClient(env_name, api_key=praxos_api_key)

        source_data = await praxos_client.add_conversation(
            user_id=conversation_user_id,
            source='conversation_summary',
            messages=messages,
            metadata={ ... },
            user_record=user_record,
            conversation_id=conversation_id
        )

    # ... rest of existing code ...
```

---

## Phase 3: Testing (Week 5-6)

### Step 1: Create Test User

```python
# Create test script: test_letta_integration.py

import asyncio
from src.core.letta_memory_client import LettaMemoryClient

async def test_letta():
    # Create test agent
    client = LettaMemoryClient(
        environment_name="env_for_test@example.com",
        api_key=None  # Local server
    )

    # Test conversation addition
    result = await client.add_conversation(
        user_id="test_user",
        source="test",
        messages=[
            {"role": "user", "content": "Hello, my name is Alice"},
            {"role": "assistant", "content": "Nice to meet you, Alice!"}
        ],
        metadata={},
        user_record={"first_name": "Alice", "last_name": "Test"},
        conversation_id="test_conv_1"
    )

    print(f"Conversation added: {result}")

    # Test memory search
    search_result = await client.search_memory("What is my name?")
    print(f"Search results: {search_result}")

if __name__ == "__main__":
    asyncio.run(test_letta())
```

Run test:
```bash
python test_letta_integration.py
```

Expected output:
```
Conversation added: {'success': True, 'id': 'msg_xxx', 'source': ...}
Search results: {'success': True, 'sentences': ['...'], 'count': ...}
```

### Step 2: Parallel Testing with Real User

```bash
# Update .env for a single test user
TEST_USER_EMAIL=your_email@example.com
TEST_USER_MEMORY_BACKEND=letta  # Just for this user
```

Update code to check per-user backend:
```python
# In _get_long_term_memory:
user_email = user_context.user_record.get('email')

# Check if this user is testing Letta
if user_email == settings.TEST_USER_EMAIL:
    backend = "letta"
else:
    backend = settings.MEMORY_BACKEND  # Global default
```

### Step 3: Compare Results

Create comparison script:
```python
# compare_backends.py

async def compare_memory_search(query, user_context):
    # Search with Praxos
    praxos_result = await get_praxos_memory(user_context, query)

    # Search with Letta
    letta_result = await get_letta_memory(user_context, query)

    # Compare
    print(f"Query: {query}")
    print(f"Praxos results: {len(praxos_result.get('sentences', []))}")
    print(f"Letta results: {len(letta_result.get('sentences', []))}")

    # Check for similar content
    # (implementation depends on your needs)
```

---

## Phase 4: Data Migration (Week 7-8)

### Step 1: Export Praxos Data

```python
# export_praxos_data.py

import asyncio
from src.core.praxos_client import PraxosClient

async def export_user_data(user_email, praxos_api_key):
    env_name = f"env_for_{user_email}"
    client = PraxosClient(env_name, api_key=praxos_api_key)

    # Export conversations (if Praxos supports export)
    # This depends on Praxos API - you may need to contact support
    conversations = await client.export_all_data()  # If available

    # Save to file
    import json
    with open(f"praxos_export_{user_email}.json", "w") as f:
        json.dump(conversations, f, indent=2)

    print(f"Exported data for {user_email}")
```

### Step 2: Import into Letta

```python
# import_to_letta.py

import asyncio
import json
from src.core.letta_memory_client import LettaMemoryClient

async def import_user_data(user_email, data_file):
    env_name = f"env_for_{user_email}"
    client = LettaMemoryClient(env_name, api_key=None)

    # Load exported data
    with open(data_file) as f:
        data = json.load(f)

    # Import conversations
    for conversation in data.get('conversations', []):
        await client.add_conversation(
            user_id=conversation['user_id'],
            source='praxos_import',
            messages=conversation['messages'],
            metadata=conversation.get('metadata', {}),
            user_record={},
            conversation_id=conversation['id']
        )

    print(f"Imported {len(data['conversations'])} conversations for {user_email}")
```

Run migration:
```bash
python export_praxos_data.py
python import_to_letta.py
```

---

## Phase 5: Gradual Rollout (Week 9-10)

### Week 9: 10% of users on Letta

```python
# In settings.py
LETTA_ROLLOUT_PERCENTAGE = 10  # 10% of users

# In _get_long_term_memory:
import random
user_hash = hash(user_context.user_id)
if (user_hash % 100) < settings.LETTA_ROLLOUT_PERCENTAGE:
    backend = "letta"
else:
    backend = "praxos"
```

Monitor:
- Error rates
- Query latency
- User feedback
- Memory retrieval accuracy

### Week 10: 50% of users on Letta

```python
LETTA_ROLLOUT_PERCENTAGE = 50
```

Continue monitoring. If issues arise, roll back:
```python
LETTA_ROLLOUT_PERCENTAGE = 10  # or 0
```

### Week 11: 100% on Letta

```bash
# Update .env
MEMORY_BACKEND=letta
```

Remove rollout percentage logic. Keep Praxos running for 1 week as hot standby.

### Week 12: Decommission Praxos

```bash
# Remove Praxos imports
# Cancel Praxos subscription
# Archive Praxos data for compliance (30-90 days)
```

---

## Troubleshooting

### Letta Server Not Starting

```bash
# Check logs
docker-compose logs letta-server

# Common issues:
# 1. PostgreSQL not ready
docker-compose restart postgres
sleep 10
docker-compose restart letta-server

# 2. Port conflict
# Change port in docker-compose.yml
```

### Memory Search Returns Empty Results

```python
# Verify agent has archival memory
client = LettaMemoryClient(...)
agent = client.client.get_agent(client.agent_id)
print(f"Archival memory size: {agent.archival_memory_size}")

# If zero, data wasn't indexed
# Re-run consolidation
```

### High Query Latency

```bash
# Check embedding model
# Switch to faster model in .env:
EMBEDDING_MODEL=text-embedding-3-small  # Faster

# Or use local embeddings:
EMBEDDING_MODEL=local
EMBEDDING_MODEL_PATH=BAAI/bge-small-en-v1.5
```

---

## Rollback Plan

If critical issues occur:

### Emergency Rollback (< 1 hour)

```bash
# 1. Switch backend immediately
export MEMORY_BACKEND=praxos

# 2. Restart hetairos services
# (method depends on your deployment)

# 3. Verify Praxos is responding
curl -X POST https://api.mypraxos.com/health
```

### Planned Rollback (1-2 days)

1. Set `LETTA_ROLLOUT_PERCENTAGE = 0`
2. Monitor for 24 hours
3. Document issues
4. Plan fixes
5. Re-attempt rollout

---

## Cost Analysis

### Praxos Monthly Cost (Estimated)

- Free tier: 1,000 credits/day
- Paid: $500-2,000/month (usage-based)

### Letta Monthly Cost

**Option 1: Fully Local (Recommended)**
- Letta Server: $0 (open source)
- Compute: $50-200/month (cloud VM or on-premise)
- PostgreSQL: $0-50/month
- ChromaDB: $0 (self-hosted)
- LLM API (OpenAI): $100-500/month
- **Total: $150-750/month**

**Option 2: Letta Cloud**
- Pricing: ~$100-500/month (estimated, contact Letta)
- **Total: $100-500/month**

**Savings: $350-1,500/month** 🎉

---

## Support

- Letta Documentation: https://docs.letta.com
- Letta Discord: https://discord.gg/letta
- This Repository Issues: https://github.com/your-org/agents-jetson-oss/issues

---

**Last Updated:** 2025-10-25
**Version:** 1.0
