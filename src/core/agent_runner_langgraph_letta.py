"""
LangGraph agent runner with Letta memory integration.

Shows how to integrate Letta memory retrieval into hetairos LangGraphAgentRunner.
"""

import structlog
from typing import List
from src.core.letta_memory_client_v2 import LettaMemoryClient
from src.config.settings import settings

logger = structlog.get_logger()


class LangGraphAgentRunnerWithLetta:
    """
    Extension of hetairos LangGraphAgentRunner with Letta memory integration.

    This is a reference implementation showing how to replace the Praxos
    _get_long_term_memory method with Letta-based memory retrieval.
    """

    async def _get_long_term_memory(
        self,
        user_context,
        input_text: str
    ) -> str:
        """
        Fetch long-term memory using Letta instead of Praxos.

        This replaces the original hetairos method:
        ```python
        async def _get_long_term_memory(
            self,
            praxos_client: PraxosClient,
            input_text: str
        ) -> List[str]:
            praxos_history = await praxos_client.search_memory(input_text, 10)
            long_term_memory_context = ''
            for i, itm in enumerate(praxos_history['sentences']):
                long_term_memory_context += f"Context Info{i+1}: {itm}\n"
            if long_term_memory_context:
                long_term_memory_context = (
                    "\n\nThe following relevant information is known about "
                    "this user from their long-term memory:\n" +
                    long_term_memory_context
                )
            return long_term_memory_context
        ```

        Args:
            user_context: User context object
            input_text: User's input query

        Returns:
            Formatted memory context string
        """
        try:
            # Create Letta client for user
            user_email = user_context.user_record.get('email', f"user_{user_context.user_id}")
            env_name = f"env_for_{user_email}"

            # Get API key (per-user or global)
            if settings.memory_backend == "letta":
                letta_api_key = settings.letta_api_key
            else:
                letta_api_key = user_context.user_record.get("letta_api_key") or settings.letta_api_key

            letta_client = LettaMemoryClient(
                environment_name=env_name,
                api_key=letta_api_key
            )

            # Search Letta memory
            letta_history = await letta_client.search_memory(
                query=input_text,
                top_k=10
            )

            # Format memory context (matching original format)
            long_term_memory_context = ''

            for i, itm in enumerate(letta_history.get('sentences', [])):
                long_term_memory_context += f"Context Info{i+1}: {itm}\n"

            if long_term_memory_context:
                long_term_memory_context = (
                    "\n\nThe following relevant information is known about "
                    "this user from their long-term memory:\n" +
                    long_term_memory_context
                )

            logger.info(
                "long_term_memory_retrieved",
                user_id=user_context.user_id,
                results_count=len(letta_history.get('sentences', []))
            )

            return long_term_memory_context

        except Exception as e:
            logger.error(
                "long_term_memory_retrieval_failed",
                user_id=user_context.user_id if user_context else "unknown",
                error=str(e),
                exc_info=True
            )
            return ""  # Return empty string on failure

    def _create_letta_client_for_user(self, user_context) -> LettaMemoryClient:
        """
        Helper method to create Letta client for a user.

        Can be called from anywhere in the agent runner where you need
        to access long-term memory.

        Args:
            user_context: User context object

        Returns:
            Configured LettaMemoryClient instance
        """
        user_email = user_context.user_record.get(
            'email',
            f"user_{user_context.user_id}"
        )
        env_name = f"env_for_{user_email}"

        # Get API key
        if settings.memory_backend == "letta":
            letta_api_key = settings.letta_api_key
        else:
            letta_api_key = (
                user_context.user_record.get("letta_api_key") or
                settings.letta_api_key
            )

        return LettaMemoryClient(
            environment_name=env_name,
            api_key=letta_api_key
        )


# ============================================================================
# Integration Example for Existing Hetairos Code
# ============================================================================

"""
To integrate Letta into existing hetairos LangGraphAgentRunner:

1. Update imports:
```python
# Replace:
from src.core.praxos_client import PraxosClient

# With:
from src.core.letta_memory_client_v2 import LettaMemoryClient
from src.config.settings import settings
```

2. Replace _get_long_term_memory method:

```python
async def _get_long_term_memory(self, user_context, input_text: str) -> str:
    '''Fetch long-term memory using Letta.'''

    # Create user-specific environment name
    user_email = user_context.user_record.get('email', f"user_{user_context.user_id}")
    env_name = f"env_for_{user_email}"

    # Get API key (check settings for backend selection)
    if settings.memory_backend == "letta":
        letta_api_key = settings.letta_api_key
    else:
        letta_api_key = user_context.user_record.get("letta_api_key") or settings.letta_api_key

    # Create Letta client
    letta_client = LettaMemoryClient(
        environment_name=env_name,
        api_key=letta_api_key
    )

    # Search memory
    letta_history = await letta_client.search_memory(input_text, 10)

    # Format results (same format as Praxos)
    long_term_memory_context = ''
    for i, itm in enumerate(letta_history.get('sentences', [])):
        long_term_memory_context += f"Context Info{i+1}: {itm}\n"

    if long_term_memory_context:
        long_term_memory_context = (
            "\n\nThe following relevant information is known about "
            "this user from their long-term memory:\n" +
            long_term_memory_context
        )

    return long_term_memory_context
```

3. Update run() method call site:

```python
# In the run() method, around line 121-124, replace:
# praxos_client = PraxosClient(
#     environment_name=f"user_{user_context.user_id}",
#     api_key=settings.PRAXOS_API_KEY
# )
# memory_context = await self._get_long_term_memory(praxos_client, input_text)

# With:
memory_context = await self._get_long_term_memory(user_context, input_text)
```

4. Update settings.py:

```python
# Add to Settings class:
MEMORY_BACKEND = os.getenv("MEMORY_BACKEND", "letta")  # or "praxos"
LETTA_SERVER_URL = os.getenv("LETTA_SERVER_URL", "http://localhost:8283")
LETTA_API_KEY = os.getenv("LETTA_API_KEY")  # Can be None for local server
```

5. Update .env file:

```bash
# Memory backend selection
MEMORY_BACKEND=letta  # Options: letta, praxos

# Letta configuration
LETTA_SERVER_URL=http://localhost:8283
LETTA_API_KEY=  # Leave empty for local server
```

That's it! The agent runner now uses Letta for long-term memory while maintaining
the same API surface and behavior.
"""
