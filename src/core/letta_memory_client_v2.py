"""
Letta-based memory client matching hetairos PraxosClient API.

This implementation provides drop-in replacement for PraxosClient using Letta
for sophisticated agent memory management.
"""

import structlog
import time
from typing import List, Dict, Optional, Any
from datetime import datetime
from pymemgpt import create_client
from pymemgpt.client.client import RESTClient

logger = structlog.get_logger()


class LettaMemoryClient:
    """
    Drop-in replacement for PraxosClient using Letta.

    Provides same API surface as PraxosClient but uses Letta's self-editing
    agent memory architecture instead of proprietary Praxos knowledge graph.
    """

    def __init__(self, environment_name: str = None, api_key: str = None):
        """
        Initialize Letta client.

        Args:
            environment_name: User-specific environment (e.g., "env_for_user@email.com")
            api_key: Letta API key (can be None for local server)
        """
        self.environment_name = environment_name
        self.api_key = api_key

        # Extract user identifier from environment name
        # Format: "env_for_{email}" -> use email as user_id
        self.user_id = environment_name.replace("env_for_", "") if environment_name else "default"

        try:
            # Connect to Letta server
            from src.config.settings import settings
            self.client: RESTClient = create_client(
                base_url=settings.letta_server_url,
                token=api_key
            )

            # Get or create user-specific agent
            self.agent_id = self._get_or_create_agent()
            self.email_source_id = self._get_or_create_email_source()

            logger.info(
                "letta_client_initialized",
                environment=environment_name,
                agent_id=self.agent_id
            )

        except Exception as e:
            logger.error(
                "letta_client_init_failed",
                environment=environment_name,
                error=str(e)
            )
            raise

    def _get_or_create_agent(self) -> str:
        """Get or create Letta agent for this environment."""
        agent_name = f"hetairos_{self.environment_name}"

        try:
            # Check existing agents
            agents = self.client.list_agents()
            for agent in agents:
                if agent.name == agent_name:
                    logger.info("found_existing_agent", agent_id=agent.id)
                    return agent.id

            # Create new agent
            logger.info("creating_new_agent", environment=self.environment_name)

            agent = self.client.create_agent(
                name=agent_name,
                memory={
                    "persona": {
                        "value": self._get_agent_persona(),
                        "limit": 2000
                    },
                    "human": {
                        "value": self._get_default_human_context(),
                        "limit": 2000
                    }
                },
                system=self._get_system_prompt(),
                tools=self._get_agent_tools()
            )

            logger.info("agent_created", agent_id=agent.id)
            return agent.id

        except Exception as e:
            logger.error("agent_creation_failed", error=str(e))
            raise

    def _get_or_create_email_source(self) -> str:
        """Get or create email data source for archival memory."""
        source_name = f"emails_{self.user_id}"

        try:
            sources = self.client.list_sources()
            for source in sources:
                if source.name == source_name:
                    return source.id

            source = self.client.create_source(
                name=source_name,
                description=f"Email archive for {self.user_id}"
            )

            # Attach to agent
            self.client.attach_source_to_agent(
                agent_id=self.agent_id,
                source_id=source.id
            )

            return source.id

        except Exception as e:
            logger.error("email_source_creation_failed", error=str(e))
            raise

    def _get_agent_persona(self) -> str:
        """Get agent persona matching hetairos behavior."""
        return """I am the Hetairos AI assistant.

Capabilities:
- Email management (Gmail, Outlook)
- Calendar coordination
- Multi-channel communication (WhatsApp, Telegram, Email)
- Task automation
- Information retrieval from conversation history

I actively learn from interactions and update my memory of user preferences.
"""

    def _get_default_human_context(self) -> str:
        """Get default human context block."""
        return f"""User: {self.user_id}

What I know:
- Just started interaction
- No preferences learned yet

I will learn and update this as we interact.
"""

    def _get_system_prompt(self) -> str:
        """Get system prompt for agent."""
        return """You are a personal AI assistant with advanced memory.

Memory types:
- Core Memory: Edit with core_memory_append/core_memory_replace
- Recall Memory: Search with conversation_search
- Archival Memory: Search emails/docs with archival_memory_search

Always:
- Update core memory when learning important facts
- Be concise and helpful
- Use tools to accomplish tasks
"""

    def _get_agent_tools(self) -> List[str]:
        """Get tools available to agent."""
        return [
            "send_message",
            "core_memory_append",
            "core_memory_replace",
            "conversation_search",
            "conversation_search_date",
            "archival_memory_insert",
            "archival_memory_search",
        ]

    # =========================================================================
    # API Methods Matching PraxosClient
    # =========================================================================

    async def add_conversation(
        self,
        user_id: str,
        source: str,
        metadata: Dict = None,
        user_record: Dict[str, Any] = None,
        messages: List[Dict] = None,
        conversation_id: str = 'no_conversation_id'
    ) -> Dict:
        """
        Add conversation to Letta agent's memory.

        Matches PraxosClient.add_conversation API.

        Args:
            user_id: User identifier
            source: Source platform (whatsapp, telegram, etc.)
            metadata: Additional metadata
            user_record: User profile information
            messages: List of message dicts with role, content, timestamp
            conversation_id: Conversation identifier

        Returns:
            Dict with success status and id
        """
        if not messages:
            return {"error": "No messages provided"}

        start_time = time.time()
        logger.info(
            "adding_conversation",
            user_id=user_id,
            source=source,
            message_count=len(messages)
        )

        try:
            # Format conversation for agent
            conversation_text = self._format_conversation_for_letta(
                messages=messages,
                user_record=user_record,
                source=source,
                conversation_id=conversation_id
            )

            # Send to agent for processing
            response = self.client.send_message(
                agent_id=self.agent_id,
                message=conversation_text,
                role="system"
            )

            duration = time.time() - start_time

            logger.info(
                "conversation_added",
                user_id=user_id,
                conversation_id=conversation_id,
                duration_sec=duration
            )

            return {
                "success": True,
                "id": response.id,
                "source": response
            }

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                "conversation_add_failed",
                user_id=user_id,
                error=str(e),
                duration_sec=duration
            )
            return {"error": str(e)}

    async def add_email_conversation(
        self,
        messages: List,
        name: str,
        description: str,
        metadata: Dict = None,
        user_record: Dict[str, Any] = None
    ) -> Dict:
        """
        Add email as conversation using Message objects.

        Matches PraxosClient.add_email_conversation API.

        Args:
            messages: List of Message objects or dicts
            name: Email identifier/subject
            description: Email description
            metadata: Additional metadata
            user_record: User profile

        Returns:
            Dict with success status and id
        """
        start_time = time.time()
        logger.info("adding_email_conversation", name=name)

        try:
            # Format email for archival memory
            email_text = self._format_email_for_archival(
                messages=messages,
                name=name,
                description=description,
                metadata=metadata
            )

            # Add to archival memory
            passage = self.client.insert_archival_memory(
                agent_id=self.agent_id,
                memory=email_text
            )

            duration = time.time() - start_time

            logger.info(
                "email_conversation_added",
                name=name,
                duration_sec=duration
            )

            return {
                "success": True,
                "id": passage.id,
                "source": passage
            }

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                "email_conversation_add_failed",
                name=name,
                error=str(e),
                duration_sec=duration
            )
            return {"error": str(e)}

    async def search_memory(
        self,
        query: str,
        top_k: int = 10,
        search_modality: str = "node_vec",
        exclude_seen: List[str] = None
    ) -> Dict:
        """
        Search agent's memory.

        Matches PraxosClient.search_memory API.

        Args:
            query: Search query
            top_k: Number of results
            search_modality: Search type (unused in Letta, kept for compatibility)
            exclude_seen: Node IDs to exclude (unused in Letta)

        Returns:
            Dict with sentences and results matching Praxos format
        """
        start_time = time.time()
        logger.info("searching_memory", query=query, top_k=top_k)

        try:
            # Query agent
            response = self.client.send_message(
                agent_id=self.agent_id,
                message=f"Search your memory for: {query}",
                role="user"
            )

            # Extract archival memory searches from response
            sentences = []
            results = []

            # Parse agent's memory accesses
            for msg in response.messages:
                if hasattr(msg, 'tool_calls'):
                    for tool_call in msg.tool_calls:
                        if tool_call.function.name == 'archival_memory_search':
                            # Extract search results
                            # (Implementation depends on Letta's response format)
                            pass

            duration = time.time() - start_time

            logger.info(
                "memory_search_completed",
                query=query,
                results_count=len(sentences),
                duration_sec=duration
            )

            # Match Praxos response format
            return {
                "success": True,
                "sentences": sentences,
                "results": results,
                "count": len(results),
                "sentences_count": len(sentences)
            }

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                "memory_search_failed",
                query=query,
                error=str(e),
                duration_sec=duration
            )
            return {"error": str(e)}

    async def add_file(
        self,
        file_path: str,
        name: str,
        description: str = None
    ) -> Dict:
        """
        Add file to memory.

        Matches PraxosClient.add_file API.
        """
        start_time = time.time()
        logger.info("adding_file", file_path=file_path, name=name)

        try:
            # Read file
            with open(file_path, 'r') as f:
                content = f.read()

            # Format for archival memory
            file_text = f"""File: {name}
Description: {description or 'No description'}

{content}
"""

            passage = self.client.insert_archival_memory(
                agent_id=self.agent_id,
                memory=file_text
            )

            duration = time.time() - start_time

            logger.info(
                "file_added",
                name=name,
                duration_sec=duration
            )

            return {
                "success": True,
                "id": passage.id,
                "source": passage
            }

        except Exception as e:
            duration = time.time() - start_time
            logger.error("file_add_failed", name=name, error=str(e))
            return {"error": str(e)}

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _format_conversation_for_letta(
        self,
        messages: List[Dict],
        user_record: Dict,
        source: str,
        conversation_id: str
    ) -> str:
        """Format conversation for Letta agent processing."""
        user_name = f"{user_record.get('first_name', '')} {user_record.get('last_name', '')}".strip()

        formatted = f"""New conversation from {source} (ID: {conversation_id})
User: {user_name}

Messages:
"""

        for msg in messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            timestamp = msg.get('timestamp', '')

            if isinstance(timestamp, datetime):
                timestamp = timestamp.isoformat()

            formatted += f"\n[{timestamp}] {role}: {content}\n"

        formatted += "\nAnalyze this conversation and update your memory with any important information you learn about the user."

        return formatted

    def _format_email_for_archival(
        self,
        messages: List,
        name: str,
        description: str,
        metadata: Dict
    ) -> str:
        """Format email for archival memory storage."""
        formatted = f"""Email: {name}
Description: {description}

"""

        for msg in messages:
            # Handle both Message objects and dicts
            if hasattr(msg, 'content'):
                content = msg.content
                role = msg.role if hasattr(msg, 'role') else 'unknown'
                timestamp = msg.timestamp if hasattr(msg, 'timestamp') else ''
            else:
                content = msg.get('content', '')
                role = msg.get('role', 'unknown')
                timestamp = msg.get('timestamp', '')

            if timestamp:
                formatted += f"[{timestamp}] {role}: {content}\n\n"
            else:
                formatted += f"{role}: {content}\n\n"

        return formatted
