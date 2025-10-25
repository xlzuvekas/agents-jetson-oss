"""
Letta-based memory client for hetairos.

Implements self-editing agent memory with LLM Operating System architecture.
Each user gets a dedicated Letta agent with hierarchical memory:
- Core Memory: Editable persona and user context
- Recall Memory: Conversation history
- Archival Memory: Emails, documents, knowledge
"""

import structlog
from typing import List, Dict, Optional
from pymemgpt import create_client
from pymemgpt.client.client import RESTClient

from src.config.settings import settings

logger = structlog.get_logger()


class LettaMemoryClient:
    """
    Letta-based memory client providing stateful agent memory.

    Each user has a dedicated Letta agent that:
    - Manages its own memory through self-editing
    - Learns from interactions over time
    - Maintains context across conversations
    - Intelligently pages memory in/out of context window
    """

    def __init__(self, user_id: str):
        """
        Initialize Letta memory client for a specific user.

        Args:
            user_id: Unique identifier for the user
        """
        self.user_id = user_id
        self.client: RESTClient = create_client(
            base_url=settings.letta_server_url,
            token=settings.letta_api_key
        )

        # Get or create user-specific agent
        self.agent_id = self._get_or_create_agent()
        self.email_source_id = self._get_or_create_email_source()

        logger.info(
            "letta_client_initialized",
            user_id=user_id,
            agent_id=self.agent_id,
            email_source_id=self.email_source_id
        )

    def _get_or_create_agent(self) -> str:
        """Get existing agent or create new one for user."""
        agent_name = f"hetairos_agent_{self.user_id}"

        try:
            # Check if agent exists
            agents = self.client.list_agents()
            for agent in agents:
                if agent.name == agent_name:
                    logger.info("found_existing_agent", user_id=self.user_id, agent_id=agent.id)
                    return agent.id

            # Create new agent with initial memory blocks
            logger.info("creating_new_agent", user_id=self.user_id)

            agent = self.client.create_agent(
                name=agent_name,
                # Core memory blocks
                memory={
                    "persona": {
                        "value": self._get_default_persona(),
                        "limit": 2000
                    },
                    "human": {
                        "value": self._get_default_human_context(),
                        "limit": 2000
                    }
                },
                # System prompt
                system=self._get_system_prompt(),
                # Tools available to agent
                tools=self._get_agent_tools(),
            )

            logger.info("agent_created", user_id=self.user_id, agent_id=agent.id)
            return agent.id

        except Exception as e:
            logger.error("agent_creation_failed", user_id=self.user_id, error=str(e))
            raise

    def _get_or_create_email_source(self) -> str:
        """Get or create data source for user's emails."""
        source_name = f"emails_{self.user_id}"

        try:
            # Check if source exists
            sources = self.client.list_sources()
            for source in sources:
                if source.name == source_name:
                    logger.info("found_existing_email_source", user_id=self.user_id)
                    return source.id

            # Create new source
            logger.info("creating_email_source", user_id=self.user_id)

            source = self.client.create_source(
                name=source_name,
                description=f"Email archive for user {self.user_id}"
            )

            # Attach source to agent's archival memory
            self.client.attach_source_to_agent(
                agent_id=self.agent_id,
                source_id=source.id
            )

            logger.info("email_source_created", user_id=self.user_id, source_id=source.id)
            return source.id

        except Exception as e:
            logger.error("email_source_creation_failed", user_id=self.user_id, error=str(e))
            raise

    def _get_default_persona(self) -> str:
        """Get default agent persona."""
        return """I am the Hetairos AI assistant, a sophisticated personal AI agent.

My capabilities:
- Manage emails across Gmail and Outlook
- Search and retrieve information from conversation history
- Learn about user preferences and context over time
- Coordinate across multiple communication channels (WhatsApp, Telegram, Email)
- Execute tasks using available tools

My approach:
- I actively learn from interactions and update my memory
- I prioritize user preferences and context
- I search my memory when relevant to provide personalized assistance
- I am concise and action-oriented
"""

    def _get_default_human_context(self) -> str:
        """Get default human context block."""
        return f"""User ID: {self.user_id}

What I know about the user:
- Just started using Hetairos
- No preferences learned yet

I will learn more as we interact and update this memory block.
"""

    def _get_system_prompt(self) -> str:
        """Get agent system prompt."""
        return """You are a personal AI assistant with advanced memory capabilities.

You have access to four types of memory:
1. Core Memory (in-context, editable):
   - persona: Your identity and capabilities
   - human: User information and preferences
   You can edit these using core_memory_append() and core_memory_replace()

2. Recall Memory (searchable conversation history):
   Search using conversation_search() and conversation_search_date()

3. Archival Memory (emails, documents, knowledge):
   Search using archival_memory_search()
   Add using archival_memory_insert()

4. Message Memory (current conversation buffer):
   Recent messages automatically maintained

When to use each memory type:
- Update core memory when you learn important facts about the user
- Search recall memory for recent conversation context
- Search archival memory for emails, documents, and long-term knowledge
- Message memory gives you immediate conversation context

Always:
- Be helpful and concise
- Learn from interactions
- Update your memory when appropriate
- Use tools to accomplish tasks
- Provide accurate information
"""

    def _get_agent_tools(self) -> List[str]:
        """Get list of tools available to agent."""
        return [
            "send_message",           # Send message to user
            "core_memory_append",     # Append to core memory block
            "core_memory_replace",    # Replace content in core memory
            "conversation_search",    # Search conversation history
            "conversation_search_date",  # Search by date
            "archival_memory_insert",    # Add to archival memory
            "archival_memory_search",    # Search archival memory
        ]

    def add_conversation(
        self,
        messages: List[Dict],
        metadata: Dict
    ) -> str:
        """
        Add conversation to agent's memory.

        The agent will process the conversation and decide what to remember.

        Args:
            messages: List of message dicts with 'role' and 'content'
            metadata: Additional context (platform, timestamp, etc.)

        Returns:
            ID of the message sent to agent
        """
        try:
            # Format conversation for agent
            conversation_text = self._format_conversation(messages)

            # Send to agent as system message for processing
            response = self.client.send_message(
                agent_id=self.agent_id,
                message=conversation_text,
                role="system"
            )

            logger.info(
                "conversation_added",
                user_id=self.user_id,
                message_count=len(messages),
                platform=metadata.get("platform")
            )

            return response.id

        except Exception as e:
            logger.error("conversation_add_failed", user_id=self.user_id, error=str(e))
            raise

    def add_email_conversation(self, email_data: Dict) -> str:
        """
        Add email to archival memory.

        Args:
            email_data: Email dict with sender, subject, body, date, etc.

        Returns:
            ID of the archival memory entry
        """
        try:
            # Format email text
            email_text = f"""Email from {email_data.get('sender', 'Unknown')}
To: {', '.join(email_data.get('recipients', []))}
Subject: {email_data.get('subject', 'No Subject')}
Date: {email_data.get('date', 'Unknown')}

{email_data.get('body', '')}
"""

            # Add to email data source
            # This will be indexed and searchable via archival_memory_search
            passage = self.client.insert_archival_memory(
                agent_id=self.agent_id,
                memory=email_text
            )

            logger.info(
                "email_added",
                user_id=self.user_id,
                sender=email_data.get('sender'),
                subject=email_data.get('subject')
            )

            return passage.id

        except Exception as e:
            logger.error("email_add_failed", user_id=self.user_id, error=str(e))
            raise

    def search_memory(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> Dict:
        """
        Query agent's memory.

        The agent will:
        1. Check core memory for relevant context
        2. Search recall memory (conversations)
        3. Search archival memory (emails, documents)
        4. Synthesize response

        Args:
            query: Natural language query
            top_k: Number of results (influences agent's search)
            filters: Optional metadata filters

        Returns:
            Dict with agent's response and memory accesses
        """
        try:
            # Send query to agent
            response = self.client.send_message(
                agent_id=self.agent_id,
                message=query,
                role="user"
            )

            # Extract agent's response and which memories it accessed
            result = {
                "response": self._extract_agent_response(response),
                "memories_accessed": self._extract_memory_accesses(response),
                "tool_calls": self._extract_tool_calls(response)
            }

            logger.info(
                "memory_search_completed",
                user_id=self.user_id,
                query=query,
                memories_count=len(result["memories_accessed"])
            )

            return result

        except Exception as e:
            logger.error("memory_search_failed", user_id=self.user_id, error=str(e))
            raise

    def add_file(self, file_path: str, metadata: Dict) -> str:
        """
        Add file to archival memory.

        Args:
            file_path: Path to file
            metadata: File metadata

        Returns:
            ID of the archival memory entry
        """
        try:
            # Read file content
            with open(file_path, 'r') as f:
                content = f.read()

            # Add to archival memory with metadata
            memory_text = f"""File: {metadata.get('filename', 'unknown')}
Type: {metadata.get('type', 'unknown')}

{content}
"""

            passage = self.client.insert_archival_memory(
                agent_id=self.agent_id,
                memory=memory_text
            )

            logger.info(
                "file_added",
                user_id=self.user_id,
                filename=metadata.get('filename')
            )

            return passage.id

        except Exception as e:
            logger.error("file_add_failed", user_id=self.user_id, error=str(e))
            raise

    def get_memory_state(self) -> Dict:
        """
        Get current state of agent's memory.

        Returns:
            Dict with memory blocks, tool usage, and statistics
        """
        try:
            agent = self.client.get_agent(self.agent_id)

            state = {
                "user_id": self.user_id,
                "agent_id": self.agent_id,
                "core_memory": agent.memory,
                "message_count": len(self.client.get_messages(agent_id=self.agent_id)),
                "archival_memory_count": len(
                    self.client.get_archival_memory(agent_id=self.agent_id)
                ),
            }

            return state

        except Exception as e:
            logger.error("get_memory_state_failed", user_id=self.user_id, error=str(e))
            raise

    # Helper methods

    def _format_conversation(self, messages: List[Dict]) -> str:
        """Format messages as conversation text."""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _extract_agent_response(self, response) -> str:
        """Extract agent's text response from messages."""
        # Find the last assistant message with send_message tool call
        for msg in reversed(response.messages):
            if hasattr(msg, 'tool_calls'):
                for tool_call in msg.tool_calls:
                    if tool_call.function.name == 'send_message':
                        import json
                        args = json.loads(tool_call.function.arguments)
                        return args.get('message', '')
        return ""

    def _extract_memory_accesses(self, response) -> List[Dict]:
        """Extract which memories the agent accessed."""
        accesses = []

        for msg in response.messages:
            if hasattr(msg, 'tool_calls'):
                for tool_call in msg.tool_calls:
                    if 'memory' in tool_call.function.name or 'search' in tool_call.function.name:
                        accesses.append({
                            "tool": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        })

        return accesses

    def _extract_tool_calls(self, response) -> List[Dict]:
        """Extract all tool calls made by agent."""
        tool_calls = []

        for msg in response.messages:
            if hasattr(msg, 'tool_calls'):
                for tool_call in msg.tool_calls:
                    tool_calls.append({
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    })

        return tool_calls
