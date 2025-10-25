"""
Abstract memory interface for supporting multiple backends.
"""

from typing import Protocol, List, Dict, Optional


class MemoryClient(Protocol):
    """
    Abstract interface for memory clients.

    Supports multiple backends:
    - Letta (self-editing agent memory)
    - LlamaIndex (simple RAG)
    - Praxos (proprietary knowledge graph)
    """

    def add_conversation(
        self,
        messages: List[Dict],
        metadata: Dict
    ) -> str:
        """
        Add a conversation to long-term memory.

        Args:
            messages: List of message dicts with 'role' and 'content'
            metadata: Additional context (platform, timestamp, etc.)

        Returns:
            ID of the stored conversation
        """
        ...

    def add_email_conversation(
        self,
        email_data: Dict
    ) -> str:
        """
        Add email to memory as structured conversation.

        Args:
            email_data: Email dict with sender, subject, body, date, etc.

        Returns:
            ID of the stored email
        """
        ...

    def search_memory(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search long-term memory.

        Args:
            query: Natural language query
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of relevant memory entries with content and metadata
        """
        ...

    def add_file(
        self,
        file_path: str,
        metadata: Dict
    ) -> str:
        """
        Add file to memory.

        Args:
            file_path: Path to file
            metadata: File metadata

        Returns:
            ID of the stored file
        """
        ...

    def get_memory_state(self) -> Dict:
        """
        Get current memory state.

        Returns:
            Dict with memory statistics and state
        """
        ...
