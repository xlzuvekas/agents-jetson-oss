"""
Integrated Letta + Knowledge Graph client.

Complete replacement for Praxos providing both:
- Vector-based memory (Letta archival)
- Structured knowledge graph (entities and relationships)

This provides ALL Praxos functionality in open-source form.
"""

import structlog
import time
from typing import List, Dict, Optional, Any
from datetime import datetime

from src.core.letta_memory_client_v2 import LettaMemoryClient
from src.core.knowledge_graph import KnowledgeGraph, GraphNode, GraphRelationship
from src.core.entity_extraction import EntityExtractor
from src.config.settings import settings

logger = structlog.get_logger()


class LettaKGClient:
    """
    Integrated Letta + Knowledge Graph client.

    Provides complete Praxos replacement with:
    1. Letta: Vector memory for conversations and emails
    2. KG: Structured entities and relationships
    3. Entity extraction: Automatic graph population from text
    4. Hybrid search: Vector + graph traversal

    API matches PraxosClient for drop-in replacement.
    """

    def __init__(self, environment_name: str = None, api_key: str = None):
        """
        Initialize integrated client.

        Args:
            environment_name: User environment (e.g., "env_for_user@email.com")
            api_key: Letta API key (optional for local server)
        """
        self.environment_name = environment_name
        self.user_id = environment_name.replace("env_for_", "") if environment_name else "default"

        # Initialize Letta memory client
        self.letta = LettaMemoryClient(
            environment_name=environment_name,
            api_key=api_key
        )

        # Initialize knowledge graph
        self.kg = KnowledgeGraph(
            user_id=self.user_id,
            persist_dir="./kg_data"
        )

        # Initialize entity extractor
        self.entity_extractor = EntityExtractor(
            api_key=settings.openai_api_key,
            model="gpt-4o-mini"
        )

        logger.info(
            "letta_kg_client_initialized",
            environment=environment_name,
            user_id=self.user_id
        )

    # ========================================================================
    # Conversation Methods (Praxos-compatible)
    # ========================================================================

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
        Add conversation to both Letta memory and knowledge graph.

        Matches PraxosClient.add_conversation API.

        Process:
        1. Store full conversation in Letta archival memory
        2. Extract entities and relationships from conversation
        3. Add entities/relationships to knowledge graph
        4. Link conversation to entities

        Args:
            user_id: User identifier
            source: Source platform
            metadata: Additional metadata
            user_record: User profile
            messages: List of message dicts
            conversation_id: Conversation ID

        Returns:
            Dict with success status and IDs
        """
        start_time = time.time()

        logger.info(
            "adding_conversation_with_kg",
            user_id=user_id,
            source=source,
            message_count=len(messages) if messages else 0
        )

        try:
            # 1. Add to Letta memory
            letta_result = await self.letta.add_conversation(
                user_id=user_id,
                source=source,
                metadata=metadata,
                user_record=user_record,
                messages=messages,
                conversation_id=conversation_id
            )

            if not letta_result.get("success"):
                return letta_result

            # 2. Extract entities from conversation
            extraction = await self.entity_extractor.extract_from_conversation(
                messages=messages,
                user_context=user_record
            )

            # 3. Add entities to knowledge graph
            entity_ids = []
            for entity in extraction.entities:
                entity_id = self.kg.add_node(entity)
                entity_ids.append(entity_id)

            # 4. Add relationships to knowledge graph
            for relationship in extraction.relationships:
                self.kg.add_relationship(relationship)

            # 5. Create conversation node in graph
            conversation_node = GraphNode(
                type="schema:Conversation",
                label=f"Conversation on {source} - {conversation_id}",
                properties={
                    "conversation_id": conversation_id,
                    "source": source,
                    "platform": source,
                    "user_id": user_id,
                    "message_count": len(messages) if messages else 0,
                    "letta_id": letta_result.get("id"),
                    "summary": extraction.text_summary
                }
            )

            conv_node_id = self.kg.add_node(conversation_node)

            # 6. Link conversation to entities
            for entity_id in entity_ids:
                self.kg.add_relationship(GraphRelationship(
                    type="mentioned_in",
                    source_id=entity_id,
                    target_id=conv_node_id
                ))

            duration = time.time() - start_time

            logger.info(
                "conversation_added_with_kg",
                conversation_id=conversation_id,
                letta_id=letta_result.get("id"),
                kg_node_id=conv_node_id,
                entities_extracted=len(entity_ids),
                duration_sec=duration
            )

            return {
                "success": True,
                "id": letta_result.get("id"),
                "kg_node_id": conv_node_id,
                "entities_extracted": len(entity_ids),
                "source": letta_result.get("source")
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
        Add email to both Letta memory and knowledge graph.

        Matches PraxosClient.add_email_conversation API.

        Process:
        1. Store email in Letta archival memory
        2. Extract entities (sender, recipients, organizations, topics)
        3. Add entities to knowledge graph
        4. Create email node and relationships

        Args:
            messages: Email messages
            name: Email identifier/subject
            description: Email description
            metadata: Additional metadata
            user_record: User profile

        Returns:
            Dict with success status and IDs
        """
        start_time = time.time()

        logger.info("adding_email_with_kg", name=name)

        try:
            # 1. Add to Letta memory
            letta_result = await self.letta.add_email_conversation(
                messages=messages,
                name=name,
                description=description,
                metadata=metadata,
                user_record=user_record
            )

            if not letta_result.get("success"):
                return letta_result

            # 2. Format email data for extraction
            email_data = self._format_email_for_extraction(
                messages,
                name,
                description,
                metadata
            )

            # 3. Extract entities from email
            extraction = await self.entity_extractor.extract_from_email(
                email_data=email_data
            )

            # 4. Add entities to knowledge graph
            entity_ids = []
            for entity in extraction.entities:
                entity_id = self.kg.add_node(entity)
                entity_ids.append(entity_id)

            # 5. Add relationships
            for relationship in extraction.relationships:
                self.kg.add_relationship(relationship)

            # 6. Create email node
            email_node = GraphNode(
                type="schema:Email",
                label=name,
                properties={
                    "subject": email_data.get("subject", name),
                    "sender": email_data.get("sender"),
                    "recipients": email_data.get("recipients", []),
                    "date": email_data.get("date"),
                    "letta_id": letta_result.get("id"),
                    "summary": extraction.text_summary
                }
            )

            email_node_id = self.kg.add_node(email_node)

            # 7. Link email to entities
            for entity_id in entity_ids:
                self.kg.add_relationship(GraphRelationship(
                    type="mentioned_in",
                    source_id=entity_id,
                    target_id=email_node_id
                ))

            duration = time.time() - start_time

            logger.info(
                "email_added_with_kg",
                name=name,
                letta_id=letta_result.get("id"),
                kg_node_id=email_node_id,
                entities_extracted=len(entity_ids),
                duration_sec=duration
            )

            return {
                "success": True,
                "id": letta_result.get("id"),
                "kg_node_id": email_node_id,
                "entities_extracted": len(entity_ids),
                "source": letta_result.get("source")
            }

        except Exception as e:
            duration = time.time() - start_time
            logger.error("email_add_failed", name=name, error=str(e))
            return {"error": str(e)}

    # ========================================================================
    # Search Methods (Praxos-compatible)
    # ========================================================================

    async def search_memory(
        self,
        query: str,
        top_k: int = 10,
        search_modality: str = "hybrid",
        exclude_seen: List[str] = None
    ) -> Dict:
        """
        Hybrid search: Letta vector + KG traversal.

        Matches PraxosClient.search_memory API.

        Process:
        1. Search Letta archival memory (vector search)
        2. Search knowledge graph (keyword + structure)
        3. Combine and rank results

        Args:
            query: Search query
            top_k: Number of results
            search_modality: "vector", "graph", or "hybrid"
            exclude_seen: Node IDs to exclude

        Returns:
            Dict with sentences and results (Praxos format)
        """
        start_time = time.time()

        logger.info("searching_memory_hybrid", query=query, modality=search_modality)

        try:
            results = []
            sentences = []

            # Vector search via Letta
            if search_modality in ["vector", "hybrid"]:
                letta_results = await self.letta.search_memory(
                    query=query,
                    top_k=top_k
                )

                if letta_results.get("success"):
                    sentences.extend(letta_results.get("sentences", []))
                    results.extend(letta_results.get("results", []))

            # Graph search via KG
            if search_modality in ["graph", "hybrid"]:
                # Search for entities matching query
                kg_results = self._search_graph(query, top_k)

                # Add to results
                for kg_result in kg_results:
                    sentences.append(kg_result["label"])
                    results.append({
                        "text": self._format_graph_result(kg_result),
                        "node_id": kg_result["node_id"],
                        "score": kg_result["score"],
                        "source": "knowledge_graph"
                    })

            # Combine and deduplicate
            unique_sentences = list(dict.fromkeys(sentences))

            duration = time.time() - start_time

            logger.info(
                "memory_search_completed",
                query=query,
                results_count=len(results),
                sentences_count=len(unique_sentences),
                duration_sec=duration
            )

            return {
                "success": True,
                "sentences": unique_sentences[:top_k],
                "results": results[:top_k],
                "count": len(results),
                "sentences_count": len(unique_sentences)
            }

        except Exception as e:
            duration = time.time() - start_time
            logger.error("memory_search_failed", query=query, error=str(e))
            return {"error": str(e)}

    async def search_from_anchors(
        self,
        user_id: str,
        query: str,
        max_hops: int = 3,
        top_k: int = 3,
        node_types: List[str] = None
    ) -> Dict:
        """
        Graph traversal search from anchor nodes.

        Matches PraxosClient.search_from_anchors API.

        Uses knowledge graph to find entities connected to user anchors.

        Args:
            user_id: User identifier (anchor)
            query: Query string
            max_hops: Maximum graph traversal hops
            top_k: Results to return
            node_types: Filter by node types

        Returns:
            Dict with results and anchor paths
        """
        start_time = time.time()

        # Create anchors from user_id
        anchors = [
            {"value": user_id},
            {"value": self.user_id}
        ]

        logger.info(
            "searching_from_anchors",
            user_id=user_id,
            query=query,
            max_hops=max_hops
        )

        try:
            # Use knowledge graph search
            node_type = node_types[0] if node_types else None

            kg_results = self.kg.search_from_anchors(
                anchors=anchors,
                query=query,
                max_hops=max_hops,
                node_type=node_type,
                top_k=top_k
            )

            duration = time.time() - start_time

            logger.info(
                "anchor_search_completed",
                results_count=len(kg_results),
                duration_sec=duration
            )

            return {
                "success": True,
                "results": kg_results,
                "count": len(kg_results)
            }

        except Exception as e:
            duration = time.time() - start_time
            logger.error("anchor_search_failed", error=str(e))
            return {"error": str(e)}

    # ========================================================================
    # Business Data Methods (Praxos-compatible)
    # ========================================================================

    async def add_business_data(
        self,
        data: Dict[str, Any],
        name: str = None,
        description: str = None,
        root_entity_type: str = "schema:Thing",
        metadata: Dict[str, Any] = None
    ) -> Dict:
        """
        Add structured business data to knowledge graph.

        Matches PraxosClient.add_business_data API.

        Args:
            data: Structured data dictionary
            name: Root entity name
            description: Description
            root_entity_type: Schema type for root
            metadata: Additional metadata

        Returns:
            Dict with node IDs and counts
        """
        start_time = time.time()

        logger.info("adding_business_data", name=name)

        try:
            # Add to knowledge graph
            result = self.kg.add_business_data(
                data=data,
                name=name or "Business Data",
                description=description,
                root_entity_type=root_entity_type
            )

            duration = time.time() - start_time

            logger.info(
                "business_data_added",
                name=name,
                nodes_created=result["nodes_created"],
                duration_sec=duration
            )

            return {
                "success": True,
                **result
            }

        except Exception as e:
            duration = time.time() - start_time
            logger.error("business_data_add_failed", error=str(e))
            return {"error": str(e)}

    # ========================================================================
    # Graph Query Methods (Praxos-compatible)
    # ========================================================================

    async def get_nodes_by_type(
        self,
        type_name: str,
        include_literals: bool = True,
        max_results: int = 100
    ) -> List[Dict]:
        """Get all nodes of a specific type from graph."""
        try:
            results = self.kg.get_nodes_by_type(type_name, max_results)

            logger.info(
                "nodes_retrieved_by_type",
                type_name=type_name,
                count=len(results)
            )

            return results

        except Exception as e:
            logger.error("get_nodes_by_type_failed", error=str(e))
            return []

    async def enrich_nodes(
        self,
        node_ids: List[str],
        k_hops: int = 2
    ) -> Dict:
        """Enrich nodes with k-hop neighborhood."""
        try:
            result = self.kg.enrich_nodes(node_ids, k_hops)

            logger.info(
                "nodes_enriched",
                node_count=len(node_ids),
                k_hops=k_hops
            )

            return result

        except Exception as e:
            logger.error("enrich_nodes_failed", error=str(e))
            return {}

    # ========================================================================
    # File Methods (Praxos-compatible)
    # ========================================================================

    async def add_file(
        self,
        file_path: str,
        name: str,
        description: str = None
    ) -> Dict:
        """
        Add file to Letta memory.

        Matches PraxosClient.add_file API.
        """
        return await self.letta.add_file(file_path, name, description)

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _search_graph(self, query: str, top_k: int) -> List[Dict]:
        """Search knowledge graph for query."""
        # Simple implementation: iterate nodes and score by text match
        # For production: integrate with embedding search or graph algorithms

        query_lower = query.lower()
        results = []

        for node_id, node_data in self.kg.graph.nodes(data=True):
            score = 0.0

            # Label match
            label = node_data.get('label', '').lower()
            if query_lower in label:
                score += 1.0

            # Property matches
            for prop_value in node_data.get('properties', {}).values():
                if isinstance(prop_value, str) and query_lower in prop_value.lower():
                    score += 0.5

            if score > 0:
                results.append({
                    "node_id": node_id,
                    "type": node_data['type'],
                    "label": node_data['label'],
                    "properties": node_data['properties'],
                    "score": score
                })

        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)

        return results[:top_k]

    def _format_graph_result(self, result: Dict) -> str:
        """Format graph search result as text."""
        return f"{result['label']} ({result['type']}) - {result.get('properties', {})}"

    def _format_email_for_extraction(
        self,
        messages: List,
        name: str,
        description: str,
        metadata: Optional[Dict]
    ) -> Dict:
        """Format email messages for entity extraction."""
        # Extract email metadata from messages
        email_data = {
            "subject": name,
            "description": description
        }

        if metadata:
            email_data.update(metadata)

        # Parse first message for sender/recipients
        if messages and len(messages) > 0:
            first_msg = messages[0]

            if hasattr(first_msg, 'content'):
                content = first_msg.content
            else:
                content = first_msg.get('content', '')

            # Try to extract sender/recipients from content
            # This is a simple parser - enhance as needed
            lines = content.split('\n')
            for line in lines:
                if line.startswith('From:'):
                    email_data['sender'] = line.replace('From:', '').strip()
                elif line.startswith('To:'):
                    email_data['recipients'] = [
                        r.strip() for r in line.replace('To:', '').split(',')
                    ]

            email_data['body'] = content

        return email_data
