"""
Knowledge Graph implementation for Letta memory system.

Replicates Praxos knowledge graph capabilities:
- Entity and relationship storage
- Typed nodes (schema:Person, schema:Organization, etc.)
- Graph traversal and anchor-based search
- Business data ingestion
- Entity extraction from conversations
"""

import json
import uuid
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
from pathlib import Path
import networkx as nx
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger()


# ============================================================================
# Graph Schema Models
# ============================================================================

class GraphNode(BaseModel):
    """A node in the knowledge graph."""
    id: str = Field(default_factory=lambda: f"node_{uuid.uuid4().hex[:12]}")
    type: str  # e.g., "schema:Person", "schema:Email", "schema:Organization"
    label: str  # Human-readable label
    properties: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    class Config:
        json_schema_extra = {
            "example": {
                "id": "node_abc123",
                "type": "schema:Person",
                "label": "John Doe",
                "properties": {
                    "email": "john@example.com",
                    "role": "CEO",
                    "company": "Acme Inc"
                }
            }
        }


class GraphRelationship(BaseModel):
    """A relationship between two nodes."""
    id: str = Field(default_factory=lambda: f"rel_{uuid.uuid4().hex[:12]}")
    type: str  # e.g., "sent_email", "works_at", "mentioned_in"
    source_id: str  # Node ID
    target_id: str  # Node ID
    properties: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    class Config:
        json_schema_extra = {
            "example": {
                "id": "rel_xyz789",
                "type": "works_at",
                "source_id": "node_abc123",
                "target_id": "node_org456",
                "properties": {"since": "2020-01-01"}
            }
        }


class EntityExtractionResult(BaseModel):
    """Result of entity extraction from text."""
    entities: List[GraphNode]
    relationships: List[GraphRelationship]
    text_summary: str


# ============================================================================
# Knowledge Graph Storage
# ============================================================================

class KnowledgeGraph:
    """
    Knowledge graph implementation using NetworkX.

    Provides Praxos-style capabilities:
    - Entity and relationship management
    - Graph traversal
    - Anchor-based search
    - Business data ingestion
    """

    def __init__(self, user_id: str, persist_dir: str = "./kg_data"):
        """
        Initialize knowledge graph for a user.

        Args:
            user_id: User identifier
            persist_dir: Directory for persisting graph data
        """
        self.user_id = user_id
        self.persist_dir = Path(persist_dir) / user_id
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # NetworkX directed multigraph (allows multiple edges between nodes)
        self.graph = nx.MultiDiGraph()

        # Load existing graph if available
        self._load_graph()

        logger.info("knowledge_graph_initialized", user_id=user_id)

    # ========================================================================
    # Core Graph Operations
    # ========================================================================

    def add_node(self, node: GraphNode) -> str:
        """
        Add a node to the graph.

        Args:
            node: GraphNode to add

        Returns:
            Node ID
        """
        self.graph.add_node(
            node.id,
            type=node.type,
            label=node.label,
            properties=node.properties,
            created_at=node.created_at,
            updated_at=node.updated_at
        )

        self._persist_graph()

        logger.info(
            "node_added",
            node_id=node.id,
            node_type=node.type,
            label=node.label
        )

        return node.id

    def add_relationship(self, relationship: GraphRelationship) -> str:
        """
        Add a relationship between two nodes.

        Args:
            relationship: GraphRelationship to add

        Returns:
            Relationship ID
        """
        # Verify nodes exist
        if relationship.source_id not in self.graph:
            raise ValueError(f"Source node {relationship.source_id} not found")
        if relationship.target_id not in self.graph:
            raise ValueError(f"Target node {relationship.target_id} not found")

        self.graph.add_edge(
            relationship.source_id,
            relationship.target_id,
            key=relationship.id,
            type=relationship.type,
            properties=relationship.properties,
            created_at=relationship.created_at
        )

        self._persist_graph()

        logger.info(
            "relationship_added",
            rel_id=relationship.id,
            rel_type=relationship.type,
            source=relationship.source_id,
            target=relationship.target_id
        )

        return relationship.id

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID."""
        if node_id not in self.graph:
            return None

        data = self.graph.nodes[node_id]

        return GraphNode(
            id=node_id,
            type=data['type'],
            label=data['label'],
            properties=data['properties'],
            created_at=data['created_at'],
            updated_at=data['updated_at']
        )

    def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """
        Update node properties.

        Args:
            node_id: Node to update
            properties: New properties (merged with existing)

        Returns:
            True if successful
        """
        if node_id not in self.graph:
            return False

        # Merge properties
        current_props = self.graph.nodes[node_id]['properties']
        current_props.update(properties)

        # Update timestamp
        self.graph.nodes[node_id]['updated_at'] = datetime.utcnow().isoformat()

        self._persist_graph()

        logger.info("node_updated", node_id=node_id)

        return True

    def delete_node(self, node_id: str, cascade: bool = True) -> bool:
        """
        Delete a node from the graph.

        Args:
            node_id: Node to delete
            cascade: If True, also delete connected relationships

        Returns:
            True if successful
        """
        if node_id not in self.graph:
            return False

        self.graph.remove_node(node_id)
        self._persist_graph()

        logger.info("node_deleted", node_id=node_id, cascade=cascade)

        return True

    # ========================================================================
    # Graph Traversal and Search
    # ========================================================================

    def search_from_anchors(
        self,
        anchors: List[Dict[str, str]],
        query: str,
        max_hops: int = 3,
        node_type: Optional[str] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search graph starting from anchor nodes (like Praxos).

        Args:
            anchors: List of anchor specifications, e.g., [{"value": "user@email.com"}]
            query: Natural language query
            max_hops: Maximum graph hops from anchors
            node_type: Filter by node type (e.g., "schema:Person")
            top_k: Maximum results to return

        Returns:
            List of matching nodes with paths from anchors
        """
        # Find anchor nodes by matching properties
        anchor_nodes = self._find_anchor_nodes(anchors)

        if not anchor_nodes:
            logger.warning("no_anchor_nodes_found", anchors=anchors)
            return []

        # Traverse graph from anchors
        reachable_nodes = self._traverse_from_anchors(
            anchor_nodes,
            max_hops,
            node_type
        )

        # Score nodes by relevance to query
        scored_results = self._score_nodes_by_query(
            reachable_nodes,
            query,
            top_k
        )

        logger.info(
            "anchor_search_completed",
            anchors_found=len(anchor_nodes),
            reachable_nodes=len(reachable_nodes),
            results=len(scored_results)
        )

        return scored_results

    def _find_anchor_nodes(
        self,
        anchors: List[Dict[str, str]]
    ) -> List[str]:
        """Find nodes matching anchor specifications."""
        anchor_nodes = []

        for anchor in anchors:
            anchor_value = anchor.get("value", "")

            # Search all nodes for matching properties
            for node_id, node_data in self.graph.nodes(data=True):
                # Check label
                if node_data.get('label', '').lower() == anchor_value.lower():
                    anchor_nodes.append(node_id)
                    continue

                # Check properties
                for prop_value in node_data.get('properties', {}).values():
                    if isinstance(prop_value, str):
                        if prop_value.lower() == anchor_value.lower():
                            anchor_nodes.append(node_id)
                            break

        return list(set(anchor_nodes))  # Remove duplicates

    def _traverse_from_anchors(
        self,
        anchor_nodes: List[str],
        max_hops: int,
        node_type: Optional[str] = None
    ) -> List[str]:
        """Traverse graph from anchor nodes up to max_hops."""
        reachable = set(anchor_nodes)
        frontier = set(anchor_nodes)

        for hop in range(max_hops):
            next_frontier = set()

            for node_id in frontier:
                # Get outgoing neighbors
                neighbors = set(self.graph.successors(node_id))

                # Get incoming neighbors (bidirectional traversal)
                neighbors.update(self.graph.predecessors(node_id))

                # Filter by type if specified
                if node_type:
                    neighbors = {
                        n for n in neighbors
                        if self.graph.nodes[n].get('type') == node_type
                    }

                next_frontier.update(neighbors)

            reachable.update(next_frontier)
            frontier = next_frontier - reachable

            if not frontier:
                break  # No more nodes to explore

        return list(reachable)

    def _score_nodes_by_query(
        self,
        node_ids: List[str],
        query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Score nodes by relevance to query.

        Simple implementation using keyword matching.
        For production, integrate with LLM or semantic search.
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored = []

        for node_id in node_ids:
            node_data = self.graph.nodes[node_id]

            # Calculate relevance score
            score = 0.0

            # Label match
            label = node_data.get('label', '').lower()
            if query_lower in label:
                score += 1.0

            # Word overlap
            label_words = set(label.split())
            overlap = len(query_words & label_words)
            score += overlap * 0.5

            # Property matches
            for prop_key, prop_value in node_data.get('properties', {}).items():
                if isinstance(prop_value, str):
                    if query_lower in prop_value.lower():
                        score += 0.3

            if score > 0:
                scored.append({
                    "node_id": node_id,
                    "type": node_data['type'],
                    "label": node_data['label'],
                    "properties": node_data['properties'],
                    "score": score
                })

        # Sort by score and return top_k
        scored.sort(key=lambda x: x['score'], reverse=True)

        return scored[:top_k]

    # ========================================================================
    # Business Data Ingestion
    # ========================================================================

    def add_business_data(
        self,
        data: Dict[str, Any],
        name: str,
        description: str = None,
        root_entity_type: str = "schema:Thing"
    ) -> Dict[str, Any]:
        """
        Add structured business data to the graph.

        Mirrors Praxos add_business_data functionality.
        Recursively creates entities and relationships from nested JSON.

        Args:
            data: Structured data dictionary
            name: Name for the root entity
            description: Optional description
            root_entity_type: Schema type for root entity

        Returns:
            Dict with created node IDs and relationship count
        """
        created_nodes = []
        created_relationships = []

        # Create root node
        root_node = GraphNode(
            type=root_entity_type,
            label=name,
            properties={
                "description": description or "",
                "source": "business_data"
            }
        )

        root_id = self.add_node(root_node)
        created_nodes.append(root_id)

        # Recursively process data
        self._process_business_data_recursive(
            data=data,
            parent_id=root_id,
            parent_key="",
            created_nodes=created_nodes,
            created_relationships=created_relationships
        )

        logger.info(
            "business_data_added",
            name=name,
            nodes_created=len(created_nodes),
            relationships_created=len(created_relationships)
        )

        return {
            "root_node_id": root_id,
            "nodes_created": len(created_nodes),
            "relationships_created": len(created_relationships),
            "node_ids": created_nodes
        }

    def _process_business_data_recursive(
        self,
        data: Any,
        parent_id: str,
        parent_key: str,
        created_nodes: List[str],
        created_relationships: List[str]
    ):
        """Recursively process business data structure."""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    # Create nested entity
                    child_node = GraphNode(
                        type=f"schema:{key.capitalize()}",
                        label=f"{parent_key}.{key}" if parent_key else key,
                        properties={"source": "business_data"}
                    )

                    child_id = self.add_node(child_node)
                    created_nodes.append(child_id)

                    # Create relationship
                    rel = GraphRelationship(
                        type=f"has_{key}",
                        source_id=parent_id,
                        target_id=child_id
                    )

                    self.add_relationship(rel)
                    created_relationships.append(rel.id)

                    # Recurse
                    self._process_business_data_recursive(
                        value,
                        child_id,
                        child_node.label,
                        created_nodes,
                        created_relationships
                    )
                else:
                    # Add as property
                    self.update_node(parent_id, {key: value})

        elif isinstance(data, list):
            for idx, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    # Create list item node
                    item_node = GraphNode(
                        type="schema:ListItem",
                        label=f"{parent_key}[{idx}]",
                        properties={"index": idx}
                    )

                    item_id = self.add_node(item_node)
                    created_nodes.append(item_id)

                    # Create relationship
                    rel = GraphRelationship(
                        type="contains",
                        source_id=parent_id,
                        target_id=item_id,
                        properties={"index": idx}
                    )

                    self.add_relationship(rel)
                    created_relationships.append(rel.id)

                    # Recurse
                    self._process_business_data_recursive(
                        item,
                        item_id,
                        item_node.label,
                        created_nodes,
                        created_relationships
                    )

    # ========================================================================
    # Query Methods (Praxos-compatible)
    # ========================================================================

    def get_nodes_by_type(
        self,
        node_type: str,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """Get all nodes of a specific type."""
        results = []

        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get('type') == node_type:
                results.append({
                    "id": node_id,
                    "type": node_data['type'],
                    "label": node_data['label'],
                    "properties": node_data['properties']
                })

                if len(results) >= max_results:
                    break

        return results

    def enrich_nodes(
        self,
        node_ids: List[str],
        k_hops: int = 2
    ) -> Dict[str, Any]:
        """
        Enrich nodes with their k-hop neighborhood.

        Args:
            node_ids: Nodes to enrich
            k_hops: Number of hops to include

        Returns:
            Subgraph containing nodes and their neighborhoods
        """
        # Get k-hop neighborhood
        all_nodes = set(node_ids)

        for node_id in node_ids:
            if node_id in self.graph:
                neighbors = nx.single_source_shortest_path_length(
                    self.graph,
                    node_id,
                    cutoff=k_hops
                )
                all_nodes.update(neighbors.keys())

        # Extract subgraph
        subgraph = self.graph.subgraph(all_nodes)

        # Format result
        result = {
            "nodes": [
                {
                    "id": nid,
                    **self.graph.nodes[nid]
                }
                for nid in all_nodes
            ],
            "relationships": [
                {
                    "source": u,
                    "target": v,
                    "type": self.graph[u][v][k].get('type'),
                    "properties": self.graph[u][v][k].get('properties', {})
                }
                for u, v, k in subgraph.edges(keys=True)
            ]
        }

        return result

    # ========================================================================
    # Persistence
    # ========================================================================

    def _persist_graph(self):
        """Save graph to disk."""
        graph_file = self.persist_dir / "graph.json"

        # Convert to JSON-serializable format
        data = {
            "nodes": [
                {
                    "id": node_id,
                    **node_data
                }
                for node_id, node_data in self.graph.nodes(data=True)
            ],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "key": k,
                    **edge_data
                }
                for u, v, k, edge_data in self.graph.edges(keys=True, data=True)
            ]
        }

        with open(graph_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_graph(self):
        """Load graph from disk."""
        graph_file = self.persist_dir / "graph.json"

        if not graph_file.exists():
            return

        try:
            with open(graph_file, 'r') as f:
                data = json.load(f)

            # Add nodes
            for node in data.get('nodes', []):
                node_id = node.pop('id')
                self.graph.add_node(node_id, **node)

            # Add edges
            for edge in data.get('edges', []):
                source = edge.pop('source')
                target = edge.pop('target')
                key = edge.pop('key')
                self.graph.add_edge(source, target, key=key, **edge)

            logger.info(
                "graph_loaded",
                nodes=self.graph.number_of_nodes(),
                edges=self.graph.number_of_edges()
            )

        except Exception as e:
            logger.error("graph_load_failed", error=str(e))
