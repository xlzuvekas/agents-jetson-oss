"""
LangGraph tools for knowledge graph operations.

Exposes knowledge graph capabilities to LangGraph agents for:
- Entity search and retrieval
- Relationship traversal
- Business data queries
- Graph-based reasoning
"""

from typing import List, Dict, Any, Optional
from langchain.tools import Tool, tool
from pydantic import BaseModel, Field
import structlog

from src.core.letta_kg_client import LettaKGClient

logger = structlog.get_logger()


# ============================================================================
# Tool Input Schemas
# ============================================================================

class SearchEntitiesInput(BaseModel):
    """Input for searching entities in knowledge graph."""
    query: str = Field(description="What to search for (person, organization, topic, etc.)")
    entity_type: Optional[str] = Field(
        default=None,
        description="Filter by entity type: Person, Organization, Email, Event, etc."
    )
    max_results: int = Field(default=5, description="Maximum number of results")


class FindRelatedEntitiesInput(BaseModel):
    """Input for finding entities related to a specific entity."""
    entity_name: str = Field(description="Name of the entity to start from")
    max_hops: int = Field(default=2, description="How many relationship hops to traverse")
    relationship_type: Optional[str] = Field(
        default=None,
        description="Filter by relationship type: works_at, sent_email, etc."
    )


class GetEntityDetailsInput(BaseModel):
    """Input for getting detailed information about an entity."""
    entity_name: str = Field(description="Name of the entity to get details for")


class QueryBusinessDataInput(BaseModel):
    """Input for querying structured business data."""
    query: str = Field(description="What business data to query")
    data_type: Optional[str] = Field(
        default=None,
        description="Type of business data: Customer, Product, Transaction, etc."
    )


# ============================================================================
# Knowledge Graph Tools
# ============================================================================

class KnowledgeGraphTools:
    """
    Tools for knowledge graph operations in LangGraph agents.

    These tools allow agents to:
    - Search for entities
    - Traverse relationships
    - Query business data
    - Reason about connections
    """

    def __init__(self, kg_client: LettaKGClient):
        """
        Initialize KG tools.

        Args:
            kg_client: LettaKGClient instance for the user
        """
        self.kg_client = kg_client

    def get_tools(self) -> List[Tool]:
        """
        Get all knowledge graph tools for LangGraph.

        Returns:
            List of Tool objects
        """
        return [
            self._create_search_entities_tool(),
            self._create_find_related_tool(),
            self._create_get_entity_details_tool(),
            self._create_query_business_data_tool(),
            self._create_find_people_tool(),
            self._create_find_organizations_tool(),
        ]

    # ========================================================================
    # Tool Creators
    # ========================================================================

    def _create_search_entities_tool(self) -> Tool:
        """Create tool for searching entities."""

        @tool(args_schema=SearchEntitiesInput)
        async def search_entities(
            query: str,
            entity_type: Optional[str] = None,
            max_results: int = 5
        ) -> str:
            """
            Search for entities in the knowledge graph.

            Use this when you need to find people, organizations, topics, or other
            entities mentioned in past conversations or emails.

            Examples:
            - "Search for people who work at Acme Corp"
            - "Find emails about the Phoenix project"
            - "Search for meetings with John"

            Args:
                query: What to search for
                entity_type: Optional filter (Person, Organization, Email, Event)
                max_results: Maximum results to return

            Returns:
                Formatted string with entity information
            """
            try:
                # Search knowledge graph
                results = await self.kg_client.search_memory(
                    query=query,
                    top_k=max_results,
                    search_modality="graph"
                )

                # Filter by type if specified
                if entity_type and results.get("success"):
                    filtered_results = [
                        r for r in results.get("results", [])
                        if entity_type.lower() in r.get("text", "").lower()
                    ]
                    results["results"] = filtered_results

                # Format results
                if results.get("success") and results.get("results"):
                    output = f"Found {len(results['results'])} entities:\n\n"

                    for i, result in enumerate(results['results'][:max_results], 1):
                        output += f"{i}. {result.get('text', 'N/A')}\n"
                        output += f"   Score: {result.get('score', 0):.2f}\n\n"

                    return output
                else:
                    return f"No entities found matching '{query}'"

            except Exception as e:
                logger.error("search_entities_failed", error=str(e))
                return f"Error searching entities: {str(e)}"

        return search_entities

    def _create_find_related_tool(self) -> Tool:
        """Create tool for finding related entities."""

        @tool(args_schema=FindRelatedEntitiesInput)
        async def find_related_entities(
            entity_name: str,
            max_hops: int = 2,
            relationship_type: Optional[str] = None
        ) -> str:
            """
            Find entities related to a specific entity by traversing the knowledge graph.

            Use this to discover connections between people, organizations, and topics.

            Examples:
            - "Find people related to John Doe"
            - "Find organizations Alice works with"
            - "Find topics related to the Phoenix project"

            Args:
                entity_name: Name of entity to start from
                max_hops: How many relationship hops to traverse (1-3 recommended)
                relationship_type: Optional filter for relationship type

            Returns:
                Formatted string with related entities
            """
            try:
                # Search from anchor
                results = await self.kg_client.search_from_anchors(
                    user_id=self.kg_client.user_id,
                    query=entity_name,
                    max_hops=max_hops,
                    top_k=10
                )

                if results.get("success") and results.get("results"):
                    output = f"Found {len(results['results'])} entities related to '{entity_name}':\n\n"

                    for i, result in enumerate(results['results'][:10], 1):
                        output += f"{i}. {result.get('label', 'N/A')}\n"
                        output += f"   Type: {result.get('type', 'Unknown')}\n"

                        if result.get('properties'):
                            output += f"   Properties: {result['properties']}\n"

                        output += "\n"

                    return output
                else:
                    return f"No related entities found for '{entity_name}'"

            except Exception as e:
                logger.error("find_related_failed", error=str(e))
                return f"Error finding related entities: {str(e)}"

        return find_related_entities

    def _create_get_entity_details_tool(self) -> Tool:
        """Create tool for getting entity details."""

        @tool(args_schema=GetEntityDetailsInput)
        async def get_entity_details(entity_name: str) -> str:
            """
            Get detailed information about a specific entity.

            Use this to learn more about a person, organization, or topic mentioned
            in conversations.

            Args:
                entity_name: Name of the entity

            Returns:
                Formatted string with entity details
            """
            try:
                # Search for exact entity
                results = await self.kg_client.search_memory(
                    query=entity_name,
                    top_k=1,
                    search_modality="graph"
                )

                if results.get("success") and results.get("results"):
                    entity = results['results'][0]

                    output = f"Entity: {entity_name}\n\n"
                    output += f"Details:\n{entity.get('text', 'No details available')}\n\n"

                    # Get relationships
                    related = await self.kg_client.search_from_anchors(
                        user_id=self.kg_client.user_id,
                        query=entity_name,
                        max_hops=1,
                        top_k=5
                    )

                    if related.get("success") and related.get("results"):
                        output += "Related entities:\n"
                        for rel in related['results']:
                            output += f"- {rel.get('label', 'N/A')} ({rel.get('type', 'Unknown')})\n"

                    return output
                else:
                    return f"Entity '{entity_name}' not found in knowledge graph"

            except Exception as e:
                logger.error("get_entity_details_failed", error=str(e))
                return f"Error getting entity details: {str(e)}"

        return get_entity_details

    def _create_query_business_data_tool(self) -> Tool:
        """Create tool for querying business data."""

        @tool(args_schema=QueryBusinessDataInput)
        async def query_business_data(
            query: str,
            data_type: Optional[str] = None
        ) -> str:
            """
            Query structured business data in the knowledge graph.

            Use this to find information about customers, products, transactions,
            or other business entities.

            Args:
                query: What business data to query
                data_type: Optional type filter

            Returns:
                Formatted string with business data
            """
            try:
                # Get nodes by type if specified
                if data_type:
                    results = await self.kg_client.get_nodes_by_type(
                        type_name=f"schema:{data_type}",
                        max_results=10
                    )

                    if results:
                        output = f"Found {len(results)} {data_type} entities:\n\n"

                        for i, node in enumerate(results[:10], 1):
                            output += f"{i}. {node.get('label', 'N/A')}\n"

                            if node.get('properties'):
                                for key, value in node['properties'].items():
                                    output += f"   {key}: {value}\n"

                            output += "\n"

                        return output
                    else:
                        return f"No {data_type} entities found"
                else:
                    # General search
                    results = await self.kg_client.search_memory(
                        query=query,
                        top_k=10,
                        search_modality="graph"
                    )

                    if results.get("success") and results.get("results"):
                        output = f"Found {len(results['results'])} results:\n\n"

                        for i, result in enumerate(results['results'][:10], 1):
                            output += f"{i}. {result.get('text', 'N/A')}\n\n"

                        return output
                    else:
                        return f"No business data found for '{query}'"

            except Exception as e:
                logger.error("query_business_data_failed", error=str(e))
                return f"Error querying business data: {str(e)}"

        return query_business_data

    def _create_find_people_tool(self) -> Tool:
        """Create specialized tool for finding people."""

        @tool
        async def find_people(query: str) -> str:
            """
            Find people mentioned in conversations or emails.

            Use this to search for specific people or people matching criteria.

            Examples:
            - "Find people from Acme Corp"
            - "Find people who attended the Q4 meeting"
            - "Find John Smith"

            Args:
                query: Description of people to find

            Returns:
                List of people found
            """
            try:
                results = await self.kg_client.get_nodes_by_type(
                    type_name="schema:Person",
                    max_results=20
                )

                # Filter by query
                query_lower = query.lower()
                filtered = [
                    r for r in results
                    if query_lower in r.get('label', '').lower() or
                       any(query_lower in str(v).lower()
                           for v in r.get('properties', {}).values())
                ]

                if filtered:
                    output = f"Found {len(filtered)} people:\n\n"

                    for i, person in enumerate(filtered[:10], 1):
                        output += f"{i}. {person.get('label', 'N/A')}\n"

                        props = person.get('properties', {})
                        if props.get('email'):
                            output += f"   Email: {props['email']}\n"
                        if props.get('company'):
                            output += f"   Company: {props['company']}\n"
                        if props.get('role'):
                            output += f"   Role: {props['role']}\n"

                        output += "\n"

                    return output
                else:
                    return f"No people found matching '{query}'"

            except Exception as e:
                logger.error("find_people_failed", error=str(e))
                return f"Error finding people: {str(e)}"

        return find_people

    def _create_find_organizations_tool(self) -> Tool:
        """Create specialized tool for finding organizations."""

        @tool
        async def find_organizations(query: str) -> str:
            """
            Find organizations mentioned in conversations or emails.

            Use this to search for companies, institutions, or groups.

            Args:
                query: Description of organizations to find

            Returns:
                List of organizations found
            """
            try:
                results = await self.kg_client.get_nodes_by_type(
                    type_name="schema:Organization",
                    max_results=20
                )

                # Filter by query
                query_lower = query.lower()
                filtered = [
                    r for r in results
                    if query_lower in r.get('label', '').lower() or
                       any(query_lower in str(v).lower()
                           for v in r.get('properties', {}).values())
                ]

                if filtered:
                    output = f"Found {len(filtered)} organizations:\n\n"

                    for i, org in enumerate(filtered[:10], 1):
                        output += f"{i}. {org.get('label', 'N/A')}\n"

                        props = org.get('properties', {})
                        if props.get('industry'):
                            output += f"   Industry: {props['industry']}\n"
                        if props.get('location'):
                            output += f"   Location: {props['location']}\n"

                        output += "\n"

                    return output
                else:
                    return f"No organizations found matching '{query}'"

            except Exception as e:
                logger.error("find_organizations_failed", error=str(e))
                return f"Error finding organizations: {str(e)}"

        return find_organizations


# ============================================================================
# Factory Function
# ============================================================================

def create_kg_tools(user_context, settings) -> List[Tool]:
    """
    Factory function to create KG tools for a user.

    Use this in hetairos AgentToolsFactory.

    Args:
        user_context: User context object
        settings: Application settings

    Returns:
        List of LangGraph Tool objects
    """
    # Create KG client for user
    user_email = user_context.user_record.get('email', f"user_{user_context.user_id}")
    env_name = f"env_for_{user_email}"

    # Get API key
    letta_api_key = (
        user_context.user_record.get("letta_api_key") or
        settings.letta_api_key
    )

    # Create integrated client
    kg_client = LettaKGClient(
        environment_name=env_name,
        api_key=letta_api_key
    )

    # Create tools
    kg_tools = KnowledgeGraphTools(kg_client)

    return kg_tools.get_tools()
