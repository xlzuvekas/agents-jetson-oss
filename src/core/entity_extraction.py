"""
Entity extraction from conversations and emails.

Uses LLM to extract entities and relationships from unstructured text,
creating structured knowledge graph entries.
"""

import json
from typing import List, Dict, Any, Optional
import structlog
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

from src.core.knowledge_graph import GraphNode, GraphRelationship, EntityExtractionResult

logger = structlog.get_logger()


# ============================================================================
# Entity Schemas
# ============================================================================

class ExtractedEntity(BaseModel):
    """Entity extracted from text."""
    type: str  # e.g., "Person", "Organization", "Email", "Event"
    name: str  # Entity name/label
    properties: Dict[str, str] = Field(default_factory=dict)
    mentions: List[str] = Field(default_factory=list)  # Text snippets where mentioned


class ExtractedRelationship(BaseModel):
    """Relationship extracted from text."""
    type: str  # e.g., "works_at", "sent_email", "attended_meeting"
    source_entity: str  # Entity name
    target_entity: str  # Entity name
    properties: Dict[str, str] = Field(default_factory=dict)


class ExtractionSchema(BaseModel):
    """Schema for LLM extraction output."""
    entities: List[ExtractedEntity]
    relationships: List[ExtractedRelationship]
    summary: str


# ============================================================================
# Entity Extractor
# ============================================================================

class EntityExtractor:
    """
    Extract entities and relationships from text using LLM.

    Supports multiple extraction strategies:
    - Conversation extraction
    - Email extraction
    - Business data extraction
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """
        Initialize entity extractor.

        Args:
            api_key: OpenAI API key
            model: Model to use for extraction
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def extract_from_conversation(
        self,
        messages: List[Dict[str, str]],
        user_context: Optional[Dict] = None
    ) -> EntityExtractionResult:
        """
        Extract entities from a conversation.

        Args:
            messages: List of message dicts with 'role' and 'content'
            user_context: Optional user context for better extraction

        Returns:
            EntityExtractionResult with entities and relationships
        """
        # Format conversation
        conversation_text = "\n".join([
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
            for msg in messages
        ])

        # Create extraction prompt
        prompt = self._create_conversation_extraction_prompt(
            conversation_text,
            user_context
        )

        # Extract using LLM
        extraction = await self._extract_with_llm(prompt)

        # Convert to graph nodes and relationships
        return self._extraction_to_graph(extraction)

    async def extract_from_email(
        self,
        email_data: Dict[str, Any]
    ) -> EntityExtractionResult:
        """
        Extract entities from an email.

        Args:
            email_data: Email dict with sender, recipients, subject, body

        Returns:
            EntityExtractionResult with entities and relationships
        """
        # Format email
        email_text = f"""
From: {email_data.get('sender', 'Unknown')}
To: {', '.join(email_data.get('recipients', []))}
Subject: {email_data.get('subject', 'No Subject')}

{email_data.get('body', '')}
"""

        # Create extraction prompt
        prompt = self._create_email_extraction_prompt(email_text, email_data)

        # Extract using LLM
        extraction = await self._extract_with_llm(prompt)

        # Convert to graph nodes and relationships
        return self._extraction_to_graph(extraction, email_data)

    async def extract_from_business_data(
        self,
        data: Dict[str, Any],
        context: str = ""
    ) -> EntityExtractionResult:
        """
        Extract entities from structured business data.

        Args:
            data: Structured business data
            context: Additional context about the data

        Returns:
            EntityExtractionResult with entities and relationships
        """
        # Format data as JSON
        data_json = json.dumps(data, indent=2)

        # Create extraction prompt
        prompt = f"""Extract entities and relationships from this business data.

Context: {context}

Data:
```json
{data_json}
```

Identify:
- Key business entities (customers, products, transactions, etc.)
- Relationships between entities
- Important attributes

Return structured extraction following the schema."""

        # Extract using LLM
        extraction = await self._extract_with_llm(prompt)

        # Convert to graph nodes and relationships
        return self._extraction_to_graph(extraction)

    # ========================================================================
    # LLM Extraction
    # ========================================================================

    async def _extract_with_llm(self, prompt: str) -> ExtractionSchema:
        """
        Use LLM to extract entities and relationships.

        Args:
            prompt: Extraction prompt

        Returns:
            ExtractionSchema with extracted data
        """
        try:
            response = await self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format=ExtractionSchema
            )

            extraction = response.choices[0].message.parsed

            logger.info(
                "entities_extracted",
                entities=len(extraction.entities),
                relationships=len(extraction.relationships)
            )

            return extraction

        except Exception as e:
            logger.error("entity_extraction_failed", error=str(e))
            # Return empty extraction on error
            return ExtractionSchema(
                entities=[],
                relationships=[],
                summary="Extraction failed"
            )

    def _get_system_prompt(self) -> str:
        """Get system prompt for entity extraction."""
        return """You are an expert entity extraction system.

Your task is to extract entities and relationships from text.

Entity types you should recognize:
- Person: People mentioned in the text
- Organization: Companies, institutions, groups
- Email: Email addresses
- Event: Meetings, calls, appointments
- Document: Files, reports, documents
- Product: Products or services
- Location: Physical or virtual locations
- Topic: Subjects or themes discussed

Relationship types you should identify:
- works_at: Person works at Organization
- sent_email: Person sent email to Person
- attended: Person attended Event
- mentioned_in: Entity mentioned in Document/Email
- located_at: Entity located at Location
- related_to: Generic relationship

Extract:
1. All entities with their properties
2. Relationships between entities
3. A brief summary of the text

Be thorough but accurate. Only extract entities you're confident about."""

    # ========================================================================
    # Prompt Creation
    # ========================================================================

    def _create_conversation_extraction_prompt(
        self,
        conversation_text: str,
        user_context: Optional[Dict] = None
    ) -> str:
        """Create prompt for conversation extraction."""
        user_info = ""
        if user_context:
            user_info = f"""
Known user context:
- Name: {user_context.get('first_name', '')} {user_context.get('last_name', '')}
- Email: {user_context.get('email', '')}
"""

        return f"""Extract entities and relationships from this conversation.

{user_info}

Conversation:
{conversation_text}

Focus on:
- People mentioned (extract names, roles, companies)
- Topics discussed
- Actions or tasks mentioned
- Time-sensitive information (meetings, deadlines)
- Relationships between entities

Extract comprehensive information."""

    def _create_email_extraction_prompt(
        self,
        email_text: str,
        email_data: Dict
    ) -> str:
        """Create prompt for email extraction."""
        return f"""Extract entities and relationships from this email.

Email:
{email_text}

Focus on:
- Sender and recipients (create Person entities)
- Organizations mentioned
- Topics and subjects
- Action items or tasks
- Meetings or events mentioned
- Documents referenced

Create relationships:
- sender → sent_email → recipients
- people → works_at → organizations
- people/organizations → mentioned_in → email

Extract all relevant information."""

    # ========================================================================
    # Conversion to Graph Format
    # ========================================================================

    def _extraction_to_graph(
        self,
        extraction: ExtractionSchema,
        source_metadata: Optional[Dict] = None
    ) -> EntityExtractionResult:
        """
        Convert extraction result to graph nodes and relationships.

        Args:
            extraction: LLM extraction result
            source_metadata: Optional metadata about the source

        Returns:
            EntityExtractionResult with graph-ready data
        """
        # Convert entities to graph nodes
        nodes = []
        entity_map = {}  # name -> node_id mapping

        for entity in extraction.entities:
            node = GraphNode(
                type=f"schema:{entity.type}",
                label=entity.name,
                properties={
                    **entity.properties,
                    "mentions": entity.mentions,
                    "source": "entity_extraction"
                }
            )

            if source_metadata:
                node.properties["source_metadata"] = source_metadata

            nodes.append(node)
            entity_map[entity.name] = node.id

        # Convert relationships to graph relationships
        relationships = []

        for rel in extraction.relationships:
            # Look up node IDs
            source_id = entity_map.get(rel.source_entity)
            target_id = entity_map.get(rel.target_entity)

            if source_id and target_id:
                graph_rel = GraphRelationship(
                    type=rel.type,
                    source_id=source_id,
                    target_id=target_id,
                    properties={
                        **rel.properties,
                        "source": "entity_extraction"
                    }
                )
                relationships.append(graph_rel)

        return EntityExtractionResult(
            entities=nodes,
            relationships=relationships,
            text_summary=extraction.summary
        )


# ============================================================================
# Convenience Functions
# ============================================================================

async def extract_entities_from_text(
    text: str,
    api_key: str,
    context: Optional[Dict] = None
) -> EntityExtractionResult:
    """
    Convenience function to extract entities from any text.

    Args:
        text: Text to extract from
        api_key: OpenAI API key
        context: Optional context

    Returns:
        EntityExtractionResult
    """
    extractor = EntityExtractor(api_key)

    prompt = f"""Extract entities and relationships from this text.

Text:
{text}

Extract all relevant entities and their relationships."""

    extraction = await extractor._extract_with_llm(prompt)

    return extractor._extraction_to_graph(extraction, context)
