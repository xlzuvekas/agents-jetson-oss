"""
Conversation consolidator for Letta memory system.

Drop-in replacement for hetairos ConversationConsolidator using Letta instead of Praxos.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional

import structlog

# These imports assume hetairos structure - update paths as needed
# from src.utils.database import ConversationDatabase
# from src.services.user_service import user_service
# from src.services.ai_service.ai_service import ai_service

from src.core.letta_memory_client_v2 import LettaMemoryClient
from src.config.settings import settings

logger = structlog.get_logger()


class ConversationConsolidatorLetta:
    """
    Consolidates conversations from MongoDB to Letta agent memory.

    Compatible with hetairos ConversationConsolidator interface.
    """

    def __init__(self, db_manager):
        """
        Initialize consolidator.

        Args:
            db_manager: Database manager (hetairos ConversationDatabase)
        """
        self.db = db_manager
        logger.info("conversation_consolidator_initialized", backend="letta")

    async def consolidate_conversation(self, conversation_id: int) -> bool:
        """
        Consolidate a single conversation to Letta.

        Matches hetairos ConversationConsolidator.consolidate_conversation API.

        Args:
            conversation_id: Conversation ID from MongoDB

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get conversation from MongoDB
            conversation = await self.db.get_conversation_info(conversation_id)

            if not conversation:
                logger.warning("conversation_not_found", conversation_id=conversation_id)
                return False

            logger.info("consolidating_conversation", conversation_id=conversation_id)

            # Get messages
            messages = await self.db.get_conversation_messages(conversation_id)

            # Process media descriptions (matching hetairos pattern)
            try:
                message_dict = {}
                file_message_idx = []
                tasks = []

                # Import AI service if available
                try:
                    from src.services.ai_service.ai_service import ai_service

                    for idx, message in enumerate(messages):
                        if message.get('metadata', {}).get('inserted_id'):
                            file_message_idx.append(idx)
                            tasks.append(
                                asyncio.create_task(
                                    ai_service.multi_modal_by_doc_id(
                                        'provide a full description of this media',
                                        message['metadata']['inserted_id']
                                    )
                                )
                            )

                    if tasks:
                        descriptions = await asyncio.gather(*tasks)

                        for i, idx in enumerate(file_message_idx):
                            message = messages[idx]
                            description = descriptions[i]
                            message['content'] = (
                                f'Description of media with id '
                                f'{message["metadata"]["inserted_id"]}: {description}'
                            )
                            message_dict[str(message['_id'])] = {
                                'content': message['content']
                            }

                        # Update messages with descriptions
                        await self.db.bulk_update_messages(message_dict)
                        logger.info(
                            "updated_media_descriptions",
                            count=len(message_dict)
                        )

                except ImportError:
                    logger.warning("ai_service_not_available",
                                 msg="Skipping media description generation")

            except Exception as e:
                logger.error("media_description_failed", error=str(e), exc_info=True)

            # Get search attempts (optional, for context)
            try:
                search_attempts = await self.db.get_recent_search_attempts(
                    conversation_id,
                    limit=100
                )
                logger.info(
                    "retrieved_search_attempts",
                    count=len(search_attempts)
                )
            except:
                search_attempts = []

            # Check if already consolidated
            if not messages:
                logger.warning(
                    "no_messages_found",
                    conversation_id=conversation_id
                )
                await self.db.mark_conversation_consolidated(conversation_id)
                return True

            new_consolidation = await self.db.mark_conversation_consolidated(
                conversation_id
            )

            if not new_consolidation:
                logger.info(
                    "already_consolidated",
                    conversation_id=conversation_id
                )
                return True

            # Get user info
            conversation_user_id = conversation['user_id']

            # Import user service if available
            try:
                from src.services.user_service import user_service
                user_record = user_service.get_user_by_id(conversation_user_id)
            except ImportError:
                user_record = {"email": f"user_{conversation_user_id}"}

            user_email = user_record.get('email', f"user_{conversation_user_id}")
            env_name = f"env_for_{user_email}"

            # Get Letta API key
            # In cloud mode, use per-user key; in local mode, use global key
            if settings.memory_backend == "letta":
                letta_api_key = settings.letta_api_key
            else:
                letta_api_key = user_record.get("letta_api_key") or settings.letta_api_key

            if not letta_api_key:
                logger.warning("no_letta_api_key", msg="Using None (local server)")
                letta_api_key = None

            # Create Letta client
            letta_client = LettaMemoryClient(
                environment_name=env_name,
                api_key=letta_api_key
            )

            # Add conversation to Letta
            source_data = await letta_client.add_conversation(
                user_id=conversation_user_id,
                source='conversation_summary',
                messages=messages,
                metadata={
                    'conversation_id': conversation_id,
                    'message_count': len(messages),
                    'search_attempts': len(search_attempts),
                    'platform': conversation['platform'],
                    'start_time': conversation['start_time'],
                    'end_time': conversation['last_activity']
                },
                user_record=user_record,
                conversation_id=conversation_id
            )

            source_id = source_data.get('id', '')

            # Update conversation with Letta source ID
            await self.db.update_conversation_praxos_source_id(
                conversation_id,
                source_id
            )

            logger.info(
                "conversation_consolidated",
                conversation_id=conversation_id,
                message_count=len(messages),
                source_id=source_id
            )

            return True

        except Exception as e:
            logger.error(
                "consolidation_failed",
                conversation_id=conversation_id,
                error=str(e),
                exc_info=True
            )
            return False

    async def consolidate_all_ready_conversations(self) -> Dict:
        """
        Consolidate all conversations ready for consolidation.

        Matches hetairos ConversationConsolidator.consolidate_all_ready_conversations API.

        Returns:
            Dict with consolidation statistics
        """
        conversations = await self.db.get_conversations_to_consolidate()

        results = {
            'total': len(conversations),
            'successful': 0,
            'failed': 0,
            'errors': []
        }

        logger.info("consolidating_batch", count=len(conversations))

        for conversation in conversations:
            conversation_id = str(conversation['_id'])

            try:
                success = await self.consolidate_conversation(conversation_id)

                if success:
                    results['successful'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append(
                        f"Conversation {conversation_id}: Unknown error"
                    )

            except Exception as e:
                results['failed'] += 1
                results['errors'].append(
                    f"Conversation {conversation_id}: {str(e)}"
                )

        logger.info(
            "batch_consolidation_complete",
            total=results['total'],
            successful=results['successful'],
            failed=results['failed']
        )

        return results

    async def consolidate_user_conversations(self, user_id: str) -> Dict:
        """
        Consolidate all ready conversations for a specific user.

        Matches hetairos ConversationConsolidator.consolidate_user_conversations API.

        Args:
            user_id: User identifier

        Returns:
            Dict with consolidation statistics for user
        """
        conversations = await self.db.get_conversations_to_consolidate()
        user_conversations = [
            conv for conv in conversations
            if conv['user_id'] == user_id
        ]

        results = {
            'user_id': user_id,
            'total': len(user_conversations),
            'successful': 0,
            'failed': 0,
            'errors': []
        }

        logger.info(
            "consolidating_user_conversations",
            user_id=user_id,
            count=len(user_conversations)
        )

        for conversation in user_conversations:
            conversation_id = str(conversation['_id'])

            try:
                success = await self.consolidate_conversation(conversation_id)

                if success:
                    results['successful'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append(
                        f"Conversation {conversation_id}: Unknown error"
                    )

            except Exception as e:
                results['failed'] += 1
                results['errors'].append(
                    f"Conversation {conversation_id}: {str(e)}"
                )

        logger.info(
            "user_consolidation_complete",
            user_id=user_id,
            successful=results['successful'],
            failed=results['failed']
        )

        return results

    async def get_consolidation_statistics(self) -> Dict:
        """
        Get statistics about conversations ready for consolidation.

        Matches hetairos ConversationConsolidator.get_consolidation_statistics API.

        Returns:
            Dict with consolidation statistics
        """
        conversations = await self.db.get_conversations_to_consolidate()

        # Group by user and platform
        user_counts = {}
        platform_counts = {}

        for conv in conversations:
            user_id = conv['user_id']
            platform = conv['platform']

            user_counts[user_id] = user_counts.get(user_id, 0) + 1
            platform_counts[platform] = platform_counts.get(platform, 0) + 1

        return {
            'total_conversations': len(conversations),
            'unique_users': len(user_counts),
            'conversations_by_user': user_counts,
            'conversations_by_platform': platform_counts,
            'oldest_conversation': (
                min(conversations, key=lambda x: x['start_time'])['start_time']
                if conversations else None
            ),
            'newest_conversation': (
                max(conversations, key=lambda x: x['start_time'])['start_time']
                if conversations else None
            )
        }
