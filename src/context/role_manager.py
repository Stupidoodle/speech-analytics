from typing import Dict, Callable

from src.conversation.types import Role
from src.context.context_manager import ContextManager, ContextUpdate

class RoleBasedContextManager:
    def __init__(self, context_manager: ContextManager):
        self.context_manager = context_manager
        self.role_strategies: Dict[Role, Callable] = {
            Role.INTERVIEWER: self._interviewer_strategy,
            Role.INTERVIEWEE: self._interviewee_strategy,
            Role.SUPPORT_AGENT: self._support_agent_strategy,
            Role.CUSTOMER: self._customer_strategy,
            Role.MEETING_HOST: self._meeting_host_strategy,
            Role.MEETING_PARTICIPANT: self._meeting_participant_strategy
        }

    async def handle_context_update(self, role: Role, context_id: str, update: ContextUpdate):
        if role in self.role_strategies:
            await self.role_strategies[role](context_id, update)
        else:
            await self.context_manager.update_context(
                context_id,
                updates=update.content,
                source=update.source,
                priority=update.priority
            )

    async def _interviewer_strategy(self, context_id: str, update: ContextUpdate):
        # Interviewer-specific context update handling
        if update.source == "cv_analysis":
            update.priority = 2.0  # Increase priority for CV analysis updates
        await self.context_manager.update_context(
            context_id,
            updates=update.content,
            source=update.source,
            priority=update.priority
        )

    async def _interviewee_strategy(self, context_id: str, update: ContextUpdate):
        # Interviewee-specific context update handling
        if update.source == "self_assessment":
            update.priority = 1.5  # Increase priority for self-assessment updates
        await self.context_manager.update_context(
            context_id,
            updates=update.content,
            source=update.source,
            priority=update.priority
        )

    async def _support_agent_strategy(self, context_id: str, update: ContextUpdate):
        # Support agent-specific context update handling
        if update.source == "customer_history":
            update.priority = 1.8  # Increase priority for customer history updates
        await self.context_manager.update_context(
            context_id,
            updates=update.content,
            source=update.source,
            priority=update.priority
        )

    async def _customer_strategy(self, context_id: str, update: ContextUpdate):
        # Customer-specific context update handling
        if update.source == "product_docs":
            update.priority = 1.6  # Increase priority for product documentation updates
        await self.context_manager.update_context(
            context_id,
            updates=update.content,
            source=update.source,
            priority=update.priority
        )

    async def _meeting_host_strategy(self, context_id: str, update: ContextUpdate):
        # Meeting host-specific context update handling
        if update.source == "agenda":
            update.priority = 1.9  # Increase priority for agenda updates
        await self.context_manager.update_context(
            context_id,
            updates=update.content,
            source=update.source,
            priority=update.priority
        )

    async def _meeting_participant_strategy(self, context_id: str, update: ContextUpdate):
        # Meeting participant-specific context update handling
        if update.source == "meeting_notes":
            update.priority = 1.7  # Increase priority for meeting notes updates
        await self.context_manager.update_context(
            context_id,
            updates=update.content,
            source=update.source,
            priority=update.priority
        )