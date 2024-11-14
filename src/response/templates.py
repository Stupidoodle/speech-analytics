"""Response template management and rendering."""
from typing import Dict, Any, List, Optional, AsyncIterator
from datetime import datetime
import re
import asyncio

from src.conversation.types import Role, MessageRole
from src.context.types import ContextEntry
from src.events.bus import EventBus
from src.events.types import Event, EventType

from .types import (
    ResponseType,
    ResponseTemplate,
    GenerationConfig,
    ResponseRequest,
    GeneratedResponse
)
from .exceptions import TemplateError


class TemplateManager:
    """Manages response templates."""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.templates: Dict[str, ResponseTemplate] = {}
        self._variable_pattern = re.compile(r'\{([^}]+)\}')
        self._setup_default_templates()

    async def render_template(
            self,
            template_name: str,
            variables: Dict[str, Any],
            role: Optional[Role] = None
    ) -> str:
        """Render template with variables."""
        if template_name not in self.templates:
            raise TemplateError(
                f"Template not found: {template_name}",
                template_name,
                variables
            )

        template = self.templates[template_name]

        # Get role-specific content if available
        content = (
            template.role_specific.get(role, template.content)
            if role else template.content
        )

        # Validate variables
        missing = template.variables - set(variables.keys())
        if missing:
            raise TemplateError(
                f"Missing variables: {missing}",
                template_name,
                variables
            )

        try:
            return content.format(**variables)
        except KeyError as e:
            raise TemplateError(
                f"Invalid variable: {e}",
                template_name,
                variables
            )

    async def add_template(
            self,
            template: ResponseTemplate
    ) -> None:
        """Add new template."""
        # Extract variables from content
        variables = set(
            self._variable_pattern.findall(template.content)
        )
        for role_content in template.role_specific.values():
            variables.update(
                self._variable_pattern.findall(role_content)
            )

        template.variables = variables
        self.templates[template.name] = template

        await self.event_bus.publish(Event(
            type=EventType.RESPONSE_RECEIVED,
            data={
                "action": "template_added",
                "template": template.name
            }
        ))

    def get_template(
            self,
            template_name: str
    ) -> Optional[ResponseTemplate]:
        """Get template by name."""
        return self.templates.get(template_name)

    def find_templates(
            self,
            response_type: ResponseType,
            role: Optional[Role] = None
    ) -> List[ResponseTemplate]:
        """Find templates matching type and role."""
        matches = []
        for template in self.templates.values():
            type_match = template.metadata.get(
                "response_type"
            ) == response_type
            role_match = (
                    not role or
                    role in template.role_specific
            )
            if type_match and role_match:
                matches.append(template)
        return matches

    def _setup_default_templates(self) -> None:
        """Set up default response templates."""
        defaults = [
            ResponseTemplate(
                name="clarification",
                content="Could you please clarify {topic}?",
                variables=set(),  # Will be extracted
                conditions={"type": ResponseType.CLARIFICATION},
                role_specific={
                    Role.INTERVIEWER: (
                        "Could you elaborate on {topic}, "
                        "particularly regarding {aspect}?"
                    ),
                    Role.SUPPORT_AGENT: (
                        "I need to better understand {topic}. "
                        "Could you provide more details about {aspect}?"
                    )
                },
                metadata={"response_type": ResponseType.CLARIFICATION}
            ),
            ResponseTemplate(
                name="follow_up",
                content=(
                    "Based on {context}, "
                    "what are your thoughts about {topic}?"
                ),
                variables=set(),
                conditions={"type": ResponseType.FOLLOW_UP},
                role_specific={
                    Role.INTERVIEWER: (
                        "Given your experience with {context}, "
                        "how would you approach {topic}?"
                    ),
                    Role.SUPPORT_AGENT: (
                        "Now that we've addressed {context}, "
                        "let me ask about {topic}."
                    )
                },
                metadata={"response_type": ResponseType.FOLLOW_UP}
            ),
            ResponseTemplate(
                name="suggestion",
                content="Have you considered {suggestion}?",
                variables=set(),
                conditions={"type": ResponseType.SUGGESTION},
                role_specific={
                    Role.SUPPORT_AGENT: (
                        "One solution would be to {suggestion}. "
                        "Would you like to try that?"
                    ),
                    Role.MEETING_HOST: (
                        "I suggest we {suggestion}. "
                        "What do you think?"
                    )
                },
                metadata={"response_type": ResponseType.SUGGESTION}
            ),
            ResponseTemplate(
                name="summary",
                content=(
                    "To summarize the key points:\n"
                    "{points}"
                ),
                variables=set(),
                conditions={"type": ResponseType.SUMMARY},
                role_specific={
                    Role.MEETING_HOST: (
                        "Let me recap our discussion:\n{points}\n"
                        "Have I missed anything important?"
                    ),
                    Role.INTERVIEWER: (
                        "Based on our discussion:\n{points}\n"
                        "Is this an accurate summary?"
                    )
                },
                metadata={"response_type": ResponseType.SUMMARY}
            )
        ]

        for template in defaults:
            asyncio.create_task(self.add_template(template))


class TemplateRenderer:
    """Renders templates with context and analysis."""

    def __init__(
            self,
            template_manager: TemplateManager,
            event_bus: EventBus
    ):
        self.templates = template_manager
        self.event_bus = event_bus

    async def render_response(
            self,
            request: ResponseRequest,
            template_name: Optional[str] = None
    ) -> AsyncIterator[str]:
        """Render response from template."""
        # Find appropriate template
        if not template_name:
            templates = self.templates.find_templates(
                request.response_type,
                request.role
            )
            if not templates:
                raise TemplateError(
                    "No matching template found",
                    "unknown",
                    {}
                )
            template = templates[0]  # Use first matching template
        else:
            template = self.templates.get_template(template_name)
            if not template:
                raise TemplateError(
                    f"Template not found: {template_name}",
                    template_name,
                    {}
                )

        # Prepare variables
        variables = await self._prepare_variables(
            template,
            request
        )

        # Render template
        try:
            rendered = await self.templates.render_template(
                template.name,
                variables,
                request.role
            )

            yield rendered

            # Emit render event
            await self.event_bus.publish(Event(
                type=EventType.RESPONSE_RECEIVED,
                data={
                    "action": "template_rendered",
                    "template": template.name,
                    "role": request.role
                }
            ))

        except Exception as e:
            raise TemplateError(
                f"Render failed: {str(e)}",
                template.name,
                variables
            )

    async def _prepare_variables(
            self,
            template: ResponseTemplate,
            request: ResponseRequest
    ) -> Dict[str, Any]:
        """Prepare variables for template.

        Args:
            template: Template to render
            request: Response request

        Returns:
            Template variables
        """
        variables = {}

        # Extract from request
        variables["query"] = request.query
        if request.role:
            variables["role"] = request.role.value

        # Extract from context
        if request.context:
            for key, value in request.context.content.items():
                if isinstance(value, (str, int, float, bool)):
                    variables[key] = value

        # Extract from analysis
        if request.analysis:
            for key, value in request.analysis.content.items():
                if isinstance(value, (str, int, float, bool)):
                    variables[f"analysis_{key}"] = value

        return variables