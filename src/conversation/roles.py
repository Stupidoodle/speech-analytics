"""Role-specific behavior and configuration."""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from .types import (
    Role,
    SystemPrompt,
    ToolConfig,
    MessageContent,
    MessageType
)


@dataclass
class RolePrompts:
    """Collection of prompts for a role."""
    base_prompt: str
    context_prompts: Dict[str, str]
    tool_prompts: Dict[str, str]
    error_prompts: Dict[str, str]
    custom_prompts: Dict[str, str] = field(default_factory=dict)


@dataclass
class RoleConfig:
    """Configuration for role behavior."""
    allowed_tools: List[str]
    required_context: List[str]
    response_guidelines: Dict[str, Any]
    validation_rules: Dict[str, Any]
    metadata_requirements: Dict[str, Any]


class RoleManager:
    """Manages role-specific behavior and configuration."""

    def __init__(self):
        """Initialize role manager."""
        self.role_prompts = self._initialize_prompts()
        self.role_configs = self._initialize_configs()
        self.tool_configs = self._initialize_tools()

    def get_system_prompts(
        self,
        role: Role,
        context: Optional[Dict[str, Any]] = None
    ) -> List[SystemPrompt]:
        """Get system prompts for role.

        Args:
            role: User role
            context: Optional context variables

        Returns:
            List of system prompts
        """
        prompts = []
        role_prompts = self.role_prompts.get(role)
        if not role_prompts:
            return prompts

        # Add base role prompt
        prompts.append(SystemPrompt(
            text=role_prompts.base_prompt,
            metadata={"type": "base_prompt"},
            priority=1
        ))

        # Add context-specific prompts
        if context:
            for context_type, prompt_template in role_prompts.context_prompts.items():
                if context_type in context:
                    try:
                        prompt_text = prompt_template.format(**context)
                        prompts.append(SystemPrompt(
                            text=prompt_text,
                            metadata={
                                "type": "context_prompt",
                                "context_type": context_type
                            },
                            priority=2
                        ))
                    except KeyError:
                        continue

        # Add tool-specific prompts
        config = self.role_configs.get(role)
        if config:
            for tool in config.allowed_tools:
                tool_prompt = role_prompts.tool_prompts.get(tool)
                if tool_prompt:
                    prompts.append(SystemPrompt(
                        text=tool_prompt,
                        metadata={
                            "type": "tool_prompt",
                            "tool": tool
                        },
                        priority=3
                    ))

        return sorted(prompts, key=lambda x: x.priority, reverse=True)

    def get_tools(
        self,
        role: Role
    ) -> List[ToolConfig]:
        """Get allowed tools for role.

        Args:
            role: User role

        Returns:
            List of tool configurations
        """
        config = self.role_configs.get(role)
        if not config:
            return []

        return [
            self.tool_configs[tool]
            for tool in config.allowed_tools
            if tool in self.tool_configs
        ]

    def validate_message(
        self,
        role: Role,
        content: List[MessageContent]
    ) -> List[Dict[str, Any]]:
        """Validate message for role.

        Args:
            role: User role
            content: Message content

        Returns:
            List of validation errors
        """
        errors = []
        config = self.role_configs.get(role)
        if not config:
            return errors

        # Check tool usage
        for item in content:
            if item.type == MessageType.TOOL_USE:
                tool_name = item.tool_use.get("name")
                if tool_name not in config.allowed_tools:
                    errors.append({
                        "type": "tool_error",
                        "message": f"Tool not allowed for role: {tool_name}"
                    })

        # Check metadata requirements
        metadata_reqs = config.metadata_requirements
        for item in content:
            for field, requirement in metadata_reqs.items():
                if requirement.get("required", False):
                    if not item.metadata.get(field):
                        errors.append({
                            "type": "metadata_error",
                            "message": f"Required metadata missing: {field}"
                        })

        return errors

    def get_error_prompt(
        self,
        role: Role,
        error_type: str
    ) -> Optional[str]:
        """Get error handling prompt for role.

        Args:
            role: User role
            error_type: Type of error

        Returns:
            Error prompt if found
        """
        role_prompts = self.role_prompts.get(role)
        if not role_prompts:
            return None

        return role_prompts.error_prompts.get(error_type)

    def _initialize_prompts(self) -> Dict[Role, RolePrompts]:
        """Initialize role-specific prompts.

        Returns:
            Role prompts mapping
        """
        return {
            Role.INTERVIEWER: RolePrompts(
                base_prompt=(
                    "You are conducting an interview. Focus on:\n"
                    "- Evaluating candidate responses\n"
                    "- Asking targeted follow-up questions\n"
                    "- Assessing technical and soft skills\n"
                    "- Maintaining structured interview flow\n"
                    "- Documenting key insights and red flags"
                ),
                context_prompts={
                    "cv": (
                        "Review the candidate's CV focusing on:\n"
                        "- Relevant experience: {experience}\n"
                        "- Technical skills: {skills}\n"
                        "- Project history: {projects}\n"
                        "- Areas to explore: {areas_of_interest}"
                    ),
                    "job": (
                        "Consider the job requirements:\n"
                        "- Required skills: {required_skills}\n"
                        "- Experience level: {experience_level}\n"
                        "- Team context: {team_context}\n"
                        "- Project scope: {project_scope}"
                    )
                },
                tool_prompts={
                    "note_taking": "Document important points and red flags.",
                    "skill_assessment": "Use structured assessment criteria.",
                    "question_bank": "Access standard interview questions."
                },
                error_prompts={
                    "off_topic": "Redirect to relevant interview topics.",
                    "unclear_response": "Ask for clarification.",
                    "technical_detail": "Probe for specific examples."
                }
            ),
            Role.INTERVIEWEE: RolePrompts(
                base_prompt=(
                    "You are a job candidate in an interview. Focus on:\n"
                    "- Providing clear, structured responses\n"
                    "- Using STAR method for experiences\n"
                    "- Demonstrating relevant skills\n"
                    "- Asking insightful questions\n"
                    "- Showing enthusiasm and cultural fit"
                ),
                context_prompts={
                    "job": (
                        "Consider the job requirements:\n"
                        "- Required skills: {required_skills}\n"
                        "- Experience level: {experience_level}\n"
                        "- Role responsibilities: {responsibilities}\n"
                        "- Team structure: {team_structure}"
                    ),
                    "company": (
                        "Consider company context:\n"
                        "- Company culture: {culture}\n"
                        "- Industry focus: {industry}\n"
                        "- Growth stage: {stage}\n"
                        "- Tech stack: {tech_stack}"
                    )
                },
                tool_prompts={
                    "experience_sharing": "Share relevant experiences clearly.",
                    "skill_demonstration": "Demonstrate technical knowledge.",
                    "question_asking": "Ask informed questions about the role."
                },
                error_prompts={
                    "unclear_question": "Ask for clarification.",
                    "technical_gap": "Acknowledge and show learning ability.",
                    "missing_context": "Request additional information."
                }
            ),
            Role.SUPPORT_AGENT: RolePrompts(
                base_prompt=(
                    "You are providing customer support. Focus on:\n"
                    "- Understanding customer issues\n"
                    "- Providing clear solutions\n"
                    "- Following up on resolution\n"
                    "- Maintaining empathy\n"
                    "- Documenting interactions"
                ),
                context_prompts={
                    "ticket": (
                        "Consider ticket details:\n"
                        "- Issue type: {issue_type}\n"
                        "- Priority: {priority}\n"
                        "- Previous interactions: {history}\n"
                        "- Current status: {status}"
                    ),
                    "customer": (
                        "Consider customer context:\n"
                        "- Account type: {account_type}\n"
                        "- Service level: {service_level}\n"
                        "- Region: {region}\n"
                        "- Language: {language}"
                    )
                },
                tool_prompts={
                    "knowledge_base": "Access support documentation.",
                    "ticket_management": "Update ticket status and notes.",
                    "escalation": "Follow escalation procedures."
                },
                error_prompts={
                    "system_error": "Explain technical difficulties.",
                    "policy_violation": "Explain policy constraints.",
                    "service_unavailable": "Provide alternative solutions."
                }
            ),
            Role.CUSTOMER: RolePrompts(
                base_prompt=(
                    "You are seeking customer support. Focus on:\n"
                    "- Describing issues clearly\n"
                    "- Providing relevant details\n"
                    "- Following troubleshooting steps\n"
                    "- Confirming resolution\n"
                    "- Giving feedback on solutions"
                ),
                context_prompts={
                    "product": (
                        "Regarding your product:\n"
                        "- Product type: {product_type}\n"
                        "- Version: {version}\n"
                        "- Purchase date: {purchase_date}\n"
                        "- Previous issues: {issue_history}"
                    ),
                    "support_history": (
                        "Previous support context:\n"
                        "- Recent tickets: {recent_tickets}\n"
                        "- Known issues: {known_issues}\n"
                        "- Applied solutions: {applied_solutions}"
                    )
                },
                tool_prompts={
                    "issue_reporting": "Report problems clearly.",
                    "solution_verification": "Verify solution effectiveness.",
                    "feedback_submission": "Provide solution feedback."
                },
                error_prompts={
                    "solution_failed": "Report solution ineffectiveness.",
                    "missing_steps": "Request detailed instructions.",
                    "technical_difficulty": "Describe technical issues."
                }
            ),
            Role.MEETING_HOST: RolePrompts(
                base_prompt=(
                    "You are hosting a meeting. Focus on:\n"
                    "- Managing discussion flow\n"
                    "- Ensuring participation\n"
                    "- Tracking action items\n"
                    "- Maintaining schedule\n"
                    "- Documenting decisions"
                ),
                context_prompts={
                    "agenda": (
                        "Follow meeting agenda:\n"
                        "- Topics: {topics}\n"
                        "- Time allocations: {time_slots}\n"
                        "- Required decisions: {decisions}\n"
                        "- Participants: {participants}"
                    ),
                    "previous": (
                        "Consider previous meeting:\n"
                        "- Action items: {action_items}\n"
                        "- Pending decisions: {pending_decisions}\n"
                        "- Follow-ups: {follow_ups}"
                    )
                },
                tool_prompts={
                    "time_tracking": "Monitor agenda progress.",
                    "action_items": "Track tasks and assignments.",
                    "minutes": "Record key points and decisions."
                },
                error_prompts={
                    "off_track": "Redirect to agenda items.",
                    "time_overrun": "Adjust schedule accordingly.",
                    "missing_participant": "Manage absent participant items."
                }
            ),
            Role.MEETING_PARTICIPANT: RolePrompts(
                base_prompt=(
                    "You are participating in a meeting. Focus on:\n"
                    "- Contributing relevant insights\n"
                    "- Taking notes on key points\n"
                    "- Engaging in discussions\n"
                    "- Following up on actions\n"
                    "- Supporting meeting objectives"
                ),
                context_prompts={
                    "agenda": (
                        "Meeting context:\n"
                        "- Topics: {topics}\n"
                        "- Your role: {participant_role}\n"
                        "- Expected input: {expected_input}\n"
                        "- Time allocation: {time_allocation}"
                    ),
                    "preparation": (
                        "Preparation materials:\n"
                        "- Required reading: {required_reading}\n"
                        "- Discussion points: {discussion_points}\n"
                        "- Data/metrics: {relevant_data}"
                    )
                },
                tool_prompts={
                    "note_taking": "Record important points.",
                    "contribution_tracking": "Track your contributions.",
                    "action_management": "Manage assigned actions."
                },
                error_prompts={
                    "missed_point": "Request topic revisit.",
                    "technical_issues": "Report connection problems.",
                    "clarification_needed": "Ask for clarification."
                }
            )
        }

    def _initialize_configs(self) -> Dict[Role, RoleConfig]:
        """Initialize role configurations.

        Returns:
            Role configurations mapping
        """
        return {
            Role.INTERVIEWER: RoleConfig(
                allowed_tools=[
                    "note_taking",
                    "skill_assessment",
                    "question_bank"
                ],
                required_context=[
                    "cv",
                    "job_description"
                ],
                response_guidelines={
                    "max_question_length": 100,
                    "follow_up_required": True,
                    "assessment_required": True
                },
                validation_rules={
                    "must_document_response": True,
                    "technical_validation_required": True
                },
                metadata_requirements={
                    "skill_area": {"required": True},
                    "assessment_score": {"required": True}
                }
            ),
            Role.SUPPORT_AGENT: RoleConfig(
                allowed_tools=[
                    "knowledge_base",
                    "ticket_management",
                    "escalation"
                ],
                required_context=[
                    "ticket_history",
                    "customer_info"
                ],
                response_guidelines={
                    "max_response_time": 300,
                    "solution_required": True,
                    "follow_up_required": True
                },
                validation_rules={
                    "must_verify_solution": True,
                    "satisfaction_check_required": True
                },
                metadata_requirements={
                    "ticket_id": {"required": True},
                    "resolution_status": {"required": True}
                }
            ),
            Role.MEETING_HOST: RoleConfig(
                allowed_tools=[
                    "time_tracking",
                    "action_items",
                    "minutes"
                ],
                required_context=[
                    "agenda",
                    "participant_list"
                ],
                response_guidelines={
                    "time_management_required": True,
                    "decision_documentation_required": True
                },
                validation_rules={
                    "must_track_decisions": True,
                    "must_assign_actions": True
                },
                metadata_requirements={
                    "agenda_item": {"required": True},
                    "time_remaining": {"required": True}
                }
            ),
            Role.INTERVIEWEE: RoleConfig(
                allowed_tools=[
                    "experience_sharing",
                    "skill_demonstration",
                    "question_asking"
                ],
                required_context=[
                    "job_description",
                    "company_info"
                ],
                response_guidelines={
                    "use_star_method": True,
                    "example_required": True,
                    "question_preparation": True
                },
                validation_rules={
                    "must_provide_example": True,
                    "relevant_experience_required": True
                },
                metadata_requirements={
                    "experience_area": {"required": True},
                    "skill_relevance": {"required": True}
                }
            ),
            Role.CUSTOMER: RoleConfig(
                allowed_tools=[
                    "issue_reporting",
                    "solution_verification",
                    "feedback_submission"
                ],
                required_context=[
                    "product_info",
                    "issue_history"
                ],
                response_guidelines={
                    "provide_details": True,
                    "follow_instructions": True,
                    "verify_results": True
                },
                validation_rules={
                    "must_describe_issue": True,
                    "must_confirm_steps": True
                },
                metadata_requirements={
                    "product_id": {"required": True},
                    "issue_type": {"required": True}
                }
            ),
            Role.MEETING_PARTICIPANT: RoleConfig(
                allowed_tools=[
                    "note_taking",
                    "contribution_tracking",
                    "action_management"
                ],
                required_context=[
                    "agenda",
                    "preparation_materials"
                ],
                response_guidelines={
                    "relevant_contribution": True,
                    "action_tracking": True,
                    "time_awareness": True
                },
                validation_rules={
                    "must_track_actions": True,
                    "must_engage_appropriately": True
                },
                metadata_requirements={
                    "topic": {"required": True},
                    "contribution_type": {"required": True}
                }
            )
        }

    def _initialize_tools(self) -> Dict[str, ToolConfig]:
        """Initialize tool configurations.

        Returns:
            Tool configurations mapping
        """
        return {
            "note_taking": ToolConfig(
                name="note_taking",
                description="Document interview notes and observations",
                parameters={
                    "topic": "string",
                    "content": "string",
                    "tags": ["string"]
                }
            ),
            "skill_assessment": ToolConfig(
                name="skill_assessment",
                description="Evaluate technical and soft skills",
                parameters={
                    "skill": "string",
                    "score": "integer",
                    "evidence": "string"
                }
            ),
            "knowledge_base": ToolConfig(
                name="knowledge_base",
                description="Search support documentation",
                parameters={
                    "query": "string",
                    "category": "string",
                    "max_results": "integer"
                }
            ),
            "ticket_management": ToolConfig(
                name="ticket_management",
                description="Manage support tickets",
                parameters={
                    "ticket_id": "string",
                    "status": "string",
                    "notes": "string"
                }
            ),
            "time_tracking": ToolConfig(
                name="time_tracking",
                description="Track meeting time usage",
                parameters={
                    "agenda_item": "string",
                    "duration": "integer",
                    "status": "string"
                }
            ),
            "action_items": ToolConfig(
                name="action_items",
                description="Track meeting action items",
                parameters={
                    "description": "string",
                    "assignee": "string",
                    "due_date": "string"
                }
            )
        }