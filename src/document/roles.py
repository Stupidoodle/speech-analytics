"""Role-specific document processing behaviors."""
from typing import Dict, Any, List
from dataclasses import dataclass

from .types import DocumentType


@dataclass
class DocumentRole:
    """Role configuration for document processing."""
    priorities: List[str]
    required_fields: List[str]
    extraction_rules: Dict[str, Any]
    system_prompts: Dict[DocumentType, str]
    context_prompts: Dict[DocumentType, str]


class DocumentRoles:
    """Document processing configurations by role."""

    @staticmethod
    def get_interviewer() -> DocumentRole:
        """Get interviewer role configuration."""
        return DocumentRole(
            priorities=[
                "technical_skills",
                "experience",
                "achievements",
                "project_details",
                "education"
            ],
            required_fields=[
                "skills",
                "experience_timeline",
                "technical_expertise",
                "project_responsibilities"
            ],
            extraction_rules={
                "skills": {"min_confidence": 0.8},
                "experience": {"require_dates": True},
                "achievements": {"require_metrics": True}
            },
            system_prompts={
                DocumentType.CV: """Analyze this CV focusing on interview preparation:

Key Areas:
1. Technical Expertise
   - Skill levels and proficiency
   - Technology exposure
   - Tools and frameworks
   - Recent vs. historical experience

2. Experience Validation
   - Role responsibilities
   - Project complexities
   - Team dynamics
   - Problem-solving examples

3. Achievement Assessment
   - Quantifiable impacts
   - Project successes
   - Technical challenges overcome
   - Innovation examples

4. Growth Indicators
   - Career progression
   - Learning patterns
   - Initiative examples
   - Leadership experience

Provide structured JSON with confidence scores and areas for deeper questioning.
Flag any inconsistencies or areas needing clarification.""",

                DocumentType.JOB_DESCRIPTION: """Analyze this job description for interview planning:

Extract:
1. Core Requirements
   - Must-have skills
   - Essential experience
   - Technical prerequisites
   - Team fit criteria

2. Assessment Areas
   - Technical competencies
   - Project experience
   - Architectural knowledge
   - Problem-solving capabilities

3. Role Context
   - Team structure
   - Project scope
   - Technical environment
   - Growth opportunities

4. Discussion Points
   - Key responsibilities
   - Technical challenges
   - Success criteria
   - Cultural alignment

Provide structured JSON with priority levels for each requirement and suggested validation approaches."""
            },
            context_prompts={
                DocumentType.CV: """Consider for next questions:
- Technical depth in {skill_area}
- Project complexity in {experience_area}
- Team leadership in {project_context}
- Problem-solving approach for {technical_challenge}""",

                DocumentType.JOB_DESCRIPTION: """Focus next questions on:
- Experience alignment with {requirement}
- Technical expertise in {technology_stack}
- Similar challenges to {project_type}
- Team fit for {team_context}"""
            }
        )

    @staticmethod
    def get_interviewee() -> DocumentRole:
        """Get interviewee role configuration."""
        return DocumentRole(
            priorities=[
                "job_requirements",
                "technical_stack",
                "team_context",
                "growth_opportunities"
            ],
            required_fields=[
                "requirements",
                "responsibilities",
                "team_structure",
                "technical_environment"
            ],
            extraction_rules={
                "requirements": {"separate_must_have": True},
                "technical_stack": {"include_versions": True}
            },
            system_prompts={
                DocumentType.JOB_DESCRIPTION: """Analyze this job description from a candidate perspective:

Focus Areas:
1. Requirement Matching
   - Core technical requirements
   - Experience alignment
   - Skill prerequisites
   - Team structure

2. Opportunity Assessment
   - Growth potential
   - Technical challenges
   - Project scope
   - Learning opportunities

3. Discussion Points
   - Technical environment
   - Team dynamics
   - Project responsibilities
   - Success metrics

4. Preparation Areas
   - Required demos
   - Technical questions
   - Project examples
   - Achievement metrics

Provide structured JSON with preparation points and question suggestions."""
            },
            context_prompts={
                DocumentType.JOB_DESCRIPTION: """Prepare responses about:
- Experience with {technology}
- Similar projects to {project_type}
- Achievements related to {requirement}
- Leadership in {team_context}"""
            }
        )

    @staticmethod
    def get_support_agent() -> DocumentRole:
        """Get support agent role configuration."""
        return DocumentRole(
            priorities=[
                "troubleshooting_steps",
                "technical_specifications",
                "common_issues",
                "solution_paths"
            ],
            required_fields=[
                "issue_resolution",
                "technical_details",
                "limitations",
                "prerequisites"
            ],
            extraction_rules={
                "troubleshooting": {"require_steps": True},
                "solutions": {"include_alternatives": True}
            },
            system_prompts={
                DocumentType.TECHNICAL_SPEC: """Analyze this technical documentation for support:

Extract:
1. Issue Resolution
   - Common problems
   - Troubleshooting steps
   - Solution paths
   - Validation checks

2. Technical Context
   - System requirements
   - Configuration details
   - Dependencies
   - Limitations

3. User Guidance
   - Setup steps
   - Usage patterns
   - Best practices
   - Common mistakes

4. Escalation Paths
   - Critical issues
   - Known limitations
   - Complex scenarios
   - Security concerns

Provide structured JSON with solution paths and escalation criteria."""
            },
            context_prompts={
                DocumentType.TECHNICAL_SPEC: """Guide resolution for:
- Configuration of {feature}
- Troubleshooting {issue_type}
- Limitations around {capability}
- Setup for {environment}"""
            }
        )

    @staticmethod
    def get_customer() -> DocumentRole:
        """Get customer role configuration."""
        return DocumentRole(
            priorities=[
                "user_guides",
                "feature_descriptions",
                "limitations",
                "requirements"
            ],
            required_fields=[
                "setup_steps",
                "prerequisites",
                "usage_instructions",
                "troubleshooting"
            ],
            extraction_rules={
                "instructions": {"step_by_step": True},
                "requirements": {"clear_prerequisites": True}
            },
            system_prompts={
                DocumentType.PRODUCT_MANUAL: """Analyze this product documentation for usage:

Focus on:
1. Setup & Configuration
   - System requirements
   - Installation steps
   - Initial setup
   - Configuration options

2. Feature Usage
   - Core functionality
   - Common workflows
   - Best practices
   - Limitations

3. Troubleshooting
   - Common issues
   - Resolution steps
   - Warning signs
   - Support contacts

4. Requirements
   - Prerequisites
   - Dependencies
   - Compatibility
   - Resource needs

Provide structured JSON with clear instructions and troubleshooting paths."""
            },
            context_prompts={
                DocumentType.PRODUCT_MANUAL: """Consider for usage:
- Setup requirements for {feature}
- Steps to configure {capability}
- Solutions for {issue_type}
- Requirements for {environment}"""
            }
        )

    @staticmethod
    def get_meeting_host() -> DocumentRole:
        """Get meeting host role configuration."""
        return DocumentRole(
            priorities=[
                "agenda_items",
                "discussion_points",
                "action_items",
                "decisions"
            ],
            required_fields=[
                "objectives",
                "participants",
                "timelines",
                "outcomes"
            ],
            extraction_rules={
                "agenda": {"time_allocation": True},
                "actions": {"assign_owners": True}
            },
            system_prompts={
                DocumentType.MEETING_NOTES: """Analyze these meeting documents for facilitation:

Focus Areas:
1. Agenda Management
   - Key topics
   - Time allocations
   - Priority items
   - Discussion points

2. Action Tracking
   - Task assignments
   - Deadlines
   - Dependencies
   - Follow-ups

3. Decision Points
   - Key decisions
   - Approval needs
   - Impact areas
   - Next steps

4. Participant Engagement
   - Role assignments
   - Input needs
   - Preparation requirements
   - Expected outputs

Provide structured JSON with facilitation points and tracking items."""
            },
            context_prompts={
                DocumentType.MEETING_NOTES: """Guide discussion on:
- Progress of {action_item}
- Decision on {topic}
- Updates from {team}
- Timeline for {deliverable}"""
            }
        )

    @staticmethod
    def get_meeting_participant() -> DocumentRole:
        """Get meeting participant role configuration."""
        return DocumentRole(
            priorities=[
                "preparation_needs",
                "contribution_areas",
                "action_items",
                "follow_ups"
            ],
            required_fields=[
                "agenda",
                "preparation",
                "contributions",
                "actions"
            ],
            extraction_rules={
                "preparation": {"required_materials": True},
                "actions": {"personal_tasks": True}
            },
            system_prompts={
                DocumentType.MEETING_NOTES: """Analyze these meeting documents for participation:

Focus Areas:
1. Preparation Needs
   - Required reading
   - Data preparation
   - Discussion points
   - Updates needed

2. Contribution Areas
   - Update points
   - Decision inputs
   - Expertise needs
   - Discussion topics

3. Action Items
   - Personal tasks
   - Team dependencies
   - Deadlines
   - Deliverables

4. Follow-up Points
   - Next steps
   - Validation needs
   - Collaboration points
   - Report backs

Provide structured JSON with preparation needs and contribution points."""
            },
            context_prompts={
                DocumentType.MEETING_NOTES: """Prepare updates on:
- Progress of {task}
- Status of {deliverable}
- Input for {decision}
- Collaboration on {project}"""
            }
        )

    @classmethod
    def get_role_config(cls, role_name: str) -> DocumentRole:
        """Get role configuration by name."""
        role_map = {
            "interviewer": cls.get_interviewer,
            "interviewee": cls.get_interviewee,
            "support_agent": cls.get_support_agent,
            "customer": cls.get_customer,
            "meeting_host": cls.get_meeting_host,
            "meeting_participant": cls.get_meeting_participant
        }

        if role_name not in role_map:
            raise ValueError(f"Unknown role: {role_name}")

        return role_map[role_name]()