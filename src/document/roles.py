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
    response_format: Dict[DocumentType, str]


class DocumentRoles:
    """Document processing configurations by role."""

    @staticmethod
    def get_interviewer() -> DocumentRole:
        return DocumentRole(
            priorities=[
                "technical_skills",
                "experience",
                "achievements",
                "project_details",
                "education",
            ],
            required_fields=[
                "skills",
                "experience_timeline",
                "technical_expertise",
                "project_responsibilities",
            ],
            extraction_rules={
                "skills": {"min_confidence": 0.8},
                "experience": {"require_dates": True},
                "achievements": {"require_metrics": True},
            },
            system_prompts={
                DocumentType.CV: """Analyze this CV for interview preparation. Focus on:
    1. Technical expertise and skills
    2. Experience validation points
    3. Achievement verification
    4. Growth and potential indicators""",
                DocumentType.JOB_DESCRIPTION: """Analyze this job description for interview planning. Focus on:
    1. Core requirements and must-haves
    2. Technical competency requirements
    3. Project experience needs
    4. Key success criteria""",
            },
            context_prompts={
                DocumentType.CV: "Focus next questions on {skill_area} and {experience_area}",
                DocumentType.JOB_DESCRIPTION: "Align questions with {requirement} and {technology_stack}",
            },
            response_format={
                DocumentType.CV: """{
                        "technical_skills": [
                            {
                                "skill": "skill name",
                                "years_experience": 0,
                                "level": "expert/intermediate/beginner",
                                "recent_usage": "description",
                                "verification_points": ["point1", "point2"]
                            }
                        ],
                        "experience": [
                            {
                                "role": "role title",
                                "duration": "timeframe",
                                "key_projects": ["project1", "project2"],
                                "verification_questions": ["question1", "question2"]
                            }
                        ],
                        "suggested_questions": [
                            {
                                "topic": "topic area",
                                "question": "question text",
                                "follow_ups": ["followup1", "followup2"],
                                "expected_detail_level": "technical depth expected"
                            }
                        ]
                    }"""
            },
        )

    @staticmethod
    def get_interviewee() -> DocumentRole:
        return DocumentRole(
            priorities=[
                "job_requirements",
                "technical_stack",
                "team_context",
                "growth_opportunities",
            ],
            required_fields=[
                "requirements",
                "responsibilities",
                "team_structure",
                "technical_environment",
            ],
            extraction_rules={
                "requirements": {"separate_must_have": True},
                "technical_stack": {"include_versions": True},
            },
            system_prompts={
                DocumentType.JOB_DESCRIPTION: """Analyze this job description from a candidate perspective. Focus on:
1. Key technical requirements and how to demonstrate them
2. Project experience to highlight
3. Potential discussion points
4. Growth opportunities to explore""",
                DocumentType.CV: """Review CV for interview preparation. Focus on:
1. Alignment with job requirements
2. Key achievements to highlight
3. Project details to expand on
4. Technical expertise demonstration points""",
            },
            context_prompts={
                DocumentType.JOB_DESCRIPTION: "Prepare responses about {requirement} and {technical_stack}",
                DocumentType.CV: "Highlight experience with {skill} in {project_context}",
            },
            response_format={
                DocumentType.JOB_DESCRIPTION: """{
                    "key_requirements": [
                        {
                            "requirement": "requirement description",
                            "your_experience": "relevant experience",
                            "talking_points": ["point1", "point2"],
                            "potential_questions": ["question1", "question2"]
                        }
                    ],
                    "technical_preparation": [
                        {
                            "area": "technical area",
                            "experience_highlights": ["highlight1", "highlight2"],
                            "example_scenarios": ["scenario1", "scenario2"]
                        }
                    ],
                    "discussion_topics": [
                        {
                            "topic": "topic area",
                            "your_experience": "experience summary",
                            "key_points": ["point1", "point2"]
                        }
                    ]
                }""",
                DocumentType.CV: """{
                    "alignment_with_requirements": [
                        {
                            "requirement": "requirement description",
                            "your_experience": "relevant experience or achievement",
                            "action_plan": ["action1", "action2"]
                        }
                    ],
                    "key_achievements": [
                        {
                            "achievement": "achievement description",
                            "context": "context in which the achievement was made",
                            "impact": "result or outcome",
                            "talking_points": ["point1", "point2"]
                        }
                    ],
                    "project_expansion": [
                        {
                            "project_name": "name of the project",
                            "your_role": "specific role in the project",
                            "key_contributions": ["contribution1", "contribution2"],
                            "technologies_used": ["tech1", "tech2"],
                            "learning_opportunities": ["learning1", "learning2"]
                        }
                    ],
                    "technical_expertise": [
                        {
                            "area": "technical area",
                            "tools_and_technologies": ["tool1", "tool2"],
                            "relevant_experience": "summary of relevant experience",
                            "example_projects": ["project1", "project2"]
                        }
                    ]
                }""",
            },
        )

    @staticmethod
    def get_support_agent() -> DocumentRole:
        """Support agent configuration with structured response format."""
        return DocumentRole(
            priorities=[
                "troubleshooting_steps",
                "technical_specifications",
                "common_issues",
                "solution_paths",
            ],
            required_fields=[
                "issue_resolution",
                "technical_details",
                "limitations",
                "prerequisites",
            ],
            extraction_rules={
                "troubleshooting": {"require_steps": True},
                "solutions": {"include_alternatives": True},
            },
            system_prompts={
                DocumentType.TECHNICAL_SPEC: """Analyze this technical documentation for support. Focus on:
1. Common issues and resolutions
2. Technical requirements and limitations
3. Troubleshooting procedures
4. Solution alternatives""",
                DocumentType.SUPPORT_GUIDE: """Review support documentation. Focus on:
1. Issue identification steps
2. Resolution procedures
3. Escalation criteria
4. Customer communication points""",
            },
            context_prompts={
                DocumentType.TECHNICAL_SPEC: "Guide resolution for {issue_type} in {environment}",
                DocumentType.SUPPORT_GUIDE: "Follow procedure for {issue} with {configuration}",
            },
            response_format={
                DocumentType.TECHNICAL_SPEC: """{
                    "issues": [
                        {
                            "problem": "issue description",
                            "symptoms": ["symptom1", "symptom2"],
                            "resolution_steps": ["step1", "step2"],
                            "verification": ["check1", "check2"]
                        }
                    ],
                    "technical_requirements": {
                        "prerequisites": ["req1", "req2"],
                        "limitations": ["limit1", "limit2"],
                        "compatibility": ["comp1", "comp2"]
                    },
                    "troubleshooting_guides": [
                        {
                            "scenario": "problem scenario",
                            "diagnosis": ["step1", "step2"],
                            "solutions": ["solution1", "solution2"],
                            "escalation_criteria": ["criterion1", "criterion2"]
                        }
                    ]
                }"""
            },
        )

    @staticmethod
    def get_meeting_host() -> DocumentRole:
        """Get meeting host role configuration."""
        return DocumentRole(
            priorities=[
                "agenda_items",
                "discussion_points",
                "action_items",
                "decisions",
            ],
            required_fields=["objectives", "participants", "timelines", "outcomes"],
            extraction_rules={
                "agenda": {"time_allocation": True},
                "actions": {"assign_owners": True},
            },
            system_prompts={
                DocumentType.MEETING_NOTES: """Analyze these meeting documents for facilitation:
1. Agenda Management
2. Action Tracking
3. Decision Points
4. Participant Engagement"""
            },
            context_prompts={
                DocumentType.MEETING_NOTES: "Track progress on {action_item} and {deliverable}"
            },
            response_format={
                DocumentType.MEETING_NOTES: """{
        	    "agenda_items": [
        	        {
        	            "topic": "discussion topic",
        	            "time_allocated": "duration in minutes",
        	            "presenter": "person responsible"
        	        }
        	    ],
        	    "discussion_points": [
        	        {
        	            "topic": "discussion topic",
        	            "key_arguments": ["point1", "point2"],
        	            "decisions": "conclusions reached"
        	        }
        	    ],
        	    "action_items": [
        	        {
        	            "task": "task description",
        	            "assignee": "person responsible",
        	            "due_date": "deadline",
        	            "follow_up": ["follow_up_action1", "follow_up_action2"]
        	        }
        	    ]
        	}"""
            },
        )

    @staticmethod
    def get_meeting_participant() -> DocumentRole:
        return DocumentRole(
            priorities=[
                "preparation_needs",
                "contribution_areas",
                "action_items",
                "follow_ups",
            ],
            required_fields=["agenda", "preparation", "contributions", "actions"],
            extraction_rules={
                "preparation": {"required_materials": True},
                "actions": {"personal_tasks": True},
            },
            system_prompts={
                DocumentType.MEETING_NOTES: """Analyze these meeting documents from a participant perspective. Focus on:
1. Required preparation and materials
2. Expected contributions
3. Action items and responsibilities
4. Follow-up requirements""",
                DocumentType.TECHNICAL_SPEC: """Review technical documentation for meeting participation. Focus on:
1. Technical requirements
2. Discussion points
3. Decision requirements
4. Implementation considerations""",
            },
            context_prompts={
                DocumentType.MEETING_NOTES: "Prepare updates on {task} and {deliverable}",
                DocumentType.TECHNICAL_SPEC: "Review technical aspects of {feature} for discussion",
            },
            response_format={
                DocumentType.MEETING_NOTES: """{
                    "preparation_needs": [
                        {
                            "topic": "topic area",
                            "materials_needed": ["item1", "item2"],
                            "review_points": ["point1", "point2"]
                        }
                    ],
                    "contribution_areas": [
                        {
                            "topic": "discussion topic",
                            "your_input": "planned contribution",
                            "required_info": ["info1", "info2"]
                        }
                    ],
                    "action_items": [
                        {
                            "task": "task description",
                            "deadline": "timeframe",
                            "dependencies": ["dep1", "dep2"]
                        }
                    ]
                }"""
            },
        )

    @classmethod
    def get_role_config(cls, role_name: str) -> DocumentRole:
        """Get role configuration by name."""
        role_map = {
            "interviewer": cls.get_interviewer,
            "interviewee": cls.get_interviewee,
            "support_agent": cls.get_support_agent,
            "meeting_host": cls.get_meeting_host,
            "meeting_participant": cls.get_meeting_participant,
        }

        if role_name not in role_map:
            raise ValueError(f"Unknown role: {role_name}")

        return role_map[role_name]()
