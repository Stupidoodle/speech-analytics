import pytest
from unittest.mock import AsyncMock
import json
from datetime import datetime

from src.assistance.enhanced_assistant import (
    EnhancedAssistant,
    AssistanceResponse,
    Role
)
from src.conversation.context import ConversationContext
from src.conversation.manager import ConversationManager


@pytest.fixture
async def mock_ai_responses():
    return {
        'interviewer': [
            {
                "type": "follow_up",
                "suggestion": "Can you describe a challenging project?",
                "confidence": 0.9,
                "priority": 0.8,
                "context": {"skill_area": "experience"}
            }
        ],
        'support_agent': [
            {
                "type": "clarification",
                "suggestion": "Could you specify the error message?",
                "confidence": 0.85,
                "priority": 0.9,
                "context": {"issue_type": "technical"}
            }
        ]
    }


@pytest.fixture
async def mock_conversation_manager(mock_ai_responses):
    manager = AsyncMock(spec=ConversationManager)

    async def mock_send_message(prompt):
        if "interviewer" in prompt.lower():
            yield AsyncMock(text=json.dumps(mock_ai_responses['interviewer']))
        else:
            yield AsyncMock(
                text=json.dumps(
                    mock_ai_responses['support_agent']
                    )
                )

    manager.send_message.side_effect = mock_send_message
    return manager


@pytest.fixture
async def context():
    return ConversationContext()


@pytest.mark.asyncio
async def test_role_specific_assistance(mock_conversation_manager, context):
    """Test role-specific assistance generation."""
    # Test interviewer role
    interviewer_assistant = EnhancedAssistant(
        context,
        mock_conversation_manager,
        Role.INTERVIEWER
    )

    responses = []
    async for response in interviewer_assistant.process_turn(
        "I worked on several projects",
        "candidate"
    ):
        responses.append(response)

    assert len(responses) == 1
    assert responses[0].type == "follow_up"
    assert "challenging project" in responses[0].suggestion

    # Test support agent role
    support_assistant = EnhancedAssistant(
        context,
        mock_conversation_manager,
        Role.SUPPORT_AGENT
    )

    responses = []
    async for response in support_assistant.process_turn(
        "I'm having an issue with the system",
        "customer"
    ):
        responses.append(response)

    assert len(responses) == 1
    assert responses[0].type == "clarification"
    assert "error message" in responses[0].suggestion


@pytest.mark.asyncio
async def test_assistance_history(mock_conversation_manager, context):
    """Test assistance history tracking."""
    assistant = EnhancedAssistant(
        context,
        mock_conversation_manager,
        Role.INTERVIEWER
    )

    # Process multiple turns
    turns = [
        "Tell me about your experience",
        "What technologies do you use?",
        "How do you handle challenges?"
    ]

    for turn in turns:
        async for response in assistant.process_turn(turn, "interviewer"):
            pass

    assert len(assistant.assistance_history) == len(turns)
    assert all(
        isinstance(r, AssistanceResponse)
        for r in assistant.assistance_history
    )
    assert all(
        isinstance(r.timestamp, datetime)
        for r in assistant.assistance_history
    )


@pytest.mark.asyncio
async def test_invalid_ai_response(mock_conversation_manager, context):
    """Test handling of invalid AI responses."""
    # Mock AI returning invalid JSON
    mock_conversation_manager.send_message.side_effect = [
        [AsyncMock(text="invalid json")]
    ]

    assistant = EnhancedAssistant(
        context,
        mock_conversation_manager,
        Role.INTERVIEWER
    )

    responses = []
    async for response in assistant.process_turn(
        "Test message",
        "speaker"
    ):
        responses.append(response)

    assert len(responses) == 0
    assert len(assistant.assistance_history) == 0


@pytest.mark.asyncio
async def test_assistance_priority_filtering(mock_conversation_manager,
                                             context
                                             ):
    """Test filtering of low-priority assistance."""
    # Mock AI returning mixed priority responses
    mock_responses = [
        {
            "type": "follow_up",
            "suggestion": "High priority question",
            "confidence": 0.9,
            "priority": 0.9,
            "context": {}
        },
        {
            "type": "insight",
            "suggestion": "Low priority insight",
            "confidence": 0.7,
            "priority": 0.3,
            "context": {}
        }
    ]

    mock_conversation_manager.send_message.side_effect = [
        [AsyncMock(text=json.dumps(mock_responses))]
    ]

    assistant = EnhancedAssistant(
        context,
        mock_conversation_manager,
        Role.INTERVIEWER
    )
    assistant.priority_threshold = 0.5

    responses = []
    async for response in assistant.process_turn(
        "Test message",
        "speaker"
    ):
        responses.append(response)

    assert len(responses) == 1
    assert responses[0].suggestion == "High priority question"
