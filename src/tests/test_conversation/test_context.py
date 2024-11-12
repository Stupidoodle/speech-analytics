import pytest
from unittest.mock import AsyncMock
import json

from src.conversation.context import (
    ConversationContext,
    ConversationTurn,
    ConversationManager
)


@pytest.fixture
async def mock_conversation_manager():
    manager = AsyncMock(spec=ConversationManager)

    async def mock_send_message(prompt):
        if "analyze" in prompt.lower():
            yield AsyncMock(text=json.dumps({
                "intent": "question",
                "tone": "professional",
                "key_points": ["technical inquiry"],
                "topics": ["experience", "skills"],
                "flow_impact": "initiating discussion"
            }))

    manager.send_message.side_effect = mock_send_message
    return manager


@pytest.fixture
async def context(mock_conversation_manager):
    return ConversationContext(mock_conversation_manager)


@pytest.mark.asyncio
async def test_turn_analysis(context):
    """Test conversation turn analysis."""
    turn = await context.add_turn(
        "interviewer",
        "What's your experience with Python?",
        {"topic": "technical_skills"}
    )

    assert turn.analysis is not None
    assert "intent" in turn.analysis
    assert "topics" in turn.analysis
    assert turn.analysis["tone"] == "professional"
    assert any("technical" in point for point in turn.analysis["key_points"])


@pytest.mark.asyncio
async def test_context_history(context):
    """Test conversation history management."""
    # Add multiple turns
    turns = [
        ("interviewer", "Tell me about yourself", {"phase": "intro"}),
        ("candidate", "I'm a software engineer", {"phase": "intro"}),
        ("interviewer", "What's your tech stack?", {"phase": "technical"})
    ]

    for speaker, content, ctx in turns:
        await context.add_turn(speaker, content, ctx)

    # Test recent context retrieval
    recent = context.get_recent_context(2)
    assert len(recent) == 2
    assert recent[-1].content == "What's your tech stack?"

    # Test full history
    assert len(context.turns) == 3
    assert all(isinstance(turn, ConversationTurn) for turn in context.turns)


@pytest.mark.asyncio
async def test_topic_tracking(context):
    """Test conversation topic tracking."""
    # Add turns with different topics
    await context.add_turn(
        "interviewer",
        "Let's discuss Python",
        {"topic": "python"}
    )
    await context.add_turn(
        "candidate",
        "I use Python for ML projects",
        {"topic": "python"}
    )
    await context.add_turn(
        "interviewer",
        "What about cloud services?",
        {"topic": "cloud"}
    )

    # Check topic transitions
    topics = [turn.context.get("topic") for turn in context.turns]
    assert "python" in topics
    assert "cloud" in topics
    assert topics.count("python") == 2


@pytest.mark.asyncio
async def test_error_handling(context):
    """Test error handling in turn analysis."""
    # Mock AI returning invalid JSON
    context.ai.send_message.side_effect = [
        [AsyncMock(text="invalid json")]
    ]

    turn = await context.add_turn(
        "interviewer",
        "Test message",
        {}
    )

    assert turn.analysis == {}
    assert turn in context.turns


@pytest.mark.asyncio
async def test_conversation_summary(context):
    """Test conversation summarization."""
    # Add some conversation turns
    await context.add_turn(
        "interviewer",
        "What's your background?",
        {"phase": "intro"}
    )
    await context.add_turn(
        "candidate",
        "I have a CS degree and 5 years experience",
        {"phase": "intro"}
    )

    summary = await context.get_summary()
    assert isinstance(summary, dict)
    assert "topics" in summary
    assert any(
        "experience" in topic.lower()
        for topic in summary["topics"]
    )


@pytest.mark.asyncio
async def test_context_updates(context):
    """Test handling of context updates during conversation."""
    # Add initial turn
    turn1 = await context.add_turn(
        "interviewer",
        "What's your experience?",
        {"phase": "initial"}
    )

    # Update context
    new_context = {"phase": "technical", "focus": "python"}
    turn2 = await context.add_turn(
        "interviewer",
        "Tell me about Python",
        new_context
    )

    assert turn1.context["phase"] != turn2.context["phase"]
    assert "focus" in turn2.context
    assert turn2.context["focus"] == "python"
