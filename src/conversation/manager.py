"""Conversation manager with client pool and context handling."""

import asyncio
from typing import Dict, List, Optional, AsyncIterator

from src.events.bus import EventBus
from .exceptions import StreamError

from .types import (
    Message,
    Request,
    ContentBlock,
    SystemContent,
    InferenceConfig,
    Metadata,
)
from .client_pool import BedrockClientPool


class ConversationManager:
    """Manages conversations with parallel processing."""

    def __init__(
        self,
        event_bus: EventBus,
        client_pool: BedrockClientPool,
        model_id: str,
        config: Optional[InferenceConfig] = None,
    ):
        self.event_bus = event_bus
        self.client_pool = client_pool
        self.model_id = model_id
        self.config = config or InferenceConfig()

        # Track conversations and context
        self._conversations: Dict[str, Dict[str, List[Message]]] = {}
        self._context: Dict[str, List[Message]] = {}

    async def send_message(
        self,
        message: Request,
        system: List[SystemContent] = None,
        client_type: str = None,
        force_no_response=False,
    ) -> AsyncIterator[str | Metadata | None]:
        """Send message with parallel sentiment analysis.

        Args:
            message: New message
            system: Optional system messages
            client_type: Optional client type (pre_processing, response, etc.)
            force_no_response: Force no response
        """
        try:
            if message.modelId != self.model_id:
                message.modelId = self.model_id
            if system:
                for system_content, system_text in zip(message.system, system):
                    system_content.text += (
                        "\n"
                        + system_text.text
                        + "\n"
                        + "Do not generate a response to the context message."
                        if force_no_response
                        else ""
                    )
            for key, value in self.config.model_dump().items():
                if getattr(message.inferenceConfig, key) is None:
                    setattr(message.inferenceConfig, key, value)
            if client_type:
                if client_type == "pre_processing":
                    async with self.client_pool.get_client("pre_processing") as client:
                        response = ""
                        async for event in client.invoke_model_with_response_stream(
                            **message.model_dump()
                        ):
                            try:
                                if "contentBlockDelta" in event:
                                    delta = event["contentBlockDelta"]["delta"]
                                    response += delta.get("text")
                                    yield response
                                if "messageStop" in event:
                                    print(response)
                                    response_text = response  # We might need to
                                    # append the last delta

                            except Exception as e:
                                yield StreamError(
                                    message="Error processing event: " + str(e)
                                )

                        tasks = {
                            "response": self.send_message(
                                message=Request(
                                    modelId=self.config.model_id,
                                    messages=[
                                        Message(
                                            role="user",
                                            content=[
                                                ContentBlock(
                                                    text=response_text,
                                                )
                                            ],
                                        )
                                    ],
                                    system=[
                                        SystemContent(text="This is a context update.")
                                    ],
                                    inferenceConfig=InferenceConfig(
                                        temperature=0.0,
                                        topP=1.0,
                                    ),
                                ),
                                system=system,
                                force_no_response=True,
                            ),
                            "sentiment": self.send_message(
                                message=Request(
                                    modelId=self.config.model_id,
                                    messages=[
                                        Message(
                                            role="user",
                                            content=[
                                                ContentBlock(
                                                    text=response_text,
                                                )
                                            ],
                                        )
                                    ],
                                    system=[
                                        SystemContent(text="This is a context update.")
                                    ],
                                    inferenceConfig=InferenceConfig(
                                        temperature=0.0,
                                        topP=1.0,
                                    ),
                                ),
                                system=system,
                                force_no_response=True,
                            ),
                            "feedback": self.send_message(
                                message=Request(
                                    modelId=self.config.model_id,
                                    messages=[
                                        Message(
                                            role="user",
                                            content=[
                                                ContentBlock(
                                                    text=response_text,
                                                )
                                            ],
                                        )
                                    ],
                                    system=[
                                        SystemContent(text="This is a context update.")
                                    ],
                                    inferenceConfig=InferenceConfig(
                                        temperature=0.0,
                                        topP=1.0,
                                    ),
                                ),
                                system=system,
                                force_no_response=True,
                            ),
                        }

                        results = await asyncio.gather(
                            *tasks.values(), return_exceptions=True
                        )
                        # TODO: Add return statement
                        print(results)

                if client_type == "response":
                    async with self.client_pool.get_client("response") as client:
                        async for event in client.converse_stream(
                            **message.model_dump()
                        ):
                            try:
                                if "contentBlockDelta" in event:
                                    delta = event["contentBlockDelta"]["delta"]
                                    response = delta.get("text")
                                    yield response
                                elif "metadata" in event:
                                    yield Metadata(
                                        usage=event["metadata"]["usage"],
                                        metrics=event["metadata"]["metrics"],
                                        trace=event["metadata"].get("trace"),
                                    )
                            except Exception as e:
                                yield StreamError(
                                    message="Error processing event: " + str(e)
                                )

                if client_type == "sentiment":
                    async with self.client_pool.get_client("sentiment") as client:
                        async for event in client.converse_stream(
                            **message.model_dump()
                        ):
                            try:
                                if "contentBlockDelta" in event:
                                    delta = event["contentBlockDelta"]["delta"]
                                    response = delta.get("text")
                                    yield response
                                elif "metadata" in event:
                                    yield Metadata(
                                        usage=event["metadata"]["usage"],
                                        metrics=event["metadata"]["metrics"],
                                        trace=event["metadata"].get("trace"),
                                    )
                            except Exception as e:
                                yield StreamError(
                                    message="Error processing event: " + str(e)
                                )

                if client_type == "feedback":
                    async with self.client_pool.get_client("feedback") as client:
                        async for event in client.converse_stream(
                            **message.model_dump()
                        ):
                            try:
                                if "contentBlockDelta" in event:
                                    delta = event["contentBlockDelta"]["delta"]
                                    response = delta.get("text")
                                    yield response
                                elif "metadata" in event:
                                    yield Metadata(
                                        usage=event["metadata"]["usage"],
                                        metrics=event["metadata"]["metrics"],
                                        trace=event["metadata"].get("trace"),
                                    )
                            except Exception as e:
                                yield StreamError(
                                    message="Error processing event: " + str(e)
                                )

        except Exception as e:
            yield StreamError(message=str(e))
            raise
