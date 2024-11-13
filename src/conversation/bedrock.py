"""Bedrock service client wrapper."""
from typing import Dict, Any, List, Optional, AsyncIterator, Union
import json
import aioboto3
from pydantic import ValidationError

from .types import (
    BedrockConfig,
    StreamResponse,
)
from .exceptions import (
    ServiceError,
    ServiceConnectionError,
    ServiceQuotaError,
    ValidationError as AnalyticsValidationError
)


class BedrockClient:
    """Client for AWS Bedrock service."""

    def __init__(
            self,
            region: str,
            config: BedrockConfig
    ) -> None:
        """Initialize Bedrock client.

        Args:
            region: AWS region
            config: Bedrock configuration
        """
        self.region = region
        self.config = config
        self.session = aioboto3.Session()
        self._client = None

    async def __aenter__(self):
        """Enter async context."""
        self._client = await self.session.client(
            "bedrock-runtime",
            region_name=self.region
        ).__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context."""
        if self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)
            self._client = None

    def _format_messages(
            self,
            messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Format messages for Bedrock API.

        Args:
            messages: List of conversation messages

        Returns:
            Formatted messages for API
        """
        formatted = []
        for msg in messages:
            content = []
            for item in msg.get("content", []):
                if isinstance(item, str):
                    content.append({"text": item})
                elif isinstance(item, dict):
                    if "tool_use" in item:
                        content.append({
                            "toolUse": {
                                "toolUseId": item["tool_use"]["tool_use_id"],
                                "name": item["tool_use"]["name"],
                                "input": item["tool_use"]["input"]
                            }
                        })
                    elif "tool_result" in item:
                        content.append({
                            "toolResult": {
                                "toolUseId": item["tool_result"]["tool_use_id"],
                                "content": item["tool_result"]["content"],
                                "status": item["tool_result"]["status"]
                            }
                        })
                    else:
                        content.append(item)
            formatted.append({
                "role": msg["role"],
                "content": content
            })
        return formatted

    async def generate_stream(
            self,
            messages: List[Dict[str, Any]],
            system_prompt: Optional[str] = None
    ) -> AsyncIterator[StreamResponse]:
        """Generate streaming response from Bedrock.

        Args:
            messages: Conversation messages
            system_prompt: Optional system prompt

        Yields:
            Streaming responses

        Raises:
            ServiceError: On service errors
            ValidationError: On validation errors
        """
        if not self._client:
            raise ServiceConnectionError(
                "Bedrock client not initialized",
                service="bedrock"
            )

        try:
            request = {
                "modelId": self.config.model_id,
                "messages": self._format_messages(messages),
                "inferenceConfig": {
                    "maxTokens": self.config.inference_config.max_tokens,
                    "temperature": self.config.inference_config.temperature,
                    "topP": self.config.inference_config.top_p
                }
            }

            if system_prompt:
                request["system"] = [{"text": system_prompt}]

            if self.config.tool_config:
                request["toolConfig"] = self.config.tool_config

            if self.config.guardrail_config:
                request["guardrailConfig"] = self.config.guardrail_config

            response = await self._client.invoke_model_with_response_stream(
                **request
            )

            async for event in response["stream"]:
                try:
                    if "messageStart" in event:
                        continue

                    if "modelStreamErrorException" in event:
                        error = event["modelStreamErrorException"]
                        raise ServiceError(
                            error["message"],
                            service="bedrock",
                            error_code=str(error.get("originalStatusCode")),
                            details={"original_message": error.get("originalMessage")}
                        )

                    if "throttlingException" in event:
                        raise ServiceQuotaError(
                            event["throttlingException"]["message"],
                            service="bedrock"
                        )

                    # Parse content delta
                    if "contentBlockDelta" in event:
                        delta = event["contentBlockDelta"]["delta"]
                        yield StreamResponse(
                            content=StreamResponse.Delta(
                                text=delta.get("text"),
                                tool_use=delta.get("toolUse")
                            )
                        )

                    # Parse stop reason
                    if "messageStop" in event:
                        yield StreamResponse(
                            stop_reason=event["messageStop"]["stopReason"]
                        )

                    # Parse metadata
                    if "metadata" in event:
                        yield StreamResponse(
                            metadata=StreamResponse.Metadata(
                                usage=event["metadata"]["usage"],
                                metrics=event["metadata"]["metrics"],
                                trace=event["metadata"].get("trace")
                            )
                        )

                except ValidationError as e:
                    raise AnalyticsValidationError(
                        f"Invalid response format: {str(e)}"
                    )

        except Exception as e:
            if isinstance(e, (ServiceError, AnalyticsValidationError, ServiceQuotaError)):
                raise
            raise ServiceError(
                f"Bedrock service error: {str(e)}",
                service="bedrock",
                details={"original_error": str(e)}
            )
