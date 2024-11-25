"""Bedrock runtime client pool for parallel processing."""

from typing import Dict, Any, Optional, AsyncIterator
import asyncio
import aioboto3
from contextlib import asynccontextmanager


class BedrockClientPool:
    """Pool of Bedrock runtime clients."""

    def __init__(self, region: str, pool_size: int = 3):
        """Initialize client pool.

        Args:
            region: AWS region
            pool_size: Number of clients in pool
        """
        self.region = region
        self.pool_size = pool_size
        self.session = aioboto3.Session()
        self._clients: Dict[str, Any] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    async def __aenter__(self):
        """Initialize client pool."""
        # Create clients for different purposes
        client_types = [
            "pre_processing",  # For pre-processing generation
            "response",  # For main response generation
            "sentiment",  # For sentiment analysis
            "feedback",  # For other analysis
        ]

        for client_type in client_types:
            # Create client
            client = await self.session.client(
                "bedrock-runtime", region_name=self.region
            ).__aenter__()

            # Store client and its lock
            self._clients[client_type] = client
            self._locks[client_type] = asyncio.Lock()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup client pool."""
        for client in self._clients.values():
            await client.__aexit__(exc_type, exc_val, exc_tb)
        self._clients.clear()
        self._locks.clear()

    @asynccontextmanager
    async def get_client(self, client_type: str):
        """Get a client from the pool.

        Args:
            client_type: Type of client needed ("response", "sentiment", "analysis")
        """
        if client_type not in self._clients:
            raise ValueError(f"Unknown client type: {client_type}")

        async with self._locks[client_type]:
            yield self._clients[client_type]
