"""Document storage management."""

from typing import Dict, Any, List, Optional, Protocol
import json
import aiofiles
from pathlib import Path
from datetime import datetime

from .types import ProcessedDocument, DocumentType
from .exceptions import StorageError, DocumentNotFoundError, DocumentValidationError


class StorageBackend(Protocol):
    """Protocol for storage backend implementations."""

    async def store(self, key: str, data: Dict[str, Any]) -> None:
        """Store data with key."""
        ...

    async def retrieve(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve data by key."""
        ...

    async def delete(self, key: str) -> None:
        """Delete data by key."""
        ...

    async def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        """List stored keys."""
        ...


class FileSystemBackend:
    """File system storage implementation."""

    def __init__(self, base_path: str):
        """Initialize storage.

        Args:
            base_path: Base storage directory
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def store(self, key: str, data: Dict[str, Any]) -> None:
        """Store data in file."""
        file_path = self.base_path / f"{key}.json"
        try:
            async with aiofiles.open(file_path, "w") as f:
                await f.write(json.dumps(data, indent=2))
        except Exception as e:
            raise StorageError(f"Failed to store document: {e}")

    async def retrieve(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve data from file."""
        file_path = self.base_path / f"{key}.json"
        if not file_path.exists():
            return None

        try:
            async with aiofiles.open(file_path, "r") as f:
                content = await f.read()
                return json.loads(content)
        except Exception as e:
            raise StorageError(f"Failed to retrieve document: {e}")

    async def delete(self, key: str) -> None:
        """Delete file."""
        file_path = self.base_path / f"{key}.json"
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            raise StorageError(f"Failed to delete document: {e}")

    async def list_keys(self, prefix: Optional[str] = None) -> List[str]:
        """List stored document keys."""
        try:
            files = list(self.base_path.glob("*.json"))
            keys = [f.stem for f in files]
            if prefix:
                keys = [k for k in keys if k.startswith(prefix)]
            return sorted(keys)
        except Exception as e:
            raise StorageError(f"Failed to list documents: {e}")


class DocumentStore:
    """Manages document storage and retrieval."""

    def __init__(self, backend: StorageBackend, cache_size: int = 100):
        """Initialize document store.

        Args:
            backend: Storage backend
            cache_size: Maximum cache entries
        """
        self.backend = backend
        self.cache_size = cache_size
        self.cache: Dict[str, ProcessedDocument] = {}
        self.access_times: Dict[str, datetime] = {}

    async def store_document(self, document: ProcessedDocument) -> None:
        """Store processed document.

        Args:
            document: Document to store

        Raises:
            StorageError: If storage fails
            DocumentValidationError: If document invalid
        """
        try:
            # Validate document
            if not document.id or not document.type:
                raise DocumentValidationError("Missing required fields")

            # Store in backend
            await self.backend.store(document.id, document.model_dump())

            # Update cache
            self._update_cache(document.id, document)

        except DocumentValidationError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to store document: {e}")

    async def get_document(self, document_id: str) -> ProcessedDocument:
        """Retrieve document by ID.

        Args:
            document_id: Document ID

        Returns:
            Retrieved document

        Raises:
            DocumentNotFoundError: If document not found
            StorageError: If retrieval fails
        """
        try:
            # Check cache
            if document_id in self.cache:
                self.access_times[document_id] = datetime.now()
                return self.cache[document_id]

            # Get from backend
            data = await self.backend.retrieve(document_id)
            if not data:
                raise DocumentNotFoundError(f"Document not found: {document_id}")

            # Create document and cache
            document = ProcessedDocument(**data)
            self._update_cache(document_id, document)
            return document

        except DocumentNotFoundError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to retrieve document: {e}")

    async def delete_document(self, document_id: str) -> None:
        """Delete document.

        Args:
            document_id: Document ID

        Raises:
            DocumentNotFoundError: If document not found
            StorageError: If deletion fails
        """
        try:
            # Remove from cache
            self.cache.pop(document_id, None)
            self.access_times.pop(document_id, None)

            # Delete from backend
            await self.backend.delete(document_id)

        except Exception as e:
            raise StorageError(f"Failed to delete document: {e}")

    async def list_documents(
        self, doc_type: Optional[DocumentType] = None
    ) -> List[str]:
        """List stored document IDs.

        Args:
            doc_type: Optional type filter

        Returns:
            List of document IDs

        Raises:
            StorageError: If listing fails
        """
        try:
            prefix = f"{doc_type.value}_" if doc_type else None
            return await self.backend.list_keys(prefix)
        except Exception as e:
            raise StorageError(f"Failed to list documents: {e}")

    def _update_cache(self, doc_id: str, document: ProcessedDocument) -> None:
        """Update cache with document.

        Args:
            doc_id: Document ID
            document: Document to cache
        """
        # Remove the oldest if cache full
        if len(self.cache) >= self.cache_size:
            oldest = min(self.access_times.items(), key=lambda x: x[1])[0]
            self.cache.pop(oldest)
            self.access_times.pop(oldest)

        # Add to cache
        self.cache[doc_id] = document
        self.access_times[doc_id] = datetime.now()
