from typing import List
from src.conversation.types import Document
from src.context.context_manager import ContextManager, ContextEntry
from src.document.processor import DocumentProcessor

class DocumentContextUtils:
    def __init__(self, context_manager: ContextManager):
        self.context_manager = context_manager

    async def add_document_reference(self, context_id: str, document: Document, entry_key: str):
        """Add a reference from a document to a context entry."""
        entry = await self.context_manager.store.get_entry(context_id, entry_key)
        if entry:
            entry.references.add(document.name)
            await self.context_manager.store.add_entry(context_id, entry_key, entry)

    async def get_document_related_entries(self, context_id: str, document_name: str) -> List[ContextEntry]:
        """Retrieve context entries related to a specific document."""
        related_entries = []
        entries = await self.context_manager.store.get_context(context_id)
        for entry in entries.values():
            if document_name in entry.references:
                related_entries.append(entry)
        return related_entries

    async def update_document_references(self, context_id: str, old_doc_name: str, new_doc_name: str):
        """Update references when a document name changes."""
        entries = await self.context_manager.store.get_context(context_id)
        for entry_key, entry in entries.items():
            if old_doc_name in entry.references:
                entry.references.remove(old_doc_name)
                entry.references.add(new_doc_name)
                await self.context_manager.store.add_entry(context_id, entry_key, entry)

class DocumentSyncService:
    def __init__(self, context_manager: ContextManager, document_processor: DocumentProcessor):
        self.context_manager = context_manager
        self.document_processor = document_processor

    async def sync_document(self, context_id: str, document: Document):
        """Synchronize a document with the context store."""
        # Process the document
        # TODO: ProcessingContext
        async for result in self.document_processor.process_document(
                document.content,
                document.doc_type
        ):
            # Create context entry
            entry = ContextEntry(
                content=result.analysis,
                source=document.name,
                priority=2.0,
                metadata={
                    "doc_type": document.doc_type.value,
                    "format": document.format,
                },
                references={document.name}
            )

            # Add or update the entry in the store
            entry_key = f"doc_{document.name}_{result.timestamp.isoformat()}"
            await self.context_manager.store.add_entry(context_id, entry_key, entry)

    async def handle_document_update(self, context_id: str, old_doc: Document, new_doc: Document):
        """Handle updates to a document."""
        # Update document references
        await DocumentContextUtils(self.context_manager).update_document_references(
            context_id,
            old_doc.name,
            new_doc.name
        )

        # Sync the updated document
        await self.sync_document(context_id, new_doc)