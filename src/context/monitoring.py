"""Monitoring and metrics for context management."""
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio
import time

from src.events.bus import EventBus
from src.events.types import Event, EventType

from .types import (
    ContextSource,
    ContextPriority,
    ContextState,
    ContextMetadata,
    ContextConfig
)
from .exceptions import ContextError


class ContextMetrics:
    """Collects and manages context metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self.reset_metrics()

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.operation_counts = defaultdict(int)
        self.operation_times = defaultdict(list)
        self.source_counts = defaultdict(int)
        self.priority_counts = defaultdict(int)
        self.state_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.session_metrics: Dict[str, Dict[str, Any]] = {}
        self.last_reset = datetime.now()

    async def track_operation(
            self,
            operation: str,
            duration: float,
            metadata: ContextMetadata,
            session_id: Optional[str] = None
    ) -> None:
        """Track context operation.

        Args:
            operation: Operation name
            duration: Operation duration in seconds
            metadata: Operation metadata
            session_id: Optional session identifier
        """
        # Update general metrics
        self.operation_counts[operation] += 1
        self.operation_times[operation].append(duration)
        self.source_counts[metadata.source] += 1
        self.priority_counts[metadata.priority] += 1
        self.state_counts[metadata.state] += 1

        # Update session metrics
        if session_id:
            if session_id not in self.session_metrics:
                self.session_metrics[session_id] = {
                    "operations": defaultdict(int),
                    "total_duration": 0.0,
                    "start_time": datetime.now(),
                    "last_operation": None
                }

            session = self.session_metrics[session_id]
            session["operations"][operation] += 1
            session["total_duration"] += duration
            session["last_operation"] = datetime.now()

    async def track_error(
            self,
            error_type: str,
            details: Dict[str, Any]
    ) -> None:
        """Track context error.

        Args:
            error_type: Type of error
            details: Error details
        """
        self.error_counts[error_type] += 1

    async def get_metrics(
            self,
            since: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get current metrics.

        Args:
            since: Optional start time for metrics

        Returns:
            Current metrics
        """
        if not since:
            since = self.last_reset

        return {
            "operations": {
                "counts": dict(self.operation_counts),
                "average_times": {
                    op: sum(times) / len(times) if times else 0
                    for op, times in self.operation_times.items()
                }
            },
            "sources": dict(self.source_counts),
            "priorities": dict(self.priority_counts),
            "states": dict(self.state_counts),
            "errors": dict(self.error_counts),
            "sessions": len(self.session_metrics),
            "time_period": {
                "start": since.isoformat(),
                "end": datetime.now().isoformat()
            }
        }


class ContextMonitor:
    """Monitors context operations and health."""

    def __init__(
            self,
            event_bus: EventBus,
            config: Optional[ContextConfig] = None
    ):
        """Initialize context monitor.

        Args:
            event_bus: Event bus instance
            config: Optional configuration
        """
        self.event_bus = event_bus
        self.config = config or ContextConfig()
        self.metrics = ContextMetrics()

        # Operation tracking
        self._operations: Dict[str, Dict[str, Any]] = {}
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None

        # Set up event handlers
        self._setup_event_handlers()

    def _setup_event_handlers(self) -> None:
        """Set up event handlers."""
        self.event_bus.subscribe(
            EventType.CONTEXT_UPDATE,
            self._handle_context_event
        )
        self.event_bus.subscribe(
            EventType.ERROR,
            self._handle_error_event
        )

    async def start_monitoring(self) -> None:
        """Start context monitoring."""
        if not self._running:
            self._running = True
            self._monitor_task = asyncio.create_task(
                self._monitor_loop()
            )

            await self.event_bus.publish(Event(
                type=EventType.METRICS,
                data={
                    "status": "monitoring_started",
                    "config": self.config.model_dump()
                }
            ))

    async def stop_monitoring(self) -> None:
        """Stop context monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

        await self.event_bus.publish(Event(
            type=EventType.METRICS,
            data={"status": "monitoring_stopped"}
        ))

    async def track_operation_start(
            self,
            operation_id: str,
            operation_type: str,
            metadata: Dict[str, Any]
    ) -> None:
        """Track start of context operation.

        Args:
            operation_id: Operation identifier
            operation_type: Type of operation
            metadata: Operation metadata
        """
        self._operations[operation_id] = {
            "type": operation_type,
            "start_time": time.time(),
            "metadata": metadata,
            "completed": False
        }

    async def track_operation_end(
            self,
            operation_id: str,
            status: str = "completed",
            results: Optional[Dict[str, Any]] = None
    ) -> None:
        """Track end of context operation.

        Args:
            operation_id: Operation identifier
            status: Operation status
            results: Optional operation results
        """
        if operation_id in self._operations:
            operation = self._operations[operation_id]
            duration = time.time() - operation["start_time"]

            # Track metrics
            await self.metrics.track_operation(
                operation["type"],
                duration,
                ContextMetadata(**operation["metadata"]),
                operation["metadata"].get("session_id")
            )

            # Mark as completed
            operation["completed"] = True
            operation["status"] = status
            operation["duration"] = duration
            operation["results"] = results

            # Clean up after delay
            asyncio.create_task(
                self._cleanup_operation(operation_id)
            )

    async def get_active_operations(self) -> Dict[str, Dict[str, Any]]:
        """Get currently active operations.

        Returns:
            Active operations
        """
        active = {}
        current_time = time.time()

        for op_id, operation in self._operations.items():
            if not operation["completed"]:
                active[op_id] = {
                    "type": operation["type"],
                    "duration": current_time - operation["start_time"],
                    "metadata": operation["metadata"]
                }

        return active

    async def get_operation_stats(
            self,
            operation_type: Optional[str] = None,
            time_window: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get operation statistics.

        Args:
            operation_type: Optional operation type filter
            time_window: Optional time window in seconds

        Returns:
            Operation statistics
        """
        # Get relevant operations
        start_time = (
            datetime.now() - timedelta(seconds=time_window)
            if time_window
            else self.metrics.last_reset
        )

        return await self.metrics.get_metrics(since=start_time)

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                # Check for stalled operations
                await self._check_stalled_operations()

                # Publish metrics
                metrics = await self.metrics.get_metrics()
                await self.event_bus.publish(Event(
                    type=EventType.METRICS,
                    data={
                        "type": "context_metrics",
                        "metrics": metrics
                    }
                ))

                await asyncio.sleep(self.config.update_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Monitor loop error: {e}")
                await asyncio.sleep(1.0)

    async def _check_stalled_operations(self) -> None:
        """Check for stalled operations."""
        current_time = time.time()
        stalled_threshold = 60.0  # 1 minute

        for op_id, operation in self._operations.items():
            if not operation["completed"]:
                duration = current_time - operation["start_time"]
                if duration > stalled_threshold:
                    # Track as error
                    await self.metrics.track_error(
                        "stalled_operation",
                        {
                            "operation_id": op_id,
                            "type": operation["type"],
                            "duration": duration,
                            "metadata": operation["metadata"]
                        }
                    )

                    # Mark as completed with error
                    await self.track_operation_end(
                        op_id,
                        status="stalled"
                    )

    async def _cleanup_operation(
            self,
            operation_id: str
    ) -> None:
        """Clean up completed operation.

        Args:
            operation_id: Operation identifier
        """
        await asyncio.sleep(300)  # Keep for 5 minutes
        self._operations.pop(operation_id, None)

    async def _handle_context_event(
            self,
            event: Event
    ) -> None:
        """Handle context events.

        Args:
            event: Context event
        """
        if "operation_id" in event.data:
            op_id = event.data["operation_id"]

            if event.data.get("status") == "started":
                await self.track_operation_start(
                    op_id,
                    event.data.get("type", "unknown"),
                    event.data.get("metadata", {})
                )
            elif event.data.get("status") in ["completed", "failed"]:
                await self.track_operation_end(
                    op_id,
                    event.data.get("status"),
                    event.data.get("results")
                )

    async def _handle_error_event(
            self,
            event: Event
    ) -> None:
        """Handle error events.

        Args:
            event: Error event
        """
        await self.metrics.track_error(
            event.data.get("error_type", "unknown"),
            event.data
        )


class HealthCheck:
    """Context system health monitoring."""

    def __init__(
            self,
            monitor: ContextMonitor,
            check_interval: float = 60.0
    ):
        """Initialize health check.

        Args:
            monitor: Context monitor instance
            check_interval: Check interval in seconds
        """
        self.monitor = monitor
        self.check_interval = check_interval
        self._status = {"healthy": True}
        self._running = False
        self._check_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start health checking."""
        if not self._running:
            self._running = True
            self._check_task = asyncio.create_task(
                self._check_loop()
            )

    async def stop(self) -> None:
        """Stop health checking."""
        self._running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
            self._check_task = None

    async def get_health_status(self) -> Dict[str, Any]:
        """Get current health status.

        Returns:
            Health status
        """
        return {
            **self._status,
            "last_check": datetime.now().isoformat()
        }

    async def _check_loop(self) -> None:
        """Main health check loop."""
        while self._running:
            try:
                # Get metrics
                metrics = await self.monitor.get_operation_stats(
                    time_window=300  # Last 5 minutes
                )

                # Check error rate
                total_ops = sum(
                    metrics["operations"]["counts"].values()
                )
                total_errors = sum(
                    metrics["errors"].values()
                )
                error_rate = (
                    total_errors / total_ops
                    if total_ops > 0
                    else 0.0
                )

                # Update status
                self._status = {
                    "healthy": error_rate < 0.1,  # Less than 10% errors
                    "error_rate": error_rate,
                    "active_operations": len(
                        await self.monitor.get_active_operations()
                    ),
                    "metrics": metrics
                }

                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._status = {
                    "healthy": False,
                    "error": str(e)
                }
                await asyncio.sleep(self.check_interval)