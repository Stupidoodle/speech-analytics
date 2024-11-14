"""Core analysis engine for managing analysis tasks and pipelines."""
from typing import Dict, Any, List, Optional, Set, AsyncIterator
from datetime import datetime
import asyncio
import uuid

from src.events.bus import EventBus
from src.events.types import Event, EventType
from src.conversation.manager import ConversationManager
from src.context.manager import ContextManager
from src.document.processor import DocumentProcessor
from .compliance_analyzer import AnalysisAggregator
from .registry import BaseAnalyzer, analyzer_registry

from .types import (
    AnalysisType,
    AnalysisPriority,
    AnalysisState,
    AnalysisTask,
    AnalysisResult,
    AnalysisConfig,
    AnalysisPipeline,
    AnalysisRequest,
    AnalysisMetrics,
    AnalysisInsight
)
from .exceptions import (
    AnalysisError,
    AnalysisTaskError,
    AnalysisPipelineError,
    AnalysisTimeoutError,
    AnalysisResourceError, AnalyzerNotFoundError
)


class AnalysisEngine:
    """Manages analysis tasks and pipelines."""

    def __init__(
            self,
            event_bus: EventBus,
            conversation_manager: ConversationManager,
            context_manager: ContextManager,
            document_processor: DocumentProcessor,
            config: Optional[AnalysisConfig] = None
    ):
        """Initialize analysis engine.

        Args:
            event_bus: Event bus instance
            conversation_manager: Conversation manager
            context_manager: Context manager
            document_processor: Document processor
            config: Optional analysis configuration
        """
        self.aggregator = AnalysisAggregator(event_bus)
        self.event_bus = event_bus
        self.conversation = conversation_manager
        self.context = context_manager
        self.doc_processor = document_processor
        self.config = config or AnalysisConfig(
            enabled_analyzers={
                AnalysisType.CONVERSATION,
                AnalysisType.SENTIMENT,
                AnalysisType.TOPIC,
                AnalysisType.QUALITY
            }
        )

        # Task management
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.task_results: Dict[str, List[AnalysisResult]] = {}
        self.running_pipelines: Dict[str, Dict[str, Any]] = {}

        # Resource tracking
        self.resource_usage = {
            "tasks": 0,
            "pipelines": 0,
            "memory": 0.0
        }

        # Analysis state
        self._running = False
        self._task_queue = asyncio.Queue()
        self._workers: Set[asyncio.Task] = set()

    async def start(self) -> None:
        """Start analysis engine."""
        if not self._running:
            self._running = True

            # Start worker tasks
            for _ in range(self.config.max_concurrent_tasks):
                worker = asyncio.create_task(self._task_worker())
                self._workers.add(worker)

            await self.event_bus.publish(Event(
                type=EventType.ANALYSIS,
                data={
                    "status": "engine_started",
                    "config": self.config.model_dump()
                }
            ))

    async def stop(self) -> None:
        """Stop analysis engine."""
        self._running = False

        # Cancel all workers
        for worker in self._workers:
            worker.cancel()

        # Wait for workers to complete
        if self._workers:
            await asyncio.gather(
                *self._workers,
                return_exceptions=True
            )
        self._workers.clear()

        await self.event_bus.publish(Event(
            type=EventType.ANALYSIS,
            data={"status": "engine_stopped"}
        ))

    async def submit_request(
            self,
            request: AnalysisRequest
    ) -> AsyncIterator[AnalysisResult]:
        """Submit analysis request.

        Args:
            request: Analysis request

        Yields:
            Analysis results

        Raises:
            AnalysisError: If request processing fails
        """
        try:
            # Validate request
            await self._validate_request(request)

            # Create pipeline if not provided
            pipeline = request.pipeline or await self._create_default_pipeline(
                request
            )

            # Initialize pipeline tracking
            pipeline_id = str(uuid.uuid4())
            self.running_pipelines[pipeline_id] = {
                "request": request,
                "pipeline": pipeline,
                "start_time": datetime.now(),
                "current_stage": 0,
                "completed_tasks": set(),
                "failed_tasks": set()
            }

            # Process pipeline stages
            async for result in self._process_pipeline(pipeline_id):
                yield result

        except Exception as e:
            if isinstance(e, AnalysisError):
                raise
            raise AnalysisError(
                f"Failed to process analysis request: {str(e)}"
            )
        finally:
            # Cleanup pipeline
            if pipeline_id in self.running_pipelines:
                del self.running_pipelines[pipeline_id]

    async def _validate_request(
            self,
            request: AnalysisRequest
    ) -> None:
        """Validate analysis request.

        Args:
            request: Request to validate

        Raises:
            AnalysisError: If request is invalid
        """
        # Check resource limits
        if self.resource_usage["tasks"] >= self.config.max_concurrent_tasks:
            raise AnalysisResourceError(
                "Maximum concurrent tasks exceeded",
                "tasks",
                self.resource_usage,
                {"max_tasks": self.config.max_concurrent_tasks}
            )

        # Validate analyzers
        if request.config:
            invalid_analyzers = (
                    request.config.enabled_analyzers -
                    self.config.enabled_analyzers
            )
            if invalid_analyzers:
                raise AnalysisError(
                    "Invalid analyzers requested",
                    {"invalid_analyzers": list(invalid_analyzers)}
                )

    async def _create_default_pipeline(
            self,
            request: AnalysisRequest
    ) -> AnalysisPipeline:
        """Create default analysis pipeline.

        Args:
            request: Analysis request

        Returns:
            Analysis pipeline configuration
        """
        # Create default tasks
        tasks = []
        for analyzer in self.config.enabled_analyzers:
            task_id = f"{analyzer}_{uuid.uuid4()}"
            tasks.append(
                AnalysisTask(
                    id=task_id,
                    type=analyzer,
                    priority=AnalysisPriority.MEDIUM,
                    role=request.metadata.role if request.metadata else None
                )
            )

        # Create single-stage pipeline
        return AnalysisPipeline(
            stages=[{"default": tasks}],
            parallel_stages=True
        )

    async def _process_pipeline(
            self,
            pipeline_id: str
    ) -> AsyncIterator[AnalysisResult]:
        """Process analysis pipeline.

        Args:
            pipeline_id: Pipeline identifier

        Yields:
            Analysis results

        Raises:
            AnalysisPipelineError: If pipeline processing fails
        """
        pipeline_info = self.running_pipelines[pipeline_id]
        pipeline = pipeline_info["pipeline"]
        request = pipeline_info["request"]

        try:
            # Process each stage
            for stage_idx, stage in enumerate(pipeline.stages):
                pipeline_info["current_stage"] = stage_idx

                # Submit stage tasks
                tasks = []
                for group_name, group_tasks in stage.items():
                    for task in group_tasks:
                        if await self._can_run_task(
                                task,
                                pipeline_info["completed_tasks"]
                        ):
                            tasks.append(
                                self._submit_task(
                                    task,
                                    request,
                                    pipeline_id
                                )
                            )

                # Wait for stage completion
                if pipeline.parallel_stages:
                    # Run tasks in parallel
                    results = await asyncio.gather(
                        *tasks,
                        return_exceptions=True
                    )
                    for result in results:
                        if isinstance(result, Exception):
                            if pipeline.error_handling == "fail":
                                raise result
                        else:
                            yield result
                else:
                    # Run tasks sequentially
                    for task_future in tasks:
                        try:
                            result = await task_future
                            yield result
                        except Exception as e:
                            if pipeline.error_handling == "fail":
                                raise

        except Exception as e:
            if isinstance(e, AnalysisPipelineError):
                raise
            raise AnalysisPipelineError(
                f"Pipeline processing failed: {str(e)}",
                str(pipeline_info["current_stage"]),
                list(pipeline_info["failed_tasks"])
            )

    async def _can_run_task(
            self,
            task: AnalysisTask,
            completed_tasks: Set[str]
    ) -> bool:
        """Check if task can be run.

        Args:
            task: Task to check
            completed_tasks: Set of completed task IDs

        Returns:
            Whether task can be run
        """
        # Check dependencies
        return all(
            dep in completed_tasks
            for dep in task.dependencies
        )

    async def _submit_task(
            self,
            task: AnalysisTask,
            request: AnalysisRequest,
            pipeline_id: str
    ) -> AnalysisResult:
        """Submit task for processing.

        Args:
            task: Task to submit
            request: Original analysis request
            pipeline_id: Pipeline identifier

        Returns:
            Task result

        Raises:
            AnalysisTaskError: If task submission fails
        """
        try:
            # Track task
            self.active_tasks[task.id] = {
                "task": task,
                "pipeline_id": pipeline_id,
                "start_time": datetime.now(),
                "state": AnalysisState.PENDING
            }
            self.resource_usage["tasks"] += 1

            # Submit to queue
            await self._task_queue.put((task, request))

            # Wait for result
            timeout = task.timeout or self.config.default_timeout
            try:
                async with asyncio.timeout(timeout):
                    while True:
                        if task.id in self.task_results:
                            results = self.task_results[task.id]
                            if results:
                                return results[-1]
                        await asyncio.sleep(0.1)
            except asyncio.TimeoutError:
                raise AnalysisTimeoutError(
                    f"Task timed out after {timeout}s",
                    timeout,
                    task.id
                )

        except Exception as e:
            if isinstance(e, (AnalysisTaskError, AnalysisTimeoutError)):
                raise
            raise AnalysisTaskError(
                f"Task submission failed: {str(e)}",
                task.id,
                task.type
            )
        finally:
            # Cleanup task
            self.active_tasks.pop(task.id, None)
            self.resource_usage["tasks"] -= 1

    async def _task_worker(self) -> None:
        """Worker for processing analysis tasks."""
        while self._running:
            try:
                # Get next task
                task, request = await self._task_queue.get()

                # Update state
                self.active_tasks[task.id]["state"] = AnalysisState.RUNNING

                # Process task
                start_time = datetime.now()
                try:
                    # Execute analysis
                    result = await self._execute_analysis(
                        task,
                        request
                    )

                    # Store result
                    if task.id not in self.task_results:
                        self.task_results[task.id] = []
                    self.task_results[task.id].append(result)

                    # Update pipeline state
                    pipeline_id = self.active_tasks[task.id]["pipeline_id"]
                    self.running_pipelines[pipeline_id]["completed_tasks"].add(
                        task.id
                    )

                    # Update state
                    self.active_tasks[task.id]["state"] = AnalysisState.COMPLETED

                except Exception as e:
                    # Update error state
                    self.active_tasks[task.id]["state"] = AnalysisState.FAILED
                    pipeline_id = self.active_tasks[task.id]["pipeline_id"]
                    self.running_pipelines[pipeline_id]["failed_tasks"].add(
                        task.id
                    )
                    raise

                finally:
                    # Emit task event
                    await self.event_bus.publish(Event(
                        type=EventType.ANALYSIS,
                        data={
                            "status": "task_completed",
                            "task_id": task.id,
                            "duration": (
                                    datetime.now() - start_time
                            ).total_seconds()
                        }
                    ))

                    self._task_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Task worker error: {e}")
                await asyncio.sleep(1.0)

    async def _execute_analysis(
            self,
            task: AnalysisTask,
            request: AnalysisRequest
    ) -> AnalysisResult:
        """Execute analysis task with full integration.

        Args:
            task: Task to execute
            request: Analysis request

        Returns:
            Analysis result

        Raises:
            AnalysisTaskError: If analysis fails
        """
        start_time = datetime.now()

        try:
            # Get and validate analyzer
            analyzer = self._get_analyzer(task.type)
            if not analyzer:
                raise AnalysisTaskError(
                    f"No analyzer found for type: {task.type}",
                    task.id,
                    task.type
                )

            # Execute analysis
            insights = await analyzer.analyze(
                request.content,
                request.context,
                task.config
            )

            # Create metrics
            metrics = await self._calculate_metrics(insights, request)

            # Create result
            result = AnalysisResult(
                task_id=task.id,
                type=task.type,
                insights=insights,
                metrics=metrics,
                confidence=self._calculate_confidence(insights),
                duration=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now()
            )

            # Add to aggregator
            await self.aggregator.add_result(result)

            # Emit analysis completion event
            await self.event_bus.publish(Event(
                type=EventType.ANALYSIS,
                data={
                    "status": "analysis_completed",
                    "task_id": task.id,
                    "type": task.type,
                    "metrics": metrics
                }
            ))

            return result

        except Exception as e:
            await self.event_bus.publish(Event(
                type=EventType.ERROR,
                data={
                    "status": "analysis_failed",
                    "task_id": task.id,
                    "type": task.type,
                    "error": str(e)
                }
            ))
            raise AnalysisTaskError(
                f"Analysis execution failed: {str(e)}",
                task.id,
                task.type
            )

    async def _calculate_metrics(
            self,
            insights: List[AnalysisInsight],
            request: AnalysisRequest
    ) -> AnalysisMetrics:
        """Calculate analysis metrics.

        Args:
            insights: Analysis insights
            request: Analysis request

        Returns:
            Analysis metrics
        """
        # Implement metrics calculation based on insights
        return AnalysisMetrics(
            duration=0.0,  # Calculate from insights
            turn_count=0,  # Calculate from conversation
            speaker_ratio={},  # Calculate from conversation
            avg_response_time=0.0,  # Calculate from conversation
            topic_distribution={},  # Calculate from insights
            engagement_score=0.0,  # Calculate from insights
            clarity_score=0.0  # Calculate from insights
        )

    def _get_analyzer(
            self,
            analyzer_type: AnalysisType
    ) -> Optional[BaseAnalyzer]:
        """Get analyzer instance for type.

        Args:
            analyzer_type: Type of analyzer

        Returns:
            Analyzer instance if found
        """
        try:
            return analyzer_registry.get_analyzer(
                analyzer_type,
                self.conversation,
                self.config.role_configs.get(analyzer_type.value, {})
            )
        except AnalyzerNotFoundError:
            return None

    def _calculate_confidence(
            self,
            insights: List[AnalysisInsight]
    ) -> float:
        """Calculate overall confidence from insights.

        Args:
            insights: Analysis insights

        Returns:
            Overall confidence score
        """
        if not insights:
            return 0.0

        # Weight insights by type
        type_weights = {
            AnalysisType.SENTIMENT: 1.0,
            AnalysisType.TOPIC: 0.8,
            AnalysisType.QUALITY: 1.0,
            AnalysisType.ENGAGEMENT: 0.9,
            AnalysisType.BEHAVIORAL: 0.7,
            AnalysisType.COMPLIANCE: 1.0
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for insight in insights:
            weight = type_weights.get(insight.type, 0.5)
            weighted_sum += insight.confidence * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    async def get_analysis_summary(
            self,
            session_id: str
    ) -> Dict[str, Any]:
        """Get comprehensive analysis summary.

        Args:
            session_id: Session identifier

        Returns:
            Analysis summary
        """
        try:
            # Get aggregated summary
            summary = await self.aggregator.get_summary(session_id)

            # Add additional context
            if session_id in self.running_pipelines:
                pipeline_info = self.running_pipelines[session_id]
                summary.update({
                    "pipeline_status": {
                        "current_stage": pipeline_info["current_stage"],
                        "completed_tasks": len(
                            pipeline_info["completed_tasks"]
                        ),
                        "failed_tasks": len(
                            pipeline_info["failed_tasks"]
                        ),
                        "duration": (
                                datetime.now() -
                                pipeline_info["start_time"]
                        ).total_seconds()
                    }
                })

            # Add metrics summary
            metrics_summary = {
                analyzer_type: await self._get_analyzer_metrics(
                    session_id,
                    analyzer_type
                )
                for analyzer_type in self.config.enabled_analyzers
            }
            summary["metrics"] = metrics_summary

            return summary

        except Exception as e:
            raise AnalysisError(
                f"Failed to get analysis summary: {str(e)}"
            )

    async def _get_analyzer_metrics(
            self,
            session_id: str,
            analyzer_type: AnalysisType
    ) -> Dict[str, Any]:
        """Get metrics for specific analyzer.

        Args:
            session_id: Session identifier
            analyzer_type: Analyzer type

        Returns:
            Analyzer metrics
        """
        metrics = {
            "task_count": 0,
            "success_rate": 0.0,
            "avg_duration": 0.0,
            "avg_confidence": 0.0
        }

        # Count relevant tasks
        relevant_tasks = [
            task for task in self.task_results.get(session_id, [])
            if task.type == analyzer_type
        ]

        if relevant_tasks:
            metrics.update({
                "task_count": len(relevant_tasks),
                "success_rate": sum(
                    1 for t in relevant_tasks
                    if t.state == AnalysisState.COMPLETED
                ) / len(relevant_tasks),
                "avg_duration": sum(
                    t.duration for t in relevant_tasks
                ) / len(relevant_tasks),
                "avg_confidence": sum(
                    t.confidence for t in relevant_tasks
                ) / len(relevant_tasks)
            })

        return metrics

    async def cancel_analysis(
            self,
            session_id: str
    ) -> None:
        """Cancel ongoing analysis for session.

        Args:
            session_id: Session identifier
        """
        if session_id in self.running_pipelines:
            pipeline_info = self.running_pipelines[session_id]

            # Cancel active tasks
            active_tasks = set(self.active_tasks.keys())
            completed_tasks = pipeline_info["completed_tasks"]
            tasks_to_cancel = active_tasks - completed_tasks

            for task_id in tasks_to_cancel:
                if task_id in self.active_tasks:
                    self.active_tasks[task_id]["state"] = (
                        AnalysisState.CANCELED
                    )

            # Update pipeline status
            pipeline_info["current_stage"] = -1

            # Emit cancellation event
            await self.event_bus.publish(Event(
                type=EventType.ANALYSIS,
                data={
                    "status": "analysis_canceled",
                    "session_id": session_id,
                    "canceled_tasks": len(tasks_to_cancel)
                }
            ))

    async def cleanup_session(
            self,
            session_id: str
    ) -> None:
        """Clean up session data.

        Args:
            session_id: Session identifier
        """
        # Remove pipeline data
        self.running_pipelines.pop(session_id, None)

        # Clean up task results
        self.task_results = {
            task_id: results
            for task_id, results in self.task_results.items()
            if not task_id.startswith(f"{session_id}_")
        }

        # Clean up active tasks
        self.active_tasks = {
            task_id: task
            for task_id, task in self.active_tasks.items()
            if not task_id.startswith(f"{session_id}_")
        }

        # Update resource usage
        self.resource_usage["tasks"] = len(self.active_tasks)

        # Emit cleanup event
        await self.event_bus.publish(Event(
            type=EventType.ANALYSIS,
            data={
                "status": "session_cleaned",
                "session_id": session_id
            }
        ))
