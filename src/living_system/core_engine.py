"""Core orchestration engine composing all living-system layers."""

from __future__ import annotations

import traceback
from typing import Any

from src.living_system.diagnostics import DiagnosticsService
from src.living_system.embedding import HashEmbeddingService
from src.living_system.evolution import EvolutionService
from src.living_system.feedback import FeedbackService
from src.living_system.knowledge_sql import KnowledgeSQLStore
from src.living_system.models import FailureRecord, ReasoningTrace, TelemetryEvent
from src.living_system.monitoring import MonitoringService
from src.living_system.prompt_brain import PromptBrain
from src.living_system.reasoning import SemanticReasoningService
from src.living_system.recovery import RecoveryService
from src.living_system.universal_knowledge import UniversalKnowledgeAgent
from src.living_system.visualization import GraphVisualizationService


class LivingSystemEngine:
    """Long-living intelligent-agent runtime with fault-aware layered design."""

    def __init__(
        self,
        *,
        db_path: str = "data/living_system.db",
        workspace_root: str = ".",
        prompt_llm_fn: Any | None = None,
    ):
        self.store = KnowledgeSQLStore(db_path)
        self.store.initialize()

        self.monitoring = MonitoringService(self.store)
        self.embedding = HashEmbeddingService(self.store)
        self.reasoning = SemanticReasoningService(self.store, self.embedding)
        self.visualization = GraphVisualizationService(self.store)
        self.feedback = FeedbackService(self.store)
        self.diagnostics = DiagnosticsService(self.store)
        self.recovery = RecoveryService(self.store)
        self.evolution = EvolutionService(self.store)
        self.universal_knowledge = UniversalKnowledgeAgent(self.store, self.embedding)
        self.prompt_brain = PromptBrain(
            self.store,
            workspace_root=workspace_root,
            llm_fn=prompt_llm_fn,
        )

        self._bootstrap_localization()
        self.evolution.record_release(
            "living_system",
            "1.0.0",
            metadata={"checksum": "ls-100", "notes": "initial modular runtime"},
        )

    def _bootstrap_localization(self) -> None:
        texts = {
            "system.ready": {
                "hy": "Համակարգը պատրաստ է",
                "ru": "Система готова",
                "en": "System is ready",
                "pt": "Sistema pronto",
                "ar": "النظام جاهز",
                "zh": "系统已就绪",
            },
            "system.safe_mode": {
                "hy": "Համակարգը անվտանգ ռեժիմում է",
                "ru": "Система в безопасном режиме",
                "en": "System is in safe mode",
                "pt": "Sistema em modo seguro",
                "ar": "النظام في الوضع الآمن",
                "zh": "系统处于安全模式",
            },
            "reasoning.needs_confirmation": {
                "hy": "Պահանջվում է հաստատում անորոշության պատճառով",
                "ru": "Требуется подтверждение из-за неопределенности",
                "en": "Confirmation is required due to uncertainty",
                "pt": "Confirmação necessária devido à incerteza",
                "ar": "المطلوب تأكيد بسبب عدم اليقين",
                "zh": "由于不确定性需要确认",
            },
        }
        for key, variants in texts.items():
            for lang, value in variants.items():
                self.store.ensure_localized_text(lang, key, value)

    def _localized(self, key: str, *, language: str, fallback: str = "") -> str:
        return self.store.get_localized_text(key, language_code=language, fallback=fallback)

    def bootstrap_user(
        self,
        *,
        user_id: str,
        display_name: str,
        primary_language: str = "hy",
        secondary_languages: list[str] | None = None,
    ) -> None:
        self.store.upsert_user_profile(
            {
                "user_id": user_id,
                "display_name": display_name,
                "primary_language": primary_language,
                "secondary_languages": list(secondary_languages or ["ru", "en", "pt", "ar", "zh"]),
                "preferences": {},
                "behavior_model": {},
                "timeline": [],
            }
        )

    def process_input(
        self,
        *,
        text: str,
        user_id: str,
        language: str = "hy",
        session_id: str = "default",
        auto_snapshot: bool = True,
    ) -> dict[str, Any]:
        if self.recovery.state.safe_mode and not self.recovery.state.human_override:
            message = self._localized(
                "system.safe_mode",
                language=language,
                fallback="System is in safe mode",
            )
            return {
                "ok": False,
                "safe_mode": True,
                "message": message,
                "reason": self.recovery.state.reason,
            }

        self.monitoring.emit(
            TelemetryEvent(
                component="core_engine",
                event_type="input_received",
                message="user text accepted",
                details={"session_id": session_id, "language": language, "length": len(text)},
                level="info",
                user_id=user_id,
            )
        )

        try:
            reasoning = self.reasoning.process(text, user_id=user_id, language=language)
            trace = ReasoningTrace(
                user_id=user_id,
                session_id=session_id,
                input_text=text,
                output_text=reasoning.get("explanation", ""),
                confidence=float(reasoning.get("confidence", 0.0)),
                trace=reasoning,
            )
            self.store.store_reasoning_trace(trace)

            prompt_result = self.prompt_brain.run_prompt(
                prompt_name="explain_decision",
                variables={
                    "language": language,
                    "input_text": text,
                    "decision": reasoning.get("explanation", ""),
                },
                user_id=user_id,
                session_id=session_id,
            )

            snapshot_id: int | None = None
            if auto_snapshot:
                snapshot_id = self.recovery.create_snapshot(reason="post_input", user_id=user_id)

            health = self.diagnostics.health_check()
            integrity = self.diagnostics.integrity_check()
            drift = self.diagnostics.drift_check()
            uncertainty = self.diagnostics.uncertainty_monitor()

            needs_confirmation = bool(reasoning.get("requires_confirmation", False))
            confirmation_message = self._localized(
                "reasoning.needs_confirmation",
                language=language,
                fallback="Confirmation is required",
            )

            return {
                "ok": True,
                "message": self._localized("system.ready", language=language, fallback="System is ready"),
                "reasoning": reasoning,
                "needs_confirmation": needs_confirmation,
                "confirmation_message": confirmation_message if needs_confirmation else "",
                "prompt_explanation": prompt_result.get("output", ""),
                "snapshot_id": snapshot_id,
                "health": [
                    {
                        "layer": row.layer,
                        "status": row.status,
                        "score": row.score,
                        "details": row.details,
                    }
                    for row in health
                ],
                "integrity": integrity,
                "drift": drift,
                "uncertainty": uncertainty,
            }

        except Exception as exc:
            failure = FailureRecord(
                signature=f"core_engine:{type(exc).__name__}:{str(exc)}",
                error_type=type(exc).__name__,
                message=str(exc),
                traceback=traceback.format_exc(),
                component="core_engine",
                severity="critical",
            )
            error_id = self.store.capture_failure(failure)
            self.recovery.set_safe_mode(True, reason="critical failure")
            return {
                "ok": False,
                "error_id": error_id,
                "error": str(exc),
                "safe_mode": True,
            }

    def architecture_overview(self) -> dict[str, Any]:
        return {
            "layers": [
                "core_engine",
                "monitoring",
                "knowledge_sql",
                "embedding",
                "reasoning",
                "visualization",
                "feedback",
                "diagnostics",
                "recovery",
                "evolution",
                "prompt_brain",
                "universal_knowledge",
            ],
            "replaceable": True,
            "storage": "sqlite",
            "snapshot_reconstructable": True,
        }

    def graph_view(self, *, user_id: str = "") -> dict[str, Any]:
        return self.visualization.graph_view(user_id=user_id)

    def health_report(self) -> dict[str, Any]:
        return {
            "health": [
                {
                    "layer": item.layer,
                    "status": item.status,
                    "score": item.score,
                    "details": item.details,
                }
                for item in self.diagnostics.health_check()
            ],
            "integrity": self.diagnostics.integrity_check(),
            "drift": self.diagnostics.drift_check(),
            "uncertainty": self.diagnostics.uncertainty_monitor(),
            "safe_mode": self.recovery.state.safe_mode,
        }

    def create_snapshot(self, *, reason: str, user_id: str = "") -> int:
        return self.recovery.create_snapshot(reason=reason, user_id=user_id)

    def rollback(self, snapshot_id: int) -> dict[str, Any]:
        action = self.recovery.rollback(snapshot_id)
        return {
            "action": action.action,
            "status": action.status,
            "details": action.details,
        }

    def set_safe_mode(self, enabled: bool, *, reason: str = "") -> dict[str, Any]:
        action = self.recovery.set_safe_mode(enabled, reason=reason)
        return {
            "action": action.action,
            "status": action.status,
            "details": action.details,
        }

    def set_human_override(self, enabled: bool, *, reason: str = "") -> dict[str, Any]:
        action = self.recovery.set_human_override(enabled, reason=reason)
        return {
            "action": action.action,
            "status": action.status,
            "details": action.details,
        }

    def feedback_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.feedback.capture(payload)

    def evolution_plan(self) -> dict[str, Any]:
        return self.evolution.plan()

    def prompt_catalog(self) -> list[dict[str, Any]]:
        return self.store.list_active_prompts()

    def run_prompt(
        self,
        *,
        prompt_name: str,
        variables: dict[str, Any],
        user_id: str = "",
        session_id: str = "",
    ) -> dict[str, Any]:
        return self.prompt_brain.run_prompt(
            prompt_name=prompt_name,
            variables=variables,
            user_id=user_id,
            session_id=session_id,
        )

    def project_map(self, *, max_files: int = 600) -> dict[str, Any]:
        return self.prompt_brain.index_project(max_files=max_files)

    def create_file(self, *, relative_path: str, content: str, user_id: str = "") -> dict[str, Any]:
        return self.prompt_brain.create_file(relative_path=relative_path, content=content, user_id=user_id)

    def update_file(self, *, relative_path: str, content: str, user_id: str = "") -> dict[str, Any]:
        return self.prompt_brain.update_file(relative_path=relative_path, content=content, user_id=user_id)

    def delete_file(self, *, relative_path: str, user_id: str = "") -> dict[str, Any]:
        return self.prompt_brain.delete_file(relative_path=relative_path, user_id=user_id)

    def analyze_knowledge(
        self,
        *,
        text: str,
        user_id: str,
        sources: list[dict[str, Any]] | None = None,
        branch_id: str = "main",
        apply_changes: bool = False,
    ) -> dict[str, Any]:
        return self.universal_knowledge.analyze_input(
            text=text,
            user_id=user_id,
            sources=list(sources or []),
            branch_id=branch_id,
            apply_changes=apply_changes,
        )

    def initialize_foundational_knowledge(
        self,
        *,
        user_id: str,
        branch_id: str = "foundation",
        apply_changes: bool = True,
    ) -> dict[str, Any]:
        return self.universal_knowledge.initialize_foundational_domains(
            user_id=user_id,
            branch_id=branch_id,
            apply_changes=apply_changes,
        )

    def evaluate_knowledge_graph(self, *, user_id: str = "") -> dict[str, Any]:
        return self.universal_knowledge.evaluate_graph(user_id=user_id)

    def create_knowledge_branch(self, *, user_id: str, branch_name: str) -> dict[str, Any]:
        return self.universal_knowledge.branch_graph(user_id=user_id, branch_name=branch_name)

    def merge_knowledge_branches(
        self,
        *,
        user_id: str,
        base_snapshot_id: int,
        target_snapshot_id: int,
        apply_changes: bool = False,
    ) -> dict[str, Any]:
        return self.universal_knowledge.merge_branches(
            user_id=user_id,
            base_snapshot_id=base_snapshot_id,
            target_snapshot_id=target_snapshot_id,
            apply_changes=apply_changes,
        )
