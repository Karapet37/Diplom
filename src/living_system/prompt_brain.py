"""Prompt-brain layer for local GGUF code cognition and controlled file operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.living_system.knowledge_sql import KnowledgeSQLStore
from src.living_system.models import OperationPolicy, PromptRun


class _SafeDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class PromptBrain:
    """Prompt-driven local brain with project awareness and guarded code actions."""

    def __init__(
        self,
        store: KnowledgeSQLStore,
        *,
        workspace_root: str | Path,
        llm_fn: Any | None = None,
        role_llm_resolver: Any | None = None,
        policy: OperationPolicy | None = None,
    ):
        self.store = store
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.policy = policy or OperationPolicy(workspace_root=str(self.workspace_root))
        self.session_context: dict[str, dict[str, Any]] = {}
        self.llm_fn = llm_fn if llm_fn is not None else self._build_local_llm()
        self.role_llm_resolver = (
            role_llm_resolver
            if role_llm_resolver is not None
            else self._build_role_llm_resolver()
        )
        self.bootstrap_default_prompts()

    @staticmethod
    def _build_local_llm() -> Any | None:
        try:
            from src.utils.local_llm_provider import build_local_llm_fn
        except Exception:
            return None
        try:
            return build_local_llm_fn()
        except Exception:
            return None

    @staticmethod
    def _build_role_llm_resolver() -> Any | None:
        try:
            from src.utils.local_llm_provider import build_role_llm_fn
        except Exception:
            return None
        return build_role_llm_fn

    def bootstrap_default_prompts(self) -> None:
        prompts = [
            {
                "name": "code_architect",
                "language": "en",
                "description": "Design robust code changes for long-living systems.",
                "template": (
                    "You are a reliability-first software architect.\n"
                    "Goal: {goal}\n"
                    "Project map:\n{project_map}\n"
                    "Session memory:\n{session_memory}\n"
                    "Return a concise implementation plan and code patch strategy."
                ),
            },
            {
                "name": "code_patch",
                "language": "en",
                "description": "Generate concrete code updates.",
                "template": (
                    "Task: {task}\n"
                    "Target file: {target_file}\n"
                    "Constraints: {constraints}\n"
                    "Project context:\n{project_map}\n"
                    "Output only the updated code block."
                ),
            },
            {
                "name": "coder_architect_advisor",
                "language": "en",
                "description": "Mini advisor for architecture decisions in code changes.",
                "template": (
                    "Role: coder architect mini advisor.\n"
                    "Objective: {goal}\n"
                    "Constraints: {constraints}\n"
                    "Project map:\n{project_map}\n"
                    "Return 5 concise architecture recommendations."
                ),
            },
            {
                "name": "coder_reviewer_advisor",
                "language": "en",
                "description": "Mini advisor for code review risk detection.",
                "template": (
                    "Role: coder reviewer mini advisor.\n"
                    "Change summary:\n{change_summary}\n"
                    "Return: bugs, regressions, security concerns and missing tests."
                ),
            },
            {
                "name": "coder_refactor_advisor",
                "language": "en",
                "description": "Mini advisor for safe refactoring strategy.",
                "template": (
                    "Role: coder refactor mini advisor.\n"
                    "Target module: {target_module}\n"
                    "Current pain points: {pain_points}\n"
                    "Return a small-step reversible refactor plan."
                ),
            },
            {
                "name": "coder_debug_advisor",
                "language": "en",
                "description": "Mini advisor for debugging strategy.",
                "template": (
                    "Role: coder debug mini advisor.\n"
                    "Bug report:\n{bug_report}\n"
                    "Runtime context:\n{runtime_context}\n"
                    "Return probable root causes and next diagnostic steps."
                ),
            },
            {
                "name": "translate_text",
                "language": "en",
                "description": "Translator prompt (special MADLAD path when available).",
                "template": (
                    "Translate text.\n"
                    "Source language: {source_language}\n"
                    "Target language: {target_language}\n"
                    "Text:\n{text}\n"
                    "Output translation only."
                ),
            },
            {
                "name": "explain_decision",
                "language": "hy",
                "description": "Explain system decision in Armenian.",
                "template": (
                    "Դու բացատրող համակարգ ես։\n"
                    "Մուտք՝ {input_text}\n"
                    "Որոշում՝ {decision}\n"
                    "Հիմնավորում՝ հակիրճ, բայց պատճառահետևանքային։"
                ),
            },
            {
                "name": "explain_decision",
                "language": "en",
                "description": "Explain system decision in English.",
                "template": (
                    "You explain system decisions with causal clarity.\n"
                    "Input: {input_text}\n"
                    "Decision: {decision}\n"
                    "Return a concise explanation."
                ),
            },
        ]
        for item in prompts:
            self.store.upsert_prompt(
                name=item["name"],
                template_text=item["template"],
                version=1,
                language_code=item["language"],
                description=item["description"],
                is_active=True,
            )

    @staticmethod
    def _prompt_role(prompt_name: str) -> str:
        name = str(prompt_name or "").strip().lower()
        mapping = {
            "coder_architect_advisor": "coder_architect",
            "coder_reviewer_advisor": "coder_reviewer",
            "coder_refactor_advisor": "coder_refactor",
            "coder_debug_advisor": "coder_debug",
            "code_architect": "coder_architect",
            "code_patch": "coder_refactor",
            "translate_text": "translator",
        }
        return mapping.get(name, "general")

    def _resolve_prompt_llm(self, *, prompt_name: str) -> Any | None:
        role = self._prompt_role(prompt_name)
        if role == "translator":
            if self.role_llm_resolver is None:
                return None
            try:
                return self.role_llm_resolver(role)
            except Exception:
                return None
        if self.llm_fn is not None:
            return self.llm_fn
        if self.role_llm_resolver is None:
            return None
        try:
            return self.role_llm_resolver(role)
        except Exception:
            return None

    def _resolve_path(self, relative_path: str) -> Path:
        rel = str(relative_path or "").strip().replace("\\", "/")
        if not rel:
            raise ValueError("relative_path is required")
        path = (self.workspace_root / rel).resolve()
        if self.workspace_root not in path.parents and path != self.workspace_root:
            raise PermissionError("target path escapes workspace root")

        normalized_rel = str(path.relative_to(self.workspace_root)).replace("\\", "/")
        for blocked in self.policy.blocked_prefixes:
            prefix = str(blocked).strip("/")
            if prefix and (normalized_rel == prefix or normalized_rel.startswith(prefix + "/")):
                raise PermissionError(f"target path is blocked by policy: {prefix}")
        return path

    def index_project(self, *, max_files: int = 1200) -> dict[str, Any]:
        files: list[str] = []
        for path in self.workspace_root.rglob("*"):
            if len(files) >= max_files:
                break
            if not path.is_file():
                continue
            rel = str(path.relative_to(self.workspace_root)).replace("\\", "/")
            if rel.startswith(".git/"):
                continue
            files.append(rel)
        files.sort()
        buckets: dict[str, int] = {}
        for rel in files:
            root = rel.split("/", 1)[0]
            buckets[root] = buckets.get(root, 0) + 1
        return {
            "workspace_root": str(self.workspace_root),
            "files": files,
            "buckets": dict(sorted(buckets.items())),
            "count": len(files),
        }

    def remember_context(self, *, session_id: str, key: str, value: Any) -> None:
        if not session_id:
            return
        bucket = self.session_context.setdefault(session_id, {})
        bucket[str(key)] = value

    def context_for_session(self, session_id: str) -> dict[str, Any]:
        return dict(self.session_context.get(session_id, {}))

    def run_prompt(
        self,
        *,
        prompt_name: str,
        variables: dict[str, Any],
        user_id: str = "",
        session_id: str = "",
    ) -> dict[str, Any]:
        language = str(variables.get("language", "en") or "en").strip() or "en"
        prompt = self.store.get_prompt(name=prompt_name, language_code=language)
        if prompt is None:
            raise ValueError(f"prompt not found: {prompt_name}")

        session_memory = self.context_for_session(session_id)
        project_map = self.index_project(max_files=400)
        payload = {
            **variables,
            "project_map": "\n".join(project_map["files"][:120]),
            "session_memory": str(session_memory),
        }
        rendered = str(prompt["template_text"]).format_map(_SafeDict(payload))

        llm_callable = self._resolve_prompt_llm(prompt_name=prompt_name)
        if llm_callable is None:
            if self._prompt_role(prompt_name) == "translator":
                output = (
                    "Translator GGUF model is unavailable. "
                    "Set LOCAL_TRANSLATOR_GGUF_MODEL or place MADLAD translator GGUF in models/gguf."
                )
                confidence = 0.0
            else:
                output = (
                    "Local GGUF model is unavailable. "
                    "Prompt rendered successfully; execute fallback deterministic workflow."
                )
                confidence = 0.35
        else:
            try:
                output = str(llm_callable(rendered) or "").strip()
                confidence = 0.78 if output else 0.22
            except Exception as exc:
                output = f"LLM inference failed: {exc}"
                confidence = 0.1

        run = PromptRun(
            prompt_id=int(prompt["prompt_id"]),
            input_payload=payload,
            output_text=output,
            confidence=confidence,
            user_id=user_id,
        )
        run_id = self.store.record_prompt_run(run)

        if session_id:
            self.remember_context(session_id=session_id, key="last_prompt", value=prompt_name)
            self.remember_context(session_id=session_id, key="last_output", value=output[:1000])

        return {
            "run_id": run_id,
            "prompt": prompt,
            "rendered": rendered,
            "output": output,
            "confidence": confidence,
        }

    def _validate_file_content(self, content: str) -> int:
        if "\x00" in str(content):
            raise ValueError("file content contains null byte")
        size_bytes = len(str(content).encode("utf-8"))
        limit = max(1, int(self.policy.max_file_bytes))
        if size_bytes > limit:
            raise ValueError(f"file content is too large: {size_bytes} bytes > {limit} bytes")
        return size_bytes

    def _write_file(self, *, action: str, relative_path: str, content: str, user_id: str = "") -> dict[str, Any]:
        target = self._resolve_path(relative_path)
        if action == "create" and not self.policy.allow_create:
            raise PermissionError("create operation disabled by policy")
        if action == "update" and not self.policy.allow_update:
            raise PermissionError("update operation disabled by policy")
        if target.exists() and target.is_symlink():
            raise PermissionError("symlink targets are blocked by policy")

        size_bytes = self._validate_file_content(content)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        audit_id = self.store.record_audit_action(
            user_id=user_id,
            action_type=f"file_{action}",
            target_path=str(target.relative_to(self.workspace_root)).replace("\\", "/"),
            content=content,
            status="ok",
            details={"size_bytes": size_bytes},
        )
        return {
            "ok": True,
            "audit_id": audit_id,
            "path": str(target.relative_to(self.workspace_root)).replace("\\", "/"),
            "bytes": size_bytes,
        }

    def create_file(self, *, relative_path: str, content: str, user_id: str = "") -> dict[str, Any]:
        target = self._resolve_path(relative_path)
        if target.exists():
            raise FileExistsError("target file already exists")
        return self._write_file(action="create", relative_path=relative_path, content=content, user_id=user_id)

    def update_file(self, *, relative_path: str, content: str, user_id: str = "") -> dict[str, Any]:
        target = self._resolve_path(relative_path)
        if not target.exists():
            raise FileNotFoundError("target file does not exist")
        return self._write_file(action="update", relative_path=relative_path, content=content, user_id=user_id)

    def delete_file(self, *, relative_path: str, user_id: str = "") -> dict[str, Any]:
        if not self.policy.allow_delete:
            raise PermissionError("delete operation disabled by policy")

        target = self._resolve_path(relative_path)
        if not target.exists():
            raise FileNotFoundError("target file does not exist")
        if target.is_dir():
            raise IsADirectoryError("directory deletion is not supported")

        existing_content = target.read_text(encoding="utf-8") if target.exists() else ""
        target.unlink()
        audit_id = self.store.record_audit_action(
            user_id=user_id,
            action_type="file_delete",
            target_path=str(target.relative_to(self.workspace_root)).replace("\\", "/"),
            content=existing_content,
            status="ok",
            details={},
        )
        return {
            "ok": True,
            "audit_id": audit_id,
            "path": str(target.relative_to(self.workspace_root)).replace("\\", "/"),
        }
