import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
import tempfile
from pathlib import Path
from unittest.mock import patch

from src.utils import local_llm_provider as provider


class _FakeLlama:
    calls_lock = threading.Lock()
    init_calls = 0

    def __init__(self, *args, **kwargs):
        with _FakeLlama.calls_lock:
            _FakeLlama.init_calls += 1
        # Simulate expensive init to expose race conditions.
        time.sleep(0.03)

    def create_chat_completion(self, *args, **kwargs):
        return {"choices": [{"message": {"content": "{}"}}]}


class LocalLlmProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        provider._LLM_INSTANCE = None
        provider._LLM_FN = None
        provider._LLM_UNAVAILABLE = False
        provider._LAST_ERROR = ""
        provider._ROLE_MODEL_MAP.clear()
        provider._PATH_LLM_INSTANCE.clear()
        provider._PATH_LLM_FN.clear()
        provider._ROLE_LLM_FN.clear()
        provider._ROLE_ERRORS_WARNED.clear()
        _FakeLlama.init_calls = 0

    def test_missing_model_warns_once_and_short_circuits(self) -> None:
        with patch.object(provider, "_get_model_path", return_value=None), patch.object(provider, "_warn") as warn_mock:
            self.assertIsNone(provider.build_local_llm_fn())
            self.assertIsNone(provider.build_local_llm_fn())

        self.assertEqual(warn_mock.call_count, 1)
        self.assertTrue(provider._LLM_UNAVAILABLE)

    def test_concurrent_initialization_creates_single_instance(self) -> None:
        with patch.object(provider, "_get_model_path", return_value="/tmp/fake.gguf"), patch.object(provider, "Llama", _FakeLlama):
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(lambda _: provider.build_local_llm_fn(), range(16)))

        self.assertEqual(_FakeLlama.init_calls, 1)
        self.assertTrue(all(callable(result) for result in results))

    def test_role_mapping_prefers_madlad_for_translator_and_coder_for_coder_roles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gguf = root / "gguf"
            gguf.mkdir(parents=True, exist_ok=True)
            (gguf / "mistral-7b-instruct.gguf").write_text("", encoding="utf-8")
            (gguf / "deepseek-coder-6.7b.gguf").write_text("", encoding="utf-8")
            madlad_dir = gguf / "madlad400-3b-mt"
            madlad_dir.mkdir(parents=True, exist_ok=True)
            (madlad_dir / "model-q4k.gguf").write_text("", encoding="utf-8")

            with patch.dict("os.environ", {"LOCAL_MODELS_DIR": str(gguf)}, clear=False):
                roles = provider._resolve_model_role_paths()

        self.assertIn("general", roles)
        self.assertIn("translator", roles)
        self.assertIn("coder_architect", roles)
        self.assertIn("madlad", roles["translator"].lower())
        self.assertIn("coder", roles["coder_architect"].lower())

    def test_build_role_llm_fn_uses_role_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gguf = root / "gguf"
            gguf.mkdir(parents=True, exist_ok=True)
            (gguf / "deepseek-coder-6.7b.gguf").write_text("", encoding="utf-8")
            (gguf / "mistral-7b-instruct.gguf").write_text("", encoding="utf-8")

            with patch.dict("os.environ", {"LOCAL_MODELS_DIR": str(gguf)}, clear=False), patch.object(provider, "Llama", _FakeLlama):
                fn = provider.build_role_llm_fn("coder_reviewer")
                self.assertTrue(callable(fn))
                out = fn("return json")
                self.assertIsInstance(out, str)

    def test_translator_role_does_not_fallback_to_general(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gguf = root / "gguf"
            gguf.mkdir(parents=True, exist_ok=True)
            (gguf / "mistral-7b-instruct.gguf").write_text("", encoding="utf-8")

            with patch.dict(
                "os.environ",
                {
                    "LOCAL_MODELS_DIR": str(gguf),
                    "LOCAL_TRANSLATOR_GGUF_MODEL": "",
                },
                clear=False,
            ), patch.object(provider, "_iter_model_dirs", return_value=[gguf]), patch.object(provider, "Llama", _FakeLlama):
                fn = provider.build_role_llm_fn("translator")
                self.assertIsNone(fn)


if __name__ == "__main__":
    unittest.main()
