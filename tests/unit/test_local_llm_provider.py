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

    def test_split_gguf_discovery_uses_only_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gguf = root / "gguf"
            gguf.mkdir(parents=True, exist_ok=True)
            first = gguf / "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
            second = gguf / "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf"
            first.write_text("", encoding="utf-8")
            second.write_text("", encoding="utf-8")

            with patch.dict("os.environ", {"LOCAL_MODELS_DIR": str(gguf)}, clear=False), patch.object(
                provider, "_iter_model_dirs", return_value=[gguf]
            ):
                discovered = provider._discover_gguf_paths()

        paths = [str(item).replace("\\", "/") for item in discovered]
        self.assertEqual(len(paths), 1)
        self.assertIn("00001-of-00002.gguf", paths[0])
        self.assertTrue(all("00002-of-00002.gguf" not in item for item in paths))

    def test_explicit_split_second_shard_is_remapped_to_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            gguf = root / "gguf"
            gguf.mkdir(parents=True, exist_ok=True)
            first = gguf / "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
            second = gguf / "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf"
            first.write_text("", encoding="utf-8")
            second.write_text("", encoding="utf-8")

            with patch.dict("os.environ", {"LOCAL_GGUF_MODEL": str(second)}, clear=False):
                resolved = provider._get_model_path()

        self.assertIsNotNone(resolved)
        self.assertIn("00001-of-00002.gguf", str(resolved))

    def test_discovery_ignores_llama_cpp_internal_vocab_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            good = root / "models" / "gguf" / "mistral-7b-instruct.gguf"
            bad = root / "models" / "gguf" / "llama.cpp" / "models" / "ggml-vocab-refact.gguf"
            good.parent.mkdir(parents=True, exist_ok=True)
            bad.parent.mkdir(parents=True, exist_ok=True)
            good.write_text("", encoding="utf-8")
            bad.write_text("", encoding="utf-8")

            with patch.object(provider, "_iter_model_dirs", return_value=[root / "models" / "gguf"]):
                discovered = provider._discover_gguf_paths()

        paths = [str(item).replace("\\", "/").lower() for item in discovered]
        self.assertEqual(len(paths), 1)
        self.assertIn("mistral-7b-instruct.gguf", paths[0])


if __name__ == "__main__":
    unittest.main()
