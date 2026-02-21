import os
import tempfile
import unittest
from pathlib import Path

from src.utils.env_loader import load_local_env


class EnvLoaderTests(unittest.TestCase):
    def test_loads_env_values_and_ignores_comments(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "FOO=bar",
                        "BAR=' spaced value '",
                        'BAZ="quoted # not comment"',
                        "IGNORED=abc # trailing comment",
                    ]
                ),
                encoding="utf-8",
            )

            for key in ("FOO", "BAR", "BAZ", "IGNORED"):
                os.environ.pop(key, None)

            loaded = load_local_env(env_path)
            self.assertEqual(loaded, 4)
            self.assertEqual(os.getenv("FOO"), "bar")
            self.assertEqual(os.getenv("BAR"), "spaced value")
            self.assertEqual(os.getenv("BAZ"), "quoted # not comment")
            self.assertEqual(os.getenv("IGNORED"), "abc")

    def test_respects_existing_env_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("EXISTING=from_file\n", encoding="utf-8")

            os.environ["EXISTING"] = "from_process"
            loaded = load_local_env(env_path, override=False)
            self.assertEqual(loaded, 0)
            self.assertEqual(os.getenv("EXISTING"), "from_process")

            loaded = load_local_env(env_path, override=True)
            self.assertEqual(loaded, 1)
            self.assertEqual(os.getenv("EXISTING"), "from_file")


if __name__ == "__main__":
    unittest.main()
