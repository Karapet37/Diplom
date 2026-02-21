import unittest

from src.web.client_introspection import build_client_profile


class ClientIntrospectionTests(unittest.TestCase):
    def test_profile_contains_network_device_and_locale(self):
        profile = build_client_profile(
            request_headers={
                "user-agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/120.0",
                "x-forwarded-for": "198.51.100.10, 10.0.0.2",
                "x-real-ip": "198.51.100.10",
            },
            request_client_ip="198.51.100.10",
            payload={
                "session_id": "sess_1",
                "user_id": "u1",
                "client": {
                    "platform": "Linux x86_64",
                    "language": "en-US",
                    "languages": ["en-US", "ru"],
                    "timezone": "Asia/Yerevan",
                    "screen": {"width": 1920, "height": 1080},
                    "viewport": {"width": 1280, "height": 860},
                },
            },
        )

        self.assertEqual(profile["session_id"], "sess_1")
        self.assertEqual(profile["network"]["ip"]["ip"], "198.51.100.10")
        self.assertTrue(profile["network"]["vpn_proxy_suspected"])
        self.assertIn("device", profile)
        self.assertIn("locale", profile)
        self.assertIn("browser_signals", profile)


if __name__ == "__main__":
    unittest.main()
