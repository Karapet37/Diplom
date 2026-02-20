import unittest

from src.web.observability import RuntimeMetrics, is_inference_path
from src.web.privacy_noise import PrivacyNoiseConfig, PrivacyNoisePlugin
from src.web.security import (
    InMemoryRateLimiter,
    SecuritySettings,
    create_access_token,
    decode_access_token,
    extract_client_ip,
    is_strong_password,
    is_strong_secret,
    requires_auth,
)


class ApiSecurityObservabilityTests(unittest.TestCase):
    def test_metrics_registry_prometheus_output(self):
        metrics = RuntimeMetrics()
        metrics.record_request(
            method="POST",
            path="/api/living/process",
            status_code=200,
            latency_seconds=0.42,
            is_inference=True,
        )
        rendered = metrics.render_prometheus()
        self.assertIn("autograph_requests_total", rendered)
        self.assertIn("autograph_inference_latency_seconds_sum", rendered)
        self.assertIn('/api/living/process', rendered)

    def test_jwt_roundtrip(self):
        token = create_access_token(
            subject="tester",
            secret="s3cr3t",
            issuer="autograph",
            audience="",
            expires_minutes=5,
        )
        payload = decode_access_token(
            token,
            secret="s3cr3t",
            issuer="autograph",
            audience="",
        )
        self.assertEqual(payload["sub"], "tester")
        self.assertEqual(payload["iss"], "autograph")

    def test_auth_write_only_behavior(self):
        settings = SecuritySettings(
            auth_enable=True,
            auth_protect_write_only=True,
            jwt_secret="x",
            jwt_issuer="autograph",
            jwt_audience="",
            jwt_exp_minutes=60,
            auth_exempt_paths={"/api/health", "/metrics"},
            rate_limit_enable=False,
            rate_limit_backend="memory",
            rate_limit_per_minute=120,
            rate_limit_exempt_paths={"/api/health", "/metrics"},
        )
        self.assertFalse(requires_auth(settings=settings, method="GET", path="/api/graph/snapshot"))
        self.assertTrue(requires_auth(settings=settings, method="POST", path="/api/graph/seed-demo"))
        self.assertFalse(requires_auth(settings=settings, method="POST", path="/api/health"))

    def test_in_memory_rate_limiter(self):
        limiter = InMemoryRateLimiter(per_minute=2)
        self.assertTrue(limiter.allow("127.0.0.1"))
        self.assertTrue(limiter.allow("127.0.0.1"))
        self.assertFalse(limiter.allow("127.0.0.1"))
        self.assertTrue(limiter.allow("127.0.0.2"))

    def test_privacy_noise_plugin(self):
        disabled = PrivacyNoisePlugin(PrivacyNoiseConfig(enabled=False, intensity=0.4, seed=123))
        self.assertFalse(disabled.enabled())
        self.assertEqual(disabled.synthetic_metrics(), {})

        enabled = PrivacyNoisePlugin(PrivacyNoiseConfig(enabled=True, intensity=0.4, seed=123))
        self.assertTrue(enabled.enabled())
        report = enabled.report()
        self.assertTrue(report["enabled"])
        self.assertIn("autograph_privacy_synthetic_events_total", report["metrics"])

    def test_inference_path_classifier(self):
        self.assertTrue(is_inference_path("/api/living/process"))
        self.assertFalse(is_inference_path("/api/health"))

    def test_secret_strength_helpers(self):
        self.assertFalse(is_strong_secret("change-me"))
        self.assertFalse(is_strong_password("password"))
        self.assertTrue(is_strong_secret("A_very_long_secret_value_1234567890"))
        self.assertTrue(is_strong_password("StrongPassword_123!"))

    def test_extract_client_ip_ignores_spoofed_xff_by_default(self):
        class _Client:
            host = "10.0.0.5"

        class _Req:
            headers = {"x-forwarded-for": "8.8.8.8"}
            client = _Client()

        ip = extract_client_ip(_Req(), settings=SecuritySettings())
        self.assertEqual(ip, "10.0.0.5")

    def test_extract_client_ip_uses_xff_only_for_trusted_proxy(self):
        class _Client:
            host = "127.0.0.1"

        class _Req:
            headers = {"x-forwarded-for": "8.8.8.8, 127.0.0.1"}
            client = _Client()

        settings = SecuritySettings(
            trust_proxy_headers=True,
            trusted_proxy_ips={"127.0.0.1"},
        )
        ip = extract_client_ip(_Req(), settings=settings)
        self.assertEqual(ip, "8.8.8.8")


if __name__ == "__main__":
    unittest.main()
