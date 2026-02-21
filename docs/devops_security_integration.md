# DevOps and Security Integration Notes

## Scope

This integration extends the existing backend with:

- runtime metrics at `/metrics`
- JWT auth middleware (optional, env-driven)
- per-IP rate limiting (slowapi when available, in-memory fallback)
- Nginx reverse proxy config
- Prometheus scraping and Grafana dashboard provisioning
- experimental privacy noise telemetry plugin

No existing API routes were removed.

## Feature Flags

```env
AUTH_ENABLE=0
AUTH_PROTECT_WRITE_ONLY=1
AUTH_JWT_SECRET=
AUTH_JWT_ISSUER=autograph
AUTH_JWT_AUDIENCE=
AUTH_JWT_EXP_MINUTES=60
AUTH_USER=admin
AUTH_PASSWORD=

RATE_LIMIT_ENABLE=0
RATE_LIMIT_BACKEND=slowapi
RATE_LIMIT_PER_MINUTE=120
RATE_LIMIT_EXEMPT_PATHS=/api/health,/metrics

PRIVACY_NOISE_ENABLE=0
PRIVACY_NOISE_INTENSITY=0.05
PRIVACY_NOISE_SEED=20260217
```

## Rollback

### Code Rollback

```bash
git checkout -- src/web/api.py src/web/security.py src/web/observability.py src/web/privacy_noise.py Dockerfile requirements.txt README.md
git checkout -- docker-compose.yml infra/
git checkout -- tests/unit/test_api_security_observability.py docs/devops_security_integration.md
```

### Runtime Rollback

1. Set flags to disabled:

```env
AUTH_ENABLE=0
RATE_LIMIT_ENABLE=0
PRIVACY_NOISE_ENABLE=0
```

2. Restart backend.

### Container Rollback

```bash
docker compose down
```

Then run previous deployment mode (`python3 start.py ...`) without compose.
