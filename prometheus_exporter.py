import os
import time
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# =============== CONFIG ===============
# Target MLflow scoring server (yang jalan dari `mlflow models serve`)
MLFLOW_SCORING_URL = os.getenv("MLFLOW_SCORING_URL", "http://127.0.0.1:5001/invocations")

# Label supaya kebukti "model version" (isi dengan run_id/versi model)
MODEL_NAME = os.getenv("MODEL_NAME", "creditscoring")
MODEL_VERSION = os.getenv("MODEL_VERSION", "e62a3c7b79ff4917b875b53f82e4665e")  # run_id Anda

# Histogram bucket latency (detik)
LATENCY_BUCKETS = (
    0.01, 0.025, 0.05, 0.1, 0.25,
    0.5, 1.0, 2.5, 5.0, 10.0
)

# =============== METRICS ===============
REQ_TOTAL = Counter(
    "inference_requests_total",
    "Total inference requests received",
    ["model_name", "model_version", "status_code"],
)

REQ_ERRORS = Counter(
    "inference_request_errors_total",
    "Total inference requests that returned non-2xx from upstream",
    ["model_name", "model_version", "error_type"],
)

REQ_LATENCY = Histogram(
    "inference_request_latency_seconds",
    "End-to-end latency for inference requests (proxy -> mlflow -> response)",
    ["model_name", "model_version"],
    buckets=LATENCY_BUCKETS,
)

INPROGRESS = Gauge(
    "inference_requests_inprogress",
    "Number of inference requests currently in progress",
    ["model_name", "model_version"],
)

UPSTREAM_UP = Gauge(
    "mlflow_upstream_up",
    "Whether upstream MLflow scoring server is reachable (1=up, 0=down)",
    ["model_name", "model_version"],
)

PAYLOAD_BYTES = Histogram(
    "inference_payload_bytes",
    "Request payload size in bytes",
    ["model_name", "model_version"],
    buckets=(100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000),
)

app = FastAPI(title="Prometheus Exporter Proxy for MLflow")

_http_timeout = httpx.Timeout(30.0, connect=5.0)
client = httpx.Client(timeout=_http_timeout)

def _is_2xx(code: int) -> bool:
    return 200 <= code < 300

@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
def health():
    # quick ping upstream
    try:
        # MLflow scoring server tidak punya /health, jadi kita ping root saja
        r = client.get(MLFLOW_SCORING_URL.replace("/invocations", "/"))
        UPSTREAM_UP.labels(MODEL_NAME, MODEL_VERSION).set(1)
        return {"status": "ok", "upstream": "reachable", "code": r.status_code}
    except Exception:
        UPSTREAM_UP.labels(MODEL_NAME, MODEL_VERSION).set(0)
        return {"status": "degraded", "upstream": "unreachable"}

@app.post("/invocations")
async def invocations(request: Request):
    start = time.perf_counter()
    INPROGRESS.labels(MODEL_NAME, MODEL_VERSION).inc()

    body: bytes = await request.body()
    PAYLOAD_BYTES.labels(MODEL_NAME, MODEL_VERSION).observe(len(body))

    # header: pertahankan content-type (MLflow biasanya application/json)
    headers = {"Content-Type": request.headers.get("content-type", "application/json")}

    try:
        r = client.post(MLFLOW_SCORING_URL, content=body, headers=headers)
        latency = time.perf_counter() - start
        REQ_LATENCY.labels(MODEL_NAME, MODEL_VERSION).observe(latency)

        REQ_TOTAL.labels(MODEL_NAME, MODEL_VERSION, str(r.status_code)).inc()

        if not _is_2xx(r.status_code):
            REQ_ERRORS.labels(MODEL_NAME, MODEL_VERSION, "upstream_non_2xx").inc()

        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))

    except httpx.ConnectError:
        latency = time.perf_counter() - start
        REQ_LATENCY.labels(MODEL_NAME, MODEL_VERSION).observe(latency)
        REQ_TOTAL.labels(MODEL_NAME, MODEL_VERSION, "502").inc()
        REQ_ERRORS.labels(MODEL_NAME, MODEL_VERSION, "connect_error").inc()
        UPSTREAM_UP.labels(MODEL_NAME, MODEL_VERSION).set(0)
        return PlainTextResponse("Upstream MLflow server unreachable", status_code=502)

    except Exception as e:
        latency = time.perf_counter() - start
        REQ_LATENCY.labels(MODEL_NAME, MODEL_VERSION).observe(latency)
        REQ_TOTAL.labels(MODEL_NAME, MODEL_VERSION, "500").inc()
        REQ_ERRORS.labels(MODEL_NAME, MODEL_VERSION, "unexpected_exception").inc()
        return PlainTextResponse(f"Internal proxy error: {type(e).__name__}", status_code=500)

    finally:
        INPROGRESS.labels(MODEL_NAME, MODEL_VERSION).dec()
