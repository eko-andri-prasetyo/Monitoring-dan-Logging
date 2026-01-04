import os
import time
import threading
from typing import Any, Dict

import psutil
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import Counter, Gauge, Histogram, CONTENT_TYPE_LATEST, generate_latest

MLFLOW_INVOCATIONS_URL = os.getenv("MLFLOW_INVOCATIONS_URL", "http://127.0.0.1:5002/invocations")

HTTP_REQUESTS_TOTAL = Counter("http_requests_total", "Total HTTP requests", ["endpoint", "method", "status_code"])
HTTP_REQUEST_ERRORS_TOTAL = Counter("http_request_errors_total", "Total HTTP request errors", ["endpoint", "method", "error_type"])
REQUEST_LATENCY_SECONDS = Histogram("request_latency_seconds", "Latency (seconds)", ["endpoint", "method"])

SYSTEM_CPU_USAGE = Gauge("system_cpu_usage", "System CPU usage percentage (0-100)")
SYSTEM_RAM_USAGE = Gauge("system_ram_usage", "System RAM usage percentage (0-100)")
MODEL_PREDICTIONS_TOTAL = Counter("model_predictions_total", "Total model predictions forwarded to MLflow", ["status"])

app = FastAPI(title="ML Model Prometheus Exporter", version="1.0.0")

def update_system_metrics_loop(interval_sec: int = 5) -> None:
    psutil.cpu_percent(interval=None)
    while True:
        try:
            SYSTEM_CPU_USAGE.set(psutil.cpu_percent(interval=None))
            SYSTEM_RAM_USAGE.set(psutil.virtual_memory().percent)
        except Exception:
            pass
        time.sleep(interval_sec)

@app.on_event("startup")
def on_startup() -> None:
    threading.Thread(target=update_system_metrics_loop, daemon=True).start()

@app.get("/metrics")
def metrics() -> PlainTextResponse:
    data = generate_latest()
    return PlainTextResponse(content=data.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "mlflow_invocations_url": MLFLOW_INVOCATIONS_URL}

@app.post("/predict")
async def predict(request: Request) -> JSONResponse:
    endpoint = "/predict"
    method = "POST"
    start = time.perf_counter()

    try:
        payload = await request.json()
    except Exception as e:
        HTTP_REQUEST_ERRORS_TOTAL.labels(endpoint=endpoint, method=method, error_type="invalid_json").inc()
        HTTP_REQUESTS_TOTAL.labels(endpoint=endpoint, method=method, status_code="400").inc()
        return JSONResponse(status_code=400, content={"status": "error", "message": f"Invalid JSON: {e}"})

    try:
        resp = requests.post(MLFLOW_INVOCATIONS_URL, json=payload, timeout=30)
        REQUEST_LATENCY_SECONDS.labels(endpoint=endpoint, method=method).observe(time.perf_counter() - start)
        HTTP_REQUESTS_TOTAL.labels(endpoint=endpoint, method=method, status_code=str(resp.status_code)).inc()

        if resp.ok:
            MODEL_PREDICTIONS_TOTAL.labels(status="success").inc()
            try:
                return JSONResponse(status_code=200, content=resp.json())
            except Exception:
                return JSONResponse(status_code=200, content={"result": resp.text})
        else:
            MODEL_PREDICTIONS_TOTAL.labels(status="failed").inc()
            HTTP_REQUEST_ERRORS_TOTAL.labels(endpoint=endpoint, method=method, error_type="mlflow_non_2xx").inc()
            return JSONResponse(
                status_code=502,
                content={"status": "error", "message": "Upstream MLflow non-2xx", "upstream_status": resp.status_code, "upstream_body": resp.text},
            )

    except requests.exceptions.Timeout:
        REQUEST_LATENCY_SECONDS.labels(endpoint=endpoint, method=method).observe(time.perf_counter() - start)
        HTTP_REQUESTS_TOTAL.labels(endpoint=endpoint, method=method, status_code="504").inc()
        HTTP_REQUEST_ERRORS_TOTAL.labels(endpoint=endpoint, method=method, error_type="timeout").inc()
        MODEL_PREDICTIONS_TOTAL.labels(status="failed").inc()
        return JSONResponse(status_code=504, content={"status": "error", "message": "Timeout to MLflow invocations"})

    except Exception as e:
        REQUEST_LATENCY_SECONDS.labels(endpoint=endpoint, method=method).observe(time.perf_counter() - start)
        HTTP_REQUESTS_TOTAL.labels(endpoint=endpoint, method=method, status_code="500").inc()
        HTTP_REQUEST_ERRORS_TOTAL.labels(endpoint=endpoint, method=method, error_type="exception").inc()
        MODEL_PREDICTIONS_TOTAL.labels(status="failed").inc()
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("3.prometheus_exporter:app", host="0.0.0.0", port=8000, reload=False)
