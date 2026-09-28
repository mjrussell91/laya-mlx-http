import threading
from contextlib import asynccontextmanager
from typing import Any

import laya_mlx as laya
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_ID = "aac6fef/laya-mlx"

state: dict[str, Any] = {}
# MLX isn't thread-safe; FastAPI runs sync endpoints in a thread pool, so serialize predictions.
lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup: a cold load takes seconds, a prediction takes milliseconds.
    state["agent"] = laya.load(MODEL_ID)
    yield
    state.clear()


app = FastAPI(title="laya-mlx", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str
    questions: dict[str, dict[str, Any]]


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        with lock:
            result = state["agent"].predict(req.text, req.questions)
    except (ValueError, KeyError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result["answers"]
