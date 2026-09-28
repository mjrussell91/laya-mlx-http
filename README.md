# laya-mlx HTTP endpoint

A small FastAPI server around [`aac6fef/laya-mlx`](https://huggingface.co/aac6fef/laya-mlx), the MLX port of the Laya question-answering model. The model loads once at startup, so each request after that takes milliseconds.

## Requirements

- A Mac with Apple silicon (MLX doesn't run on Intel Macs)
- Python 3.14
- [uv](https://docs.astral.sh/uv/)

## Setup

```sh
uv sync
```

This creates `.venv` and installs the pinned dependencies from `uv.lock`. The model weights download from Hugging Face on the first run and are cached in `~/.cache/huggingface/hub`.

## Start the server

```sh
uv run uvicorn server:app --port 8000
```

The server only accepts connections from this machine. To make it reachable from other machines, add `--host 0.0.0.0`. The server has no authentication.

Interactive API docs are at http://localhost:8000/docs.

## Endpoints

### `GET /health`

```json
{"status": "ok", "model": "aac6fef/laya-mlx"}
```

### `POST /predict`

Send the text to classify and a set of named questions. Each question has a `type` (`choice`, `score` or `noul`) and `instructions`; `choice` questions also take a list of `criteria`.

```sh
curl -s localhost:8000/predict -H 'content-type: application/json' -d '{
  "text": "I was billed twice. Please refund the duplicate today.",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which department should handle this request?",
      "criteria": ["billing", "technical", "sales"]
    },
    "refund": {
      "type": "noul",
      "instructions": "Does the customer ask for money back?"
    }
  }
}'
```

The response has one answer per question:

```json
{
  "department": {"type": "choice", "confidence": 0.7577, "action": {"act_probability": 1.0}, "choice": "billing", "probabilities": {"billing": 0.9386, "technical": 0.0453, "sales": 0.0161}},
  "refund": {"type": "noul", "confidence": 0.8217, "action": {"act_probability": 1.0}, "noul": 0.8217}
}
```

Invalid questions, such as an unknown `type`, return `400` with the reason in `detail`.

## Notes

- Requests are handled one at a time, because MLX isn't safe to call from several threads at once.
- On startup, laya-mlx warns that this checkpoint's calibration temperatures are out of range and clamps them, so `confidence` values for some `choice` questions may be uncalibrated. This comes from the model checkpoint, not this server.
