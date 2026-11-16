# Android coding assistant API

An asynchronous FastAPI endpoint backed by Hugging Face inference. Each request
is independent; multiple inference calls can run concurrently.

## Run

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=your_hugging_face_token
uvicorn main:app --host 0.0.0.0 --port 8000
```

Send `POST /chat` with `{"message": "Create a Kotlin activity"}`. The response
is `{"response": "..."}`. The token may instead be placed in a local `.env`
file; do not commit it.

## Concurrency and overload

The async client is shared for the lifetime of each server worker. Configure
these optional environment variables before starting the server:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MAX_CONCURRENT_REQUESTS` | `10` | Maximum simultaneous upstream inference calls; positive integer. |
| `MAX_QUEUED_REQUESTS` | `50` | Maximum additional requests waiting for a slot; non-negative integer. |
| `QUEUE_TIMEOUT_SECONDS` | `5` | Maximum time a request may wait for a slot; positive number. |
| `INFERENCE_TIMEOUT_SECONDS` | `120` | Maximum time for an upstream call; positive number. |

A full or timed-out queue returns HTTP 503 with `Retry-After`; an inference
timeout returns 504, and other upstream failures return 502. Clients should use
bounded retries with backoff for transient failures. Limits apply **per worker**:
with `uvicorn --workers 4`, the default permits up to 40 concurrent calls and
200 queued requests in total. Tune these values and worker count to available
resources and Hugging Face provider rate limits; they do not enforce a global
limit across processes or instances. A timeout/cancellation frees local capacity,
but does not guarantee that the provider has stopped processing its request.

## Tests

```sh
pip install httpx
python -m unittest discover -s tests -v
```

The tests mock the inference provider and do not require a token.
