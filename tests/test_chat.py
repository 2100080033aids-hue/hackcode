import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx

import main


class FakeInferenceClient:
    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(completions=self)
        self.started = asyncio.Queue()
        self.release = asyncio.Event()
        self.active = 0
        self.peak = 0
        self.closed = False
        self.error = None

    async def create(self, **kwargs):
        self.active += 1
        self.peak = max(self.peak, self.active)
        await self.started.put(kwargs)
        try:
            await self.release.wait()
            if self.error:
                raise self.error
            message = kwargs["messages"][-1]["content"]
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=message))])
        finally:
            self.active -= 1

    async def close(self):
        self.closed = True


class ChatTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.env = patch.dict("os.environ", {
            "MAX_CONCURRENT_REQUESTS": "2", "MAX_QUEUED_REQUESTS": "1",
            "QUEUE_TIMEOUT_SECONDS": "0.1", "INFERENCE_TIMEOUT_SECONDS": "1",
        })
        self.env.start()
        self.factory = patch.object(main, "AsyncInferenceClient", FakeInferenceClient)
        self.factory.start()
        self.lifespan = main.app.router.lifespan_context(main.app)
        await self.lifespan.__aenter__()
        self.fake = main.app.state.client
        self.http = httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test")

    async def asyncTearDown(self):
        await self.http.aclose()
        await self.lifespan.__aexit__(None, None, None)
        self.assertTrue(self.fake.closed)
        self.factory.stop()
        self.env.stop()

    def send(self, text):
        return asyncio.create_task(self.http.post("/chat", json={"message": text}))

    async def test_parallel_queue_and_rejection(self):
        first, second = self.send("first"), self.send("second")
        calls = [await asyncio.wait_for(self.fake.started.get(), 1) for _ in range(2)]
        self.assertEqual({call["messages"][-1]["content"] for call in calls}, {"first", "second"})
        self.assertEqual(self.fake.peak, 2)
        queued = self.send("third")
        # Let the queued request be admitted before trying to exceed capacity.
        await asyncio.sleep(0.01)
        rejected = await self.http.post("/chat", json={"message": "fourth"})
        self.assertEqual(rejected.status_code, 503)
        self.assertEqual(rejected.headers["retry-after"], "1")
        self.fake.release.set()
        results = await asyncio.gather(first, second, queued)
        self.assertEqual([r.json()["response"] for r in results], ["first", "second", "third"])
        self.assertTrue(all(r.status_code == 200 for r in results))
        self.assertEqual(main.app.state.admitted, 0)

    async def test_queue_timeout_releases_capacity(self):
        first, second = self.send("first"), self.send("second")
        await self.fake.started.get()
        await self.fake.started.get()
        response = await self.http.post("/chat", json={"message": "waiting"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(main.app.state.admitted, 2)
        self.fake.release.set()
        await asyncio.gather(first, second)
        self.assertEqual(main.app.state.admitted, 0)

    async def test_cancelling_waiter_releases_queue_capacity(self):
        first, second = self.send("first"), self.send("second")
        await self.fake.started.get()
        await self.fake.started.get()
        queued = self.send("cancel waiting")
        await asyncio.sleep(0.01)
        self.assertEqual(main.app.state.admitted, 3)
        queued.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await queued
        self.assertEqual(main.app.state.admitted, 2)
        self.fake.release.set()
        await asyncio.gather(first, second)
        self.assertEqual((await self.http.post("/chat", json={"message": "next"})).status_code, 200)

    async def test_upstream_timeout_releases_slot(self):
        main.app.state.inference_timeout = 0.01
        response = await self.http.post("/chat", json={"message": "slow"})
        self.assertEqual(response.status_code, 504)
        self.assertEqual(main.app.state.admitted, 0)
        self.assertEqual(self.fake.active, 0)
        self.fake.release.set()
        self.assertEqual((await self.http.post("/chat", json={"message": "next"})).status_code, 200)

    async def test_cancellation_releases_capacity(self):
        request = self.send("cancel")
        await self.fake.started.get()
        request.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await request
        self.assertEqual(main.app.state.admitted, 0)
        self.fake.release.set()
        self.assertEqual((await self.http.post("/chat", json={"message": "next"})).status_code, 200)

    async def test_upstream_error_does_not_expose_details(self):
        self.fake.error = RuntimeError("secret provider detail")
        self.fake.release.set()
        with self.assertLogs(main.logger, level="ERROR"):
            response = await self.http.post("/chat", json={"message": "error"})
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("secret provider detail", response.text)
        self.assertEqual(main.app.state.admitted, 0)


if __name__ == "__main__":
    unittest.main()
