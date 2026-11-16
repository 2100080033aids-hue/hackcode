import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx

import main


class FakeInferenceClient:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.chat = SimpleNamespace(completions=self)
        self.started = asyncio.Queue()
        self.release = asyncio.Event()
        self.gates = {}
        self.outcomes = {}
        self.active = 0
        self.peak = 0
        self.closed = False
        self.error = None

    async def create(self, **kwargs):
        self.active += 1
        self.peak = max(self.peak, self.active)
        message = kwargs["messages"][-1]["content"]
        await self.started.put(kwargs)
        try:
            gate = self.gates.get(message, self.release)
            await gate.wait()
            outcome = self.outcomes.get(message, self.error)
            if isinstance(outcome, BaseException):
                raise outcome
            if callable(outcome):
                return outcome(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=message))])
        finally:
            self.active -= 1

    async def close(self):
        self.closed = True


class ConfigurationTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_settings_and_missing_hf_token(self):
        with patch.dict("os.environ", {}, clear=True), patch.object(main, "AsyncInferenceClient", FakeInferenceClient):
            async with main.app.router.lifespan_context(main.app):
                client = main.app.state.client
                self.assertEqual(main.app.state.capacity, 60)
                self.assertEqual(main.app.state.slots._value, 10)
                self.assertEqual(main.app.state.queue_timeout, 5.0)
                self.assertEqual(main.app.state.inference_timeout, 120.0)
                self.assertEqual(main.app.state.admitted, 0)
                self.assertEqual(client.init_kwargs, {"api_key": None})
                self.assertFalse(client.closed)
            self.assertTrue(client.closed)

    async def test_hf_token_is_forwarded_to_client(self):
        env = {"HF_TOKEN": "hf_test_token_123"}
        with patch.dict("os.environ", env, clear=True), patch.object(main, "AsyncInferenceClient", FakeInferenceClient):
            async with main.app.router.lifespan_context(main.app):
                self.assertEqual(main.app.state.client.init_kwargs, {"api_key": "hf_test_token_123"})

    async def test_zero_queued_requests_is_valid(self):
        env = {"MAX_CONCURRENT_REQUESTS": "3", "MAX_QUEUED_REQUESTS": "0"}
        with patch.dict("os.environ", env, clear=True), patch.object(main, "AsyncInferenceClient", FakeInferenceClient):
            async with main.app.router.lifespan_context(main.app):
                self.assertEqual(main.app.state.capacity, 3)
                self.assertEqual(main.app.state.slots._value, 3)

    async def test_fractional_timeout_settings_are_valid(self):
        env = {"QUEUE_TIMEOUT_SECONDS": "0.25", "INFERENCE_TIMEOUT_SECONDS": "1.5"}
        with patch.dict("os.environ", env, clear=True), patch.object(main, "AsyncInferenceClient", FakeInferenceClient):
            async with main.app.router.lifespan_context(main.app):
                self.assertEqual(main.app.state.queue_timeout, 0.25)
                self.assertEqual(main.app.state.inference_timeout, 1.5)

    async def test_invalid_integer_settings_fail_startup(self):
        invalid_cases = [
            ("MAX_CONCURRENT_REQUESTS", "0", "must be an integer greater than 0"),
            ("MAX_CONCURRENT_REQUESTS", "-1", "must be an integer greater than 0"),
            ("MAX_CONCURRENT_REQUESTS", "2.5", "must be an integer greater than 0"),
            ("MAX_CONCURRENT_REQUESTS", "abc", "must be an integer greater than 0"),
            ("MAX_CONCURRENT_REQUESTS", "", "must be an integer greater than 0"),
            ("MAX_QUEUED_REQUESTS", "-1", "must be an integer greater than -1"),
            ("MAX_QUEUED_REQUESTS", "-5", "must be an integer greater than -1"),
            ("MAX_QUEUED_REQUESTS", "1.5", "must be an integer greater than -1"),
            ("MAX_QUEUED_REQUESTS", "abc", "must be an integer greater than -1"),
            ("MAX_QUEUED_REQUESTS", "", "must be an integer greater than -1"),
        ]
        for name, raw, expected_message in invalid_cases:
            with self.subTest(name=name, raw=raw):
                with patch.dict("os.environ", {name: raw}, clear=True), patch.object(
                    main, "AsyncInferenceClient", FakeInferenceClient
                ):
                    with self.assertRaisesRegex(ValueError, f"Invalid {name}: {expected_message}"):
                        async with main.app.router.lifespan_context(main.app):
                            pass

    async def test_invalid_timeout_settings_fail_startup(self):
        for name in ("QUEUE_TIMEOUT_SECONDS", "INFERENCE_TIMEOUT_SECONDS"):
            for raw in ("0", "-0.5", "nan", "inf", "-inf", "abc", ""):
                with self.subTest(name=name, raw=raw):
                    with patch.dict("os.environ", {name: raw}, clear=True), patch.object(
                        main, "AsyncInferenceClient", FakeInferenceClient
                    ):
                        with self.assertRaisesRegex(
                            ValueError, f"Invalid {name}: must be a number greater than 0"
                        ):
                            async with main.app.router.lifespan_context(main.app):
                                pass


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

    async def test_forwards_model_and_system_prompt_and_returns_snippet(self):
        snippet = "class MainActivity : ComponentActivity() {\n    // Kotlin activity\n}"
        self.fake.outcomes["Create a Kotlin activity"] = lambda _: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=snippet))]
        )
        self.fake.release.set()
        response = await self.http.post("/chat", json={"message": "Create a Kotlin activity"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"response": snippet})
        call = self.fake.started.get_nowait()
        self.assertEqual(call["model"], main.MODEL)
        self.assertEqual(
            call["messages"],
            [
                {"role": "system", "content": main.SYSTEM_PROMPT},
                {"role": "user", "content": "Create a Kotlin activity"},
            ],
        )
        self.assertEqual(main.app.state.admitted, 0)

    async def test_empty_message_string_is_accepted(self):
        self.fake.release.set()
        response = await self.http.post("/chat", json={"message": ""})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"response": ""})
        self.assertEqual(main.app.state.admitted, 0)

    async def test_invalid_payloads_return_422_without_consuming_capacity(self):
        invalid_payloads = [
            {"json": {}},
            {"json": {"message": None}},
            {"json": {"message": 123}},
            {"json": {"message": ["not", "a", "string"]}},
            {"content": b"{invalid-json", "headers": {"content-type": "application/json"}},
        ]
        for kwargs in invalid_payloads:
            with self.subTest(kwargs=kwargs):
                response = await self.http.post("/chat", **kwargs)
                self.assertEqual(response.status_code, 422)
                self.assertEqual(main.app.state.admitted, 0)
                self.assertTrue(self.fake.started.empty())

    async def test_unsupported_method_and_unknown_route(self):
        get_response = await self.http.get("/chat")
        self.assertEqual(get_response.status_code, 405)
        missing_response = await self.http.post("/unknown", json={"message": "hi"})
        self.assertEqual(missing_response.status_code, 404)
        self.assertEqual(main.app.state.admitted, 0)

    async def test_cors_preflight_and_error_response_headers(self):
        preflight = await self.http.options(
            "/chat",
            headers={
                "Origin": "https://android-studio.example",
                "Access-Control-Request-Method": "POST",
            },
        )
        self.assertEqual(preflight.status_code, 200)
        self.assertEqual(preflight.headers["access-control-allow-origin"], "https://android-studio.example")
        self.assertIn("POST", preflight.headers["access-control-allow-methods"])

        main.app.state.capacity = 0
        busy = await self.http.post(
            "/chat",
            json={"message": "blocked"},
            headers={"Origin": "https://android-studio.example"},
        )
        self.assertEqual(busy.status_code, 503)
        self.assertEqual(busy.headers["access-control-allow-origin"], "https://android-studio.example")

    async def test_zero_queue_capacity_rejects_immediately(self):
        main.app.state.capacity = 2
        main.app.state.queue_timeout = 10.0
        first, second = self.send("first"), self.send("second")
        await self.fake.started.get()
        await self.fake.started.get()

        rejected = await asyncio.wait_for(self.http.post("/chat", json={"message": "third"}), timeout=0.5)
        self.assertEqual(rejected.status_code, 503)
        self.assertEqual(rejected.json(), {"detail": "Server is busy. Please retry."})
        self.assertEqual(rejected.headers["retry-after"], "1")

        self.fake.release.set()
        results = await asyncio.gather(first, second)
        self.assertTrue(all(r.status_code == 200 for r in results))
        self.assertEqual(main.app.state.admitted, 0)

    async def test_error_responses_have_expected_details_and_headers(self):
        # 1. Queue timeout 503 detail + Retry-After
        first, second = self.send("first"), self.send("second")
        await self.fake.started.get()
        await self.fake.started.get()
        queue_timed_out = await self.http.post("/chat", json={"message": "queued"})
        self.assertEqual(queue_timed_out.status_code, 503)
        self.assertEqual(queue_timed_out.json(), {"detail": "Request queue timed out. Please retry."})
        self.assertEqual(queue_timed_out.headers["retry-after"], "1")
        self.fake.release.set()
        await asyncio.gather(first, second)

        # 2. Inference timeout 504 detail
        self.fake.release.clear()
        main.app.state.inference_timeout = 0.01
        inference_timed_out = await self.http.post("/chat", json={"message": "slow"})
        self.assertEqual(inference_timed_out.status_code, 504)
        self.assertEqual(inference_timed_out.json(), {"detail": "Inference request timed out."})

        # 3. Upstream failure 502 detail
        main.app.state.inference_timeout = 1.0
        self.fake.error = ValueError("internal token error")
        self.fake.release.set()
        with self.assertLogs(main.logger, level="ERROR"):
            upstream_failed = await self.http.post("/chat", json={"message": "boom"})
        self.assertEqual(upstream_failed.status_code, 502)
        self.assertEqual(upstream_failed.json(), {"detail": "Inference service request failed."})

    async def test_malformed_upstream_completion_returns_502_and_releases_slot(self):
        self.fake.outcomes["empty_choices"] = lambda _: SimpleNamespace(choices=[])
        self.fake.outcomes["null_content"] = lambda _: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
        )
        self.fake.release.set()

        for msg in ("empty_choices", "null_content"):
            with self.subTest(msg=msg):
                with self.assertLogs(main.logger, level="ERROR"):
                    response = await self.http.post("/chat", json={"message": msg})
                self.assertEqual(response.status_code, 502)
                self.assertEqual(response.json(), {"detail": "Inference service request failed."})
                self.assertEqual(main.app.state.admitted, 0)

        ok = await self.http.post("/chat", json={"message": "healthy"})
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(ok.json(), {"response": "healthy"})

    async def test_queued_request_proceeds_when_active_request_fails(self):
        main.app.state.queue_timeout = 1.0
        fail_gate = asyncio.Event()
        self.fake.gates["failing"] = fail_gate
        self.fake.outcomes["failing"] = RuntimeError("provider crash")

        failing = self.send("failing")
        second = self.send("second")
        await asyncio.wait_for(self.fake.started.get(), 1)
        await asyncio.wait_for(self.fake.started.get(), 1)

        third = self.send("third")
        await asyncio.sleep(0.01)
        self.assertEqual(main.app.state.admitted, 3)

        with self.assertLogs(main.logger, level="ERROR"):
            fail_gate.set()
            failed_resp = await failing
        self.assertEqual(failed_resp.status_code, 502)

        # The queued third request should acquire the freed slot before second finishes.
        third_call = await asyncio.wait_for(self.fake.started.get(), 1)
        self.assertEqual(third_call["messages"][-1]["content"], "third")
        self.assertEqual(self.fake.peak, 2)

        self.fake.release.set()
        second_resp, third_resp = await asyncio.gather(second, third)
        self.assertEqual(second_resp.status_code, 200)
        self.assertEqual(third_resp.status_code, 200)
        self.assertEqual(third_resp.json(), {"response": "third"})
        self.assertEqual(main.app.state.admitted, 0)

    async def test_queued_request_proceeds_when_active_request_times_out(self):
        main.app.state.queue_timeout = 1.0
        main.app.state.inference_timeout = 0.03
        slow = self.send("slow")
        await asyncio.wait_for(self.fake.started.get(), 1)

        main.app.state.inference_timeout = 1.0
        second = self.send("second")
        await asyncio.wait_for(self.fake.started.get(), 1)

        third = self.send("third")
        slow_resp = await slow
        self.assertEqual(slow_resp.status_code, 504)

        third_call = await asyncio.wait_for(self.fake.started.get(), 1)
        self.assertEqual(third_call["messages"][-1]["content"], "third")
        self.assertEqual(self.fake.peak, 2)

        self.fake.release.set()
        second_resp, third_resp = await asyncio.gather(second, third)
        self.assertEqual(second_resp.status_code, 200)
        self.assertEqual(third_resp.status_code, 200)
        self.assertEqual(main.app.state.admitted, 0)

    async def test_burst_concurrency_never_exceeds_max_concurrent_slots(self):
        main.app.state.queue_timeout = 1.0
        tasks = [self.send(f"msg-{i}") for i in range(6)]
        await asyncio.wait_for(self.fake.started.get(), 1)
        await asyncio.wait_for(self.fake.started.get(), 1)

        # Wait until the 3 excess requests beyond capacity (2 active + 1 queued) are rejected.
        pending = set(tasks)
        rejected = []
        while len(rejected) < 3:
            done, pending = await asyncio.wait(pending, timeout=1, return_when=asyncio.FIRST_COMPLETED)
            self.assertTrue(done, "Expected excess burst requests to complete with 503")
            rejected.extend(task.result() for task in done)

        self.assertTrue(all(r.status_code == 503 for r in rejected))
        self.assertEqual(main.app.state.admitted, 3)

        self.fake.release.set()
        succeeded = await asyncio.gather(*pending)
        self.assertEqual(len(succeeded), 3)
        self.assertTrue(all(r.status_code == 200 for r in succeeded))
        self.assertEqual(self.fake.peak, 2)
        self.assertEqual(main.app.state.admitted, 0)


if __name__ == "__main__":
    unittest.main()
