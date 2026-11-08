"""Offline send-boundary contracts; run under the exact deployed image deps.

The runner must clear credentials and use --network=none before importing src.
"""

import socket
import unittest
from contextlib import closing
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


def no_network(*args, **kwargs):
    raise AssertionError("provider network forbidden")


socket.socket.connect = no_network
socket.create_connection = no_network

from src.embedding_client import BatchItem, _EmbeddingClient  # noqa: E402
from src.llm import executor  # noqa: E402
from src.llm.backends.openai import OpenAIBackend  # noqa: E402
from src.llm.types import LLMTelemetryContext  # noqa: E402
from src.request_receipts import (  # noqa: E402
    PURPOSE_STAGES,
    request_headers,
    validated_headers,
)


def client():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="ok", tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=None,
    )
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=response))
        )
    )


class ReceiptHeaderTests(unittest.IsolatedAsyncioTestCase):
    def test_validation_has_no_arbitrary_or_authentication_headers(self):
        headers = request_headers("deriver.representation")
        headers["Authorization"] = "FAKE"
        self.assertNotIn("Authorization", validated_headers(headers))
        headers["X-Honcho-Stage"] = {"invalid": True}
        self.assertEqual(validated_headers(headers), {})
        self.assertEqual(request_headers(["invalid"])["X-Honcho-Stage"], "unattributed")

    async def test_executor_to_sdk_keeps_every_stage_distinct_from_model(self):
        identifiers = set()
        for purpose, expected_stage in PURPOSE_STAGES.items():
            fake = client()
            backend = OpenAIBackend(fake)
            with (
                patch.object(executor, "backend_for_provider", return_value=backend),
                patch.object(executor, "_emit_llm_call_completed"),
            ):
                await executor.honcho_llm_call_inner(
                    provider="openai",
                    model="same-model",
                    max_tokens=32,
                    client_override=fake,
                    prompt="SENSITIVE CONTENT",
                    telemetry=LLMTelemetryContext(
                        call_purpose=purpose, workspace_name="SENSITIVE WORKSPACE"
                    ),
                )
            kwargs = fake.chat.completions.create.await_args.kwargs
            headers = kwargs["extra_headers"]
            self.assertEqual(headers["X-Honcho-Stage"], expected_stage)
            self.assertEqual(headers["X-Honcho-Call-Purpose"], purpose)
            self.assertNotIn("SENSITIVE", repr(headers))
            self.assertNotIn("honcho_receipt_headers", kwargs.get("extra_body", {}))
            identifiers.add(headers["X-Honcho-Request-ID"])
        self.assertEqual(len(identifiers), len(PURPOSE_STAGES))

    async def test_streaming_send_uses_the_same_header_channel(self):
        fake = client()

        class Stream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        fake.chat.completions.create.return_value = Stream()
        backend = OpenAIBackend(fake)
        with (
            patch.object(executor, "backend_for_provider", return_value=backend),
            patch.object(executor, "_emit_llm_call_completed"),
        ):
            stream = await executor.honcho_llm_call_inner(
                provider="openai",
                model="same-model",
                max_tokens=32,
                client_override=fake,
                prompt="SENSITIVE CONTENT",
                stream=True,
                telemetry=LLMTelemetryContext(call_purpose="dialectic.answer"),
            )
            async for _ in stream:
                pass
        kwargs = fake.chat.completions.create.await_args.kwargs
        self.assertTrue(kwargs["stream"])
        self.assertEqual(kwargs["extra_headers"]["X-Honcho-Stage"], "dialectic")

    async def test_all_openai_embedding_send_paths_carry_opaque_identity(self):
        fake = SimpleNamespace(
            embeddings=SimpleNamespace(
                create=AsyncMock(
                    return_value=SimpleNamespace(
                        data=[SimpleNamespace(embedding=[0.1, 0.2])]
                    )
                )
            )
        )
        embedding = _EmbeddingClient.__new__(_EmbeddingClient)
        embedding.client = fake
        embedding.model = "embedding-model"
        embedding.transport = "openai"
        embedding.encoding = SimpleNamespace(encode=lambda text: [1])
        embedding.max_embedding_tokens = 8192
        embedding.max_batch_size = 32
        embedding.vector_dimensions = 2
        embedding.send_dimensions = False

        async def emit(**kwargs):
            return await kwargs["fn"]()

        with patch("src.embedding_client._emit_embedding_call", side_effect=emit):
            self.assertEqual(await embedding.embed("SENSITIVE CONTENT"), [0.1, 0.2])
            self.assertEqual(
                await embedding.simple_batch_embed(["SENSITIVE CONTENT"]), [[0.1, 0.2]]
            )
            await embedding._process_batch([BatchItem("id", 0, "SENSITIVE CONTENT", 1)])
        ids = set()
        self.assertEqual(fake.embeddings.create.await_count, 3)
        for call in fake.embeddings.create.await_args_list:
            headers = call.kwargs["extra_headers"]
            self.assertEqual(headers["X-Honcho-Stage"], "embedding")
            self.assertNotIn("SENSITIVE", repr(headers))
            ids.add(headers["X-Honcho-Request-ID"])
        self.assertEqual(len(ids), 3)

    async def test_real_sdk_serialization_carries_headers_not_body_fields(self):
        import httpx
        from openai import AsyncOpenAI

        seen = []

        def respond(request):
            import json

            body = json.loads(request.content)
            self.assertNotIn("honcho_receipt_headers", body)
            self.assertNotIn("extra_headers", body)
            seen.append(request.headers.get("X-Honcho-Stage"))
            return httpx.Response(
                200,
                json={
                    "id": "fixture",
                    "object": "chat.completion",
                    "model": "same-model",
                    "created": 1,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            )

        http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        sdk = AsyncOpenAI(
            api_key="FAKE", base_url="https://fixture.invalid/v1", http_client=http
        )
        try:
            for purpose in PURPOSE_STAGES:
                with patch.object(executor, "_emit_llm_call_completed"):
                    result = await executor.honcho_llm_call_inner(
                        provider="openai",
                        model="same-model",
                        max_tokens=32,
                        client_override=sdk,
                        prompt="SENSITIVE CONTENT",
                        telemetry=LLMTelemetryContext(call_purpose=purpose),
                    )
                self.assertEqual(result.content, "ok")
        finally:
            await sdk.close()
        self.assertEqual(seen, list(PURPOSE_STAGES.values()))

    async def test_caller_receipt_captures_failure_before_proxy_without_content(self):
        import os
        import sqlite3
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "caller.sqlite3"
            fake = client()
            fake.base_url = "https://user:FAKE@fixture.invalid/v1?token=SECRET"
            fake.chat.completions.create.side_effect = TimeoutError("SENSITIVE FAILURE")
            with (
                patch.dict(os.environ, {"HONCHO_REQUEST_RECEIPT_PATH": str(path)}),
                patch.object(
                    executor, "backend_for_provider", return_value=OpenAIBackend(fake)
                ),
                patch.object(executor, "_emit_llm_call_completed"),
                self.assertRaises(TimeoutError),
            ):
                await executor.honcho_llm_call_inner(
                    provider="openai",
                    model="same-model",
                    max_tokens=32,
                    client_override=fake,
                    prompt="SENSITIVE CONTENT",
                    telemetry=LLMTelemetryContext(
                        call_purpose="deriver.representation"
                    ),
                )
            headers = fake.chat.completions.create.await_args.kwargs["extra_headers"]
            with closing(sqlite3.connect(path)) as connection:
                row = connection.execute(
                    "SELECT id,stage,outcome,error_class,endpoint FROM requests"
                ).fetchone()
            self.assertEqual(
                row,
                (
                    headers["X-Honcho-Request-ID"],
                    "deriver",
                    "error",
                    "TimeoutError",
                    "https://fixture.invalid",
                ),
            )
            self.assertNotIn(b"SENSITIVE", path.read_bytes())
            self.assertNotIn(b"SECRET", path.read_bytes())
            self.assertNotIn(b"FAKE", path.read_bytes())
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    async def test_stream_cancellation_has_a_distinct_durable_outcome(self):
        import asyncio
        import os
        import sqlite3
        import tempfile
        from pathlib import Path

        fake = client()

        class CancelledStream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise asyncio.CancelledError()

        fake.chat.completions.create.return_value = CancelledStream()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "caller.sqlite3"
            with (
                patch.dict(os.environ, {"HONCHO_REQUEST_RECEIPT_PATH": str(path)}),
                patch.object(
                    executor, "backend_for_provider", return_value=OpenAIBackend(fake)
                ),
                patch.object(executor, "_emit_llm_call_completed"),
            ):
                stream = await executor.honcho_llm_call_inner(
                    provider="openai",
                    model="same-model",
                    max_tokens=32,
                    client_override=fake,
                    prompt="SENSITIVE",
                    stream=True,
                    telemetry=LLMTelemetryContext(call_purpose="dialectic.answer"),
                )
                with self.assertRaises(asyncio.CancelledError):
                    async for _ in stream:
                        pass
            with closing(sqlite3.connect(path)) as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT stage,outcome,error_class FROM requests"
                    ).fetchone(),
                    ("dialectic", "cancelled", "CancelledError"),
                )


if __name__ == "__main__":
    unittest.main()
