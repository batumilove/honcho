# Provider request receipts

OpenAI calls emit three allowlisted HTTP headers at the SDK send boundary: an
opaque request ID, processing stage, and known telemetry call purpose. The ID
survives SDK retries; the receiving proxy assigns its own ID to every HTTP attempt.
No conversation, workspace, session, content hash or credential enters these
headers. Stages come from the known call purpose, never from model names.
Embedding requests are explicitly labeled. Unknown purposes remain unattributed.

Set `HONCHO_REQUEST_RECEIPT_PATH` to enable optional durable caller receipts. Use
a private persistent directory owned by the container user, with separate SQLite
files for API and deriver. Admission is committed before SDK dispatch; successful
completion, failure and cancellation close the receipt. A process exit leaves
unfinished debt. This covers failures before a proxy is reachable. Receipt errors
produce a generic warning and do not block provider work; receipt absence is
unverified. There is no automatic replay or change to SDK retry policy.

Receipts contain opaque request identity, timestamps, stage/purpose, requested
model, sanitized endpoint origin, terminal outcome, exception class and HTTP
status. They exclude prompt/response text and exception messages. Completed rows
retain seven days with bounded cleanup; unfinished rows are retained. Readers
must use read-only bounded queries and preserve these files during code rollback.

For this owned deployment lane, layer only `src/request_receipts.py`,
`src/embedding_client.py`, `src/llm/executor.py` and
`src/llm/backends/openai.py` on each exact deployed API/deriver image. Preserve the
base image layers, CA configuration, image environment, entrypoint and command.
Do not patch a running container. Recreate through the existing Compose project
with private persistent receipt mounts, retaining the old image refs for rollback.
Gracefully stop the deriver and allow its active tasks to finish before recreation.

The standalone offline contract test is `scripts/test_request_receipts.py`.
Run it inside both deployed image dependency environments with network disabled,
credentials cleared, and the candidate source ahead of `/app` on `sys.path`.
It exercises the actual executor/backend and embedding boundaries plus real SDK
serialization with a mock HTTP transport. Natural caller/proxy request ID joins
and regular monitor output remain required deployment acceptance evidence.
