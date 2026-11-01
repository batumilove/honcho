# Recovering unavailable-memory turns

This provider can finish startup before Honcho session setup completes. One initializer per profile holds `state/honcho-init.lock` until its actual work returns, including across module reloads. A timed-out join does not cancel the SDK call or release its admission. Keep the persistent lock carrier; deleting or replacing it can admit overlapping work. Different profiles have separate locks. Shutdown still tracks and joins the worker through the existing lifecycle.

When eager/context memory is unavailable, permitted turns are retained as private, fsynced `state/honcho-pending-turns/*.json` files. A `*.partial` file means publication was interrupted; preserve it for inspection. `saveMessages:false`, disabled bot ingestion, missing bot IDs, machine-generated turns, and intentionally lazy tools-only capture remain respected. Bot records name their separate a2a session. Records bind the original profile, workspace/endpoint settings, peer inputs and alias-map hash without saving credentials.

These records are recovery evidence. `state:not_submitted` describes creation by this journal; it is not proof that no other system later submitted the turn. The provider does not automatically replay or expire records. Local retention is not remote ingestion, and normal SDK write-queue receipts remain independent.

## Inventory without contacting Honcho

Use the owning plugin source and pass the profile home explicitly:

```sh
python pending_turns.py --home /absolute/path/to/profile --limit 50
```

This is read-only and uses the standard library. Output includes filenames, byte hashes, destination binding and character counts, but no message bodies. Keep the inventory private: destination/identity metadata can also be sensitive. The scan is bounded and reports truncation. Unknown schemas, mismatched profile bindings, symlinks and oversized records are held for manual inspection, never silently migrated or submitted.

## Explicit reconciliation procedure

1. Preserve the original records and partials. Record each selected JSON's exact filename and SHA-256, the original profile/config receipts, and the operator owning recovery. Do not rename a partial to JSON merely because it parses.
2. Under the original profile's secret and identity scope, verify the actual endpoint, workspace, user/assistant peers, alias-map hash and target session against the captured binding. Verify current `saveMessages` and bot/a2a policy. A changed or disabled policy holds the record for a separate decision; it does not authorize migration into the default profile or a human session.
3. Inspect existing delivery/recovery receipts and, when separately authorized, the supported Honcho message API for the selected session. Resolve any earlier attempt before considering submission. Do not infer absence from a timeout or from only one page of history. Do not query or replay unrelated sessions.
4. Prepare a finite submission plan naming the record hashes, canonical target peer/session IDs, per-message roles/author attribution, message chunks, and intended metadata identifying the journal record. Include the exact number of mutations and the expected returned message IDs. Obtain explicit approval for this replay scope; source-fix/update approval alone does not authorize replay.
5. Before the one approved submission, create a private durable attempt receipt with record hash, target identity, operation ID and `attempting` state. Fsync it before the POST. Use a supported Honcho client/API with retries disabled unless server idempotency for that exact operation is independently verified. Preserve any response and returned IDs.
6. Confirm the returned IDs through a finite supported readback against the approved destination. Record `confirmed` only for messages actually verified. If any result is missing, partial or ambiguous, record `unresolved` and reconcile that attempt before another submission; never blindly retry the whole turn.
7. Append a private durable reconciliation receipt binding the original record hash, attempt ID, confirmed message IDs, endpoint/workspace/session and verification time. Retain the original JSON, partials and all attempt receipts. This procedure does not delete records, restore databases, or discard newer writes.

The inventory tool deliberately performs none of the mutation steps. There is no claim of automatic exactly-once ingestion or an exercised production replay. A tested, server-supported idempotency/acknowledgement contract is required before implementing an automatic replay worker.

File data is fsynced on supported platforms; POSIX additionally fsyncs the parent directory. Native Windows execution and power-loss fault injection have not been verified by the Linux test receipt.
