"""Offline harness illustration: scripted proposals, in-memory tools, no LLM/API."""
from copy import deepcopy
import json


def run(actions, checkpoint=None, max_steps=6):
    state = deepcopy(checkpoint) if checkpoint is not None else {
        "artifact": None, "checks": [], "receipts": {}, "trace": []
    }
    status = "paused"
    for step, action in enumerate(actions):
        if step >= max_steps:
            status = "budget_exhausted"
            break
        if not isinstance(action, dict) or set(action) != {"id", "tool", "args"}:
            status = "invalid_action"
            break
        key, tool, args = action["id"], action["tool"], action["args"]
        if not isinstance(key, str) or not key or not isinstance(tool, str) or not isinstance(args, dict):
            status = "invalid_action"
            break
        if tool not in {"write_draft", "check", "finish"}:
            status = "denied"
            break
        valid = (tool == "write_draft" and set(args) == {"text"}
                 and isinstance(args["text"], str)) or (tool != "write_draft" and not args)
        if not valid:
            status = "invalid_action"
            break
        fingerprint = json.dumps([tool, args], sort_keys=True)
        if key in state["receipts"]:
            if state["receipts"][key] != fingerprint:
                status = "id_conflict"
                break
            state["trace"].append("replayed:" + key)
            continue
        if tool == "finish":
            status = "complete" if state["checks"] == [True, True] else "needs_evidence"
            break
        if tool == "write_draft":
            state["artifact"] = args["text"]
            state["checks"] = []  # Editing invalidates prior checks.
        else:
            draft = state["artifact"] or ""
            state["checks"] = [draft.startswith("# "), "Source: " in draft]
        state["receipts"][key] = fingerprint
        state["trace"].append("executed:" + key)
    return status, state


if __name__ == "__main__":
    write = {"id": "draft-1", "tool": "write_draft", "args": {"text": "# Note\nSource: example"}}
    check = {"id": "check-1", "tool": "check", "args": {}}
    finish = {"id": "finish-1", "tool": "finish", "args": {}}
    first, saved = run([write])
    resumed, final = run([write, check, finish], saved)
    premature, _ = run([finish])
    denied, _ = run([{"id": "send-1", "tool": "send_email", "args": {}}])
    print(first, resumed, premature, denied)
    print(final["trace"])
    assert (first, resumed, premature, denied) == ("paused", "complete", "needs_evidence", "denied")
    assert saved["checks"] == []  # Resume does not mutate the supplied checkpoint.
    assert final["trace"].count("executed:draft-1") == 1
    changed = {**write, "args": {"text": "different"}}
    assert run([changed], saved)[0] == "id_conflict"
    assert run([write, check], max_steps=1)[0] == "budget_exhausted"
    assert run([{"id": "bad", "tool": "check", "args": {"extra": 1}}])[0] == "invalid_action"
    edit = {"id": "draft-2", "tool": "write_draft", "args": {"text": "# Changed"}}
    assert run([write, check, edit, finish])[0] == "needs_evidence"
