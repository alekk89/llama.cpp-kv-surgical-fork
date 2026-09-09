import pytest
from utils import *


server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.slot_save_path = "./tmp"
    server.server_slots = True
    server.temperature = 0.0
    server.n_predict = 4


def tokenize(content, *, add_special=False):
    res = server.make_request("POST", "/tokenize", data={
        "content": content,
        "add_special": add_special,
    })
    assert res.status_code == 200
    assert res.body["tokens"]
    return res.body["tokens"]


def managed_prefill(content):
    tokens = tokenize(content, add_special=True)
    res = server.make_request("POST", "/slots/1?action=managed_native_prefill", data={
        "expected_revision": 0,
        "tokens": tokens,
    })
    assert res.status_code == 200
    assert res.body["n_appended"] == len(tokens)
    assert res.body["managed_revision"] == 1
    assert res.body["managed_append_only"] is True
    assert res.body["managed_requires_rebuild"] is False
    return tokens


def test_slot_kv_edit_and_append_probe():
    global server
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox jumps over the lazy dog. This is a cache edit test.",
        "id_slot": 1,
        "cache_prompt": True,
        "n_predict": 1,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/tokenize", data={
        "content": " summary",
    })
    assert res.status_code == 200
    summary_tokens = res.body["tokens"]
    assert summary_tokens

    res = server.make_request("POST", "/slots/1?action=kv_edit", data={
        "expected_revision": 0,
        "edits": [{
            "start_pos": 2,
            "end_pos": 4,
            "tokens": summary_tokens[:1],
        }],
    })
    assert res.status_code == 200
    assert res.body["n_removed"] == 2
    assert res.body["n_inserted"] == 1
    assert res.body["pos_max_after"] == res.body["pos_max_before"]
    assert res.body["managed_revision"] == 1
    assert res.body["managed_append_only"] is True

    res = server.make_request("POST", "/slots/1?action=managed_generate", data={
        "expected_revision": 1,
        "tokens": summary_tokens[:1],
        "n_predict": 2,
        "temperature": 0.0,
        "stream": False,
    })
    assert res.status_code == 200
    assert res.body["n_appended"] == 1
    assert 0 <= res.body["n_generated"] <= 2
    assert res.body["pos_end"] == res.body["pos_start"] + res.body["n_appended"] + res.body["n_generated"]
    assert isinstance(res.body["tokens"], list)
    assert res.body["managed_revision"] == 2
    assert res.body["managed_append_only"] is True

    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox jumps over the lazy dog. This is a cache edit test.",
        "id_slot": 1,
        "cache_prompt": True,
        "n_predict": 1,
    })
    assert res.status_code == 400

    res = server.make_request("POST", "/slots/1?action=kv_append", data={
        "expected_revision": 0,
        "tokens": summary_tokens[:1],
    })
    assert res.status_code == 400

    res = server.make_request("POST", "/slots/1?action=kv_append", data={
        "expected_revision": 2,
        "tokens": summary_tokens[:1],
    })
    assert res.status_code == 200
    assert res.body["n_appended"] == 1
    assert res.body["pos_start"] > 0
    assert res.body["pos_end"] == res.body["pos_start"] + 1
    assert isinstance(res.body["token_probe"], int)
    assert res.body["managed_revision"] == 3
    assert res.body["managed_append_only"] is True
    appended_start = res.body["pos_start"]
    appended_end = res.body["pos_end"]

    # A non-bootstrap append must extend the position-aligned prompt ledger,
    # otherwise a later edit can see physical KV beyond its managed tokens.
    res = server.make_request("POST", "/slots/1?action=kv_edit", data={
        "expected_revision": 3,
        "edits": [{
            "start_pos": appended_start,
            "end_pos": appended_end,
            "tokens": summary_tokens[:1],
        }],
    })
    assert res.status_code == 200
    assert res.body["managed_revision"] == 4

    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200
    assert res.body["managed_revision"] == 5


def test_managed_slot_capabilities():
    global server
    server.start()

    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    assert res.body["managed_slot"] == {
        "api_version": 1,
        "requires_slot_save_path": True,
        "requires_expected_revision": True,
        "reports_rebuild_required": True,
        "supports_mmproj": False,
        "supports_context_shift": False,
        "supports_slot_save": False,
        "edit": True,
        "append": True,
        "append_generate": True,
        "legacy_generate_deprecated": True,
        "bootstrap_generate": True,
        "native_completion": True,
        "native_bootstrap": True,
        "native_prefill_recovery": True,
        "native_continuation": True,
        "dual_kv_edit": True,
        "qwen_attention_only_dflash_dual_edit": False,
        "qwen_attention_only_dflash_dual_compact": False,
        "qwen_attention_only": True,
        "qwen_compact_positions": True,
    }


def test_slot_kv_holes_compact_in_one_batch():
    global server
    server.start()

    tokens = managed_prefill(
        "One two three four five six seven eight nine ten eleven twelve thirteen fourteen."
    )
    assert len(tokens) > 10

    removed = server.make_request("POST", "/slots/1?action=kv_edit", data={
        "expected_revision": 1,
        "edits": [
            {"start_pos": 2, "end_pos": 4, "tokens": []},
            {"start_pos": 7, "end_pos": 9, "tokens": []},
        ],
    })
    assert removed.status_code == 200
    assert removed.body["positions_compacted"] is False
    assert removed.body["pos_max_after"] == removed.body["pos_max_before"]

    compacted = server.make_request("POST", "/slots/1?action=kv_edit", data={
        "expected_revision": 2,
        "compact_positions": True,
        "edits": [
            {"start_pos": 2, "end_pos": 4, "tokens": []},
            {"start_pos": 7, "end_pos": 9, "tokens": []},
        ],
    })
    assert compacted.status_code == 200
    assert compacted.body["positions_compacted"] is True
    assert compacted.body["n_removed"] == 4
    assert compacted.body["n_inserted"] == 0
    assert compacted.body["pos_max_after"] == compacted.body["pos_max_before"] - 4

    tail = tokenize(" tail")[:1]
    appended = server.make_request("POST", "/slots/1?action=kv_append", data={
        "expected_revision": 3,
        "tokens": tail,
    })
    assert appended.status_code == 200
    assert appended.body["pos_start"] == compacted.body["pos_max_after"] + 1


def test_managed_native_lifecycle_and_stream_contract():
    global server
    server.start()

    managed_prefill("Managed cache lifecycle test with enough tokens for an edit.")
    replacement = tokenize(" short")[:1]

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "managed-slot.bin",
    })
    assert res.status_code == 400

    res = server.make_request("POST", "/slots/1?action=kv_edit", data={
        "expected_revision": 1,
        "edits": [{
            "start_pos": 2,
            "end_pos": 4,
            "tokens": replacement,
        }],
    })
    assert res.status_code == 200
    assert res.body["managed_revision"] == 2

    events = list(server.make_stream_request(
        "POST",
        "/slots/1?action=managed_native_completion",
        data={
            "expected_revision": 2,
            "tokens": replacement,
            "n_predict": 2,
            "temperature": 0.0,
            "stream": True,
        },
    ))
    assert events
    assert events[0]["type"] == "managed_native_begin"
    assert all(event["type"] in {
        "managed_native_begin",
        "managed_native_delta",
        "managed_native_final",
    } for event in events)
    final = next(event for event in events if event["type"] == "managed_native_final")
    assert final["managed_native"] is True
    assert final["n_appended"] == 1
    assert final["managed_revision"] == 3
    assert final["managed_append_only"] is True
    assert final["managed_requires_rebuild"] is False

    res = server.make_request("POST", "/slots/1?action=managed_native_completion", data={
        "expected_revision": 3,
        "tokens": replacement,
        "temperature": 0.0,
        "stream": False,
    })
    assert res.status_code == 200
    assert res.body["type"] == "managed_native_final"
    assert res.body["managed_native"] is True
    assert res.body["n_appended"] == 1
    assert res.body["managed_revision"] == 4
    assert res.body["managed_requires_rebuild"] is False
    assert 0 <= res.body["tokens_predicted"] <= server.n_predict

    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200
    assert res.body["managed_revision"] == 5
    assert res.body["managed_append_only"] is False
    assert res.body["managed_requires_rebuild"] is False

    res = server.make_request("POST", "/completion", data={
        "prompt": "The erased slot is available to ordinary completion again.",
        "id_slot": 1,
        "cache_prompt": True,
        "n_predict": 1,
    })
    assert res.status_code == 200


def test_managed_mutations_require_revision_and_preserve_revision_on_rejection():
    global server
    server.start()

    tokens = managed_prefill("One two three four five six seven eight nine ten.")
    replacement = tokenize(" short")[:1]
    assert len(tokens) > 8

    invalid_requests = [
        {
            "edits": [{"start_pos": 1, "end_pos": 2, "tokens": []}],
        },
        {
            "expected_revision": 0,
            "edits": [{"start_pos": 1, "end_pos": 2, "tokens": []}],
        },
        {
            "expected_revision": 2**64 - 1,
            "edits": [{"start_pos": 1, "end_pos": 2, "tokens": []}],
        },
        {
            "expected_revision": 1,
            "edits": [
                {"start_pos": 1, "end_pos": 4, "tokens": []},
                {"start_pos": 3, "end_pos": 5, "tokens": []},
            ],
        },
        {
            "expected_revision": 1,
            "edits": [{"start_pos": 1, "end_pos": 2, "tokens": replacement * 2}],
        },
        {
            "expected_revision": 1,
            "edits": [{"start_pos": 1, "end_pos": 2, "tokens": [2**31 - 1]}],
        },
    ]
    for request in invalid_requests:
        res = server.make_request("POST", "/slots/1?action=kv_edit", data=request)
        assert res.status_code == 400

    res = server.make_request("POST", "/slots/1?action=kv_edit", data={
        "expected_revision": 1,
        "edits": [
            {"start_pos": 1, "end_pos": 2, "tokens": []},
            {"start_pos": 3, "end_pos": 4, "tokens": replacement},
        ],
    })
    assert res.status_code == 200
    assert res.body["managed_revision"] == 2

    res = server.make_request("POST", "/slots/1?action=kv_append", data={
        "tokens": replacement,
    })
    assert res.status_code == 400

    res = server.make_request("POST", "/slots/1?action=managed_native_completion", data={
        "tokens": replacement,
        "n_predict": 1,
    })
    assert res.status_code == 400


def test_partial_append_failure_requires_explicit_rebuild():
    global server
    server.start()

    managed_prefill("Managed slot failure recovery test.")
    token = tokenize(" tail")[:1]

    res = server.make_request("POST", "/slots/1?action=kv_append", data={
        "expected_revision": 1,
        "tokens": token * (server.n_ctx + 32),
    })
    assert res.status_code == 500

    res = server.make_request("GET", "/slots")
    assert res.status_code == 200
    slot = next(item for item in res.body if item["id"] == 1)
    assert slot["managed_revision"] == 2
    assert slot["managed_append_only"] is True
    assert slot["managed_draft_coherent"] is False
    assert slot["managed_requires_rebuild"] is True

    res = server.make_request("POST", "/slots/1?action=kv_append", data={
        "expected_revision": 2,
        "tokens": token,
    })
    assert res.status_code == 400

    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200
    assert res.body["managed_revision"] == 3
    assert res.body["managed_append_only"] is False
    assert res.body["managed_requires_rebuild"] is False


def test_managed_generation_stops_instead_of_shifting_absolute_positions():
    global server
    server.start()

    token = tokenize(" tail")[:1]
    res = server.make_request("POST", "/slots/1?action=managed_native_prefill", data={
        "expected_revision": 0,
        "tokens": token * 240,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/1?action=managed_native_completion", data={
        "expected_revision": 1,
        "tokens": token,
        "n_predict": 50,
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": False,
    })
    assert res.status_code == 200
    assert res.body["stop_type"] == "limit"
    assert res.body["tokens_predicted"] < 50
    assert res.body["managed_revision"] == 2
    assert res.body["managed_append_only"] is True
    assert res.body["managed_requires_rebuild"] is False
