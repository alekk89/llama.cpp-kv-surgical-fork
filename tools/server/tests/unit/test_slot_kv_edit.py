import pytest
from utils import *


server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.slot_save_path = "./tmp"
    server.temperature = 0.0


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

    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200
    assert res.body["managed_revision"] == 4


def test_managed_slot_capabilities():
    global server
    server.start()

    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    assert res.body["managed_slot"] == {
        "api_version": 1,
        "requires_slot_save_path": True,
        "edit": True,
        "append": True,
        "append_generate": True,
        "bootstrap_generate": True,
        "native_completion": True,
        "native_bootstrap": True,
        "native_prefill_recovery": True,
        "native_continuation": True,
        "dual_kv_edit": True,
        "qwen_attention_only_dflash_dual_edit": True,
        "qwen_attention_only": True,
        "qwen_compact_positions": False,
    }
