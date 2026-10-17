# Copyright (c) 2026 MiniStack Contributors. SPDX-License-Identifier: MIT
"""Conditional Lambda updates reject stale revisions before any side effects."""

import base64
import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from ministack.core.responses import AccountRegionScopedDict, request_scope
from ministack.services import lambda_svc as svc


@pytest.fixture
def functions(monkeypatch):
    store = AccountRegionScopedDict()
    for name in ("revision-test", "unrelated"):
        store[name] = {
            "config": {"RevisionId": "function-before", "Architectures": ["x86_64"],
                       "Description": "original", "Version": "$LATEST"},
            "code_zip": b"original-code", "versions": {}, "next_version": 1,
            "aliases": {alias: {"RevisionId": "alias-before", "FunctionVersion": "1",
                                "Description": "original"} for alias in ("live", "other")},
        }
    monkeypatch.setattr(svc, "_functions", store)
    for helper in ("invalidate_worker", "_pool_kill_function", "_sweep_extract_cache",
                   "_schedule_state_transition", "_snapstart_provision_version_async"):
        monkeypatch.setattr(svc, helper, Mock())
    return store


def _update(operation, revision="function-before", value="changed", name="revision-test"):
    data = {"RevisionId": revision} if revision is not None else {}
    if operation == "code":
        return svc._update_code(name, {**data, "ZipFile": base64.b64encode(value.encode()).decode()})
    if operation == "config":
        return svc._update_config(name, {**data, "Description": value})
    return svc._update_alias(name, "live", {**data, "Description": value, "FunctionVersion": "2"})


@pytest.mark.parametrize("operation", ["code", "config", "alias"])
def test_stale_revision_rejects_before_mutation_or_side_effects(functions, monkeypatch, operation):
    fetch = Mock(side_effect=AssertionError("stale request fetched code"))
    monkeypatch.setattr(svc, "_fetch_code_from_s3", fetch)
    before = copy.deepcopy(functions._data)
    if operation == "code":
        result = svc._update_code("revision-test", {
            "RevisionId": "stale", "S3Bucket": "unused", "S3Key": "unused",
            "Architectures": ["arm64"], "Publish": True,
            "S3ObjectStorageMode": "CustomerManaged", "SourceKMSKeyArn": "changed",
        })
    elif operation == "config":
        result = svc._update_config("revision-test", {
            "RevisionId": "stale", "Description": "changed", "MemorySize": 512,
            "Environment": {"Variables": {"CHANGED": "true"}},
        })
    else:
        result = svc._update_alias("revision-test", "live", {
            "RevisionId": "stale", "FunctionVersion": "2", "Description": "changed",
            "RoutingConfig": {"AdditionalVersionWeights": {"3": 0.1}},
        })
    status, _, body = result
    assert status == 412
    assert json.loads(body)["__type"] == "PreconditionFailedException"
    assert functions._data == before
    fetch.assert_not_called()
    for helper in (svc.invalidate_worker, svc._pool_kill_function, svc._sweep_extract_cache,
                   svc._schedule_state_transition, svc._snapstart_provision_version_async):
        helper.assert_not_called()


@pytest.mark.parametrize("operation", ["code", "config", "alias"])
def test_matching_and_omitted_revisions_allow_updates(functions, operation):
    revision = "alias-before" if operation == "alias" else "function-before"
    status, _, body = _update(operation, revision)
    assert status == 200
    updated = json.loads(body)
    assert updated["RevisionId"] != revision
    assert _update(operation, revision)[0] == 412
    assert _update(operation, updated["RevisionId"])[0] == 200
    assert _update(operation, None)[0] == 200


@pytest.mark.parametrize("style", ["keyword", "mixed", "data-keyword"])
def test_revision_updates_preserve_keyword_call_support(functions, style):
    assert svc._update_code(name="revision-test", data={"RevisionId": "function-before"})[0] == 200
    revision = functions["revision-test"]["config"]["RevisionId"]
    assert svc._update_config(name="revision-test", data={"RevisionId": revision})[0] == 200
    data = {"RevisionId": "alias-before"}
    if style == "keyword":
        result = svc._update_alias(func_name="revision-test", alias_name="live", data=data)
    elif style == "mixed":
        result = svc._update_alias("revision-test", alias_name="live", data=data)
    else:
        result = svc._update_alias("revision-test", "live", data=data)
    assert result[0] == 200


@pytest.mark.parametrize("first,second", [("code", "code"), ("code", "config"),
                                        ("config", "code"), ("config", "config"),
                                        ("alias", "alias")])
def test_competing_updates_with_same_revision_have_one_winner(functions, monkeypatch, first, second):
    entered, release, second_started, second_done = (threading.Event() for _ in range(4))
    helper = "new_uuid" if first == "alias" else "_now_iso"
    original = getattr(svc, helper)
    first_thread = []

    def pause_first():
        if threading.get_ident() == first_thread[0] and not entered.is_set():
            entered.set()
            assert release.wait(5), "first update was not released"
        return original()

    monkeypatch.setattr(svc, helper, pause_first)
    revision = "alias-before" if first == "alias" else "function-before"

    def run_first():
        first_thread.append(threading.get_ident())
        return _update(first, revision, "winner")

    def run_second():
        second_started.set()
        try:
            return _update(second, revision, "loser")
        finally:
            second_done.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        winner = pool.submit(run_first)
        try:
            assert entered.wait(5), "first update never reached mutation"
            loser = pool.submit(run_second)
            assert second_started.wait(5)
            # Let an unguarded contender finish while the first revision is
            # still current; a correctly serialized contender must wait.
            second_done.wait(0.3)
        finally:
            release.set()
        assert winner.result(timeout=5)[0] == 200
        assert loser.result(timeout=5)[0] == 412
    record = functions["revision-test"]
    if first == "alias":
        assert record["aliases"]["live"]["Description"] == "winner"
    elif first == "code":
        assert record["code_zip"] == b"winner"
        assert record["config"]["Description"] == "original"
    else:
        assert record["config"]["Description"] == "winner"
        assert record["code_zip"] == b"original-code"


def test_slow_code_update_does_not_block_independent_resources(functions, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    prototype = copy.deepcopy(functions["revision-test"])

    def fetch(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return b"new-code"

    monkeypatch.setattr(svc, "_fetch_code_from_s3", fetch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        slow = pool.submit(svc._update_code, "revision-test", {
            "RevisionId": "function-before", "S3Bucket": "unused", "S3Key": "unused"})
        try:
            assert entered.wait(5)
            assert pool.submit(_update, "config", name="unrelated").result(timeout=2)[0] == 200
            assert pool.submit(_update, "alias", "alias-before").result(timeout=2)[0] == 200
            for account, region in (("222222222222", "us-east-1"), ("000000000000", "eu-west-1")):
                def other_tenant():
                    with request_scope(account, region):
                        functions["revision-test"] = copy.deepcopy(prototype)
                        return _update("config")
                assert pool.submit(other_tenant).result(timeout=2)[0] == 200
        finally:
            release.set()
        assert slow.result(timeout=5)[0] == 200


@pytest.mark.parametrize("operation", ["code", "config", "alias"])
def test_missing_resource_keeps_not_found_response(functions, operation):
    assert _update(operation, "stale", name="missing")[0] == 404
    assert svc._update_alias("revision-test", "missing", {"RevisionId": "stale"})[0] == 404
