"""In-process regressions for FilterLogEvents ordering across streams."""

import json
import time
import uuid

import pytest

from ministack.services import cloudwatch_logs as logs


@pytest.fixture
def log_group():
    group = f"/test/filter-pattern/{uuid.uuid4().hex}"
    assert logs._create_log_group({"logGroupName": group})[0] == 200
    yield group
    assert logs._delete_log_group({"logGroupName": group})[0] == 200


def put_messages(group, stream, messages):
    assert logs._create_log_stream({
        "logGroupName": group, "logStreamName": stream,
    })[0] == 200
    response = logs._put_log_events({
        "logGroupName": group,
        "logStreamName": stream,
        "logEvents": [{"timestamp": timestamp, "message": message}
                      for timestamp, message in messages],
    })
    assert response[0] == 200


def test_filter_applies_limit_after_global_stream_order(log_group):
    now = int(time.time() * 1000)
    # The stream scanned first has newer events than the second stream.
    put_messages(log_group, "newer", [(now + 30, "match newer")])
    put_messages(log_group, "older", [
        (now, "unrelated"), (now + 10, "match first"), (now + 20, "match second"),
    ])
    response = logs._filter_log_events({
        "logGroupName": log_group, "filterPattern": "match", "limit": 1,
    })
    assert response[0] == 200
    assert [e["message"] for e in json.loads(response[2])["events"]] == ["match first"]


def test_filter_does_not_stop_later_stream_after_partial_global_limit(log_group):
    now = int(time.time() * 1000)
    put_messages(log_group, "newer", [(now + 30, "match newer")])
    put_messages(log_group, "older", [(now + 10, "match first"), (now + 20, "match second")])
    response = logs._filter_log_events({
        "logGroupName": log_group, "filterPattern": "match", "limit": 2,
    })
    assert response[0] == 200
    assert [e["message"] for e in json.loads(response[2])["events"]] == [
        "match first", "match second",
    ]


def filter_events(group, **parameters):
    response = logs._filter_log_events({"logGroupName": group, **parameters})
    assert response[0] == 200
    return json.loads(response[2])["events"]


def count_filter_calls(monkeypatch):
    compile_pattern = logs._compile_filter_pattern
    checked = []

    def compile_counted(pattern):
        matcher = compile_pattern(pattern)

        def matches(message):
            checked.append(message)
            return matcher(message)

        return matches

    monkeypatch.setattr(logs, "_compile_filter_pattern", compile_counted)
    return checked


def test_limit_stops_filtering_after_oldest_matches(log_group, monkeypatch):
    now = int(time.time() * 1000)
    put_messages(log_group, "a", [(now + i, f"match {i}") for i in range(2, 202, 2)])
    put_messages(log_group, "b", [(now + i, f"match {i}") for i in range(1, 202, 2)])
    # A later request can carry events older than those already stored.
    assert logs._put_log_events({
        "logGroupName": log_group, "logStreamName": "a",
        "logEvents": [{"timestamp": now, "message": "skip"}],
    })[0] == 200
    checked = count_filter_calls(monkeypatch)
    events = filter_events(log_group, filterPattern="match", limit=2)
    assert [event["message"] for event in events] == ["match 1", "match 2"]
    assert checked == ["skip", "match 1", "match 2"]


def test_time_window_seeks_without_scanning_history(log_group):
    now = int(time.time() * 1000)
    put_messages(log_group, "history", [(now + i, f"event {i}") for i in range(8192)])
    timestamp_reads = []

    class CountedEvent(dict):
        def __getitem__(self, key):
            if key == "timestamp":
                timestamp_reads.append(None)
            return super().__getitem__(key)

    stream = logs._log_groups[log_group]["streams"]["history"]
    stream["events"] = [CountedEvent(event) for event in stream["events"]]
    events = filter_events(log_group, startTime=now + 8000, endTime=now + 8001, limit=10)
    assert [event["message"] for event in events] == ["event 8000", "event 8001"]
    # Allow headroom for binary searches and merge setup, but no linear scan.
    assert len(timestamp_reads) < 100


def test_late_batch_keeps_reads_ordered_and_live_delivery_exact(log_group, monkeypatch):
    now = int(time.time() * 1000)
    put_messages(log_group, "s", [(now + 30, "newer")])
    deliveries = []
    monkeypatch.setattr(logs, "_fanout_to_live_tail_sessions",
                        lambda group, stream, events: deliveries.extend(events))
    assert logs._put_log_events({
        "logGroupName": log_group, "logStreamName": "s",
        "logEvents": [
            {"timestamp": now + 10, "message": "older"},
            {"timestamp": now + 20, "message": "middle"},
        ],
    })[0] == 200
    response = logs._get_log_events({
        "logGroupName": log_group, "logStreamName": "s", "startFromHead": True,
    })
    assert response[0] == 200
    assert [event["message"] for event in json.loads(response[2])["events"]] == [
        "older", "middle", "newer",
    ]
    assert [event["message"] for event in deliveries] == ["older", "middle"]
    stored = filter_events(log_group)
    assert [event["eventId"] for event in stored[:2]] == [event["eventId"] for event in deliveries]


@pytest.mark.parametrize("stream_order", [["a", "b"], ["b", "a"]])
def test_equal_timestamps_keep_selected_stream_and_arrival_order(log_group, stream_order):
    now = int(time.time() * 1000)
    for stream in ("a", "b"):
        put_messages(log_group, stream, [(now, f"{stream} first")])
        assert logs._put_log_events({
            "logGroupName": log_group, "logStreamName": stream,
            "logEvents": [{"timestamp": now, "message": f"{stream} second"}],
        })[0] == 200
    events = filter_events(log_group, logStreamNames=stream_order, limit=3)
    first, second = stream_order
    assert [event["message"] for event in events] == [
        f"{first} first", f"{first} second", f"{second} first",
    ]


def test_legacy_persistence_restores_order_before_limited_reads(log_group, monkeypatch, tmp_path):
    from ministack.core import persistence

    now = int(time.time() * 1000)
    put_messages(log_group, "s", [(now + i, f"event {i}") for i in range(20)])
    state = logs.get_state()
    state["log_groups"][log_group]["streams"]["s"]["events"].reverse()
    monkeypatch.setattr(persistence, "PERSIST_STATE", True)
    monkeypatch.setattr(persistence, "STATE_DIR", str(tmp_path))
    persistence.save_state("cloudwatch_logs", state)
    assert logs._delete_log_group({"logGroupName": log_group})[0] == 200
    logs.load_persisted_state(persistence.load_state("cloudwatch_logs"))
    checked = count_filter_calls(monkeypatch)
    events = filter_events(log_group, limit=1)
    assert [event["message"] for event in events] == ["event 0"]
    assert checked == ["event 0"]


@pytest.mark.parametrize("producer", ["codebuild", "dynamodb"])
def test_native_writers_keep_order_after_future_dated_api_event(log_group, monkeypatch, producer):
    now = int(time.time() * 1000)
    if producer == "codebuild":
        from ministack.services import codebuild

        stream_name = "native"
        emit = codebuild._log_sink({"logs": {"groupName": log_group, "streamName": stream_name}})
        monkeypatch.setattr(codebuild.time, "time", lambda: now / 1000)
    else:
        from ministack.services import dynamodb

        monkeypatch.setattr(dynamodb, "_IMPORT_LOG_GROUP", log_group)
        monkeypatch.setattr(dynamodb.time, "time", lambda: now / 1000)
        import_id = uuid.uuid4().hex
        stream_name = f"{import_id}/error"

        def emit(message):
            dynamodb._log_import_error(f"arn:aws:dynamodb:us-east-1:000000000000:table/t/import/{import_id}",
                                      "bucket", "key", 0, message)

    emit("initial")
    assert logs._put_log_events({
        "logGroupName": log_group, "logStreamName": stream_name,
        "logEvents": [{"timestamp": now + 3000, "message": "future API event"}],
    })[0] == 200
    emit("later native event")
    events = filter_events(log_group, limit=2)
    assert len(events) == 2
    assert [event["timestamp"] for event in events] == [now, now]
    assert "initial" in events[0]["message"]
    assert "later native event" in events[1]["message"]


def test_reader_remains_consistent_during_late_arrival(log_group, monkeypatch):
    now = int(time.time() * 1000)
    put_messages(log_group, "s", [(now + 10, "ten"), (now + 20, "twenty")])
    inserted = False

    def matcher(message):
        nonlocal inserted
        if not inserted:
            inserted = True
            assert logs._put_log_events({
                "logGroupName": log_group, "logStreamName": "s",
                "logEvents": [{"timestamp": now, "message": "zero"}],
            })[0] == 200
        return True

    monkeypatch.setattr(logs, "_compile_filter_pattern", lambda pattern: matcher)
    assert [event["message"] for event in filter_events(log_group, limit=2)] == ["ten", "twenty"]
    assert [event["message"] for event in filter_events(log_group, limit=2)] == ["zero", "ten"]


def test_mixed_batches_and_ranges_match_stable_reference(log_group):
    import random

    randomizer = random.Random(731)
    now = int(time.time() * 1000)
    arrivals = {name: [] for name in ("a", "b", "c")}
    for name in arrivals:
        assert logs._create_log_stream({"logGroupName": log_group, "logStreamName": name})[0] == 200
    for batch in range(30):
        name = randomizer.choice(list(arrivals))
        records = [{"timestamp": now + randomizer.randrange(12),
                    "message": f"{'keep' if index % 2 else 'drop'} {batch}-{index}"}
                   for index in range(4)]
        arrivals[name].extend(records)
        assert logs._put_log_events({
            "logGroupName": log_group, "logStreamName": name, "logEvents": records,
        })[0] == 200
    for selection in (["a", "b", "c"], ["c", "a"], ["missing", "b"]):
        reference = sorted((dict(event, logStreamName=name)
                            for name in selection for event in arrivals.get(name, [])),
                           key=lambda event: event["timestamp"])
        for lower, upper in ((0, 11), (3, 7), (5, 5), (9, 2)):
            expected = [event for event in reference if now + lower <= event["timestamp"] <= now + upper
                        and "keep" in event["message"]]
            for limit in (1, 7, 10000):
                actual = filter_events(log_group, logStreamNames=selection, filterPattern="keep",
                                       startTime=now + lower, endTime=now + upper, limit=limit)
                assert [(event["timestamp"], event["message"], event["logStreamName"]) for event in actual] == [
                    (event["timestamp"], event["message"], event["logStreamName"])
                    for event in expected[:limit]
                ]


def test_concurrent_late_batches_preserve_every_event(log_group):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from ministack.core.responses import get_account_id, get_region, request_scope

    now = int(time.time() * 1000)
    put_messages(log_group, "shared", [])
    account, region = get_account_id(), get_region()
    barrier = Barrier(4)

    def write(worker):
        with request_scope(account, region):
            barrier.wait(timeout=5)
            for batch in reversed(range(8)):
                timestamp = now + batch * 4 + worker
                assert logs._put_log_events({
                    "logGroupName": log_group, "logStreamName": "shared",
                    "logEvents": [{"timestamp": timestamp, "message": f"event {timestamp}"}],
                })[0] == 200

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(write, worker) for worker in range(4)]
        for future in futures:
            future.result(timeout=10)
    events = filter_events(log_group, limit=100)
    assert [event["timestamp"] for event in events] == list(range(now, now + 32))
    stream = logs._log_groups[log_group]["streams"]["shared"]
    assert stream["firstEventTimestamp"] == now
    assert stream["lastEventTimestamp"] == now + 31


def get_page(group, stream, **parameters):
    response = logs._get_log_events({"logGroupName": group, "logStreamName": stream, **parameters})
    assert response[0] == 200
    body = json.loads(response[2])
    return [event["message"] for event in body["events"]], body


def test_forward_token_survives_late_arrivals(log_group):
    now = int(time.time() * 1000)
    put_messages(log_group, "s", [(now + 10, "e10"), (now + 20, "e20"), (now + 30, "e30")])
    messages, body = get_page(log_group, "s", startFromHead=True, limit=2)
    assert messages == ["e10", "e20"]
    # One event lands behind the cursor, one between the cursor and stored history.
    assert logs._put_log_events({
        "logGroupName": log_group, "logStreamName": "s",
        "logEvents": [{"timestamp": now + 15, "message": "e15"},
                      {"timestamp": now + 25, "message": "e25"}],
    })[0] == 200
    messages, _ = get_page(log_group, "s", nextToken=body["nextForwardToken"], limit=2)
    assert messages == ["e25", "e30"]


def test_backward_token_survives_late_arrivals(log_group):
    now = int(time.time() * 1000)
    put_messages(log_group, "s", [(now + 10, "e10"), (now + 20, "e20"), (now + 30, "e30")])
    messages, body = get_page(log_group, "s", limit=2)
    assert messages == ["e20", "e30"]
    # e15 is older than everything read so far; e25 is behind the backward cursor.
    assert logs._put_log_events({
        "logGroupName": log_group, "logStreamName": "s",
        "logEvents": [{"timestamp": now + 15, "message": "e15"},
                      {"timestamp": now + 25, "message": "e25"}],
    })[0] == 200
    messages, _ = get_page(log_group, "s", nextToken=body["nextBackwardToken"], limit=2)
    assert messages == ["e10", "e15"]


def test_backward_token_from_tail_excludes_later_events(log_group):
    now = int(time.time() * 1000)
    put_messages(log_group, "s", [(now + 10, "e10"), (now + 20, "e20")])
    _, body = get_page(log_group, "s", startFromHead=True)
    messages, body = get_page(log_group, "s", nextToken=body["nextForwardToken"])
    assert messages == []
    # e30 arrives after the reader reached the tail.
    assert logs._put_log_events({
        "logGroupName": log_group, "logStreamName": "s",
        "logEvents": [{"timestamp": now + 30, "message": "e30"}],
    })[0] == 200
    messages, _ = get_page(log_group, "s", nextToken=body["nextBackwardToken"])
    assert messages == ["e10", "e20"]


def test_forward_token_splits_equal_timestamps_exactly_once(log_group):
    now = int(time.time() * 1000)
    put_messages(log_group, "s", [(now, "a"), (now, "b"), (now, "c")])
    seen = []
    messages, body = get_page(log_group, "s", startFromHead=True, limit=1)
    seen += messages
    # An equal-time arrival sorts after existing ties, so it is still ahead of the cursor.
    assert logs._put_log_events({
        "logGroupName": log_group, "logStreamName": "s",
        "logEvents": [{"timestamp": now, "message": "d"}],
    })[0] == 200
    for _ in range(5):
        messages, body = get_page(log_group, "s", nextToken=body["nextForwardToken"], limit=1)
        if not messages:
            break
        seen += messages
    assert seen == ["a", "b", "c", "d"]
