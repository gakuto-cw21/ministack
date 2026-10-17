# Copyright (c) 2026 MiniStack Contributors. SPDX-License-Identifier: MIT
"""Terraform removes queue policies with explicit empty attribute values."""

import asyncio
import copy
import json
import xml.etree.ElementTree as ET
from urllib.parse import urlencode

import pytest

from ministack.core.responses import AccountRegionScopedDict
from ministack.services import sqs


@pytest.fixture
def queue(monkeypatch):
    monkeypatch.setattr(sqs, "_queues", AccountRegionScopedDict())
    monkeypatch.setattr(sqs, "_queue_name_to_url", AccountRegionScopedDict())
    url = sqs._act_create_queue({"QueueName": "policy-removal"}, "")["QueueUrl"]
    dlq = sqs._act_create_queue({"QueueName": "policy-removal-dlq"}, "")["QueueUrl"]
    arn = sqs._act_get_queue_attributes({"QueueUrl": dlq}, "")["Attributes"]["QueueArn"]
    policies = {
        "Policy": json.dumps({"Version": "2012-10-17", "Statement": []}),
        "RedrivePolicy": json.dumps({"deadLetterTargetArn": arn, "maxReceiveCount": "3"}),
        "RedriveAllowPolicy": json.dumps({"redrivePermission": "denyAll"}),
    }
    sqs._act_set_queue_attributes({"QueueUrl": url, "Attributes": policies}, "")
    return url, policies


def _request(protocol, action, url, attributes=None, names=None):
    if protocol == "json":
        data = {"QueueUrl": url}
        if attributes is not None:
            data["Attributes"] = attributes
        if names is not None:
            data["AttributeNames"] = names
        headers = {"x-amz-target": f"AmazonSQS.{action}"}
        body = json.dumps(data).encode()
    else:
        data = {"Action": action, "QueueUrl": url}
        for i, (key, value) in enumerate((attributes or {}).items(), 1):
            data[f"Attribute.{i}.Name"] = key
            data[f"Attribute.{i}.Value"] = value
        for i, name in enumerate(names or [], 1):
            data[f"AttributeName.{i}"] = name
        headers = {"content-type": "application/x-www-form-urlencoded"}
        body = urlencode(data).encode()
    return asyncio.run(sqs.handle_request("POST", "/", headers, body, {}))


def _read(protocol, url, names):
    status, _, body = _request(protocol, "GetQueueAttributes", url, names=names)
    assert status == 200
    if protocol == "json":
        return json.loads(body)["Attributes"]
    ns = "{http://queue.amazonaws.com/doc/2012-11-05/}"
    return {entry.findtext(f"{ns}Name"): entry.findtext(f"{ns}Value")
            for entry in ET.fromstring(body).iter(f"{ns}Attribute")}


@pytest.mark.parametrize("protocol", ["json", "query"])
@pytest.mark.parametrize("attribute", ["Policy", "RedrivePolicy", "RedriveAllowPolicy"])
def test_empty_policy_removes_attribute_from_all_and_specific_reads(queue, protocol, attribute):
    url, policies = queue
    assert _read(protocol, url, [attribute]) == {attribute: policies[attribute]}
    for _ in range(2):
        status, _, body = _request(protocol, "SetQueueAttributes", url, {attribute: ""})
        assert status == 200, body
        assert _read(protocol, url, [attribute]) == {}
        remaining = _read(protocol, url, ["All"])
        assert attribute not in remaining
        for other in policies.keys() - {attribute}:
            assert remaining[other] == policies[other]
        assert remaining["VisibilityTimeout"] == "30"


@pytest.mark.parametrize("invalid", [{"RedrivePolicy": "not-json"},
                                   {"RedrivePolicy": '"double-encoded"'},
                                   {"VisibilityTimeout": "-1"}])
def test_invalid_update_preserves_all_attributes(queue, invalid):
    url, _ = queue
    before = copy.deepcopy(sqs._queues[url])
    status, _, body = _request("json", "SetQueueAttributes", url, {"Policy": "", **invalid})
    assert status == 400
    assert json.loads(body)["__type"] == "InvalidAttributeValue"
    assert sqs._queues[url] == before


def test_removed_policies_remain_absent_after_persistence_round_trip(queue):
    url, policies = queue
    assert _request("json", "SetQueueAttributes", url, dict.fromkeys(policies, ""))[0] == 200
    state = sqs.get_state()
    sqs._queues.clear()
    sqs._queue_name_to_url.clear()
    sqs.load_persisted_state(state)
    assert policies.keys().isdisjoint(_read("json", url, ["All"]))
    assert _read("json", url, ["QueueArn"])["QueueArn"].endswith(":policy-removal")
