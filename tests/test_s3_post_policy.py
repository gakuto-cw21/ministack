"""Local S3 POST contract tests using boto3 forms and registered IAM credentials.

Expected conditions come from the S3 POST policy documentation, not a live AWS run.
"""
import asyncio
import time
import uuid
from xml.etree import ElementTree

import boto3
import pytest
import requests
from botocore.config import Config
from conftest import inprocess_iam_user, routed_request

from ministack.core.responses import AccountScopedDict, request_scope
from ministack.services import iam, s3, sts


@pytest.fixture
def post_form(monkeypatch):
    import ministack.app as app_mod

    monkeypatch.setattr(app_mod, "AUTH", False)
    monkeypatch.setattr(s3, "_buckets", AccountScopedDict())
    monkeypatch.setattr(s3, "S3_PERSIST", False)
    with request_scope("000000000000", "us-east-1"):
        bucket = "post-policy-" + uuid.uuid4().hex[:12]
        key, _arn = inprocess_iam_user()
        s3._create_bucket(bucket, b"", {})

        class Form:
            def generate(self, *, fields=None, conditions=None, key_name="uploads/data.txt", expires=300,
                         secret=None, token=None, access_key=None, signature_version="s3v4"):
                client = boto3.client(
                    "s3", endpoint_url="http://localhost", region_name="us-east-1",
                    aws_access_key_id=access_key or key[0], aws_secret_access_key=secret or key[1],
                    aws_session_token=token, config=Config(signature_version=signature_version),
                )
                return client.generate_presigned_post(
                    Bucket=bucket, Key=key_name, Fields=fields or {},
                    Conditions=conditions or [], ExpiresIn=expires,
                )["fields"]

            def submit(self, fields, *, data=b"contents", filename="data.txt", routed=False):
                req = requests.Request(
                    "POST", "http://localhost/" + bucket,
                    data=fields, files={"file": (filename, data)},
                ).prepare()
                headers = {name.lower(): value for name, value in req.headers.items()}
                if routed:
                    return routed_request(None, "s3", "POST", "/" + bucket,
                                          body=req.body, headers=headers)
                return asyncio.run(s3.handle_request("POST", "/" + bucket, headers, req.body, {}))

            def assert_rejected(self, fields, code, **kwargs):
                response = self.submit(fields, **kwargs)
                assert response[0] in (400, 403)
                assert ElementTree.fromstring(response[2]).findtext("Code") == code
                assert s3._buckets[bucket]["objects"] == {}

        form = Form()
        form.bucket, form.key = bucket, key
        yield form
        iam._access_keys.pop(key[0], None)
        name = _arn.rsplit("/", 1)[-1]
        iam._users.pop(name, None)
        iam._user_inline_policies.pop(name, None)


@pytest.mark.parametrize("signature_version", ["s3v4", "s3"])
def test_signed_post_stores_bytes_and_metadata(post_form, signature_version):
    fields = post_form.generate(fields={"x-amz-meta-author": "alice", "success_action_status": "201"},
                                conditions=[{"x-amz-meta-author": "alice"}, {"success_action_status": "201"}],
                                signature_version=signature_version)
    assert post_form.submit(fields)[0] == 201
    obj = s3._buckets[post_form.bucket]["objects"]["uploads/data.txt"]
    assert obj["body"] == b"contents"
    assert obj["metadata"] == {"x-amz-meta-author": "alice"}


@pytest.mark.parametrize("name,value", [("key", "uploads/other.txt"), ("Content-Type", "text/plain"),
                                        ("x-amz-meta-author", "bob"), ("success_action_status", "204")])
def test_post_exact_conditions_reject_mismatched_fields(post_form, name, value):
    fields = {"Content-Type": "application/json", "x-amz-meta-author": "alice", "success_action_status": "201"}
    form = post_form.generate(fields=fields, conditions=[{k: v} for k, v in fields.items()])
    form[name] = value
    post_form.assert_rejected(form, "AccessDenied")


@pytest.mark.parametrize("signature_version", ["s3v4", "s3"])
@pytest.mark.parametrize("auth_enabled", [False, True])
def test_post_signature_verified_in_both_auth_modes(post_form, monkeypatch, auth_enabled, signature_version):
    import ministack.app as app_mod

    monkeypatch.setattr(app_mod, "AUTH", auth_enabled)
    form = post_form.generate(secret=uuid.uuid4().hex, signature_version=signature_version)
    post_form.assert_rejected(form, "SignatureDoesNotMatch", routed=True)


@pytest.mark.parametrize("missing", ["x-amz-signature", "x-amz-date", "x-amz-credential", "policy"])
def test_incomplete_sigv4_form_rejected(post_form, missing):
    form = post_form.generate()
    del form[missing]
    post_form.assert_rejected(form, "InvalidArgument")


def test_expired_post_policy_rejected(post_form):
    post_form.assert_rejected(post_form.generate(expires=-60), "AccessDenied")


@pytest.mark.parametrize("content_type,accepted", [("image/png,image/jpeg", True), ("image/png,text/plain", False)])
def test_post_content_type_prefix_applies_to_each_list_item(post_form, content_type, accepted):
    form = post_form.generate(fields={"Content-Type": content_type},
                              conditions=[["starts-with", "$Content-Type", "image/"]])
    if accepted:
        assert post_form.submit(form)[0] == 204
    else:
        post_form.assert_rejected(form, "AccessDenied")


def test_post_variables_expanded_before_matching(post_form):
    form = post_form.generate(key_name="uploads/${filename}", fields={"x-amz-meta-name": "${filename}"},
                              conditions=[["eq", "$x-amz-meta-name", "data.txt"]])
    assert post_form.submit(form)[0] == 204
    obj = s3._buckets[post_form.bucket]["objects"]["uploads/data.txt"]
    assert obj["metadata"]["x-amz-meta-name"] == "data.txt"


def test_post_variables_expanded_once(post_form):
    form = post_form.generate(key_name="uploads/${filename}")
    assert post_form.submit(form, filename="${filename}.txt")[0] == 204
    assert list(s3._buckets[post_form.bucket]["objects"]) == ["uploads/${filename}.txt"]


@pytest.mark.parametrize("extra", ["x-amz-meta-extra", "acl"])
def test_post_uncovered_form_fields_rejected(post_form, extra):
    form = post_form.generate(fields={extra: "private"})
    post_form.assert_rejected(form, "AccessDenied")


def test_post_missing_condition_field_rejected(post_form):
    form = post_form.generate(conditions=[["starts-with", "$x-amz-meta-required", ""]])
    post_form.assert_rejected(form, "AccessDenied")


def test_post_ignore_fields_exempt_from_conditions(post_form):
    assert post_form.submit(post_form.generate(fields={"x-ignore-client": "local"}))[0] == 204


@pytest.mark.parametrize("data,code", [(b"a", "EntityTooSmall"), (b"1234", "EntityTooLarge"), (b"12", None), (b"123", None)])
def test_post_size_range_inclusive(post_form, data, code):
    form = post_form.generate(conditions=[["content-length-range", 2, 3]])
    if code:
        post_form.assert_rejected(form, code, data=data)
    else:
        assert post_form.submit(form, data=data)[0] == 204


@pytest.mark.parametrize("low,code", [("2", None), ("x", "AccessDenied")])
def test_post_size_range_string_bounds(post_form, low, code):
    form = post_form.generate(conditions=[["content-length-range", low, "3"]])
    if code:
        post_form.assert_rejected(form, code, data=b"12")
    else:
        assert post_form.submit(form, data=b"12")[0] == 204


def test_post_modified_policy_fails_signature(post_form):
    form = post_form.generate()
    form["policy"] = form["policy"][:-2]
    post_form.assert_rejected(form, "SignatureDoesNotMatch")


def test_post_invalid_condition_rejected(post_form):
    post_form.assert_rejected(post_form.generate(conditions=[["unknown", "$key", ""]]), "InvalidPolicyDocument")


def test_post_bucket_condition_uses_destination(post_form):
    post_form.assert_rejected(post_form.generate(conditions=[{"bucket": "different-bucket"}]), "AccessDenied")


def test_post_inactive_registered_key_rejected(post_form):
    form = post_form.generate()
    iam._access_keys[post_form.key[0]]["Status"] = "Inactive"
    post_form.assert_rejected(form, "InvalidAccessKeyId")


def test_post_unexpected_session_token_rejected(post_form):
    post_form.assert_rejected(post_form.generate(token=uuid.uuid4().hex), "InvalidToken")


def test_post_registered_key_resolves_nondefault_account(post_form):
    with request_scope("123456789012", "us-east-1"):
        key, arn = inprocess_iam_user()
        try:
            bucket = "tenant-post-" + uuid.uuid4().hex[:12]
            s3._create_bucket(bucket, b"", {})
            client = boto3.client("s3", endpoint_url="http://localhost", region_name="us-east-1",
                                  aws_access_key_id=key[0], aws_secret_access_key=key[1],
                                  config=Config(signature_version="s3v4"))
            form = client.generate_presigned_post(Bucket=bucket, Key="uploads/data.txt")["fields"]
            # submit() captures the fixture bucket, so route this request explicitly.
            req = requests.Request("POST", "http://localhost/" + bucket, data=form,
                                   files={"file": ("data.txt", b"tenant")}).prepare()
            result = asyncio.run(s3.handle_request("POST", "/" + bucket,
                {k.lower(): v for k, v in req.headers.items()}, req.body, {}))
            assert result[0] == 204
            assert s3._buckets[bucket]["objects"]["uploads/data.txt"]["body"] == b"tenant"
        finally:
            iam._access_keys.pop(key[0], None)
            iam._users.pop(arn.rsplit("/", 1)[-1], None)


@pytest.mark.parametrize("signature", ["", "not-hex", "é" * 64])
def test_post_malformed_signature_is_a_client_error(post_form, signature):
    form = post_form.generate()
    form["x-amz-signature"] = signature
    post_form.assert_rejected(form, "InvalidArgument" if not signature else "SignatureDoesNotMatch")


@pytest.mark.parametrize("token_state", ["valid", "missing", "incorrect", "expired"])
def test_post_uses_sts_issued_credentials(post_form, token_state):
    response = routed_request(post_form.key, "sts", "POST", "/",
        body=b"Action=GetSessionToken&DurationSeconds=900",
        headers={"content-type": "application/x-www-form-urlencoded"})
    assert response[0] == 200
    root = ElementTree.fromstring(response[2])
    credentials = {name: root.findtext(".//{*}" + name) for name in ("AccessKeyId", "SecretAccessKey", "SessionToken")}
    access_key = credentials["AccessKeyId"]
    try:
        token = credentials["SessionToken"]
        if token_state == "missing":
            token = None
        elif token_state == "incorrect":
            token = uuid.uuid4().hex
        elif token_state == "expired":
            sts._sessions[access_key]["Expiration"] = time.time() - 60
        form = post_form.generate(access_key=access_key, secret=credentials["SecretAccessKey"], token=token)
        if token_state == "valid":
            assert post_form.submit(form)[0] == 204
        else:
            post_form.assert_rejected(form, "ExpiredToken" if token_state == "expired" else "InvalidToken")
    finally:
        sts._sessions.pop(access_key, None)


def test_post_configured_root_credentials(post_form, monkeypatch):
    key, secret = "local-root-" + uuid.uuid4().hex, uuid.uuid4().hex
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", key)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", secret)
    assert post_form.submit(post_form.generate(access_key=key, secret=secret))[0] == 204


def test_rejected_post_preserves_existing_object_and_disk(post_form, monkeypatch, tmp_path):
    monkeypatch.setattr(s3, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(s3, "S3_PERSIST", True)
    events = []
    monkeypatch.setattr(s3, "_fire_s3_event_async", lambda *a, **kw: events.append(kw))
    assert post_form.submit(post_form.generate(), data=b"original")[0] == 204
    files = {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert files
    invalid = post_form.generate(secret=uuid.uuid4().hex)
    assert post_form.submit(invalid, data=b"replacement")[0] == 403
    assert {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == files
    assert len(events) == 1
    obj = s3._buckets[post_form.bucket]["objects"]["uploads/data.txt"]
    assert s3._read_body(post_form.bucket, "uploads/data.txt", obj) == b"original"
    s3._buckets.clear()
    s3._load_persisted_data()
    obj = s3._buckets[post_form.bucket]["objects"]["uploads/data.txt"]
    assert s3._read_body(post_form.bucket, "uploads/data.txt", obj) == b"original"


def test_post_after_persisted_reload(post_form, monkeypatch, tmp_path):
    monkeypatch.setattr(s3, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(s3, "S3_PERSIST", True)
    assert post_form.submit(post_form.generate(), data=b"first")[0] == 204
    s3._buckets.clear()
    s3._load_persisted_data()
    assert post_form.submit(post_form.generate(), data=b"second")[0] == 204
