"""Real subprocess tests of the one-shot local Node executor, bypassing warm workers."""

import asyncio
import base64
import io
import json
import shutil
import uuid
import zipfile

import pytest

from ministack.services import lambda_svc as svc

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")


def _local_function(source, *, filename="index.js"):
    name = "local-node-logs-" + uuid.uuid4().hex
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr(filename, source)
    return {
        "config": {
            "FunctionName": name,
            "FunctionArn": svc._func_arn(name),
            "Runtime": "nodejs20.x",
            "Handler": "index.handler",
            "MemorySize": 128,
            "Timeout": 10,
        },
        "code_zip": archive.getvalue(),
    }


def _invoke_local(source, event=None, *, filename="index.js"):
    # Dispatch normally prefers a warm worker for Node. Call the one-shot
    # fallback directly so a working warm executor cannot hide this regression.
    return svc._execute_function_local(
        _local_function(source, filename=filename),
        event if event is not None else {},
        request_id=uuid.uuid4().hex,
    )


@pytest.mark.parametrize("filename,export", [("index.js", "exports.handler ="), ("index.mjs", "export const handler =")])
def test_local_stdout_uses_log_channel(filename, export):
    source = "console.log('module-initialization');\n" + export + """ async (event) => {
  console.log('log-message');
  console.info('info-message');
  console.debug('debug-message');
  console.warn('warn-message');
  console.error('error-message');
  process.stdout.write('stdout-message\\n');
  process.stderr.write('stderr-message\\n');
  await new Promise(resolve => setTimeout(resolve, 10));
  console.log('after-await-message');
  return {ok: true, event};
};
"""
    result = _invoke_local(source, {"value": 42}, filename=filename)

    assert not result.get("error")
    assert result["body"] == {"ok": True, "event": {"value": 42}}
    assert result["log"].splitlines() == [
        "module-initialization", "log-message", "info-message", "debug-message", "warn-message",
        "error-message", "stdout-message", "stderr-message", "after-await-message",
    ]


@pytest.mark.parametrize("filename,imports,export", [
    ("index.js", "const {writeSync, write} = require('fs');", "exports.handler ="),
    ("index.mjs", "import {writeSync, write} from 'node:fs';", "export const handler ="),
])
def test_local_fd_stdout_uses_log_channel(filename, imports, export):
    source = imports + "\n" + export + """ async () => {
  writeSync(1, 'fd-message\\n');
  writeSync(1, JSON.stringify({level: 30, msg: 'pino-message'}) + '\\n');
  await new Promise((resolve, reject) =>
    write(1, 'async-fd-message\\n', err => err ? reject(err) : resolve()));
  const fs = await import('node:fs');
  const fd = fs.openSync('fd-output.txt', 'w');
  try {
    writeSync(fd, 'sync-file');
    await new Promise((resolve, reject) =>
      write(fd, Buffer.from('-async-file'), err => err ? reject(err) : resolve()));
  } finally {
    fs.closeSync(fd);
  }
  return {ok: true, file: fs.readFileSync('fd-output.txt', 'utf8')};
};
"""
    result = _invoke_local(source, filename=filename)

    assert not result.get("error")
    assert result["body"] == {"ok": True, "file": "sync-file-async-file"}
    messages = result["log"].splitlines()
    assert messages[0] == "fd-message"
    assert json.loads(messages[1]) == {"level": 30, "msg": "pino-message"}
    assert messages[2] == "async-fd-message"
    assert len(messages) == 3


def test_local_logs_reach_invoke_tail(monkeypatch):
    function = _local_function("""
exports.handler = async () => {
  console.log('invoke-console-message');
  console.error('invoke-stderr-message');
  return {ok: true};
};
""")
    name = function["config"]["FunctionName"]
    monkeypatch.setitem(svc._functions, name, function)
    monkeypatch.setattr(svc, "LAMBDA_EXECUTOR", "local")
    monkeypatch.setattr(svc, "LAMBDA_STRICT", False)
    monkeypatch.setattr(svc, "_execute_function_warm", svc._execute_function_local)
    monkeypatch.setattr(svc, "_emit_lambda_logs", lambda *args, **kwargs: None)
    monkeypatch.setattr(svc, "_emit_lambda_metrics", lambda *args, **kwargs: None)

    status, headers, payload = asyncio.run(svc._invoke(name, {}, {"x-amz-log-type": "Tail"}, None))

    assert status == 200
    assert "X-Amz-Function-Error" not in headers
    assert json.loads(payload) == {"ok": True}
    expected_logs = ["invoke-console-message", "invoke-stderr-message"]
    assert base64.b64decode(headers["X-Amz-Log-Result"]).decode().splitlines() == expected_logs
