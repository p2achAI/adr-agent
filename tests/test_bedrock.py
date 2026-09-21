import json
import types

import pytest

from test_reconciliation import load_module


@pytest.mark.parametrize("input_region,aws_region,default_region,expected", [
    ("eu-west-1", "us-east-1", "us-west-2", "eu-west-1"),
    ("", "us-east-1", "us-west-2", "us-east-1"),
    (" ", "", "us-west-2", "us-west-2"),
    ("", "", "", ""),
])
def test_bedrock_region_precedence(monkeypatch, tmp_path, input_region, aws_region, default_region, expected):
    module = load_module(monkeypatch, tmp_path)
    for name, value in zip(("BEDROCK_AWS_REGION", "AWS_REGION", "AWS_DEFAULT_REGION"),
                           (input_region, aws_region, default_region)):
        monkeypatch.setenv(name, value)
    assert module.bedrock_region() == expected


@pytest.mark.parametrize("model,region,error", [
    ("", "us-east-1", "BEDROCK_MODEL is required"),
    ("profile", "", "AWS region is required"),
])
def test_bedrock_validates_configuration(monkeypatch, tmp_path, model, region, error):
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    monkeypatch.setenv("BEDROCK_MODEL", model)
    monkeypatch.setenv("BEDROCK_AWS_REGION", region)
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.setenv("ADR2_OPERATION", "reconcile")
    module = load_module(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match=error):
        module.main()


def test_index_does_not_require_bedrock_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    monkeypatch.setenv("ADR2_OPERATION", "index")
    monkeypatch.delenv("BEDROCK_MODEL", raising=False)
    module = load_module(monkeypatch, tmp_path)
    module.main()


def test_unknown_provider_does_not_fall_back_to_openai(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "typo")
    module = load_module(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
        module.call_llm_text(model="model", messages=[])


def setup_bedrock(monkeypatch, tmp_path, result=None, status_code=200):
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    monkeypatch.setenv("BEDROCK_MODEL", "global.openai.gpt-5.6-terra")
    monkeypatch.setenv("BEDROCK_AWS_REGION", "ap-northeast-2")
    monkeypatch.setenv("BEDROCK_REASONING_EFFORT", "medium")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "test-session-token")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    module = load_module(monkeypatch, tmp_path)
    requests = []
    closed = []

    class Transport:
        def __init__(self, **kwargs):
            pass

        def send(self, request):
            requests.append(request)
            return types.SimpleNamespace(status_code=status_code, content=json.dumps(result).encode())

        def close(self):
            closed.append(True)

    monkeypatch.setattr(module, "URLLib3Session", Transport)
    return module, requests, closed


def test_bedrock_signed_responses_request(monkeypatch, tmp_path):
    result = {"status": "completed", "output": [
        {"type": "reasoning", "summary": []},
        {"type": "message", "content": [{"type": "output_text", "text": '{"ok":true}'}]},
    ]}
    module, requests, closed = setup_bedrock(monkeypatch, tmp_path, result)
    assert module.DEFAULT_MODEL == "global.openai.gpt-5.6-terra"
    assert module.PLANNER_MODEL == module.DEFAULT_MODEL
    text = module.call_llm_text(
        model=module.DEFAULT_MODEL,
        messages=[{"role": "user", "content": "hello"}],
        instructions="instructions",
        response_format={"type": "json_object"},
    )
    assert text == '{"ok":true}'
    assert len(requests) == 1
    request = requests[0]
    assert request.url == "https://bedrock-runtime.ap-northeast-2.amazonaws.com/openai/v1/responses"
    assert request.headers["Authorization"].startswith("AWS4-HMAC-SHA256 ")
    assert request.headers["X-Amz-Security-Token"] == "test-session-token"
    assert json.loads(request.body) == {
        "model": "global.openai.gpt-5.6-terra", "input": [{"role": "user", "content": "hello"}],
        "reasoning": {"effort": "medium"}, "store": False, "instructions": "instructions",
        "text": {"format": {"type": "json_object"}},
    }
    assert closed == [True]


@pytest.mark.parametrize("result,status_code,error", [
    ({"secret": "repository content"}, 403, "HTTP 403"),
    ({"status": "incomplete", "output": []}, 200, "did not complete"),
    ({"status": "completed", "output": []}, 200, "no output text"),
])
def test_bedrock_errors_do_not_return_partial_output(monkeypatch, tmp_path, result, status_code, error):
    module, _, closed = setup_bedrock(monkeypatch, tmp_path, result, status_code)
    with pytest.raises(RuntimeError, match=error) as caught:
        module.call_llm_text(model=module.DEFAULT_MODEL, messages=[])
    assert "repository content" not in str(caught.value)
    assert closed == [True]


def test_bedrock_invalid_effort_fails_before_request(monkeypatch, tmp_path):
    module, requests, _ = setup_bedrock(monkeypatch, tmp_path)
    monkeypatch.setenv("BEDROCK_REASONING_EFFORT", "typo")
    with pytest.raises(ValueError, match="BEDROCK_REASONING_EFFORT"):
        module.call_llm_text(model=module.DEFAULT_MODEL, messages=[])
    assert not requests


def test_bedrock_missing_credentials_does_not_fall_back(monkeypatch, tmp_path):
    module, requests, _ = setup_bedrock(monkeypatch, tmp_path)
    monkeypatch.setattr(module.boto3, "Session", lambda: types.SimpleNamespace(get_credentials=lambda: None))
    with pytest.raises(RuntimeError, match="AWS credentials are required"):
        module.call_llm_text(model=module.DEFAULT_MODEL, messages=[])
    assert not requests
