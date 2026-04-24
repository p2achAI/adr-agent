import importlib.util
import sys
import types
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "adr2_agent_action.py"


def load_module(monkeypatch, tmp_path, docs_dirs=None):
    anthropic_stub = types.SimpleNamespace(Anthropic=object, APIError=Exception)
    openai_stub = types.SimpleNamespace(OpenAI=object, OpenAIError=Exception)
    monkeypatch.setitem(sys.modules, "anthropic", anthropic_stub)
    monkeypatch.setitem(sys.modules, "openai", openai_stub)
    monkeypatch.setenv("ADR2_REPO_ROOT", str(tmp_path))
    if docs_dirs is None:
        monkeypatch.delenv("ADR2_DOCS_DIRS", raising=False)
    else:
        monkeypatch.setenv("ADR2_DOCS_DIRS", docs_dirs)

    spec = importlib.util.spec_from_file_location("adr2_agent_action", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, "adr2_agent_action", module)
    spec.loader.exec_module(module)
    return module


def test_default_docs_context_points_to_root_docs(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path)

    contexts = module.resolve_docs_contexts()

    assert len(contexts) == 1
    assert contexts[0].docs_dir == tmp_path / "docs"
    assert contexts[0].aar_dir == tmp_path / "docs" / "aar"
    assert contexts[0].adr_dir == tmp_path / "docs" / "adr"
    assert contexts[0].index_path == tmp_path / "docs" / "adr" / "index.json"


def test_docs_dirs_contexts_accept_newline_and_comma_separated_relative_paths(monkeypatch, tmp_path):
    module = load_module(
        monkeypatch,
        tmp_path,
        "apps/backend/docs\napps/frontend/docs, apps/cms-mqtt-api/docs",
    )

    contexts = module.resolve_docs_contexts()

    assert [context.docs_dir for context in contexts] == [
        tmp_path / "apps" / "backend" / "docs",
        tmp_path / "apps" / "frontend" / "docs",
        tmp_path / "apps" / "cms-mqtt-api" / "docs",
    ]
    assert contexts[0].aar_dir == tmp_path / "apps" / "backend" / "docs" / "aar"
    assert contexts[1].adr_dir == tmp_path / "apps" / "frontend" / "docs" / "adr"


def test_docs_dirs_contexts_accept_absolute_paths(monkeypatch, tmp_path):
    external_docs = tmp_path.parent / "external-docs"
    module = load_module(monkeypatch, tmp_path, str(external_docs))

    contexts = module.resolve_docs_contexts()

    assert contexts[0].docs_dir == external_docs.resolve()
    assert contexts[0].aar_dir == external_docs.resolve() / "aar"
    assert module.display_path(external_docs.resolve()) == str(external_docs.resolve())


def test_candidate_decision_requires_boolean_true(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path)

    is_candidate, scope = module.normalize_candidate_decision(
        {"isCandidate": "false", "decisionScope": "api-contract"}
    )

    assert is_candidate is False
    assert scope == "api-contract"


def test_candidate_decision_rejects_missing_scope(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path)

    is_candidate, scope = module.normalize_candidate_decision({"isCandidate": True})

    assert is_candidate is False
    assert scope == ""


def test_candidate_decision_rejects_minor_change(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path)

    is_candidate, scope = module.normalize_candidate_decision(
        {"isCandidate": True, "decisionScope": "minor-change"}
    )

    assert is_candidate is False
    assert scope == "minor-change"


def test_candidate_decision_accepts_valid_candidate(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path)

    is_candidate, scope = module.normalize_candidate_decision(
        {"isCandidate": True, "decisionScope": "api-contract"}
    )

    assert is_candidate is True
    assert scope == "api-contract"
