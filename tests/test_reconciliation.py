import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "adr2_agent_action.py"


def load_module(monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        types.SimpleNamespace(Anthropic=object, APIError=Exception, APIConnectionError=ConnectionError),
    )
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=object, OpenAIError=Exception))
    monkeypatch.setenv("ADR2_REPO_ROOT", str(tmp_path))
    spec = importlib.util.spec_from_file_location("adr2_agent_action", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, "adr2_agent_action", module)
    spec.loader.exec_module(module)
    return module


def test_claude_retries_connection_errors(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path)
    calls = 0
    sleeps = []

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get_final_message(self):
            return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text="ok")])

    class Messages:
        def stream(self, **kwargs):
            nonlocal calls
            calls += 1
            if calls < 3:
                raise ConnectionError("temporary")
            return Stream()

    monkeypatch.setattr(
        module,
        "get_anthropic_client",
        lambda: types.SimpleNamespace(messages=Messages()),
    )
    monkeypatch.setattr(module.time, "sleep", sleeps.append)

    result = module.call_claude_text(model="claude", messages=[{"role": "user", "content": "hi"}])

    assert result == "ok"
    assert calls == 3
    assert sleeps == [1, 2]


def test_anthropic_api_key_is_trimmed(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path)
    received = {}
    client = object()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key\n")
    monkeypatch.setattr(
        module.anthropic,
        "Anthropic",
        lambda **kwargs: received.update(kwargs) or client,
    )

    assert module.get_anthropic_client() is client
    assert received == {"api_key": "test-key"}


def write_adr(path, **overrides):
    meta = {
        "id": "ADR-0001",
        "title": "Existing",
        "scope": "architecture",
        "domain": "device",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "decision": "기존 결정을 유지한다.",
        "validation_rules": ["기존 규칙을 지킨다."],
        "agent_playbook": ["기존 경계를 확인하라."],
        "index_terms": ["existing"],
        "owns": [{"type": "contract", "key": "device.settings"}],
        "contracts": [{"id": "device.settings.v1", "role": "producer"}],
        "applies_to": {"paths": ["apps/device/**"], "symbols": ["DeviceSettings"]},
        "relations": {"related": [], "depends_on": [], "supersedes": []},
    }
    meta.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n" + module_yaml(meta) + "---\n\n## Context (for humans)\n기존 맥락\n",
        encoding="utf-8",
    )


def module_yaml(meta):
    import yaml

    return yaml.safe_dump(meta, allow_unicode=True, sort_keys=False)


def test_reconciliation_decision_supports_all_actions(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path)

    for action in ("covered", "amend", "create", "reject", "defer"):
        normalized = module.normalize_reconciliation_decision({"action": action})
        assert normalized["action"] == action

    assert module.normalize_reconciliation_decision({"action": "unknown"})["action"] == "defer"


def test_reconciliation_shortlist_is_bounded_and_never_empty_when_catalog_exists(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path)
    catalog = [{"id": f"ADR-{number:04d}", "title": f"Decision {number}"} for number in range(1, 13)]

    selected = module.shortlist_existing_adrs("전혀 다른 요청", catalog)

    assert [item["id"] for item in selected] == [f"ADR-{number:04d}" for number in range(1, 6)]


def test_amend_preserves_existing_rules_and_applies_explicit_replacement(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path)
    path = tmp_path / "docs" / "adr" / "ADR-0001-existing.md"
    write_adr(path)

    module.amend_existing_adr(
        path,
        {
            "decision_addition": "새 예외도 함께 보존한다.",
            "add_validation_rules": ["새 규칙을 지킨다."],
            "replace_validation_rules": [
                {"existing": "기존 규칙을 지킨다.", "replacement": "기존 규칙과 예외를 지킨다."}
            ],
            "add_agent_playbook": ["새 예외를 확인하라."],
            "add_index_terms": ["exception"],
        },
    )

    meta, _ = module.parse_front_matter(path)
    assert meta["decision"] == "기존 결정을 유지한다. 새 예외도 함께 보존한다."
    assert meta["validation_rules"] == ["기존 규칙과 예외를 지킨다.", "새 규칙을 지킨다."]
    assert meta["agent_playbook"] == ["기존 경계를 확인하라.", "새 예외를 확인하라."]
    assert meta["index_terms"] == ["existing", "exception"]
    assert meta["created_at"] == "2026-01-01T00:00:00Z"


def test_index_payload_contains_retrieval_metadata_and_is_deterministic(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path)
    path = tmp_path / "docs" / "adr" / "ADR-0001-existing.md"
    write_adr(path)
    context = module.resolve_docs_contexts()[0]
    catalog = module.catalog_existing_adrs(context)

    first = module.build_index_payload(catalog)
    second = module.build_index_payload(catalog)

    assert first == second
    assert first["schema_version"] == 2
    assert first["source_hash"]
    assert first["generated_at"] == "2026-01-01T00:00:00Z"
    assert first["items"][0]["owns"] == [{"type": "contract", "key": "device.settings"}]
    assert first["items"][0]["validation_rules"] == ["기존 규칙을 지킨다."]
    assert first["items"][0]["agent_playbook"] == ["기존 경계를 확인하라."]


def test_index_operation_does_not_require_an_llm_key(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path)
    context = module.resolve_docs_contexts()[0]
    write_adr(context.adr_dir / "ADR-0001-existing.md")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ADR2_OPERATION", "index")

    module.main()

    written = json.loads(context.index_path.read_text(encoding="utf-8"))
    assert written["schema_version"] == 2


@pytest.mark.parametrize(
    ("owns", "created"),
    [
        ([{"type": "contract", "key": "device.settings"}], False),
        ([{"type": "contract", "key": "device.other"}], True),
        ([], False),
    ],
)
def test_create_requires_independent_ownership(monkeypatch, tmp_path, owns, created):
    module = load_module(monkeypatch, tmp_path)
    context = module.resolve_docs_contexts()[0]
    existing_path = context.adr_dir / "ADR-0001-existing.md"
    write_adr(existing_path)
    catalog = module.catalog_existing_adrs(context)
    monkeypatch.setattr(
        module,
        "generate_adr_payload",
        lambda *args, **kwargs: {
            "title": "Candidate",
            "scope": "architecture",
            "decision": "독립 후보",
            "validation_rules": [],
            "agent_playbook": [],
            "index_terms": [],
            "owns": owns,
            "contracts": [],
            "applies_to": {},
        },
    )

    result = module.create_adr_from_aar({}, context, "AAR", catalog, [], "architecture")

    assert (result is not None) is created


def test_consolidation_preserves_unique_rules_and_supersedes_duplicate(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path)
    context = module.resolve_docs_contexts()[0]
    canonical = context.adr_dir / "ADR-0001-existing.md"
    duplicate = context.adr_dir / "ADR-0002-duplicate.md"
    write_adr(canonical)
    write_adr(
        duplicate,
        id="ADR-0002",
        title="Duplicate",
        created_at="2026-02-01T00:00:00Z",
        decision="중복 문서의 추가 결정을 유지한다.",
        validation_rules=["중복 문서의 고유 규칙을 지킨다."],
        agent_playbook=["중복 경계를 확인하라."],
        index_terms=["duplicate"],
    )

    module.consolidate_adr_files(context, canonical, [duplicate])

    meta, _ = module.parse_front_matter(canonical)
    assert meta["validation_rules"] == ["기존 규칙을 지킨다.", "중복 문서의 고유 규칙을 지킨다."]
    assert meta["decision"] == "기존 결정을 유지한다. 중복 문서의 추가 결정을 유지한다."
    assert meta["relations"]["supersedes"] == ["ADR-0002"]
    assert not duplicate.exists()
    assert (context.adr_dir / "superseded" / duplicate.name).exists()


@pytest.mark.parametrize(
    ("action", "aar_remains", "rule_added", "new_adr_count"),
    [
        ("covered", False, False, 1),
        ("reject", False, False, 1),
        ("defer", True, False, 1),
        ("amend", False, True, 1),
        ("create", False, False, 2),
    ],
)
def test_main_reconciles_aar_without_always_creating_a_new_adr(
    monkeypatch, tmp_path, action, aar_remains, rule_added, new_adr_count
):
    module = load_module(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ADR2_OPERATION", "reconcile")
    context = module.resolve_docs_contexts()[0]
    existing_path = context.adr_dir / "ADR-0001-existing.md"
    write_adr(existing_path)
    aar_path = context.aar_dir / "change.md"
    aar_path.parent.mkdir(parents=True, exist_ok=True)
    aar_path.write_text("device.settings 계약에 예외 규칙을 추가한다.", encoding="utf-8")
    (tmp_path / "README.md").write_text("ADR2 instructions", encoding="utf-8")

    def fake_call(system_prompt, user_content, model=None, *, instructions=None):
        if "RECONCILIATION_ACTION" in system_prompt:
            return {
                "action": action,
                "target_adr_id": "ADR-0001" if action in {"covered", "amend"} else None,
                "decision_addition": "새 예외를 보존한다.",
                "add_validation_rules": ["새 예외 규칙을 지킨다."],
            }
        if "owns is an array" in system_prompt:
            return {
                "title": "Independent decision",
                "scope": "architecture",
                "decision": "독립 결정을 유지한다.",
                "domain": "device",
                "index_terms": ["independent"],
                "owns": [{"type": "contract", "key": "device.other"}],
            }
        return {"rules": []}

    monkeypatch.setattr(module, "call_openai_json_object", fake_call)

    module.main()

    assert aar_path.exists() is aar_remains
    assert len(list(context.adr_dir.glob("ADR-*.md"))) == new_adr_count
    existing, _ = module.parse_front_matter(existing_path)
    assert ("새 예외 규칙을 지킨다." in existing["validation_rules"]) is rule_added


def test_covered_keeps_aar_when_target_does_not_exist(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    context = module.resolve_docs_contexts()[0]
    aar_path = context.aar_dir / "change.md"
    aar_path.parent.mkdir(parents=True, exist_ok=True)
    aar_path.write_text("결정", encoding="utf-8")
    (tmp_path / "README.md").write_text("ADR2 instructions", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "reconciliation_decision",
        lambda *args: {"action": "covered", "target_adr_id": "ADR-9999"},
    )

    module.main()

    assert aar_path.exists()


def test_consolidate_operation_backfills_ownership_and_merges_judged_duplicates(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ADR2_OPERATION", "consolidate")
    report_path = tmp_path / "report.md"
    monkeypatch.setenv("ADR2_PR_BODY_PATH", str(report_path))
    context = module.resolve_docs_contexts()[0]
    first = context.adr_dir / "ADR-0001-existing.md"
    second = context.adr_dir / "ADR-0002-duplicate.md"
    write_adr(first, owns=[])
    write_adr(second, id="ADR-0002", title="Duplicate", owns=[], validation_rules=["고유 규칙"])
    (tmp_path / "README.md").write_text("ADR2 instructions", encoding="utf-8")

    def fake_call(system_prompt, user_content, model=None, *, instructions=None):
        if "OWNERSHIP_METADATA" in system_prompt:
            return {
                "owns": [{"type": "contract", "key": "device.settings"}],
                "contracts": [{"id": "device.settings.v1", "role": "producer"}],
                "applies_to": {"paths": ["apps/device/**"], "symbols": []},
            }
        if "CONSOLIDATION_JUDGE" in system_prompt:
            return {
                "merge": True,
                "same_authoritative_boundary": True,
                "same_validation_responsibility": True,
                "independent_lifecycle": False,
                "independent_rollback": False,
            }
        raise AssertionError(system_prompt)

    monkeypatch.setattr(module, "call_openai_json_object", fake_call)

    module.main()

    assert first.exists()
    assert not second.exists()
    assert (context.adr_dir / "superseded" / second.name).exists()
    meta, _ = module.parse_front_matter(first)
    assert meta["owns"] == [{"type": "contract", "key": "device.settings"}]
    assert "고유 규칙" in meta["validation_rules"]
    report = report_path.read_text(encoding="utf-8")
    assert "canonical: ADR-0001" in report
    assert "retired: ADR-0002" in report
    assert "보존 규칙" in report


def test_consolidate_reclassifies_ownership_when_independent_documents_collide(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path)
    context = module.resolve_docs_contexts()[0]
    first = context.adr_dir / "ADR-0001-existing.md"
    second = context.adr_dir / "ADR-0002-independent.md"
    write_adr(first, contracts=[{"id": "device.shared", "role": "producer"}])
    write_adr(second, id="ADR-0002", title="Independent", contracts=[{"id": "device.shared", "role": "producer"}])
    catalog = module.catalog_existing_adrs(context)
    monkeypatch.setattr(
        module,
        "call_openai_json_object",
        lambda *args, **kwargs: {
            "merge": False,
            "same_authoritative_boundary": False,
            "same_validation_responsibility": False,
            "independent_lifecycle": True,
            "independent_rollback": True,
            "ownership_updates": [
                {
                    "adr_id": "ADR-0001",
                    "owns": [{"type": "runtime", "key": "device.read"}],
                    "contracts": [{"id": "device.read", "role": "producer"}],
                },
                {
                    "adr_id": "ADR-0002",
                    "owns": [{"type": "runtime", "key": "device.write"}],
                    "contracts": [{"id": "device.write", "role": "producer"}],
                },
            ],
        },
    )

    changed = module.consolidate_context({"consolidate": "prompt"}, context, catalog, [])

    assert changed
    assert module.validate_catalog(module.catalog_existing_adrs(context), require_ownership=True) == []


def test_backfill_uses_valid_proposed_domain_when_terms_do_not_match(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path)
    context = module.resolve_docs_contexts()[0]
    path = context.adr_dir / "ADR-0001-existing.md"
    write_adr(path, domain="unclassified", owns=[])
    catalog = module.catalog_existing_adrs(context)
    domains = [module.Domain("forms-ux", "Forms", ("form",), None)]
    monkeypatch.setattr(
        module,
        "call_openai_json_object",
        lambda *args, **kwargs: {
            "domain": "forms-ux",
            "owns": [{"type": "runtime", "key": "image.compression"}],
            "contracts": [],
            "applies_to": {"paths": [], "symbols": []},
        },
    )

    module.backfill_ownership({"ownership": "prompt"}, catalog, domains)

    meta, _ = module.parse_front_matter(path)
    assert meta["domain"] == "forms-ux"


def test_catalog_validation_rejects_missing_and_duplicate_ownership(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path)
    catalog = [
        {"id": "ADR-0001", "domain": "device", "owns": []},
        {"id": "ADR-0002", "domain": "unclassified", "owns": [{"type": "contract", "key": "same"}]},
        {"id": "ADR-0003", "domain": "device", "owns": [{"type": "contract", "key": "same"}]},
    ]

    errors = module.validate_catalog(catalog, require_ownership=True)

    assert any("ADR-0001" in error and "ownership" in error for error in errors)
    assert any("ADR-0002" in error and "domain" in error for error in errors)
    assert any("same" in error and "ADR-0002" in error and "ADR-0003" in error for error in errors)


def test_catalog_validation_rejects_duplicate_contract_producer_across_apps(monkeypatch, tmp_path):
    module = load_module(monkeypatch, tmp_path)
    errors = module.validate_contract_producers(
        [
            [{"id": "backend/ADR-0001", "contracts": [{"id": "device.v1", "role": "producer"}]}],
            [{"id": "frontend/ADR-0002", "contracts": [{"id": "device.v1", "role": "producer"}]}],
        ]
    )

    assert errors == ["contract producer device.v1 is duplicated by backend/ADR-0001 and frontend/ADR-0002"]


def prepare_domain_reconciliation(monkeypatch, tmp_path, *, proposed=None, terms=(), taxonomy=True, required=True):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ADR2_OPERATION", "reconcile")
    monkeypatch.setenv("ADR2_REQUIRE_OWNERSHIP", str(required).lower())
    monkeypatch.delenv("ADR2_DOCS_DIRS", raising=False)
    module = load_module(monkeypatch, tmp_path)
    context = module.resolve_docs_contexts()[0]
    existing = context.adr_dir / "ADR-0159-existing.md"
    write_adr(existing, id="ADR-0159")
    if taxonomy:
        context.domains_path.write_text(module_yaml({"domains": [
            {"key": "device", "match_terms": ["device"]},
            {"key": "address-management", "match_terms": list(terms)},
        ]}), encoding="utf-8")
    aar = context.aar_dir / "address-group-optional-address-fields.md"
    aar.parent.mkdir(parents=True)
    aar.write_text("주소 그룹의 자동 보완과 수동 입력 저장", encoding="utf-8")
    payload = {
        "title": "주소 그룹의 자동 보완과 수동 입력 저장",
        "decision": "주소 그룹은 geocoding과 수동 입력의 저장 책임을 구분한다.",
        "index_terms": ["주소 그룹", "geocoding"],
        "owns": [{"type": "component", "key": "address_group"}],
    }
    if proposed is not None:
        payload["domain"] = proposed

    def fake_call(system_prompt, user_content, model=None, *, instructions=None):
        if system_prompt == "candidate":
            return {"action": "create"}
        # Exercise the real generator defaults, resolver, writer and final validation.
        return dict(payload)

    monkeypatch.setattr(module, "load_prompts", lambda: {"adr2": "instructions", "candidate": "candidate"})
    monkeypatch.setattr(module, "call_openai_json_object", fake_call)
    return module, context, aar, payload


def docs_snapshot(context):
    return {str(path.relative_to(context.docs_dir)): path.read_bytes()
            for path in context.docs_dir.rglob("*") if path.is_file()}


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("proposed", [None, "", "unclassified", "not-registered"])
def test_unresolved_domain_defers_before_any_document_change(monkeypatch, tmp_path, capsys, required, proposed):
    module, context, aar, _ = prepare_domain_reconciliation(
        monkeypatch, tmp_path, proposed=proposed, required=required,
    )
    before = docs_snapshot(context)

    module.main()
    module.main()

    assert docs_snapshot(context) == before
    assert aar.exists()
    assert "Deferred create because no registered domain could be resolved" in capsys.readouterr().out


@pytest.mark.parametrize("proposed,terms", [
    ("address-management", ()),
    (None, ("주소 그룹", "geocoding")),
    ("not-registered", ("주소 그룹", "geocoding")),
])
def test_resolved_domain_creates_valid_adr_once(monkeypatch, tmp_path, proposed, terms):
    module, context, aar, _ = prepare_domain_reconciliation(
        monkeypatch, tmp_path, proposed=proposed, terms=terms,
    )

    module.main()

    assert not aar.exists()
    created = list(context.adr_dir.glob("ADR-0160-*.md"))
    assert len(created) == 1
    meta, _ = module.parse_front_matter(created[0])
    assert meta["domain"] == "address-management"
    assert module.validate_catalog(module.catalog_existing_adrs(context), require_ownership=True) == []
    before_retry = docs_snapshot(context)
    module.main()
    assert docs_snapshot(context) == before_retry


def test_deferred_aar_can_be_retried_after_taxonomy_is_completed(monkeypatch, tmp_path):
    module, context, aar, _ = prepare_domain_reconciliation(monkeypatch, tmp_path)
    before = docs_snapshot(context)
    module.main()
    assert docs_snapshot(context) == before

    context.domains_path.write_text(module_yaml({"domains": [
        {"key": "device", "match_terms": ["device"]},
        {"key": "address-management", "match_terms": ["주소 그룹", "geocoding"]},
    ]}), encoding="utf-8")
    module.main()

    assert not aar.exists()
    created = list(context.adr_dir.glob("ADR-0160-*.md"))
    assert len(created) == 1
    assert module.parse_front_matter(created[0])[0]["domain"] == "address-management"


def test_domain_fallback_uses_canonical_index_terms(monkeypatch, tmp_path):
    module, context, aar, payload = prepare_domain_reconciliation(
        monkeypatch, tmp_path, terms=("geocoding",),
    )
    write_adr(context.adr_dir / "ADR-0159-existing.md", id="ADR-0159", index_terms=["geocoding"])
    payload.update(title="새 경계", decision="저장 값을 검증한다.", index_terms=["geo-coding"])

    module.main()

    assert not aar.exists()
    meta, _ = module.parse_front_matter(next(context.adr_dir.glob("ADR-0160-*.md")))
    assert meta["domain"] == "address-management"
    assert meta["index_terms"] == ["geocoding"]


@pytest.mark.parametrize("required", [False, True])
def test_missing_taxonomy_only_allows_creation_when_domain_is_optional(monkeypatch, tmp_path, required):
    module, context, aar, _ = prepare_domain_reconciliation(
        monkeypatch, tmp_path, proposed="address-management", taxonomy=False, required=required,
    )
    before = docs_snapshot(context)
    module.main()
    if required:
        assert docs_snapshot(context) == before
    else:
        assert not aar.exists()
        meta, _ = module.parse_front_matter(next(context.adr_dir.glob("ADR-0160-*.md")))
        assert "domain" not in meta


def test_invalid_catalog_candidate_keeps_aar_and_existing_adr(monkeypatch, tmp_path):
    module, context, aar, payload = prepare_domain_reconciliation(
        monkeypatch, tmp_path, proposed="address-management",
    )
    payload["contracts"] = [{"id": "device.settings.v1", "role": "producer"}]
    before = docs_snapshot(context)

    module.main()

    assert docs_snapshot(context) == before
    assert aar.exists()


@pytest.mark.parametrize("taxonomy", ["domains: []\n", "domains: invalid\n", "domains: [\n"])
def test_unusable_taxonomy_does_not_disable_classification(monkeypatch, tmp_path, taxonomy):
    module, context, aar, _ = prepare_domain_reconciliation(monkeypatch, tmp_path, required=False)
    context.domains_path.write_text(taxonomy, encoding="utf-8")
    before = docs_snapshot(context)

    module.main()

    assert docs_snapshot(context) == before
    assert aar.exists()


@pytest.mark.parametrize("failure", ["generation", "partial_write", "publish"])
def test_create_failure_preserves_aar_without_partial_adr_and_retry_succeeds(monkeypatch, tmp_path, failure):
    module, context, aar, _ = prepare_domain_reconciliation(
        monkeypatch, tmp_path, proposed="address-management",
    )
    before = docs_snapshot(context)

    def fail(*args, **kwargs):
        raise OSError("injected failure")

    def partial_write(path, content):
        path.write_text(content[:20], encoding="utf-8")
        fail()

    with monkeypatch.context() as patch:
        if failure == "generation":
            patch.setattr(module, "generate_adr_payload", fail)
        elif failure == "partial_write":
            patch.setattr(module, "write_file", partial_write)
        else:
            patch.setattr(Path, "replace", fail)
        with pytest.raises(OSError, match="injected failure"):
            module.main()

    assert docs_snapshot(context) == before
    assert not list(context.adr_dir.glob(".adr-*"))
    assert aar.exists()
    module.main()
    assert not aar.exists()
    assert len(list(context.adr_dir.glob("ADR-0160-*.md"))) == 1


def test_aar_removal_failure_leaves_complete_adr_and_can_be_reconciled_on_retry(monkeypatch, tmp_path):
    module, context, aar, _ = prepare_domain_reconciliation(
        monkeypatch, tmp_path, proposed="address-management",
    )
    original_aar = aar.read_bytes()
    unlink = Path.unlink

    def fail_aar_removal(path, *args, **kwargs):
        if path == aar:
            raise OSError("AAR removal failed")
        return unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_aar_removal)
        with pytest.raises(OSError, match="AAR removal failed"):
            module.main()

    assert aar.read_bytes() == original_aar
    catalog = module.catalog_existing_adrs(context)
    assert module.validate_catalog(catalog, require_ownership=True) == []
    generated = next(context.adr_dir.glob("ADR-0160-*.md"))
    original_adr = generated.read_bytes()
    # Even another create decision cannot duplicate the already published owner.
    module.main()
    assert aar.exists()
    monkeypatch.setattr(module, "reconciliation_decision", lambda *args: {
        "action": "covered", "target_adr_id": "ADR-0160",
    })
    module.main()
    assert not aar.exists()
    assert generated.read_bytes() == original_adr
    assert len(list(context.adr_dir.glob("ADR-0160-*.md"))) == 1
