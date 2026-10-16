"""AI analyst loop with a scripted provider: tool use, grounding, citations and scope."""

from loan_approval.ai import analyst
from loan_approval.ai.providers import ProviderError, ScriptedProvider


def _tool_call(name, args=None, call_id="t1"):
    return {"type": "tool_use", "id": call_id, "name": name, "input": args or {}}


def test_grounded_answer_with_citations(trained_env):
    from loan_approval.ai.tools import get_run_metrics

    ev = get_run_metrics()
    sel = ev["data"]["selected_algorithm"]
    auc = ev["data"]["test_split"][sel]["probability_metrics"]["roc_auc"]
    provider = ScriptedProvider(
        [
            {
                "content": [_tool_call("get_run_metrics")],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 10},
            },
            {
                "content": [
                    _tool_call(
                        "submit_answer",
                        {
                            "answer": f"The selected model has a test ROC-AUC of {auc:.3f}.",
                            "evidence_ids": [ev["evidence_id"]],
                            "measured_findings": ["ROC-AUC measured on the test split"],
                            "hypotheses": [],
                        },
                        "t2",
                    )
                ],
                "stop_reason": "tool_use",
            },
        ]
    )
    out = analyst.ask("How good is the model?", provider=provider)
    assert out["status"] == "ok"
    assert out["grounding"]["grounded"], out["grounding"]
    assert out["tools_used"] == ["get_run_metrics"]
    # The tool result is sent back as wrapped, untrusted data
    tool_msg = provider.calls[1]["messages"][-1]["content"][0]
    assert tool_msg["type"] == "tool_result" and "untrusted_data_label" in tool_msg["content"]


def test_fabricated_numbers_are_flagged(trained_env):
    provider = ScriptedProvider(
        [
            {"content": [_tool_call("get_model_card")], "stop_reason": "tool_use"},
            {
                "content": [
                    _tool_call(
                        "submit_answer", {"answer": "Accuracy is 99.1%.", "evidence_ids": ["made_up"]}, "t2"
                    )
                ]
            },
        ]
    )
    out = analyst.ask("How accurate is it?", provider=provider)
    assert not out["grounding"]["grounded"]
    assert "99.1%" in out["grounding"]["unverified_numbers"]
    assert out["grounding"]["invalid_citations"] == ["made_up"]


def test_disallowed_tool_is_reported_as_error(trained_env):
    provider = ScriptedProvider(
        [
            {"content": [_tool_call("delete_everything")], "stop_reason": "tool_use"},
            {"content": [{"type": "text", "text": "I could not do that."}]},
        ]
    )
    out = analyst.ask("Run a forbidden tool", provider=provider)
    assert out["status"] == "ok"
    sent = provider.calls[1]["messages"][-1]["content"][0]
    assert sent["is_error"] is True


def test_scope_refusal_skips_provider(trained_env):
    provider = ScriptedProvider([])
    out = analyst.ask("Should I approve my neighbour's mortgage?", provider=provider)
    assert out["status"] == "refused"
    assert provider.calls == []


def test_answer_drifting_into_advice_is_replaced(trained_env):
    provider = ScriptedProvider(
        [
            {
                "content": [
                    _tool_call(
                        "submit_answer",
                        {"answer": "You should approve this loan for your client.", "evidence_ids": []},
                    )
                ]
            }
        ]
    )
    out = analyst.ask("Explain the model", provider=provider)
    assert "can't help with real lending decisions" in out["answer"]


def test_provider_failure_is_graceful(trained_env):
    class Broken:
        name = "broken"

        def complete(self, *a, **k):
            raise ProviderError("timeout")

    out = analyst.ask("Explain calibration", provider=Broken())
    assert out["status"] == "error"
    assert "still work" in out["answer"]


def test_quick_answers_work_without_llm(trained_env):
    for kind in analyst.SUGGESTED_QUESTIONS:
        qa = analyst.quick_answer(kind)
        assert qa["answer"] and qa["evidence"]
