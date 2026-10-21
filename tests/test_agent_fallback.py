import backend.agents.orchestrator as orchestrator
import backend.agents.orchestrator_rule as rule_agent


def test_run_agent_falls_back_to_rule_agent(monkeypatch):
    def fail(coro):
        coro.close()
        raise RuntimeError("llm down")

    monkeypatch.setattr(orchestrator, "run_coroutine", fail)
    monkeypatch.setattr(
        rule_agent,
        "run_agent_rule_based",
        lambda question, city="beijing", **kwargs: {
            "answer": "fallback answer",
            "tool_calls": [],
            "tool_results": [],
            "city": city,
            "references": [],
            "refused": False,
        },
    )
    result = orchestrator.run_agent("北京现在空气质量", "beijing")
    assert result["answer"] == "fallback answer"
    assert result["mode"] == "rule-fallback"
    assert "llm down" in result["agent_error"]
