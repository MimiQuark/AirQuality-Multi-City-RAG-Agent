from backend.tools.implementations import is_environment_query, tool_query_policy


def test_non_environment_policy_query_is_refused():
    result = tool_query_policy("个人所得税专项附加扣除标准是多少")
    assert result["refused"] is True
    assert result["results"] == []


def test_environment_query_passes_domain_gate():
    assert is_environment_query("PM2.5 国家标准限值是多少")
    assert is_environment_query("环境空气质量标准")
