from backend.agents.orchestrator_rule import rule_based_plan, rule_based_synthesize


def test_two_city_comparison_phrasing_routes_to_comparison():
    plan = rule_based_plan("广州和深圳哪个空气质量更好", "beijing")
    assert plan[0]["tool"] == "get_cities_comparison"


def test_get_alerts_answer_is_not_blank():
    text = rule_based_synthesize(
        "北京当前告警",
        [
            {
                "tool": "get_alerts",
                "result": [
                    {
                        "station": "北京市",
                        "metric": "PM25",
                        "value": 80,
                        "threshold": 75,
                        "level": "warning",
                    }
                ],
            }
        ],
        "beijing",
    )
    assert "PM25" in text and "80" in text


def test_train_models_answer_is_not_blank():
    text = rule_based_synthesize(
        "训练模型",
        [{"tool": "train_models", "result": {"models": {"PM25": {"mae": 1, "rmse": 2, "r2": 0.9}}}}],
        "beijing",
    )
    assert "PM25" in text and "R2" in text


def test_season_and_correlation_include_city():
    season = rule_based_synthesize(
        "北京季节分析",
        [{"tool": "get_season_analysis", "result": [{"season": "春", "avg_pm25": 30}]}],
        "beijing",
    )
    assert "beijing" in season
    corr = rule_based_synthesize(
        "北京相关性",
        [{"tool": "get_correlation", "result": [{"indicator": "PM25", "temperature": 0.2}]}],
        "beijing",
    )
    assert "beijing" in corr
