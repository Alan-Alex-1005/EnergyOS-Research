from src.agents.runner import run_all_agents
from src.analysis_pipeline import run_day2_analysis
from src.evaluation.coverage import compute_explanation_coverage
from src.evaluation.dimensions import applicable_dimensions, DIMENSIONS
from src.evaluation.diversity import compute_evidence_diversity
from src.evaluation.levels import level1_isolated_analytics, level2_independent_agents, level3_collaborative_framework
from src.evaluation.recommendation_support import CONTEXTUAL, GENERIC, IMPACT_AWARE, NONE_, classify_recommendation_support
from src.orchestration.orchestrator import run_orchestration
from tests.conftest import fixture_path


def test_applicable_dimensions_excludes_schedule_always():
    applicable = applicable_dimensions(True, True)
    assert "schedule_evidence" not in applicable


def test_applicable_dimensions_excludes_cost_carbon_when_not_configured():
    applicable = applicable_dimensions(False, False)
    assert "cost_impact" not in applicable
    assert "carbon_impact" not in applicable
    assert len(applicable) == len(DIMENSIONS) - 3  # schedule + cost + carbon


def test_applicable_dimensions_same_across_scenario_regardless_of_level():
    a1 = applicable_dimensions(True, True)
    a2 = applicable_dimensions(True, True)
    assert a1 == a2


def test_level1_no_agent_dimensions_ever_true(tmp_path):
    run = run_day2_analysis(fixture_path("wide_energy.csv"), output_root=str(tmp_path), generate_figures=False)
    dims = level1_isolated_analytics(run)
    assert dims["equipment_context_evidence"] is False
    assert dims["cost_impact"] is False
    assert dims["carbon_impact"] is False
    assert dims["cross_agent_evidence_correlation"] is False
    assert dims["agreement_assessment"] is False
    assert dims["unified_explanation"] is False
    assert dims["forecast_context"] is True
    assert dims["abnormal_behavior_detection"] is True


def test_level2_has_equipment_but_no_collaboration(tmp_path):
    run = run_day2_analysis(fixture_path("wide_energy.csv"), output_root=str(tmp_path), generate_figures=False)
    agents = run_all_agents(run, None)
    dims = level2_independent_agents(run, agents)
    assert dims["equipment_context_evidence"] is True
    assert dims["cost_impact"] is False
    assert dims["cross_agent_evidence_correlation"] is False
    assert dims["agreement_assessment"] is False
    assert dims["unified_explanation"] is False


def test_level2_has_cost_carbon_when_configured(tmp_path):
    run = run_day2_analysis(fixture_path("wide_energy.csv"), output_root=str(tmp_path), generate_figures=False)
    agents = run_all_agents(run, {"impact": {"tariff_per_kwh": 0.2, "carbon_factor_kg_per_kwh": 0.4}})
    dims = level2_independent_agents(run, agents)
    assert dims["cost_impact"] is True
    assert dims["carbon_impact"] is True


def test_level3_adds_collaboration_on_top_of_level2(tmp_path):
    run = run_day2_analysis(fixture_path("wide_energy.csv"), output_root=str(tmp_path), generate_figures=False)
    collab = run_orchestration(run, None)
    dims2 = level2_independent_agents(run, collab.agents)
    dims3 = level3_collaborative_framework(run, collab)
    for d in dims2:
        if dims2[d] is True:
            assert dims3[d] is True
    assert dims3["cross_agent_evidence_correlation"] is True
    assert dims3["agreement_assessment"] is True
    assert dims3["evidence_completeness_assessment"] is True
    assert dims3["decision_support_confidence"] is True
    assert dims3["unified_explanation"] is True


def test_level3_never_fabricates_schedule_or_cost_without_config(tmp_path):
    run = run_day2_analysis(fixture_path("wide_energy.csv"), output_root=str(tmp_path), generate_figures=False)
    collab = run_orchestration(run, None)
    dims3 = level3_collaborative_framework(run, collab)
    assert dims3["cost_impact"] is False
    assert dims3["carbon_impact"] is False


def test_coverage_full_support():
    applicable = ["a", "b", "c"]
    support = {"a": True, "b": True, "c": True}
    result = compute_explanation_coverage(support, applicable)
    assert result.numerator == 3
    assert result.denominator == 3
    assert result.percentage == 100.0


def test_coverage_partial_support():
    applicable = ["a", "b", "c", "d"]
    support = {"a": True, "b": False, "c": True, "d": False}
    result = compute_explanation_coverage(support, applicable)
    assert result.numerator == 2
    assert result.denominator == 4
    assert result.percentage == 50.0


def test_coverage_denominator_identical_across_levels_same_scenario(tmp_path):
    run = run_day2_analysis(fixture_path("wide_energy.csv"), output_root=str(tmp_path), generate_figures=False)
    agents = run_all_agents(run, None)
    collab = run_orchestration(run, None)
    applicable = applicable_dimensions(False, False)

    c1 = compute_explanation_coverage(level1_isolated_analytics(run), applicable)
    c2 = compute_explanation_coverage(level2_independent_agents(run, agents), applicable)
    c3 = compute_explanation_coverage(level3_collaborative_framework(run, collab), applicable)
    assert c1.denominator == c2.denominator == c3.denominator


def test_diversity_counts_only_evidence_categories():
    applicable = ["forecast_context", "anomaly_event_information", "agreement_assessment"]
    support = {"forecast_context": True, "anomaly_event_information": True, "agreement_assessment": True}
    result = compute_evidence_diversity(support, applicable)
    assert result.count == 2
    assert set(result.categories) == {"forecast", "anomaly"}


def test_recommendation_support_none_for_empty():
    assert classify_recommendation_support([]) == NONE_
    assert classify_recommendation_support(None) == NONE_


def test_recommendation_support_generic_no_context():
    from src.orchestration.recommendations import Recommendation

    recs = [Recommendation(finding="f", possible_explanation=None, recommended_action="a", evidence_refs=["ANOMALY"])]
    assert classify_recommendation_support(recs) == GENERIC


def test_recommendation_support_contextual_with_equipment():
    from src.orchestration.recommendations import Recommendation

    recs = [Recommendation(finding="f", possible_explanation="h", recommended_action="a", evidence_refs=["ANOMALY", "EQUIPMENT_CONTEXT"])]
    assert classify_recommendation_support(recs) == CONTEXTUAL


def test_recommendation_support_impact_aware_with_cost():
    from src.orchestration.recommendations import Recommendation

    recs = [Recommendation(finding="f", possible_explanation=None, recommended_action="a", evidence_refs=["COST_IMPACT"])]
    assert classify_recommendation_support(recs) == IMPACT_AWARE


def test_recommendation_support_real_scenarios_not_gamed(tmp_path):
    run = run_day2_analysis(fixture_path("wide_energy.csv"), output_root=str(tmp_path), generate_figures=False)
    assert classify_recommendation_support(None) == NONE_
    collab = run_orchestration(run, None)
    level3_class = classify_recommendation_support(collab.recommendations)
    assert level3_class in (NONE_, GENERIC, CONTEXTUAL, IMPACT_AWARE)
