from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent
from agents.prompt_builder import (
    PLANNER_SYSTEM_FALLBACK,
    join_sections,
    load_prompt_template,
    render_json_block,
)
from agents.prompt_guard import filter_large_payload
from agents.response_parser import parse_agent_response
from core.contracts import AnalysisPlan, ProfileSummary, SchemaSummary
from core.enums import AgentName, ProblemType, SplitStrategy
from core.executor import model_to_dict, to_problem_type
from core.workflow import build_canonical_plan_steps


def _resolve_problem_type(
    *,
    schema: SchemaSummary,
    profile: ProfileSummary,
    target_column: str | None,
) -> ProblemType:
    if not target_column:
        return ProblemType.UNSUPERVISED

    row_count = max(profile.row_count, 1)
    column_lookup = {column.name: column for column in schema.columns}
    target_schema = column_lookup.get(target_column)
    unique_count = profile.unique_counts.get(target_column, target_schema.unique_count if target_schema else None)
    unique_count = int(unique_count or 0)
    unique_ratio = unique_count / row_count if row_count else 0.0
    inferred_dtype = (target_schema.inferred_dtype if target_schema else "").lower()

    if "date" in inferred_dtype or "time" in inferred_dtype:
        return ProblemType.TIME_SERIES

    if target_column in schema.numeric_columns:
        if unique_count <= 12 and unique_ratio <= 0.20:
            return ProblemType.CLASSIFICATION
        return ProblemType.REGRESSION

    return ProblemType.CLASSIFICATION


def _resolve_split_strategy(problem_type: ProblemType) -> SplitStrategy:
    if problem_type == ProblemType.CLASSIFICATION:
        return SplitStrategy.STRATIFIED
    if problem_type in {ProblemType.REGRESSION, ProblemType.TIME_SERIES}:
        return SplitStrategy.STANDARD
    return SplitStrategy.NONE


def _build_chart_suggestions(schema: SchemaSummary, target_column: str | None) -> list[str]:
    suggestions: list[str] = []
    if schema.numeric_columns:
        suggestions.extend(
            [
                "Numeric distribution histograms for the top continuous variables.",
                "Correlation heatmap for the numeric feature set.",
            ]
        )
    if schema.categorical_columns:
        suggestions.append("Categorical cardinality and top-frequency bar charts.")
    if target_column:
        suggestions.append(f"Target-aware plots for relationships against '{target_column}'.")
    if schema.datetime_columns:
        suggestions.append("Datetime trend breakdown for date-like fields.")
    return suggestions[:6]


def _build_feature_actions(schema: SchemaSummary, target_column: str | None) -> list[str]:
    actions: list[str] = []
    if schema.numeric_columns:
        actions.append("Impute and scale numeric features through the baseline preprocessing pipeline.")
    if schema.categorical_columns:
        actions.append("Encode categorical variables with a low-risk categorical transformer.")
    if schema.datetime_columns:
        actions.append("Expand datetime columns into derived calendar features where useful.")
    if schema.text_columns:
        actions.append("Treat free-text columns cautiously and avoid heavy text modeling in the baseline pass.")
    if target_column:
        actions.append(f"Exclude '{target_column}' from feature inputs and verify leakage-safe feature lists.")
    return actions[:6]


def _build_risk_flags(
    schema: SchemaSummary,
    profile: ProfileSummary,
    problem_type: ProblemType,
) -> list[str]:
    risk_flags = list(profile.warnings[:6])

    high_missing = [
        column
        for column, pct in profile.missing_percentages.items()
        if float(pct) >= 40.0
    ]
    if high_missing:
        risk_flags.append("high_missingness_columns:" + ", ".join(high_missing[:5]))

    if schema.id_like_columns:
        risk_flags.append("id_like_columns_detected:" + ", ".join(schema.id_like_columns[:5]))
    if schema.text_columns:
        risk_flags.append("free_text_columns_detected:" + ", ".join(schema.text_columns[:5]))
    if problem_type == ProblemType.CLASSIFICATION:
        risk_flags.append("confirm_class_balance_before_relying_on_accuracy_only")
    if problem_type == ProblemType.UNSUPERVISED:
        risk_flags.append("no_supervised_target_confirmed_modeling_will_be_skipped")

    return risk_flags[:8]


def _heuristic_plan(
    *,
    schema: SchemaSummary,
    profile: ProfileSummary,
    target_column: str | None,
) -> AnalysisPlan:
    resolved_target = target_column or (schema.target_candidates[0] if schema.target_candidates else None)
    problem_type = _resolve_problem_type(
        schema=schema,
        profile=profile,
        target_column=resolved_target,
    )
    split_strategy = _resolve_split_strategy(problem_type)

    return AnalysisPlan(
        problem_type=problem_type,
        split_strategy=split_strategy,
        target_column=resolved_target,
        chart_suggestions=_build_chart_suggestions(schema, resolved_target),
        feature_engineering_actions=_build_feature_actions(schema, resolved_target),
        risk_flags=_build_risk_flags(schema, profile, problem_type),
        steps=build_canonical_plan_steps(
            problem_type=problem_type,
            target_column=resolved_target,
        ),
        planner_notes=(
            "Deterministic fallback plan generated from schema and profiling signals. "
            "Use this as the baseline workflow when LLM planning is unavailable."
        ),
    )


def _coerce_split_strategy(value: Any, fallback: SplitStrategy) -> SplitStrategy:
    if isinstance(value, SplitStrategy):
        return value
    if not value:
        return fallback
    normalized = str(value).strip().lower()
    for member in SplitStrategy:
        if member.value == normalized:
            return member
    return fallback


def _plan_from_payload(
    payload: dict[str, Any],
    *,
    schema: SchemaSummary,
    profile: ProfileSummary,
    target_column: str | None,
) -> AnalysisPlan:
    fallback = _heuristic_plan(
        schema=schema,
        profile=profile,
        target_column=target_column,
    )

    resolved_problem_type = to_problem_type(payload.get("problem_type") or fallback.problem_type)
    resolved_target = payload.get("target_column") or fallback.target_column
    valid_columns = {column.name for column in schema.columns}
    if resolved_target and resolved_target not in valid_columns:
        resolved_target = fallback.target_column

    return AnalysisPlan(
        problem_type=resolved_problem_type,
        split_strategy=_coerce_split_strategy(payload.get("split_strategy"), fallback.split_strategy),
        target_column=resolved_target,
        chart_suggestions=list(payload.get("chart_suggestions") or fallback.chart_suggestions),
        feature_engineering_actions=list(
            payload.get("feature_engineering_actions") or fallback.feature_engineering_actions
        ),
        risk_flags=list(payload.get("risk_flags") or fallback.risk_flags),
        steps=build_canonical_plan_steps(
            problem_type=resolved_problem_type,
            target_column=resolved_target,
        ),
        planner_notes=str(payload.get("planner_notes") or fallback.planner_notes),
    )


def run_planner_agent(
    *,
    schema: SchemaSummary,
    profile: ProfileSummary,
    target_column: str | None = None,
    model_name: str | None = None,
) -> tuple[AnalysisPlan, object]:
    system_prompt = load_prompt_template(
        "planner_prompt.txt",
        PLANNER_SYSTEM_FALLBACK,
    )

    payload = filter_large_payload(
        {
            "schema": model_to_dict(schema),
            "profile": model_to_dict(profile),
            "target_column": target_column,
        }
    )

    fallback_plan = _heuristic_plan(
        schema=schema,
        profile=profile,
        target_column=target_column,
    )

    user_prompt = join_sections(
        "Plan the CSV analysis workflow using only the structured dataset summary below.",
        render_json_block("PLANNER_PAYLOAD", payload),
        (
            "Return valid JSON with keys: problem_type, split_strategy, target_column, "
            "chart_suggestions, feature_engineering_actions, risk_flags, planner_notes. "
            "Do not include fields outside that schema."
        ),
    )

    agent = BaseAgent(
        agent_name=AgentName.PLANNER,
        system_prompt=system_prompt,
        model_name=model_name,
        use_json_mode=True,
    )

    warnings: list[str] = []
    try:
        content = agent.run(user_prompt=user_prompt)
        _, parsed = parse_agent_response(content, expect_json=True)
        if not isinstance(parsed, dict):
            raise ValueError("Planner returned non-JSON or incompatible JSON payload.")

        plan = _plan_from_payload(
            parsed,
            schema=schema,
            profile=profile,
            target_column=target_column,
        )
        message = agent.build_message(
            content=content,
            structured_output=model_to_dict(plan),
            warnings=warnings,
        )
        return plan, message
    except Exception as exc:
        warnings.append(f"planner_agent_fallback_due_to:{type(exc).__name__}")
        message = agent.build_message(
            content=fallback_plan.planner_notes,
            structured_output=model_to_dict(fallback_plan),
            warnings=warnings,
        )
        return fallback_plan, message
