"""Deterministic, inspectable objective curriculum for GSM8K examples."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

_FRACTION_PERCENT_RE = re.compile(
    r"(?:%|percent|percentage|fraction|ratio|half|third|quarter|divide|per\b)", re.IGNORECASE
)
_OPERATOR_RE = re.compile(r"[+\-*/]")


@dataclass(frozen=True)
class CurriculumStage:
    number: int
    name: str


STAGES = (
    CurriculumStage(1, "simple_arithmetic"),
    CurriculumStage(2, "one_step_word_problem"),
    CurriculumStage(3, "fractions_percent_equations"),
    CurriculumStage(4, "multi_step_reasoning"),
)


def infer_curriculum_stage(question: str, expressions: Iterable[str] = ()) -> CurriculumStage:
    """Classify difficulty from observable text and annotated operation count."""
    expression_list = list(expressions)
    operation_count = sum(len(_OPERATOR_RE.findall(value)) for value in expression_list)
    calculation_count = len(expression_list)
    if calculation_count >= 3 or operation_count >= 4:
        return STAGES[3]
    joined = " ".join([question, *expression_list])
    if _FRACTION_PERCENT_RE.search(joined):
        return STAGES[2]
    if calculation_count <= 1:
        return STAGES[0]
    if calculation_count == 2:
        return STAGES[1]
    return STAGES[3]
