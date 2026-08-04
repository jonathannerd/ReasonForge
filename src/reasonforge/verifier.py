"""Restricted arithmetic normalization and symbolic equivalence checking."""

from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Callable
from dataclasses import dataclass

import sympy

from reasonforge.parsing import parse_completion
from reasonforge.schemas import CalculationCheck, VerificationResult

MAX_EXPRESSION_CHARS = 128
MAX_AST_NODES = 64
MAX_ABS_EXPONENT = 12
MAX_INTEGER_DIGITS = 30
MAX_ABS_RESULT = sympy.Integer(10) ** 100

_ALLOWED_RE = re.compile(r"^[0-9+\-*/().%^\s]+$")
_COMMA_NUMBER_RE = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")
_CURRENCY_RE = re.compile(r"(?:USD|CAD|EUR|GBP|dollars?|euros?|pounds?)", re.IGNORECASE)


class UnsafeExpressionError(ValueError):
    """Raised when generated arithmetic leaves the restricted language."""


@dataclass(frozen=True)
class NormalizedValue:
    """A safe SymPy numeric expression plus a stable display string."""

    value: sympy.Expr
    text: str


def _numeric_constant(value: object, source_literal: str) -> sympy.Expr:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UnsafeExpressionError("only numeric constants are supported")
    rendered = source_literal.strip()
    digits = rendered.replace("-", "").replace(".", "").replace("e", "").replace("+", "")
    if len(digits) > MAX_INTEGER_DIGITS:
        raise UnsafeExpressionError("numeric literal is too long")
    if isinstance(value, float) and not math.isfinite(value):
        raise UnsafeExpressionError("non-finite number")
    try:
        # Parsing the source spelling as a string keeps decimals exact (0.1 is
        # 1/10), unlike constructing a rational from its binary float value.
        return sympy.Rational(rendered)
    except (TypeError, ValueError) as exc:
        raise UnsafeExpressionError("invalid numeric literal") from exc


_BINARY_OPERATORS: dict[type[ast.operator], Callable[[sympy.Expr, sympy.Expr], sympy.Expr]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


def _build_expression(node: ast.AST, source: str) -> sympy.Expr:
    if isinstance(node, ast.Expression):
        return _build_expression(node.body, source)
    if isinstance(node, ast.Constant):
        literal = ast.get_source_segment(source, node)
        if literal is None:
            raise UnsafeExpressionError("numeric literal could not be read")
        return _numeric_constant(node.value, literal)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _build_expression(node.operand, source)
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _build_expression(node.left, source)
        right = _build_expression(node.right, source)
        if isinstance(node.op, ast.Div) and right == 0:
            raise UnsafeExpressionError("division by zero")
        if isinstance(node.op, ast.Pow) and (
            not bool(right.is_integer) or abs(int(right)) > MAX_ABS_EXPONENT
        ):
            raise UnsafeExpressionError("exponent must be an integer between -12 and 12")
        return _BINARY_OPERATORS[type(node.op)](left, right)
    raise UnsafeExpressionError(f"unsupported syntax: {type(node).__name__}")


def safe_arithmetic(expression: str) -> NormalizedValue:
    """Evaluate a tightly restricted arithmetic expression without ``eval``.

    Python's AST is used only as a parser; accepted nodes are rebuilt as SymPy
    numeric objects. Names, calls, attributes, subscripts, assignments, and all
    other syntax are rejected before SymPy sees the expression.
    """
    if not isinstance(expression, str) or not expression.strip():
        raise UnsafeExpressionError("expression is empty")
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise UnsafeExpressionError("expression exceeds size limit")
    if not _ALLOWED_RE.fullmatch(expression):
        raise UnsafeExpressionError("expression contains unsupported characters")
    candidate = expression.replace("^", "**")
    try:
        tree = ast.parse(candidate, mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise UnsafeExpressionError("invalid arithmetic syntax") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise UnsafeExpressionError("expression is too complex")
    try:
        value = sympy.cancel(_build_expression(tree, candidate))
    except UnsafeExpressionError:
        raise
    except Exception as exc:
        raise UnsafeExpressionError("arithmetic could not be simplified") from exc
    if value.has(sympy.zoo, sympy.nan, sympy.oo, -sympy.oo) or value.is_real is not True:
        raise UnsafeExpressionError("result must be a finite real number")
    if abs(value) > MAX_ABS_RESULT:
        raise UnsafeExpressionError("result magnitude exceeds limit")
    return NormalizedValue(value=value, text=str(value))


def normalize_answer(answer: str) -> NormalizedValue:
    """Normalize common numeric answer forms into a safe symbolic value."""
    if not isinstance(answer, str) or not answer.strip():
        raise UnsafeExpressionError("answer is empty")
    candidate = answer.strip().replace("\N{MINUS SIGN}", "-").replace("\N{EN DASH}", "-")
    candidate = candidate.replace("$", "").replace("€", "").replace("£", "")
    candidate = _CURRENCY_RE.sub("", candidate).strip()
    candidate = candidate.rstrip(". ")
    candidate = _COMMA_NUMBER_RE.sub("", candidate)
    is_percent = candidate.endswith("%")
    if is_percent:
        candidate = candidate[:-1].strip()
    normalized = safe_arithmetic(candidate)
    value = normalized.value / 100 if is_percent else normalized.value
    return NormalizedValue(value=sympy.cancel(value), text=str(sympy.cancel(value)))


def mathematically_equivalent(left: str, right: str) -> bool:
    """Return whether two supported numeric answers represent the same value."""
    try:
        left_value = normalize_answer(left).value
        right_value = normalize_answer(right).value
        return bool(sympy.cancel(left_value - right_value) == 0)
    except UnsafeExpressionError:
        return False


def verify_completion(completion: object, reference_answer: str) -> VerificationResult:
    """Verify schema, reference answer, calculations, and final-step consistency."""
    parsed = parse_completion(completion)
    categories: list[str] = []
    try:
        reference = normalize_answer(reference_answer)
    except UnsafeExpressionError as exc:
        return VerificationResult(
            parse=parsed,
            reference_answer=reference_answer,
            failure_reason=f"invalid reference answer: {exc}",
            failure_categories=["invalid_reference"],
        )

    if not parsed.json_valid:
        categories.append("invalid_json")
    elif not parsed.schema_valid:
        categories.append("schema_violation")
    elif not parsed.exact_json:
        categories.append("non_exact_json")

    result = VerificationResult(
        parse=parsed,
        reference_answer=reference_answer,
        normalized_reference=reference.text,
        failure_categories=categories,
    )
    solution = parsed.solution
    if solution is None:
        # Give mathematics its deserved dominance only for a completion that is
        # itself a concise supported answer, never by fishing numbers from prose.
        try:
            raw_value = normalize_answer(parsed.raw_text)
        except UnsafeExpressionError:
            result.failure_reason = parsed.error or "no verifiable final answer"
            return result
        result.normalized_final_answer = raw_value.text
        result.answer_correct = bool(sympy.cancel(raw_value.value - reference.value) == 0)
        if not result.answer_correct:
            result.failure_categories.append("wrong_answer")
            result.failure_reason = parsed.error or "final answer is not equivalent to reference"
        elif result.failure_categories:
            result.failure_reason = parsed.error or ", ".join(result.failure_categories)
        return result

    try:
        final_value = normalize_answer(solution.final_answer)
        result.normalized_final_answer = final_value.text
        result.answer_correct = bool(sympy.cancel(final_value.value - reference.value) == 0)
    except UnsafeExpressionError as exc:
        final_value = None
        result.failure_reason = f"unsupported final answer: {exc}"
        result.failure_categories.append("unsupported_answer")

    for index, calculation in enumerate(solution.calculations):
        check = CalculationCheck(
            index=index,
            expression=calculation.expression,
            claimed_result=calculation.result,
        )
        try:
            expression_value = safe_arithmetic(calculation.expression)
            claimed_value = normalize_answer(calculation.result)
            check.normalized_expression_value = expression_value.text
            check.normalized_claimed_result = claimed_value.text
            check.valid = bool(sympy.cancel(expression_value.value - claimed_value.value) == 0)
            if not check.valid:
                check.reason = "claimed result does not equal expression value"
        except UnsafeExpressionError as exc:
            check.reason = str(exc)
        result.calculation_checks.append(check)

    if result.calculation_checks:
        passed = sum(check.valid for check in result.calculation_checks)
        result.calculation_validity_rate = passed / len(result.calculation_checks)
        last_check = result.calculation_checks[-1]
        if last_check.valid and final_value is not None and last_check.normalized_claimed_result:
            result.final_consistent = mathematically_equivalent(
                last_check.normalized_claimed_result, final_value.text
            )

    if not result.answer_correct:
        result.failure_categories.append("wrong_answer")
    if result.calculation_validity_rate < 1.0:
        result.failure_categories.append("invalid_calculation")
    if not result.final_consistent:
        result.failure_categories.append("inconsistent_final")
    if result.failure_categories and result.failure_reason is None:
        result.failure_reason = ", ".join(result.failure_categories)
    return result
