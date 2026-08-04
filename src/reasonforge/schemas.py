"""Pydantic schemas shared by parsing, verification, and reporting."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Calculation(BaseModel):
    """One machine-verifiable arithmetic step."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    expression: str = Field(min_length=1, max_length=128)
    result: str = Field(min_length=1, max_length=128)


class StructuredSolution(BaseModel):
    """The exact response contract requested from the language model."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    method: str = Field(min_length=1, max_length=160)
    calculations: list[Calculation] = Field(min_length=1, max_length=12)
    final_answer: str = Field(min_length=1, max_length=128)

    @field_validator("method")
    @classmethod
    def method_must_be_plain_text(cls, value: str) -> str:
        """Keep the unverified method label short and free of control characters."""
        if any(ord(character) < 32 for character in value):
            raise ValueError("method contains control characters")
        return value


class ParseResult(BaseModel):
    """Structured outcome of treating a completion as untrusted JSON."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    raw_text: str
    json_valid: bool = False
    schema_valid: bool = False
    exact_json: bool = False
    data: dict[str, object] | None = None
    solution: StructuredSolution | None = None
    error: str | None = None


class CalculationCheck(BaseModel):
    """Verification details for one calculation step."""

    index: int
    expression: str
    claimed_result: str
    normalized_expression_value: str | None = None
    normalized_claimed_result: str | None = None
    valid: bool = False
    reason: str | None = None


class VerificationResult(BaseModel):
    """Auditable verification result; no hidden reasoning is claimed."""

    parse: ParseResult
    reference_answer: str
    normalized_reference: str | None = None
    normalized_final_answer: str | None = None
    answer_correct: bool = False
    calculation_checks: list[CalculationCheck] = Field(default_factory=list)
    calculation_validity_rate: float = 0.0
    final_consistent: bool = False
    failure_reason: str | None = None
    failure_categories: list[str] = Field(default_factory=list)


class RewardBreakdown(BaseModel):
    """Finite reward components for one completion."""

    schema_score: float = Field(default=0.0, serialization_alias="schema")
    answer_correctness: float = 0.0
    calculation_validity: float = 0.0
    final_consistency: float = 0.0
    conciseness: float = 0.0
    suspicious_penalty: float = 0.0
    total: float = 0.0
