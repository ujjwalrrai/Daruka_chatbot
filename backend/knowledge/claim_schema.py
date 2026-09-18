from typing import Optional

from pydantic import BaseModel, Field


class ExtractedClaim(BaseModel):
    src_node: str
    dst_node: str

    relation: str
    effect_type: str

    effect_size: Optional[float] = None
    effect_unit: Optional[str] = None

    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    ci_level: Optional[float] = None

    lag_years: Optional[float] = None
    time_to_full_effect: Optional[float] = None

    quote: str

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    notes: Optional[str] = None


class ClaimExtractionResult(BaseModel):
    claims: list[ExtractedClaim]