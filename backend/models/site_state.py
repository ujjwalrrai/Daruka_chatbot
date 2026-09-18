from typing import Optional

from pydantic import BaseModel, Field


class SiteState(BaseModel):
    soil_organic_carbon: Optional[float] = Field(
        default=None,
        description="Soil organic carbon percentage"
    )

    soil_ph: Optional[float] = Field(
        default=None,
        description="Soil pH"
    )

    soil_moisture: Optional[float] = Field(
        default=None,
        description="Soil moisture percentage"
    )

    rainfall: Optional[str] = Field(
        default=None,
        description="Rainfall condition or pattern"
    )

    crop: Optional[str] = Field(
        default=None,
        description="Primary crop"
    )

    land_use: Optional[str] = Field(
        default=None,
        description="Land use or cropping system"
    )

    region: Optional[str] = Field(
        default=None,
        description="Geographic/ecological region"
    )

    temperature: Optional[float] = Field(
        default=None,
        description="Temperature in Celsius"
    )

    pollution: Optional[str] = Field(
        default=None,
        description="Pollution condition"
    )

    def missing_fields(self) -> list[str]:
        return [
            field_name
            for field_name, value in self.model_dump().items()
            if value is None
        ]