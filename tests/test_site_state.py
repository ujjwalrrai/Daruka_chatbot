from backend.models.site_state import SiteState


site = SiteState(
    soil_organic_carbon=0.3,
    rainfall="low",
    crop="wheat",
    land_use="monoculture",
    region="semi-arid"
)


print(site.model_dump())
print()
print("Missing fields:")
print(site.missing_fields())