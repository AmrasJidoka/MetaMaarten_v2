from langchain.pydantic_v1 import BaseModel, Field
from datetime import date

class BasisInfo(BaseModel):
    auteur: str = Field(
        description="De naam van het bedrijf dat dit document heeft opgesteld"
    )
    datum: date = Field(
        description="De datum waarop dit document is opgesteld. Deze mag niet handgeschreven zijn en geen stempel"
    )
    documentNummer: str = Field(
        description="unieke identificatie voor dit document"
    )
    typeDocument: str = Field(
        description="document type: factuur of offerte"
    )
    leveringsconditie: str = Field(
        description="condities en informatie over de levering of afhaling van de goederen of diensten in dit document"
    )
    
