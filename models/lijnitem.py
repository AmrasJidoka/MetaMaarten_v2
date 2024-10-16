from langchain.pydantic_v1 import BaseModel, Field

class Lijnitem(BaseModel):
    omschrijving: str = Field(
        description=(
            "beschrijving van het goed of de dienst, dit kan meerdere lijnen zijn."
            "Wanneer er geen prijs of hoeveelheid langs de omschrijving regel staan, hoort dit nog bij het huidige goed of dienst."
            "Je mag niets weglaten, alles moet letterlijk overgenomen worden."
            )
    )
    extraInfo: str = Field(
        description=(
            "Additionele informatie over het goed of dienst."
            "Sometimes this is placed between brackets or parenthesis after the initial description."
        )
    )
    aantal: str = Field(
        description=(
            "Het aantal of de hoeveelheid van de dienst of het goed."
        )
    )
    eenheid: str = Field(
        description="De eenheid waarin het goed of de dienst verkocht wordt."
    )
    prijs: str = Field(
        description=(
            "De eenheidsprijs van het goed of de dienst. Voorbeelden van deze benaming: eenh. pr.; unit price; prijs; pr."
            "Het decimaal teken is altijd een komma, je moet dit vervangen wanneer het een punt is."
            "Duizendtallen worden gescheiden door een punt, je moet dit toevoegen als dit ontbreekt."
        )
    )
    korting: str = Field(
        description=(
            "Vermindering in prijs, meestal uitgedrukt in een percentage."
            "Als er geen korting is, of de korting is 0, dan moet je een lege string teruggeven."
        )
    )
    prijsKorting: str = Field(
        description=(
            "Eenheidsprijs van het goed of de dienst, verminderd met de korting."
            "De korting kan per lijn opgegeven zijn of op de totaalprijs zijn toegepast."
            "Bereken de individuele korting zelf met de gevonden korting en de standaard eenheidsprijs."
            "Dit moet je alleen uitvoeren als er een korting gevonden is."
        )    
    )

    chapter: str = Field(
        description=
            ""
    )