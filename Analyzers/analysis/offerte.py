from typing import Type
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableLambda, Runnable
from langchain_core.prompt_values import PromptValue

def build_prompt_arguments(text: str) -> dict:
	return {
		'document': text
	}

def build_prompt(data_model: Type[BaseModel]) -> PromptTemplate:
	template = (
		# persona
		"Je bent een document ontledingsprogramma voor facturen en offertes.\n"
		"Je hoofddoel is prijsiformatie van de producten extraheren. \n"
		"Informatie moet letterlijk overgenomen worden, je mag niets wijzigen. \n"
		"Mininum info voor een lijn is: omschrijving, prijs en eenheid.\n"
		"Dit is het document dat je zal analyseren: \n"
		"{document} \n"
		"{output_structure} \n"
	)
	parser = JsonOutputParser(pydantic_object=data_model)

	return PromptTemplate(
		template=template,
		input_variables=["document"],
		partial_variables={"output_structure": parser.get_format_instructions()})

def build_offerte_prompt(data_model: Type[BaseModel]) -> Runnable[str, PromptValue]:
	return (
		RunnableLambda(build_prompt_arguments)
		| build_prompt(data_model)
	)