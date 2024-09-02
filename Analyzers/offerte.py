from typing import Type, Any

from azure.ai.documentintelligence.models import AnalyzeResult
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableSerializable, RunnableParallel, RunnableLambda
from langchain_openai import AzureChatOpenAI
from pydantic.v1 import BaseModel
from Analyzers.analysis.offerte import build_offerte_prompt
from json_processing.string_processing import remove_trailing_commas_from_message
from json_processing.dict_processing import remove_empty_objects
from models.offerte_response import Offerte
from tools.custom.model import CustomTextExtractorTool


def build_chain(requested_data: Type[Offerte], model: AzureChatOpenAI) -> RunnableSerializable[str, Any]:
    return (
            build_offerte_prompt(data_model=requested_data)
            | model
            | RunnableLambda(remove_trailing_commas_from_message)
            | JsonOutputParser(pydantic_object=requested_data)
            | RunnableLambda(remove_empty_objects)
    )


def parse_offerte(text: str, model: AzureChatOpenAI) -> dict:
    chain = (
            RunnableParallel(
                offerte=build_chain(Offerte, model)
            )
            | RunnableLambda(lambda x: x["offerte"])
    )

    return chain.invoke(text)


def parse_offerte_file(filename: str, model: AzureChatOpenAI, ocr: CustomTextExtractorTool):
    file_text = ocr.run(filename)
    return parse_offerte(file_text, model)
