from langchain.tools.azure_cognitive_services import AzureCogsFormRecognizerTool
from configuration import get_configuration


def init_langchain_ocr_tool():
    configuration = get_configuration("azure")

    return AzureCogsFormRecognizerTool(
        azure_cogs_endpoint=configuration["cognitive_services_url"],
    )
