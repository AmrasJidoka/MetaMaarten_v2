from langchain_openai.chat_models import AzureChatOpenAI
from configuration import get_configuration

def init_azure_chat() -> AzureChatOpenAI:
    configuration = get_configuration("azure")

    return AzureChatOpenAI(
        azure_endpoint=configuration["openai_api_base"],
        openai_api_version=configuration["openai_api_version"],
        deployment_name=configuration["deployment_name"],
        temperature=0,
    )