import os
from flask import Flask, request

from Analyzers.offerte import parse_offerte_file
from chatmodels.openai_chat import init_azure_chat
from file_storage.file_storage import Document
from dotenv import load_dotenv

from responses.json_response import build_json_response
from tools.custom.model import init_custom_ocr_tool

app = Flask(__name__)


@app.route('/analyse', methods=['POST'])
def analyse():
    print('Starting analyse')
    with Document(request.files["file"]) as document:
        print("setting up ocr and model")
        ocr = init_custom_ocr_tool()
        model = init_azure_chat()

        print("parsing file")
        result_dict = parse_offerte_file(document.filename, model, ocr)
    print("retuning response")
    return build_json_response(result_dict)

if __name__ == '__main__':
    app.run()
