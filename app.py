import os
from dotenv import load_dotenv

from openai import AzureOpenAI

load_dotenv(override=True)

whisper_endpoint = os.environ["WHISPER_ENDPOINT"]
whisper_apikey = os.environ["WHISPER_API_KEY"]
whisper_apiversion = os.environ["WHISPER_API_VERSION"]
whisper_model_name = os.environ["WHISPER_DEPLOYMENT_NAME"]

# Create AOAI client for whisper
whisper_client = AzureOpenAI(
    api_version=whisper_apiversion,
    azure_endpoint=whisper_endpoint,
    api_key=whisper_apikey
)


transcription = whisper_client.audio.transcriptions.create(
    model=whisper_model_name,
    file=open('sample.mp3', "rb"),
)
