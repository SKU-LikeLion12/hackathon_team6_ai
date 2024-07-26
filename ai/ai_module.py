import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# 환경변수.env file에서 API key 호출
_ = load_dotenv(find_dotenv())

# API key로 OpenAI client 시작
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# 오디오 파일을 텍스트로 변환하는 함수
def get_transcript(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
    return transcript