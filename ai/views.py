import os
import io
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from django.conf import settings

# 환경변수.env file에서 API key 호출
load_dotenv(find_dotenv())

# API key로 OpenAI client 시작
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

@csrf_exempt
def transcribe_audio(request):
    if request.method == 'POST' and request.FILES.get('file'):
        audio_file = request.FILES['file']
        transcript = get_transcript(audio_file)
        return JsonResponse({'transcript': transcript})
    return JsonResponse({'error': 'Invalid request'}, status=400)

# def get_transcript(audio_file):

#     print(type(audio_file))
#     audio_bytes = audio_file.read()
#     audio_io = io.BytesIO(audio_bytes)
#     transcript = client.audio.transcriptions.create(
#         model="whisper-1",
#         file=audio_io
#     )
#     print('done')
#     return transcript['text']
import tempfile
def get_transcript(audio_file):
    # 임시 파일 생성
    with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_file:
        # 업로드된 파일 내용을 임시 파일에 쓰기
        for chunk in audio_file.chunks():
            temp_file.write(chunk)
    
    try:
        # OpenAI API에 임시 파일 전달
        with open(temp_file.name, 'rb') as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        print(transcript.text)
        return transcript.text
    finally:
        # 임시 파일 삭제
        os.unlink(temp_file.name)