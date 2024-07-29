import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import tempfile
import requests
import datetime
from typing import Dict, Any
import json

load_dotenv(find_dotenv())

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
SPRING_SERVER_URL = os.environ.get("SPRING_SERVER_URL", "http://localhost:8080/api/chat")

@csrf_exempt
def transcribe_and_process(request):
    if request.method != 'POST' or 'file' not in request.FILES:
        return JsonResponse({'error': 'Invalid request'}, status=400)

    audio_file = request.FILES['file']

    try:
        raw_transcript = get_transcript(audio_file)
        refined_text = refine_text(raw_transcript)
        emotions = get_sentiment(refined_text)
        situation = get_situation(refined_text)

        chat_data = {
            "userId": request.user.id,
            "message": refined_text,
            "startTime": datetime.datetime.now().isoformat(),
            "endTime": datetime.datetime.now().isoformat(),
            "emotions": emotions,
            "situation": situation
        }
  
        send_to_spring_server(chat_data)

        return JsonResponse({
            'refined_text': refined_text,
            'emotions': emotions,
            'situation': situation,
            'message': 'Chat entry saved successfully'
        })

    except Exception as e:
        return JsonResponse({'error': f'An error occurred: {str(e)}'}, status=500)

def get_transcript(audio_file) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_file:
        for chunk in audio_file.chunks():
            temp_file.write(chunk)
    
    try:
        with open(temp_file.name, 'rb') as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcript.text
    finally:
        os.unlink(temp_file.name)

def refine_text(text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that refines diary entries into well-structured sentences. And you have to answer in casual Korean."},
            {"role": "user", "content": f"Please refine this diary entry into well-structured sentences: {text}"}
        ]
    )
    return response.choices[0].message.content.strip()

def get_sentiment(text: str) -> Dict[str, int]:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI trained to analyze emotions in text. Provide emotion percentages for: happiness, anxiety, neutral, sadness, and anger. The total should sum to 100%."},
            {"role": "user", "content": f"Analyze the emotions in this text and provide percentages: {text}"}
        ],
        temperature=0.3,
        max_tokens=150
    )
    
    analysis = response.choices[0].message.content
    emotions = {"행복": 0, "불안": 0, "중립": 0, "슬픔": 0, "분노": 0}

    for line in analysis.split('\n'):
        for emotion in emotions.keys():
            if emotion in line:
                percentage = int(line.split(':')[1].strip().rstrip('%'))
                emotions[emotion] = percentage

    total = sum(emotions.values())

    if total > 0:  # 0으로 나누는 것을 방지
        if total != 100:
            factor = 100 / total
            emotions = {k: round(v * factor) for k, v in emotions.items()}
    else:
        # total이 0인 경우 처리
        # 예: 모든 감정을 균등하게 분배
        equal_value = 100 // len(emotions)
        emotions = {k: equal_value for k in emotions}
        # 남은 값을 첫 번째 감정에 할당
        remainder = 100 - (equal_value * len(emotions))
        if remainder > 0:
            first_emotion = next(iter(emotions))
            emotions[first_emotion] += remainder

    return emotions


def get_situation(text: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Analyze the emotional situations in the following text. Extract instances of happiness, anxiety, sadness, and anger. Respond in JSON format with keys '행복', '불안', '슬픔', '분노', and empty string values if not applicable. And you have to answer in casual Korean."},
            {"role": "user", "content": text}
        ]
    )
    
    gpt_response = response.choices[0].message.content
    
    situation = {
        "행복": "",
        "불안": "",
        "슬픔": "",
        "분노": ""
    }
    
    try:
        # GPT 응답을 JSON으로 파싱
        parsed_response = json.loads(gpt_response)
        
        # 파싱된 응답에서 각 감정에 해당하는 값을 가져와 situation 딕셔너리에 업데이트
        for emotion in situation.keys():
            if emotion in parsed_response:
                situation[emotion] = parsed_response[emotion]
    except json.JSONDecodeError:
        print("Failed to parse GPT response as JSON. Returning default dictionary.")
    
    return situation

def send_to_spring_server(data: Dict[str, Any]) -> None:
    response = requests.post(
        SPRING_SERVER_URL, 
        json=data,
        headers={'Content-Type': 'application/json'}
    )
    response.raise_for_status()

