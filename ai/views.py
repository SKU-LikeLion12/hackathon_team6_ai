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
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from django.conf import settings

load_dotenv(find_dotenv())
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
SPRING_SERVER_URL = os.environ.get("SPRING_SERVER_URL", "http://team6back.sku-sku.com/api/chat/")

# 모델과 토크나이저 로드
model_path = os.path.join(settings.BASE_DIR, 'ai', 'feelinsight_distilbert_model')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

@csrf_exempt
@csrf_exempt
def transcribe_and_process(request):
    if request.method != 'POST' or 'file' not in request.FILES:
        return JsonResponse({'error': 'Invalid request'}, status=400)

    audio_file = request.FILES['file']

    try:
        raw_transcript = get_transcript(audio_file)
        refined_text = refine_text(raw_transcript)
        refined_text_KOR = refine_text_KOR(raw_transcript)
        emotions = get_sentiment(refined_text)
        situation = get_situation(refined_text_KOR)

        # 사용자 인증 확인
        user_id = request.user.id if request.user.is_authenticated else None

        chat_data = {
            "userId": user_id,
            "message": refined_text,
            "startTime": datetime.datetime.now().isoformat(),
            "endTime": datetime.datetime.now().isoformat(),
            "emotions": emotions,
            "situation": situation
        }

        # 디버깅을 위한 로그 추가
        print("Refined text:", refined_text)
        print("Refined text (KOR):", refined_text_KOR)
        print("Emotions:", emotions)
        print("Situation:", situation)

        # Spring 서버로 데이터 전송
        try:
            send_to_spring_server(chat_data)
        except Exception as e:
            print(f"Error sending data to Spring server: {str(e)}")
            # Spring 서버 오류가 전체 프로세스를 중단시키지 않도록 함

        return JsonResponse({
            'refined_text': refined_text,
            'emotions': emotions,
            'situation': situation,
            'message': 'Chat entry processed successfully'
        })

    except Exception as e:
        print(f"Error in transcribe_and_process: {str(e)}")
        return JsonResponse({'error': f'An error occurred!!: {str(e)}'}, status=500)
    

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
            {"role": "system", "content": "You are a helpful assistant that makes minor corrections to diary entries, such as fixing spelling errors and removing unnecessary words, without altering the original meaning. Please ensure the revised text is clear and well-structured."},
            {"role": "user", "content": f"Please refine this diary entry with minimal changes to spelling errors and unnecessary words, while preserving the original meaning: {text}"}
        ]
    )
    return response.choices[0].message.content.strip()

def refine_text_KOR(text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that makes minor corrections to diary entries, such as fixing spelling errors and removing unnecessary words, without altering the original meaning. Please ensure the revised text is clear and well-structured. And you have to answer in casual Korean"},
            {"role": "user", "content": f"Please refine this diary entry with minimal changes to spelling errors and unnecessary words, while preserving the original meaning: {text}"}
        ]
    )
    return response.choices[0].message.content.strip()

def get_sentiment(text: str) -> Dict[str, int]:
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.sigmoid(outputs.logits).squeeze().tolist()
    
    # 학습 시 사용한 감정 순서
    original_order = ['anger', 'anxiety', 'happiness', 'sadness', 'neutral']
    
    # 원하는 출력 감정 순서
    desired_order = ['happiness', 'anxiety', 'neutral', 'sadness', 'anger']
    
    total = sum(probabilities)
    percentages = [round((prob / total) * 100) for prob in probabilities]
    
    while sum(percentages) != 100:
        if sum(percentages) > 100:
            percentages[percentages.index(max(percentages))] -= 1
        else:
            percentages[percentages.index(min(percentages))] += 1

    original_emotions = dict(zip(original_order, percentages))
    emotions = {emotion: original_emotions[emotion] for emotion in desired_order}
    
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
    try:
        response = requests.post(
            SPRING_SERVER_URL, 
            json=data,
            headers={'Content-Type': 'application/json'},
            timeout=10  # 10초 타임아웃 설정
        )
        response.raise_for_status()
        print("Data sent successfully to Spring server")
    except requests.RequestException as e:
        print(f"Error sending data to Spring server: {str(e)}")
        if hasattr(e.response, 'text'):
            print(f"Server response: {e.response.text}")
        raise

