from youtube_transcript_api import YouTubeTranscriptApi
import re
from urllib.parse import urlparse, parse_qs
from pathlib import Path
import json
import csv
import time
from datetime import datetime

def extract_video_id(url: str) -> str:
    """YouTube URL에서 비디오 ID 추출"""
    # 다양한 YouTube URL 형식 처리
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([A-Za-z0-9_-]{11})',
        r'youtube\.com/shorts/([A-Za-z0-9_-]{11})',
        r'youtube\.com/v/([A-Za-z0-9_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    # URL 파싱으로 시도
    try:
        parsed = urlparse(url)
        if 'youtu.be' in parsed.netloc:
            return parsed.path.strip('/')
        elif 'youtube.com' in parsed.netloc:
            query_params = parse_qs(parsed.query)
            if 'v' in query_params:
                return query_params['v'][0]
    except:
        pass
    
    raise ValueError(f"Could not extract video ID from: {url}")

def fetch_transcript(video_id, language="ko"):
    """
    유튜브 video_id와 언어코드로 자막(스크립트) 텍스트를 반환합니다.
    """
    try:
        transcript_list = YouTubeTranscriptApi().list(video_id)
        transcript = None
        lang_info = ""  # 기본값

        # 우선 한국어 수동 자막을 찾는다.
        try:
            transcript = transcript_list.find_transcript([language])
            lang_info = transcript.language_code
        except Exception:
            # 없으면 자동 생성 자막을 찾는다.
            try:
                transcript = transcript_list.find_generated_transcript([language])
                lang_info = f"{transcript.language_code} (auto-generated)"
            except Exception:
                # 그래도 없으면 첫 번째 사용 가능한 자막 사용
                try:
                    transcript = next(iter(transcript_list))
                    lang_info = transcript.language_code  # 예: 'en'
                except Exception:
                    return "Error: No transcript available."
        
        # transcript.fetch() 결과에서 text 속성만 추출
        transcript_data = transcript.fetch()
        text_list = []
        for entry in transcript_data:
            if hasattr(entry, 'text'):
                if entry.text:
                    text_list.append(entry.text)
            elif isinstance(entry, dict) and 'text' in entry:
                text_list.append(entry['text'])
        
        full_text = ' '.join(text_list)
        return f"[{lang_info}] {full_text}"
    
    except Exception as e:
        return f"Error fetching transcript: {e}"

# Test the function with the YouTube URL
test_url = "https://www.youtube.com/watch?v=_akid7uZrOw"
try:
    video_id = extract_video_id(test_url)
    print(f"Video ID: {video_id}")
    
    # Try Korean first, then English
    transcript = fetch_transcript(video_id, language="ko")
    if transcript.startswith("Error"):
        print("Korean transcript not available, trying English...")
        transcript = fetch_transcript(video_id, language="en")
    
    print(f"Transcript: {transcript[:200]}...")
    
except Exception as e:
    print(f"Error: {e}")