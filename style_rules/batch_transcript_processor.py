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

def extract_youtube_urls(file_path: Path):
    """파일에서 YouTube URL 추출"""
    try:
        content = file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        # UTF-8로 안되면 다른 인코딩 시도
        try:
            content = file_path.read_text(encoding='cp949')
        except:
            content = file_path.read_text(encoding='latin-1')
    
    # YouTube URL 패턴으로 검색
    youtube_patterns = [
        r'https?://(?:www\.)?youtube\.com/watch\?v=[A-Za-z0-9_-]{11}[^\s]*',
        r'https?://youtu\.be/[A-Za-z0-9_-]{11}[^\s]*',
        r'https?://(?:www\.)?youtube\.com/embed/[A-Za-z0-9_-]{11}[^\s]*',
        r'https?://(?:www\.)?youtube\.com/shorts/[A-Za-z0-9_-]{11}[^\s]*'
    ]
    
    urls = []
    for pattern in youtube_patterns:
        found = re.findall(pattern, content)
        urls.extend(found)
    
    # 중복 제거 및 정리
    unique_urls = []
    seen = set()
    for url in urls:
        # URL 끝의 따옴표나 괄호 제거
        cleaned_url = re.sub(r'["\'\)]*$', '', url)
        if cleaned_url not in seen:
            seen.add(cleaned_url)
            unique_urls.append(cleaned_url)
    
    return unique_urls

def normalize_text(text: str) -> str:
    """자막 텍스트 정리"""
    if text.startswith("Error"):
        return text
    
    # 언어 정보 추출
    lang_match = re.match(r'\[([^\]]*)\]\s*(.*)', text)
    if lang_match:
        lang_info = lang_match.group(1)
        content = lang_match.group(2)
    else:
        lang_info = "unknown"
        content = text
    
    # 텍스트 정리
    content = re.sub(r'\s+', ' ', content)  # 공백 정리
    content = re.sub(r'https?://\S+', ' ', content)  # URL 제거
    content = re.sub(r'#\w+', ' ', content)  # 해시태그 제거
    content = content.strip()
    
    return f"[{lang_info}] {content}"

def main():
    """메인 실행 함수"""
    # 설정
    INPUT_FILE = Path("youtube_urls.txt")  # YouTube 링크가 있는 파일
    OUTPUT_DIR = Path("transcripts_output")  # 결과 저장 폴더
    
    # 출력 폴더 생성
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print(f"📂 {INPUT_FILE} 파일에서 YouTube URL을 찾는 중...")
    
    if not INPUT_FILE.exists():
        print(f"❌ {INPUT_FILE} 파일을 찾을 수 없습니다.")
        return
    
    # YouTube URL 추출
    try:
        urls = extract_youtube_urls(INPUT_FILE)
        print(f"✅ {len(urls)}개의 YouTube URL을 발견했습니다.")
        
        if not urls:
            print("❌ YouTube URL을 찾을 수 없습니다.")
            return
            
    except Exception as e:
        print(f"❌ 파일 읽기 오류: {e}")
        return
    
    # 자막 수집 결과를 저장할 변수
    all_transcripts = []
    success_count = 0
    error_count = 0
    
    print(f"\n🎬 {len(urls)}개 영상의 자막을 수집합니다...")
    print("="*60)
    
    for i, url in enumerate(urls, 1):
        print(f"\n[{i:2d}/{len(urls)}] 처리 중: {url[:60]}...")
        
        try:
            # 비디오 ID 추출
            video_id = extract_video_id(url)
            print(f"   📹 Video ID: {video_id}")
            
            # 자막 추출
            transcript_text = fetch_transcript(video_id, language="ko")
            
            if transcript_text.startswith("Error"):
                print(f"   ❌ {transcript_text}")
                error_count += 1
            else:
                # 텍스트 정리
                cleaned_text = normalize_text(transcript_text)
                print(f"   ✅ 성공 - 길이: {len(cleaned_text)} 문자")
                print(f"   📝 미리보기: {cleaned_text[:80]}...")
                success_count += 1
                
                # 결과 저장
                transcript_data = {
                    'index': i,
                    'video_id': video_id,
                    'url': url,
                    'raw_transcript': transcript_text,
                    'cleaned_transcript': cleaned_text,
                    'character_count': len(cleaned_text),
                    'timestamp': datetime.now().isoformat()
                }
                all_transcripts.append(transcript_data)
                
                # 개별 파일로도 저장
                individual_file = OUTPUT_DIR / f"{i:02d}_{video_id}.txt"
                individual_file.write_text(cleaned_text, encoding='utf-8')
            
            # API 호출 제한을 위한 대기
            time.sleep(0.5)
            
        except Exception as e:
            print(f"   ❌ 오류: {e}")
            error_count += 1
            continue
    
    # 전체 결과 파일로 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. JSON 파일로 저장
    json_file = OUTPUT_DIR / f"all_transcripts_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_transcripts, f, ensure_ascii=False, indent=2)
    
    # 2. CSV 요약 파일
    csv_file = OUTPUT_DIR / f"transcripts_summary_{timestamp}.csv"
    if all_transcripts:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['index', 'video_id', 'url', 'character_count', 'timestamp'])
            writer.writeheader()
            for transcript in all_transcripts:
                writer.writerow({
                    'index': transcript['index'],
                    'video_id': transcript['video_id'],
                    'url': transcript['url'],
                    'character_count': transcript['character_count'],
                    'timestamp': transcript['timestamp']
                })
    
    # 3. 통합 텍스트 파일
    combined_file = OUTPUT_DIR / f"all_transcripts_combined_{timestamp}.txt"
    with open(combined_file, 'w', encoding='utf-8') as f:
        for transcript in all_transcripts:
            f.write(f"{'='*60}\n")
            f.write(f"[{transcript['index']:02d}] {transcript['video_id']}\n")
            f.write(f"URL: {transcript['url']}\n")
            f.write(f"{'='*60}\n")
            f.write(f"{transcript['cleaned_transcript']}\n\n")
    
    # 최종 결과 출력
    print(f"\n{'='*60}")
    print(f"🎉 자막 수집 완료!")
    print(f"✅ 성공: {success_count}개")
    print(f"❌ 실패: {error_count}개")
    print(f"📁 저장된 파일:")
    print(f"   📄 JSON: {json_file.name}")
    print(f"   📊 CSV: {csv_file.name}")
    print(f"   📝 통합: {combined_file.name}")
    print(f"   📂 개별: {OUTPUT_DIR}/*.txt")
    
    # 성공률 출력
    if len(urls) > 0:
        success_rate = (success_count / len(urls)) * 100
        print(f"📈 성공률: {success_rate:.1f}%")
    
    return all_transcripts

if __name__ == "__main__":
    print("🚀 YouTube 대량 자막 수집기 시작")
    print("📦 필요한 패키지: pip install youtube-transcript-api")
    print(f"📂 입력 파일: youtube_urls.txt")
    print(f"📁 출력 폴더: transcripts_output")
    print()
    
    # 실행
    transcripts = main()
    
    if transcripts:
        print(f"\n📊 수집된 자막 데이터:")
        print(f"   총 {len(transcripts)}개 영상")
        
        # 언어별 통계
        lang_stats = {}
        total_chars = 0
        for t in transcripts:
            # 언어 정보 추출
            lang_match = re.match(r'\[([^\]]*)\]', t['cleaned_transcript'])
            lang = lang_match.group(1) if lang_match else 'unknown'
            lang_stats[lang] = lang_stats.get(lang, 0) + 1
            total_chars += t['character_count']
        
        print(f"   총 문자수: {total_chars:,}자")
        print(f"   언어별 분포:")
        for lang, count in sorted(lang_stats.items()):
            print(f"     {lang}: {count}개")
    
    print(f"\n💡 사용 팁:")
    print(f"   - transcripts 변수에 모든 데이터가 저장됨")
    print(f"   - transcripts_output 폴더에서 결과 파일 확인")
    print(f"   - 개별 txt 파일들을 후처리에 활용 가능")