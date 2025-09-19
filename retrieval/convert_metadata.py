"""
Metadata 형식 변환 스크립트
========================
- all_transcripts_20250918_metadata.json의 metadata 필드 변환
- season, occasion을 문자열에서 리스트로 변경
- "일상, 데이트" -> ["일상", "데이트"]
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any

def parse_metadata_value(value: str) -> List[str]:
    """메타데이터 값을 파싱하여 리스트로 변환"""
    if not value or value == "미상":
        return ["미상"]
    
    # 쉼표로 구분된 경우
    if ',' in value:
        return [item.strip() for item in value.split(',') if item.strip()]
    
    # 세미콜론으로 구분된 경우
    if ';' in value:
        return [item.strip() for item in value.split(';') if item.strip()]
    
    # 공백으로 구분된 경우 (단, 단일 단어가 아닌 경우만)
    if ' ' in value and len(value.split()) > 1:
        return [item.strip() for item in value.split() if item.strip()]
    
    # 단일 값인 경우
    return [value.strip()]

def convert_metadata_format(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """메타데이터 형식을 변환"""
    converted_data = []
    
    for item in data:
        # 원본 아이템 복사
        converted_item = item.copy()
        
        # metadata 필드가 있는 경우 변환
        if 'metadata' in item and isinstance(item['metadata'], dict):
            metadata = item['metadata'].copy()
            
            # season 변환
            if 'season' in metadata:
                original_season = metadata['season']
                metadata['season'] = parse_metadata_value(str(original_season))
                print(f"Index {item.get('index', 'N/A')}: season '{original_season}' -> {metadata['season']}")
            
            # occasion 변환
            if 'occasion' in metadata:
                original_occasion = metadata['occasion']
                metadata['occasion'] = parse_metadata_value(str(original_occasion))
                print(f"Index {item.get('index', 'N/A')}: occasion '{original_occasion}' -> {metadata['occasion']}")
            
            converted_item['metadata'] = metadata
        
        converted_data.append(converted_item)
    
    return converted_data

def main():
    """메인 실행 함수"""
    input_file = Path("./style_rules/transcripts_output/all_transcripts_20250918_metadata.json")
    output_file = Path("./style_rules/transcripts_output/all_transcripts_20250918_metadata_converted.json")
    
    print("Metadata 형식 변환 시작")
    print(f"입력 파일: {input_file}")
    print(f"출력 파일: {output_file}")
    
    # JSON 파일 읽기
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"데이터 로드 완료: {len(data)}개 항목")
    except Exception as e:
        print(f"파일 읽기 실패: {e}")
        return
    
    # 변환 전 통계
    print("\n=== 변환 전 통계 ===")
    season_types = {}
    occasion_types = {}
    
    for item in data:
        if 'metadata' in item:
            season = item['metadata'].get('season', '')
            occasion = item['metadata'].get('occasion', '')
            
            season_types[season] = season_types.get(season, 0) + 1
            occasion_types[occasion] = occasion_types.get(occasion, 0) + 1
    
    print("Season 분포:")
    for season, count in sorted(season_types.items()):
        print(f"  '{season}': {count}개")
    
    print("\nOccasion 분포:")
    for occasion, count in sorted(occasion_types.items()):
        print(f"  '{occasion}': {count}개")
    
    # 메타데이터 형식 변환
    print("\n=== 메타데이터 변환 중 ===")
    converted_data = convert_metadata_format(data)
    
    # 변환 후 통계
    print("\n=== 변환 후 통계 ===")
    season_lists = {}
    occasion_lists = {}
    
    for item in converted_data:
        if 'metadata' in item:
            season = item['metadata'].get('season', [])
            occasion = item['metadata'].get('occasion', [])
            
            season_key = str(season)
            occasion_key = str(occasion)
            
            season_lists[season_key] = season_lists.get(season_key, 0) + 1
            occasion_lists[occasion_key] = occasion_lists.get(occasion_key, 0) + 1
    
    print("Season 리스트 분포:")
    for season, count in sorted(season_lists.items()):
        print(f"  {season}: {count}개")
    
    print("\nOccasion 리스트 분포:")
    for occasion, count in sorted(occasion_lists.items()):
        print(f"  {occasion}: {count}개")
    
    # 변환된 데이터 저장
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)
        print(f"\n변환 완료! 결과가 {output_file}에 저장되었습니다.")
    except Exception as e:
        print(f"파일 저장 실패: {e}")
        return
    
    # 샘플 출력
    print("\n=== 변환 샘플 ===")
    for i, item in enumerate(converted_data[:3]):
        if 'metadata' in item:
            print(f"\nIndex {item.get('index', 'N/A')}:")
            print(f"  Season: {item['metadata'].get('season', [])}")
            print(f"  Occasion: {item['metadata'].get('occasion', [])}")

if __name__ == "__main__":
    main()