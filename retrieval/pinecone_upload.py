import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import openai
from tqdm import tqdm

# -----------------------------
# 설정 관리
# -----------------------------
@dataclass
class ProcessingConfig:
    """처리 설정"""
    input_file: Path
    verbose: bool = True

# -----------------------------
# 로깅 설정
# -----------------------------
def setup_logging(verbose: bool = True) -> logging.Logger:
    """로깅 설정"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pinecone_upload.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# -----------------------------
# Pinecone 관리
# -----------------------------
class PineconeManager:
    """Pinecone 인덱스 관리 클래스"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = None
    
    def create_index(self) -> bool:
        """인덱스 생성"""
        try:
            self.pc.create_index(
                name=os.getenv("PINECONE_INDEX_STYLE"),
                dimension=int(os.getenv("PINECONE_INDEX_DIMENSION")),
                metric=os.getenv("PINECONE_INDEX_METRIC"),
                spec=ServerlessSpec(
                    cloud=os.getenv("PINECONE_INDEX_CLOUD"),
                    region=os.getenv("PINECONE_INDEX_REGION")
                )
            )
            self.logger.info(f"✓ {os.getenv('PINECONE_INDEX_STYLE')} 인덱스 생성 완료")
            return True
        except Exception as e:
            if "already exists" in str(e):
                self.logger.info(f"✓ {os.getenv('PINECONE_INDEX_STYLE')} 인덱스가 이미 존재합니다")
                return True
            else:
                self.logger.error(f"✗ {os.getenv('PINECONE_INDEX_STYLE')} 인덱스 생성 실패: {e}")
                return False
    
    def get_index(self):
        """인덱스 연결"""
        if self.index is None:
            self.index = self.pc.Index(os.getenv("PINECONE_INDEX_STYLE"))
        return self.index
    
    def upsert_vectors(self, vectors: List[Dict[str, Any]], namespace: str) -> bool:
        """벡터 배치 업로드"""
        try:
            index = self.get_index()
            
            # 배치 크기로 나누어 업로드
            batch_size = int(os.getenv("BATCH_SIZE", "100"))
            for i in tqdm(range(0, len(vectors), batch_size), 
                         desc=f"{namespace} 업로드", disable=os.getenv("VERBOSE", "true").lower() != "true"):
                batch = vectors[i:i + batch_size]
                index.upsert(vectors=batch, namespace=namespace)
            
            self.logger.info(f"✓ {namespace} {len(vectors)}개 벡터 업로드 완료")
            return True
        except Exception as e:
            self.logger.error(f"✗ {namespace} 업로드 실패: {e}")
            return False

# -----------------------------
# 데이터 처리
# -----------------------------
class DataProcessor:
    """데이터 처리 클래스"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def get_embedding(self, text: str) -> List[float]:
        """OpenAI API를 사용하여 임베딩 생성"""
        try:
            response = self.client.embeddings.create(
                model=os.getenv("EMBED_MODEL"),
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"임베딩 생성 실패: {e}")
            raise
    
    def chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """텍스트를 지정된 크기와 오버랩으로 청킹"""
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            
            if end >= len(text):
                break
                
            start = end - overlap
        
        return chunks
    
    def process_cleaned_transcript(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """cleaned_transcript 처리 (1000자 청킹, 150자 오버랩)"""
        cleaned_transcript = item.get('cleaned_transcript', '')
        
        if not cleaned_transcript:
            return []
        
        # 1000자 단위로 청킹 (오버랩 150자)
        chunks = self.chunk_text(cleaned_transcript, 1000, 150)
        
        vectors = []
        for i, chunk in enumerate(chunks):
            vector_id = f"cleaned_{item['index']}_{i}"
            metadata = {
                'index': item['index'],
                'video_id': item['video_id'],
                'url': item['url'],
                'chunk_index': i,
                'total_chunks': len(chunks),
                'character_count': item['character_count'],
                'timestamp': item['timestamp'],
                'text_type': 'cleaned_transcript'
            }
            
            # 임베딩 생성
            embedding = self.get_embedding(chunk)
            
            vectors.append({
                'id': vector_id,
                'values': embedding,
                'metadata': metadata
            })
        
        return vectors
    
    def process_summary(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """summary 처리 (metadata 포함)"""
        summary = item.get('summary', '')
        metadata_dict = item.get('metadata', {})
        
        if not summary:
            return []
        
        # 1청크로 처리
        vector_id = f"summary_{item['index']}"
        metadata = {
            'index': item['index'],
            'video_id': item['video_id'],
            'url': item['url'],
            'summary_length': item.get('summary_length', 0),
            'timestamp': item['timestamp'],
            'text_type': 'summary',
            'gender': metadata_dict.get('gender', '미상'),
            'season': metadata_dict.get('season', '미상'),
            'occasion': metadata_dict.get('occasion', '미상')
        }
        
        # 임베딩 생성
        embedding = self.get_embedding(summary)
        
        return [{
            'id': vector_id,
            'values': embedding,
            'metadata': metadata
        }]
    

# -----------------------------
# 메인 파이프라인
# -----------------------------
def load_data(input_file: Path) -> List[Dict[str, Any]]:
    """JSON 파일 로드"""
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except Exception as e:
        raise Exception(f"데이터 로드 실패: {e}")

def create_config() -> ProcessingConfig:
    """설정 생성"""
    load_dotenv()
    
    processing_config = ProcessingConfig(
        input_file=Path(os.getenv("INPUT_FILE", "./style_rules/transcripts_output/all_transcripts_20250918_metadata_converted.json")),
        verbose=os.getenv("VERBOSE", "true").lower() == "true"
    )
    
    return processing_config

def main():
    """메인 실행 함수"""
    try:
        # 설정 로드
        processing_config = create_config()
        
        # 로깅 설정
        logger = setup_logging(processing_config.verbose)
        logger.info("Pinecone 업로드 파이프라인 시작")
        
        # 설정 검증
        if not os.getenv("PINECONE_API_KEY"):
            raise ValueError("PINECONE_API_KEY가 설정되지 않았습니다")
        
        if not processing_config.input_file.exists():
            raise ValueError(f"입력 파일이 존재하지 않습니다: {processing_config.input_file}")
        
        # Pinecone 관리자 초기화
        pinecone_manager = PineconeManager(logger)
        
        # 인덱스 생성
        logger.info("Pinecone 인덱스 생성 중...")
        if not pinecone_manager.create_index():
            raise Exception("인덱스 생성 실패")
        
        # 데이터 로드
        logger.info(f"데이터 로드 중: {processing_config.input_file}")
        data = load_data(processing_config.input_file)
        logger.info(f"총 {len(data)}개의 항목을 처리합니다.")
        
        # 데이터 처리기 초기화
        processor = DataProcessor(logger)
        
        # cleaned_transcript 처리
        logger.info("cleaned_transcript 처리 중...")
        cleaned_vectors = []
        for item in tqdm(data, desc="cleaned_transcript 처리", disable=not processing_config.verbose):
            vectors = processor.process_cleaned_transcript(item)
            cleaned_vectors.extend(vectors)
        
        logger.info(f"cleaned_transcript에서 {len(cleaned_vectors)}개의 벡터를 생성했습니다.")
        
        # summary 처리 (metadata 포함)
        logger.info("summary 처리 중...")
        summary_vectors = []
        for item in tqdm(data, desc="summary 처리", disable=not processing_config.verbose):
            vectors = processor.process_summary(item)
            summary_vectors.extend(vectors)
        
        logger.info(f"summary에서 {len(summary_vectors)}개의 벡터를 생성했습니다.")
        
        # Pinecone 업로드
        logger.info("Pinecone 업로드 시작...")
        
        # cleaned_transcript를 'cleaned' namespace에 업로드
        if cleaned_vectors:
            success = pinecone_manager.upsert_vectors(cleaned_vectors, namespace='cleaned')
            if not success:
                raise Exception("cleaned_transcript 업로드 실패")
        
        # summary를 'summary' namespace에 업로드 (metadata 포함)
        if summary_vectors:
            success = pinecone_manager.upsert_vectors(summary_vectors, namespace='summary')
            if not success:
                raise Exception("summary 업로드 실패")
        
        # 결과 요약
        print(f"\n=== 처리 완료 ===")
        print(f"인덱스: {os.getenv('PINECONE_INDEX_STYLE')}")
        print(f"cleaned namespace: {len(cleaned_vectors)}개 벡터")
        print(f"summary namespace: {len(summary_vectors)}개 벡터 (metadata 포함)")
        print(f"총 벡터 수: {len(cleaned_vectors) + len(summary_vectors)}개")
        
        logger.info("파이프라인 완료")
        
    except Exception as e:
        logger.error(f"파이프라인 실행 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    main()