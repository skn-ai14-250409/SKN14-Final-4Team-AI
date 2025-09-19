"""
Pinecone Namespace별 검색 테스트
=============================
- PINECONE_INDEX_STYLE에서 namespace별로 검색
- "여름 출근룩" 쿼리로 테스트
- 검색 결과 출력 및 비교
"""

import os
import json
from typing import List, Dict, Any
import logging
from dotenv import load_dotenv
from pinecone import Pinecone
import openai

# -----------------------------
# 로깅 설정
# -----------------------------
def setup_logging() -> logging.Logger:
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# -----------------------------
# 검색 테스트 클래스
# -----------------------------
class PineconeSearchTester:
    """Pinecone 검색 테스트 클래스"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index(os.getenv("PINECONE_INDEX_STYLE"))
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
    
    def search_namespace(self, query: str, namespace: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """특정 namespace에서 검색"""
        try:
            self.logger.info(f"\n=== {namespace} namespace 검색 ===")
            query_embedding = self.get_embedding(query)
            
            # summary namespace인 경우 메타데이터 필터링 적용
            if namespace == 'summary':
                filter_conditions = {
                    "gender": "남성",
                    "occasion": {"$in": ["비즈니스", "일상"]},
                    "season": {"$in": ["여름"]}
                }
                
                self.logger.info(f"메타데이터 필터 적용: {filter_conditions}")
                
                results = self.index.query(
                    vector=query_embedding,
                    namespace=namespace,
                    top_k=top_k,
                    include_metadata=True,
                    filter=filter_conditions
                )
            else:
                results = self.index.query(
                    vector=query_embedding,
                    namespace=namespace,
                    top_k=top_k,
                    include_metadata=True
                )
            
            self.logger.info(f"검색 완료: {len(results.matches)}개 결과")
            return results.matches
            
        except Exception as e:
            self.logger.error(f"{namespace} namespace 검색 실패: {e}")
            return []
    
    def print_search_results(self, results: List[Dict[str, Any]], namespace: str):
        """검색 결과 출력"""
        print(f"\n{'='*60}")
        print(f"{namespace.upper()} NAMESPACE 검색 결과")
        print(f"{'='*60}")
        
        if not results:
            print("검색 결과가 없습니다.")
            return
        
        for i, match in enumerate(results, 1):
            print(f"\n[결과 {i}]")
            print(f"ID: {match.id}")
            print(f"Score: {match.score:.4f}")
            
            if 'metadata' in match:
                metadata = match['metadata']
                print(f"Index: {metadata.get('index', 'N/A')}")
                print(f"Video ID: {metadata.get('video_id', 'N/A')}")
                print(f"URL: {metadata.get('url', 'N/A')}")
                print(f"Text Type: {metadata.get('text_type', 'N/A')}")
                
                # 메타데이터 필터 정보 출력
                if 'gender' in metadata:
                    print(f"Gender: {metadata.get('gender', 'N/A')}")
                if 'season' in metadata:
                    print(f"Season: {metadata.get('season', 'N/A')}")
                if 'occasion' in metadata:
                    print(f"Occasion: {metadata.get('occasion', 'N/A')}")
                
                # 청크 정보 출력 (cleaned namespace인 경우)
                if 'chunk_index' in metadata:
                    print(f"Chunk: {metadata.get('chunk_index', 'N/A')}/{metadata.get('total_chunks', 'N/A')}")
                
                # 요약 길이 정보 출력 (summary namespace인 경우)
                if 'summary_length' in metadata:
                    print(f"Summary Length: {metadata.get('summary_length', 'N/A')}자")
    
    def test_all_namespaces(self, query: str, top_k: int = 5):
        """모든 namespace에서 검색 테스트"""
        self.logger.info(f"검색 쿼리: '{query}'")
        self.logger.info(f"Top-K: {top_k}")
        
        namespaces = ['cleaned', 'summary']
        all_results = {}
        
        for namespace in namespaces:
            results = self.search_namespace(query, namespace, top_k)
            all_results[namespace] = results
            self.print_search_results(results, namespace)
        
        # 결과 비교
        self.compare_results(all_results)
    
    def compare_results(self, all_results: Dict[str, List[Dict[str, Any]]]):
        """namespace별 결과 비교"""
        print(f"\n{'='*60}")
        print("NAMESPACE별 결과 비교")
        print(f"{'='*60}")
        
        for namespace, results in all_results.items():
            print(f"\n{namespace.upper()} namespace:")
            print(f"  결과 수: {len(results)}")
            
            if results:
                # 점수 분포
                scores = [match.score for match in results]
                print(f"  최고 점수: {max(scores):.4f}")
                print(f"  최저 점수: {min(scores):.4f}")
                print(f"  평균 점수: {sum(scores)/len(scores):.4f}")
                
                # 인덱스 분포
                indices = [match.metadata.get('index') for match in results if 'metadata' in match]
                print(f"  검색된 인덱스: {indices}")
                
                # 메타데이터 분포 (summary namespace인 경우)
                if namespace == 'summary' and results:
                    genders = [match.metadata.get('gender') for match in results if 'metadata' in match]
                    seasons = [match.metadata.get('season') for match in results if 'metadata' in match]
                    occasions = [match.metadata.get('occasion') for match in results if 'metadata' in match]
                    
                    # 리스트인 경우 평탄화
                    flat_seasons = []
                    for season in seasons:
                        if isinstance(season, list):
                            flat_seasons.extend(season)
                        else:
                            flat_seasons.append(season)
                    
                    flat_occasions = []
                    for occasion in occasions:
                        if isinstance(occasion, list):
                            flat_occasions.extend(occasion)
                        else:
                            flat_occasions.append(occasion)
                    
                    print(f"  성별 분포: {list(set(genders))}")
                    print(f"  계절 분포: {list(set(flat_seasons))}")
                    print(f"  상황 분포: {list(set(flat_occasions))}")
                    print(f"  [필터 적용: 남성, 여름, 비즈니스/일상]")

# -----------------------------
# 메인 실행
# -----------------------------
def main():
    """메인 실행 함수"""
    load_dotenv()
    
    # 로깅 설정
    logger = setup_logging()
    
    # 설정 검증
    required_env_vars = ['PINECONE_API_KEY', 'PINECONE_INDEX_STYLE', 'OPENAI_API_KEY', 'EMBED_MODEL']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"필수 환경변수가 설정되지 않았습니다: {missing_vars}")
        return
    
    # 검색 테스트 실행
    tester = PineconeSearchTester(logger)
    
    # 테스트 쿼리
    query = "남성 여름 출근룩"
    top_k = 5
    
    logger.info("Pinecone Namespace별 검색 테스트 시작")
    logger.info(f"인덱스: {os.getenv('PINECONE_INDEX_STYLE')}")
    logger.info(f"임베딩 모델: {os.getenv('EMBED_MODEL')}")
    
    try:
        tester.test_all_namespaces(query, top_k)
        logger.info("검색 테스트 완료")
        
    except Exception as e:
        logger.error(f"검색 테스트 실패: {e}")
        raise

if __name__ == "__main__":
    main()