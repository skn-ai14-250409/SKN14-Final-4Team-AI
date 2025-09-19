"""
Pinecone Namespace별 성능 평가
============================
- PINECONE_INDEX_STYLE에서 namespace별로 검색
- goldendataset.csv의 query_text로 검색
- 상위 5개 결과와 relevant_index 비교
- P@5, R@5, MRR, MAP 계산
- namespace별 성능 평가
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import logging
from dotenv import load_dotenv
from pinecone import Pinecone
import openai

# -----------------------------
# 설정 관리
# -----------------------------
@dataclass
class EvaluationConfig:
    """평가 설정"""
    golden_dataset_path: str
    namespace: str
    top_k: int = 5
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
            logging.FileHandler('evaluation.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# -----------------------------
# 메트릭 계산
# -----------------------------
def compute_precision_at_k(relevant_docs: List[int], retrieved_docs: List[int], k: int) -> float:
    """Precision@K 계산"""
    if k == 0:
        return 0.0
    
    retrieved_k = retrieved_docs[:k]
    relevant_retrieved = set(relevant_docs) & set(retrieved_k)
    return len(relevant_retrieved) / k

def compute_recall_at_k(relevant_docs: List[int], retrieved_docs: List[int], k: int) -> float:
    """Recall@K 계산"""
    if len(relevant_docs) == 0:
        return 0.0
    
    retrieved_k = retrieved_docs[:k]
    relevant_retrieved = set(relevant_docs) & set(retrieved_k)
    return len(relevant_retrieved) / len(relevant_docs)

def compute_mrr(relevant_docs: List[int], retrieved_docs: List[int]) -> float:
    """Mean Reciprocal Rank (MRR) 계산"""
    if len(relevant_docs) == 0:
        return 0.0
    
    for i, doc in enumerate(retrieved_docs):
        if doc in relevant_docs:
            return 1.0 / (i + 1)
    return 0.0

def compute_map(relevant_docs: List[int], retrieved_docs: List[int]) -> float:
    """Mean Average Precision (MAP) 계산"""
    if len(relevant_docs) == 0:
        return 0.0
    
    relevant_retrieved = []
    for i, doc in enumerate(retrieved_docs):
        if doc in relevant_docs:
            relevant_retrieved.append(i + 1)
    
    if not relevant_retrieved:
        return 0.0
    
    # Average Precision 계산
    precision_sum = 0.0
    for i, rank in enumerate(relevant_retrieved):
        precision_at_rank = (i + 1) / rank
        precision_sum += precision_at_rank
    
    return precision_sum / len(relevant_docs)

def compute_metrics(relevant_docs: List[int], retrieved_docs: List[int], k: int = 5) -> Dict[str, float]:
    """모든 메트릭 계산"""
    return {
        'P@5': compute_precision_at_k(relevant_docs, retrieved_docs, k),
        'R@5': compute_recall_at_k(relevant_docs, retrieved_docs, k),
        'MRR': compute_mrr(relevant_docs, retrieved_docs),
        'MAP': compute_map(relevant_docs, retrieved_docs)
    }

# -----------------------------
# 검색 및 평가
# -----------------------------
class PineconeEvaluator:
    """Pinecone 검색 평가 클래스"""
    
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
    
    def search(self, query: str, namespace: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Pinecone에서 검색"""
        try:
            query_embedding = self.get_embedding(query)
            results = self.index.query(
                vector=query_embedding,
                namespace=namespace,
                top_k=top_k,
                include_metadata=True
            )
            return results.matches
        except Exception as e:
            self.logger.error(f"검색 실패: {e}")
            return []
    
    def evaluate_query(self, query: str, relevant_indices: List[int], namespace: str, top_k: int = 5) -> Dict[str, float]:
        """단일 쿼리 평가"""
        # 검색 실행
        results = self.search(query, namespace, top_k)
        
        # 검색된 문서의 index 추출
        retrieved_indices = []
        for match in results:
            if 'metadata' in match and 'index' in match['metadata']:
                retrieved_indices.append(match['metadata']['index'])
        
        # 메트릭 계산
        metrics = compute_metrics(relevant_indices, retrieved_indices, top_k)
        
        self.logger.debug(f"Query: {query[:50]}...")
        self.logger.debug(f"Relevant: {relevant_indices}")
        self.logger.debug(f"Retrieved: {retrieved_indices}")
        self.logger.debug(f"Metrics: {metrics}")
        
        return metrics
    
    def evaluate_namespace(self, config: EvaluationConfig) -> Dict[str, float]:
        """특정 namespace에 대한 전체 평가"""
        self.logger.info(f"Namespace '{config.namespace}' 평가 시작")
        
        # Golden dataset 로드
        try:
            df = pd.read_csv(config.golden_dataset_path)
            self.logger.info(f"Golden dataset 로드 완료: {len(df)}개 쿼리")
        except Exception as e:
            self.logger.error(f"Golden dataset 로드 실패: {e}")
            return {}
        
        # 쿼리별 평가
        all_metrics = []
        for idx, row in df.iterrows():
            query = row['query_text']
            
            # relevant_index 파싱
            try:
                relevant_indices = row['relevant_index']
                if isinstance(relevant_indices, str):
                    # 세미콜론으로 구분된 경우
                    if ';' in relevant_indices:
                        relevant_indices = [int(x.strip()) for x in relevant_indices.split(';') if x.strip()]
                    # 쉼표로 구분된 경우
                    elif ',' in relevant_indices:
                        relevant_indices = [int(x.strip()) for x in relevant_indices.split(',') if x.strip()]
                    # 공백으로 구분된 경우
                    else:
                        relevant_indices = [int(x.strip()) for x in relevant_indices.split() if x.strip()]
                elif not isinstance(relevant_indices, list):
                    relevant_indices = [relevant_indices]
                
                # 정수로 변환
                relevant_indices = [int(x) for x in relevant_indices if str(x).strip()]
                
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Row {idx}: relevant_index 파싱 실패 - {relevant_indices}, 오류: {e}")
                relevant_indices = []
            
            metrics = self.evaluate_query(query, relevant_indices, config.namespace, config.top_k)
            all_metrics.append(metrics)
            
            if config.verbose and (idx + 1) % 10 == 0:
                self.logger.info(f"진행률: {idx + 1}/{len(df)}")
        
        # 전체 평균 계산
        avg_metrics = {}
        for metric in ['P@5', 'R@5', 'MRR', 'MAP']:
            values = [m[metric] for m in all_metrics]
            avg_metrics[metric] = np.mean(values)
        
        self.logger.info(f"Namespace '{config.namespace}' 평가 완료")
        return avg_metrics

# -----------------------------
# 메인 평가 함수
# -----------------------------
def evaluate_all_namespaces(golden_dataset_path: str, namespaces: List[str], top_k: int = 5, verbose: bool = True) -> Dict[str, Dict[str, float]]:
    """모든 namespace에 대한 평가"""
    logger = setup_logging(verbose)
    evaluator = PineconeEvaluator(logger)
    
    results = {}
    
    for namespace in namespaces:
        logger.info(f"\n=== {namespace} namespace 평가 ===")
        config = EvaluationConfig(
            golden_dataset_path=golden_dataset_path,
            namespace=namespace,
            top_k=top_k,
            verbose=verbose
        )
        
        metrics = evaluator.evaluate_namespace(config)
        results[namespace] = metrics
        
        # 결과 출력
        logger.info(f"\n{namespace} namespace 결과:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
    
    return results

def print_comparison_table(results: Dict[str, Dict[str, float]]) -> None:
    """결과 비교 테이블 출력"""
    print("\n" + "="*60)
    print("NAMESPACE별 성능 비교")
    print("="*60)
    
    # 테이블 헤더
    print(f"{'Namespace':<15} {'P@5':<8} {'R@5':<8} {'MRR':<8} {'MAP':<8}")
    print("-" * 60)
    
    # 각 namespace 결과 출력
    for namespace, metrics in results.items():
        print(f"{namespace:<15} {metrics['P@5']:<8.4f} {metrics['R@5']:<8.4f} {metrics['MRR']:<8.4f} {metrics['MAP']:<8.4f}")
    
    print("="*60)

# -----------------------------
# 메인 실행
# -----------------------------
def main():
    """메인 실행 함수"""
    load_dotenv()
    
    # 설정
    golden_dataset_path = os.getenv("GOLDEN_QUERIES_PATH")
    namespaces = ['cleaned', 'summary']  # 평가할 namespace들
    top_k = 5
    
    print("Pinecone Namespace별 성능 평가 시작")
    print(f"Golden dataset: {golden_dataset_path}")
    print(f"Namespaces: {namespaces}")
    print(f"Top-K: {top_k}")
    
    # 평가 실행
    results = evaluate_all_namespaces(golden_dataset_path, namespaces, top_k)
    
    # 결과 비교 테이블 출력
    print_comparison_table(results)
    
    # 최고 성능 namespace 찾기
    best_namespace = None
    best_score = 0
    
    for namespace, metrics in results.items():
        # MAP를 기준으로 최고 성능 결정
        if metrics['MAP'] > best_score:
            best_score = metrics['MAP']
            best_namespace = namespace
    
    print(f"\n최고 성능 namespace: {best_namespace} (MAP: {best_score:.4f})")

if __name__ == "__main__":
    main()