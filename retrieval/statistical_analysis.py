"""
통계적 성능 분석
===============
- golden_queries_300.csv에 대한 쿼리별 성능 분석
- Wilcoxon 검정, 평균 개선폭, 효과크기 계산
- cleaned vs summary namespace 비교
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from scipy import stats
from scipy.stats import wilcoxon
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
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

# -----------------------------
# 메트릭 계산 함수들
# -----------------------------
def parse_relevant(relevant_str: str) -> List[int]:
    """relevant_index 파싱"""
    if pd.isna(relevant_str) or not relevant_str:
        return []
    
    # 다양한 구분자 처리
    if ';' in relevant_str:
        return [int(x.strip()) for x in relevant_str.split(';') if x.strip()]
    elif ',' in relevant_str:
        return [int(x.strip()) for x in relevant_str.split(',') if x.strip()]
    else:
        return [int(x.strip()) for x in relevant_str.split() if x.strip()]

def compute_metrics(predicted: List[int], relevant: List[int], k: int = 5) -> Tuple[float, float, float, float]:
    """P@k, R@k, MRR, MAP 계산"""
    if not relevant:
        return 0.0, 0.0, 0.0, 0.0
    
    # P@k
    pred_k = predicted[:k]
    relevant_pred = set(predicted) & set(relevant)
    precision = len(relevant_pred) / k if k > 0 else 0.0
    
    # R@k
    recall = len(relevant_pred) / len(relevant) if relevant else 0.0
    
    # MRR
    reciprocal_rank = 0.0
    for i, doc in enumerate(predicted):
        if doc in relevant:
            reciprocal_rank = 1.0 / (i + 1)
            break
    
    # MAP
    if not relevant_pred:
        average_precision = 0.0
    else:
        precisions = []
        for i, doc in enumerate(predicted):
            if doc in relevant:
                relevant_so_far = set(predicted[:i+1]) & set(relevant)
                precisions.append(len(relevant_so_far) / (i + 1))
        average_precision = sum(precisions) / len(relevant) if relevant else 0.0
    
    return precision, recall, reciprocal_rank, average_precision

def per_query_metrics(method_results: Dict[int, List[int]], queries_df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """쿼리별 메트릭 계산"""
    rows = []
    for _, row in queries_df.iterrows():
        qid = row['query_id']
        rel = parse_relevant(row['relevant_index'])
        preds = method_results.get(qid, [])  # KeyError 방지
        p, r, rr, ap = compute_metrics(preds, rel, k)
        rows.append({
            "query_id": qid,
            "P@k": p,
            "R@k": r,
            "MRR": rr,
            "MAP": ap
        })
    return pd.DataFrame(rows)

# -----------------------------
# 통계 분석 클래스
# -----------------------------
class StatisticalAnalyzer:
    """통계적 성능 분석 클래스"""
    
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
    
    def search_namespace(self, query: str, namespace: str, top_k: int = 5) -> List[int]:
        """특정 namespace에서 검색하여 인덱스 리스트 반환"""
        try:
            query_embedding = self.get_embedding(query)
            
            # summary namespace인 경우 메타데이터 필터링 적용
            if namespace == 'summary':
                filter_conditions = {
                    "gender": {"$eq": "남성"},
                    "occasion": {"$in": ["비즈니스", "일상"]},
                    "season": {"$in": ["여름"]}
                }
                
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
            
            # 검색된 문서의 인덱스 추출
            indices = []
            for match in results.matches:
                if 'metadata' in match and 'index' in match['metadata']:
                    indices.append(match['metadata']['index'])
            
            return indices
            
        except Exception as e:
            self.logger.error(f"{namespace} namespace 검색 실패: {e}")
            return []
    
    def evaluate_queries(self, queries_df: pd.DataFrame, namespace: str, top_k: int = 5) -> Dict[int, List[int]]:
        """모든 쿼리에 대해 검색 결과 수집"""
        results = {}
        
        for _, row in queries_df.iterrows():
            qid = row['query_id']
            query = row['query_text']
            
            indices = self.search_namespace(query, namespace, top_k)
            results[qid] = indices
            
            if len(results) % 50 == 0:
                self.logger.info(f"진행률: {len(results)}/{len(queries_df)}")
        
        return results
    
    def compute_statistical_tests(self, df_cleaned: pd.DataFrame, df_summary: pd.DataFrame) -> Dict[str, Any]:
        """통계적 검정 수행"""
        # 같은 쿼리들만 페어링
        paired = df_cleaned.merge(df_summary, on="query_id", suffixes=("_cleaned", "_summary"))
        
        # 메트릭별 배열 준비
        metrics = ['P@k', 'R@k', 'MRR', 'MAP']
        results = {}
        
        for metric in metrics:
            cleaned_col = f"{metric}_cleaned"
            summary_col = f"{metric}_summary"
            
            cleaned_values = paired[cleaned_col].to_numpy()
            summary_values = paired[summary_col].to_numpy()
            
            # Wilcoxon 검정
            try:
                statistic, p_value = wilcoxon(cleaned_values, summary_values, alternative='two-sided')
            except ValueError as e:
                self.logger.warning(f"{metric} Wilcoxon 검정 실패: {e}")
                statistic, p_value = np.nan, np.nan
            
            # 평균 개선폭 (summary - cleaned)
            mean_improvement = np.mean(summary_values - cleaned_values)
            
            # 효과크기 (Cohen's d)
            pooled_std = np.sqrt((np.var(cleaned_values, ddof=1) + np.var(summary_values, ddof=1)) / 2)
            cohens_d = mean_improvement / pooled_std if pooled_std > 0 else 0.0
            
            # 개선된 쿼리 비율
            improved_queries = np.sum(summary_values > cleaned_values)
            total_queries = len(paired)
            improvement_ratio = improved_queries / total_queries if total_queries > 0 else 0.0
            
            results[metric] = {
                'wilcoxon_statistic': statistic,
                'wilcoxon_p_value': p_value,
                'mean_improvement': mean_improvement,
                'cohens_d': cohens_d,
                'improvement_ratio': improvement_ratio,
                'cleaned_mean': np.mean(cleaned_values),
                'summary_mean': np.mean(summary_values),
                'cleaned_std': np.std(cleaned_values),
                'summary_std': np.std(summary_values)
            }
        
        return results, paired
    
    def print_statistical_results(self, results: Dict[str, Any], paired: pd.DataFrame):
        """통계 결과 출력"""
        print("\n" + "="*80)
        print("통계적 성능 분석 결과")
        print("="*80)
        
        for metric, stats in results.items():
            print(f"\n{metric}:")
            print(f"  Wilcoxon 검정:")
            print(f"    통계량: {stats['wilcoxon_statistic']:.4f}")
            print(f"    p-value: {stats['wilcoxon_p_value']:.4f}")
            print(f"    유의성: {'유의함' if stats['wilcoxon_p_value'] < 0.05 else '유의하지 않음'}")
            
            print(f"  평균 성능:")
            print(f"    Cleaned: {stats['cleaned_mean']:.4f} ± {stats['cleaned_std']:.4f}")
            print(f"    Summary: {stats['summary_mean']:.4f} ± {stats['summary_std']:.4f}")
            
            print(f"  개선 분석:")
            print(f"    평균 개선폭: {stats['mean_improvement']:.4f}")
            print(f"    효과크기 (Cohen's d): {stats['cohens_d']:.4f}")
            print(f"    개선된 쿼리 비율: {stats['improvement_ratio']:.2%}")
            
            # 효과크기 해석
            effect_size = abs(stats['cohens_d'])
            if effect_size < 0.2:
                effect_interpretation = "작은 효과"
            elif effect_size < 0.5:
                effect_interpretation = "중간 효과"
            elif effect_size < 0.8:
                effect_interpretation = "큰 효과"
            else:
                effect_interpretation = "매우 큰 효과"
            
            print(f"    효과크기 해석: {effect_interpretation}")
        
        print(f"\n총 쿼리 수: {len(paired)}")
        print("="*80)

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
    
    # Golden queries 로드
    golden_queries_path = os.getenv("GOLDEN_QUERIES_PATH", "./golden_queries_300.csv")
    
    try:
        queries_df = pd.read_csv(golden_queries_path)
        logger.info(f"Golden queries 로드 완료: {len(queries_df)}개 쿼리")
    except Exception as e:
        logger.error(f"Golden queries 로드 실패: {e}")
        return
    
    # 통계 분석기 초기화
    analyzer = StatisticalAnalyzer(logger)
    
    # 각 namespace에 대해 검색 수행
    logger.info("Cleaned namespace 검색 시작...")
    cleaned_results = analyzer.evaluate_queries(queries_df, 'cleaned', top_k=5)
    
    logger.info("Summary namespace 검색 시작...")
    summary_results = analyzer.evaluate_queries(queries_df, 'summary', top_k=5)
    
    # 쿼리별 메트릭 계산
    logger.info("쿼리별 메트릭 계산 중...")
    df_cleaned = per_query_metrics(cleaned_results, queries_df, k=5)
    df_summary = per_query_metrics(summary_results, queries_df, k=5)
    
    # 통계적 검정 수행
    logger.info("통계적 검정 수행 중...")
    results, paired = analyzer.compute_statistical_tests(df_cleaned, df_summary)
    
    # 결과 출력
    analyzer.print_statistical_results(results, paired)
    
    # 결과를 CSV로 저장
    output_file = "statistical_analysis_results.csv"
    paired.to_csv(output_file, index=False)
    logger.info(f"상세 결과가 {output_file}에 저장되었습니다.")

if __name__ == "__main__":
    main()