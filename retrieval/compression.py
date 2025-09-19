from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal

from dotenv import load_dotenv

# LangChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# ==================================================
# 0) 설정
# ==================================================
load_dotenv()

@dataclass
class Config:
    """파이프라인 설정"""
    # 파일 경로
    input_path: Path = Path("./style_rules/transcripts_output/all_transcripts_20250909.json")
    output_path: Path = Path("./style_rules/transcripts_output/all_transcripts_20250918_metadata.json")

    # LLM 설정
    model_name: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    temperature: float = 0.3

    # 텍스트 처리
    max_chars: int = 8000         # 한 번에 모델로 보낼 최대 길이(문자)
    chunk_size: int = 3500        # 청크 크기(문자)
    chunk_overlap: int = 300      # 청크 겹침(문자)

    # 기타
    verbose: bool = True


# ==================================================
# 1) 텍스트 전처리 (클린업)
# ==================================================
_BRACKET_NOISE = re.compile(r"\[[^\]]*\]")  # [음악], [ko], [박수] 등
_PAREN_SFX = re.compile(r"\((?:[^)]{0,80}?(?:음악|박수|웃음|효과음|bgm|music|applause|sfx)[^)]{0,80})\)", re.IGNORECASE)
_TIMESTAMPS = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")  # 1:23 또는 01:23:45
_MULTI_SPACE = re.compile(r"[ \t\u00A0]+")


def clean_transcript(text: str) -> str:
    """기초 노이즈 제거 + 공백 정규화."""
    if not text:
        return ""
    s = _BRACKET_NOISE.sub(" ", text)
    s = _PAREN_SFX.sub(" ", s)
    s = _TIMESTAMPS.sub(" ", s)
    s = _MULTI_SPACE.sub(" ", s)
    return s.strip()


# ==================================================
# 2) 요약 프롬프트(한 단락, 500자)
# ==================================================
SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "자막을 읽고 전체 맥락과 타겟 성별을 정확하게 파악해 내용을 요약 및 전달하는 **커뮤니케이션 전문가**입니다."
        "주어진 텍스트(대화/자막)에서 특정 대상에게 상황/계절에 맞는 코디를 제안하는 내용만을 추려, 아래 요구사항에 따라 요약하세요.\n"
        "요구사항:\n"
        "- 반드시 대상 정보 포함(특히 **성별**; 가능하면 연령대/체형/피부 톤/성격 등)\n"
        "- 상황(데이트/결혼식/비즈니스/캠핑/일상 등)과 계절(봄/여름/가을/겨울/간절기/연중무관) 명시\n"
        "- 아이템(상의/하의/아우터/원피스 등)과 조합, 소재·컬러·패턴·실루엣 등 구체 특징 제안\n"
        "- 다음은 금지: 사회적 인삿말/메타발화(이 글은~, 이 텍스트는~ 등)/잡담, 대괄호·효과음 표기, 불릿/번호, 과장·추측·지어내기, 텍스트에 없는 워딩 사용\n"
        "- 톤: 중립·명료, 사실 위주, 부드러운 제안형 표현\n"
        "출력 형식: 한국어, 한 문단, 총 길이 500자 미만, 결과 텍스트만 출력"
    )),
    ("human", "다음 텍스트를 한 단락으로 요약해 코디 제안으로 작성하세요:\n{text}")
])


# ==================================================
# 3) 메타데이터 스키마 & 프롬프트(JSON 강제)
# ==================================================
Gender = Literal["남성", "여성"]
Season = Literal["봄", "여름", "가을", "겨울", "연중무관"]
Occasion = Literal["데이트", "일상", "비즈니스", "결혼식", "캠핑"]


class MetadataModel(BaseModel):
    gender: List[Gender] = Field(..., description="추천 대상 성별(복수 가능). 불명확하면 남성, 여성 포함 가능")
    season: List[Season] = Field(..., description="계절(복수 가능). 사계절 모두면 연중무관 권장")
    occasion: List[Occasion] = Field(..., description="상황/장소(복수 가능)")


metadata_parser = JsonOutputParser(pydantic_object=MetadataModel)

_NORMALIZE_RULES = (
    "정규화 규칙:\n"
    "- 성별 무관/남녀공용/커플룩 등 → gender: ['남성','여성']\n"
    "- 시즌:\n"
    "  · 'S/S','봄여름' → ['봄','여름']\n"
    "  · 'F/W','가을겨울' → ['가을','겨울']\n"
    "  · '간절기' → ['봄','가을']\n"
    "  · '여름철','한여름','초여름','장마철' → ['여름']\n"
    "  · '여름 전까지' → ['봄']\n"
    "  · '가을부터 봄까지' → ['가을','겨울','봄']\n"
    "  · '사계절','올시즌','All-season','연중' → ['연중무관']\n"
    "- 상황:\n"
    "  · '소개팅','나들이' → 데이트\n"
    "  · '웨딩','하객','하객룩','결혼식' → 결혼식\n"
    "  · '오피스','출근','격식','직장','면접','미팅','회의','세미포멀' → 비즈니스\n"
    "  · '캠핑','아웃도어','하이킹','트레킹','등산','자연속활동' → 캠핑\n"
    "  · '데일리','캐주얼','일상','마실','주말룩','가벼운외출','모임' → 일상\n"
    "- 금지: '미상','무관' 등의 비허용 값. 불명확하면 합리적으로 복수 선택"
)

METADATA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "당신은 글의 본질을 빠르게 파악해 메타데이터를 추출·정규화하는 분석가입니다.\n"
        "입력 요약문에서 gender/season/occasion을 추출하세요.\n"
        "허용 값:\n- gender: 남성 | 여성\n- season: 봄 | 여름 | 가을 | 겨울 | 연중무관\n- occasion: 데이트 | 일상 | 비즈니스 | 결혼식 | 캠핑\n\n"
        f"{_NORMALIZE_RULES}\n"
        "출력은 반드시 JSON 한 개로만, 아래 스키마와 일치해야 합니다."
    )),
    ("human", (
        "요약문:\n{summary}\n\n"
        "출력 지침:\n{format_instructions}"
    )),
]).partial(format_instructions=metadata_parser.get_format_instructions())


# ==================================================
# 4) 체인 생성 & 긴 텍스트 요약 유틸
# ==================================================

def setup_llm_chains(config: Config):
    llm = ChatOpenAI(model=config.model_name, temperature=config.temperature)
    return {
        "llm": llm,
        "summary_chain": SUMMARY_PROMPT | llm | StrOutputParser(),
        "metadata_chain": METADATA_PROMPT | llm | metadata_parser,
    }


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    if size <= 0:
        return [text]
    step = max(1, size - max(0, overlap))
    return [text[i : i + size] for i in range(0, len(text), step)]


def summarize_safe(text: str, summary_chain, cfg: Config) -> str:
    if not text:
        return ""
    if len(text) <= cfg.max_chars:
        return summary_chain.invoke({"text": text})
    parts = []
    for ch in chunk_text(text, cfg.chunk_size, cfg.chunk_overlap):
        parts.append(summary_chain.invoke({"text": ch}))
    merged = " \n".join(parts)
    return summary_chain.invoke({"text": merged})


# ==================================================
# 5) I/O & 통계 유틸
# ==================================================
ALL_SEASONS = {"봄", "여름", "가을", "겨울"}


def load_items(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_items(path: Path, items: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def to_joined_str(values: List[str], kind: str | None = None) -> str:
    if isinstance(values, list):
        if kind == "season" and set(values) == ALL_SEASONS:
            return "연중무관"
        values = ", ".join(dict.fromkeys([v for v in values if v]))
    return values if (isinstance(values, str) and values.strip()) else "미상"


def stats_count(items: List[Dict]):
    gender_stats: Dict[str, int] = {}
    season_stats: Dict[str, int] = {}
    occasion_stats: Dict[str, int] = {}

    for it in items:
        md = it.get("metadata", {})
        g = md.get("gender", "미상")
        s = md.get("season", "미상")
        o = md.get("occasion", "미상")
        gender_stats[g] = gender_stats.get(g, 0) + 1
        season_stats[s] = season_stats.get(s, 0) + 1
        occasion_stats[o] = occasion_stats.get(o, 0) + 1

    return gender_stats, season_stats, occasion_stats


# ==================================================
# 6) 메인 파이프라인
# ==================================================

def run(cfg: Config):
    chains = setup_llm_chains(cfg)
    summary_chain = chains["summary_chain"]
    metadata_chain = chains["metadata_chain"]

    items = load_items(cfg.input_path)

    for item in items:
        idx = item.get("index")
        vid = item.get("video_id")
        url = item.get("url")
        raw = item.get("raw_transcript", "")

        clean = clean_transcript(raw)
        item["cleaned_transcript"] = clean

        if cfg.verbose:
            print(f"\n=== Index {idx} ===")
            print(f"Video ID: {vid}")
            print(f"URL: {url}")
            print(f"원본 길이: {len(raw)} → 정제 후: {len(clean)}")

        # 요약 (안전 요약)
        summary = summarize_safe(clean, summary_chain, cfg) if clean else ""
        item["summary"] = summary
        item["summary_length"] = len(summary)

        if cfg.verbose:
            print(f"요약 길이: {len(summary)}")
            if summary:
                print(f"요약: {summary}")

        # 메타데이터 추출 (JsonOutputParser → Pydantic 또는 dict)
        try:
            if summary:
                meta_obj = metadata_chain.invoke({"summary": summary})
                if isinstance(meta_obj, BaseModel):
                    meta = meta_obj.dict()
                else:
                    meta = dict(meta_obj)
                md_dict = {
                    "gender":   to_joined_str(meta.get("gender"),   "gender"),
                    "season":   to_joined_str(meta.get("season"),   "season"),
                    "occasion": to_joined_str(meta.get("occasion"), "occasion"),
                }
            else:
                md_dict = {"gender": "미상", "season": "미상", "occasion": "미상"}
        except Exception as e:
            if cfg.verbose:
                print(f"메타데이터 추출 실패: {e}")
            md_dict = {"gender": "미상", "season": "미상", "occasion": "미상"}

        item["metadata"] = md_dict
        if cfg.verbose:
            print(f"메타데이터: {md_dict}")

    # 저장
    save_items(cfg.output_path, items)

    # 통계 출력
    gender_stats, season_stats, occasion_stats = stats_count(items)
    print("\n=== 메타데이터 통계 ===")
    print("성별 분포:")
    for k, v in gender_stats.items():
        print(f"  {k}: {v}개")
    print("\n계절 분포:")
    for k, v in season_stats.items():
        print(f"  {v}개 - {k}")  # 보고서 가독성용
    print("\n상황 분포:")
    for k, v in occasion_stats.items():
        print(f"  {k}: {v}개")

    print(f"\n처리 완료! 결과가 {cfg.output_path}에 저장되었습니다.")


if __name__ == "__main__":
    cfg = Config()
    run(cfg)
