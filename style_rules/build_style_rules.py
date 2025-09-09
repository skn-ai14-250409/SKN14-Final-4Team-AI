#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build Style Matching Rules from YouTube Subtitles (KO/EN)
- Copyright-safe: store ONLY (situation,item) labels + stats (no verbatim quotes)
- Input: youtube_urls.txt (one URL per line)
- Output: rules.json, synonyms.json (merged), audit_agg.csv, captions_meta.csv
"""

import re
import json
import time
import math
import csv
from pathlib import Path
from collections import defaultdict, Counter
from urllib.parse import urlparse, parse_qs

# ---- Dependencies you need to install:
# pip install youtube-transcript-api pandas rapidfuzz python-dateutil
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from rapidfuzz import fuzz, process
import pandas as pd

# ------------------------
# Config
# ------------------------
INPUT_URLS = Path("youtube_urls.txt")
OUT_RULES_JSON = Path("rules.json")
OUT_SYNS_JSON = Path("synonyms.json")         # final merged output
OUT_AUDIT_CSV = Path("audit_agg.csv")
OUT_CAP_META = Path("captions_meta.csv")

LANG_PRIORITIES = ["ko", "en"]  # try ko first, fallback en
MIN_SENT_LEN = 6                # very short fragments are noisy
MAX_SENT_LEN = 180              # overly long lines trimmed
TOP_K_SYNONYM_MATCH = 1
FUZZ_THRESHOLD = 88             # fuzzy match threshold for collapsing variants

# ---- Seed Dictionaries (extend as needed)
SEED_SYNS = {
    "items": {
        "린넨 셔츠": ["마 셔츠", "linen shirt", "린넨남방", "마남방", "linen shirts"],
        "와이드 팬츠": ["와이드핏 바지", "통바지", "wide pants", "와이드 슬랙스"],
        "슬랙스": ["dress pants", "슬렉스", "슬랙", "슬랙 팬츠"],
        "블레이저": ["자켓", "재킷", "blazer", "sport coat", "자켓(블레이저)"],
        "로퍼": ["loafer", "로우퍼"],
        "샌들": ["sandal", "샌달"],
        "버뮤다 쇼츠": ["버뮤다 반바지", "bermuda shorts", "버뮤다 팬츠"],
        "린넨 팬츠": ["linen pants", "마 바지", "린넨바지"],
        "폴로 셔츠": ["피케 셔츠", "polo shirt", "피케티", "피케"],
        "버킷햇": ["bucket hat", "벙거지"]
    },
    "situations": {
        "출근룩": ["오피스룩", "직장인룩", "비즈니스 캐주얼", "오피스", "회사룩"],
        "여름": ["여름철", "무더위", "한여름", "여름 시즌", "여름용"],
        "봄": ["봄철", "스프링"],
        "가을": ["가을철", "가을 시즌", "폴 시즌", "F/W 초입"],
        "겨울": ["겨울철", "한겨울", "겨울 시즌"],
        "면접": ["인터뷰", "채용 면접", "인턴 면접"],
        "데이트": ["소개팅", "약속룩", "데이트룩"],
        "결혼식": ["하객룩", "하객 코디"],
        "휴가": ["바캉스룩", "리조트룩", "해변룩", "휴양지"],
        "장마": ["비 오는 날", "우천", "레인 시즌"],
        "세미나": ["발표", "프레젠테이션", "컨퍼런스"]
    }
}

# Trigger words that imply recommendation/appropriateness (Korean + English)
TRIGGERS = [
    "추천", "좋다", "어울리", "입으면", "입기 좋", "필수", "강추", "정석", "무난", "괜찮",
    "추천한다", "어울린다", "고민 없이", "베스트", "여름에는", "겨울에는",
    "recommended", "great for", "perfect for", "works for", "go-to", "staple"
]

# Simple regex templates: (situation ... item), (item ... situation), (~룩엔 ~)
TEMPLATES = [
    re.compile(r"(여름|봄|가을|겨울|출근|오피스|직장인|면접|데이트|결혼식|휴가|장마|세미나)[^\n]{0,40}?(룩|에는|엔|때는|에는).{0,30}?([가-힣A-Za-z ]{2,15}(셔츠|팬츠|슬랙스|블레이저|자켓|자켓|로퍼|샌들|쇼츠|반바지|버뮤다|폴로|피케|버킷햇))"),
    re.compile(r"([가-힣A-Za-z ]{2,15}(셔츠|팬츠|슬랙스|블레이저|자켓|로퍼|샌들|쇼츠|반바지|버뮤다|폴로|피케|버킷햇))[^\n]{0,40}?(은|는|이|가).{0,40}?(여름|봄|가을|겨울|출근|오피스|직장인|면접|데이트|결혼식|휴가|장마|세미나)"),
]

# ------------------------
# Utilities
# ------------------------
def read_urls(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Create it with one YouTube URL per line.")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def extract_video_id(url: str) -> str:
    """
    Supports: https://www.youtube.com/watch?v=VIDEOID, youtu.be/VIDEOID, etc.
    """
    parsed = urlparse(url)
    if parsed.netloc in ("youtu.be", "www.youtu.be"):
        return parsed.path.strip("/")
    qs = parse_qs(parsed.query)
    if "v" in qs:
        return qs["v"][0]
    # shorts or embed
    m = re.search(r"/shorts/([A-Za-z0-9_-]{6,})", url)
    if m:
        return m.group(1)
    m = re.search(r"/embed/([A-Za-z0-9_-]{6,})", url)
    if m:
        return m.group(1)
    raise ValueError(f"Could not parse video id from {url}")

def fetch_transcript(video_id: str, lang_priorities=LANG_PRIORITIES):
    """
    Return concatenated text of transcript; if not found, return None.
    """
    for lang in lang_priorities:
        try:
            ts = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
            txt = " ".join([x["text"] for x in ts if x.get("text")])
            return txt, lang
        except (TranscriptsDisabled, NoTranscriptFound):
            continue
        except Exception:
            continue
    return None, None

def normalize_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[“”\"\'\(\)\[\]\{\}<>]+", " ", s)
    s = re.sub(r"#\w+", " ", s)  # hashtags
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"가입|구독|좋아요|알림설정|댓글|협찬|광고|쿠폰|링크|프로모션", " ", s)
    return s.strip()

def split_sentences(text: str):
    # Quick KO/EN sentence splitter
    # Split by ., !, ?, and also by '요 ' (Korean casual sentence end), with bounds.
    text = re.sub(r"([\.!?])", r"\1¶", text)
    text = re.sub(r"(요)(\s)", r"\1¶ ", text)
    parts = [p.strip() for p in text.split("¶") if p.strip()]
    out = []
    for p in parts:
        if len(p) < MIN_SENT_LEN:
            continue
        if len(p) > MAX_SENT_LEN:
            # try soft cut
            p = p[:MAX_SENT_LEN]
        out.append(p)
    return out

def build_flat_synlist(syn_map: dict):
    # returns list of (normalized_label, all_variants)
    out = []
    for norm, vs in syn_map.items():
        cand = [norm] + vs
        out.append((norm, list(dict.fromkeys([x.lower().strip() for x in cand]))))
    return out

def syn_normalize(token: str, synlists, threshold=FUZZ_THRESHOLD):
    t = token.lower().strip()
    best_norm = None
    best_score = -1
    for norm, variants in synlists:
        match, score, _ = process.extractOne(t, variants, scorer=fuzz.WRatio)
        if score > best_score:
            best_norm, best_score = norm, score
    return best_norm if best_score >= threshold else None

def contains_trigger(sent: str) -> bool:
    s = sent.lower()
    return any(trig in s for trig in [t.lower() for t in TRIGGERS])

# ------------------------
# Main Extraction
# ------------------------
def extract_pairs_from_sentence(sent: str, item_synlists, sit_synlists):
    pairs = []
    s = sent  # keep only for matching; DO NOT store s downstream

    # Regex template hits
    hits = []
    for rx in TEMPLATES:
        for m in rx.finditer(s):
            hits.append(m.groups())

    # Keyword proximity fallback: search any item word near any situation word
    # Keep window short to avoid noise
    window = 40
    # Candidate terms to test fuzzy-normalization
    item_terms = re.findall(r"[가-힣A-Za-z ]{2,15}", s)
    sit_terms = item_terms  # same pool, we’ll norm separately

    def pick_norms(terms, synlists):
        found = set()
        for t in terms:
            n = syn_normalize(t, synlists)
            if n:
                found.add(n)
        return list(found)

    norm_items = pick_norms(item_terms, item_synlists)
    norm_sits = pick_norms(sit_terms, sit_synlists)

    # Combine based on co-occurrence if a trigger is present (stronger signal)
    if contains_trigger(s):
        for it in norm_items:
            for si in norm_sits:
                # crude co-occurrence check by distance
                try:
                    pos_i = s.lower().index(it.replace(" ", "").lower()[:2])  # fuzzy light
                    pos_s = s.lower().index(si.replace(" ", "").lower()[:2])
                    if abs(pos_i - pos_s) <= window:
                        pairs.append((si, it, "trigger_cooccurrence"))
                except ValueError:
                    continue

    # From regex groups, normalize again
    for g in hits:
        gtxt = " ".join([x for x in g if x])
        # try to pull a situation and an item
        # naive: look up every word chunk against synlists
        cand = re.findall(r"[가-힣A-Za-z ]{2,15}", gtxt)
        n_items = pick_norms(cand, item_synlists)
        n_sits = pick_norms(cand, sit_synlists)
        for si in n_sits:
            for it in n_items:
                pairs.append((si, it, "regex"))

    # Dedup within sentence
    pairs = list({(si, it, src) for (si, it, src) in pairs})
    return pairs

def main():
    # Merge seed + optional user-provided synonyms.json if present
    user_syn_path = Path("synonyms.json")
    merged_syns = SEED_SYNS.copy()
    if user_syn_path.exists():
        try:
            user_syns = json.loads(user_syn_path.read_text(encoding="utf-8"))
            for k in ["items", "situations"]:
                if k in user_syns:
                    for norm, vars_ in user_syns[k].items():
                        merged = set(merged_syns[k].get(norm, [])) | set(vars_)
                        merged_syns[k][norm] = sorted(merged)
        except Exception:
            pass

    # Precompute syn-lists for fuzzy normalization
    item_synlists = build_flat_synlist(merged_syns["items"])
    sit_synlists = build_flat_synlist(merged_syns["situations"])

    urls = read_urls(INPUT_URLS)
    agg_counter = Counter()             # (situation_norm, item_norm) -> count
    src_counter = defaultdict(set)      # (situation_norm, item_norm) -> set(video_id)
    trig_counter = Counter()            # track trigger involvement ratio
    cap_meta_rows = []                  # for captions_meta.csv

    for idx, url in enumerate(urls, start=1):
        try:
            vid = extract_video_id(url)
        except Exception:
            continue

        txt, lang = fetch_transcript(vid)
        if not txt:
            cap_meta_rows.append({"video_id": vid, "lang": None, "sentences": 0})
            continue

        norm = normalize_text(txt)
        sents = split_sentences(norm)
        cap_meta_rows.append({"video_id": vid, "lang": lang, "sentences": len(sents)})

        for sent in sents:
            pairs = extract_pairs_from_sentence(sent, item_synlists, sit_synlists)
            if not pairs:
                continue

            trig = contains_trigger(sent)
            for (si, it, src) in pairs:
                key = (si, it)
                agg_counter[key] += 1
                src_counter[key].add(vid)
                if trig:
                    trig_counter[key] += 1

        # polite pacing
        time.sleep(0.2)

    # Build scored rules
    rules = []
    for (si, it), cnt in agg_counter.items():
        vids = src_counter[(si, it)]
        trig_hits = trig_counter.get((si, it), 0)
        # confidence: combine frequency, cross-video breadth, trigger strength
        # Simple bounded score in [0,1]
        freq = cnt
        breadth = len(vids)
        trig_strength = trig_hits / max(1, freq)
        # Log-scale freq, breadth to avoid dominance
        score = (math.tanh(freq / 6) * 0.45) + (math.tanh(breadth / 4) * 0.4) + (trig_strength * 0.15)
        rules.append({
            "situation_norm": si,
            "item_norm": it,
            "evidence": {
                "video_count": breadth,
                "mention_count": freq,
                "trigger_strength": round(trig_strength, 2),
                "video_ids": sorted(list(vids))[:50]  # avoid very long lists
            },
            "confidence": round(min(0.99, score), 2),
            "notes": []
        })

    # Sort rules by confidence then by video_count
    rules.sort(key=lambda x: (x["confidence"], x["evidence"]["video_count"], x["evidence"]["mention_count"]), reverse=True)

    # Save outputs
    OUT_RULES_JSON.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")

    # Save merged synonyms back
    OUT_SYNS_JSON.write_text(json.dumps(merged_syns, ensure_ascii=False, indent=2), encoding="utf-8")

    # Audit CSV
    audit_rows = []
    for r in rules:
        audit_rows.append({
            "situation": r["situation_norm"],
            "item": r["item_norm"],
            "video_count": r["evidence"]["video_count"],
            "mention_count": r["evidence"]["mention_count"],
            "trigger_strength": r["evidence"]["trigger_strength"],
            "confidence": r["confidence"]
        })
    pd.DataFrame(audit_rows).to_csv(OUT_AUDIT_CSV, index=False, encoding="utf-8")

    pd.DataFrame(cap_meta_rows).to_csv(OUT_CAP_META, index=False, encoding="utf-8")

    print(f"[DONE] Rules: {OUT_RULES_JSON}, Synonyms: {OUT_SYNS_JSON}, Audit: {OUT_AUDIT_CSV}, Captions meta: {OUT_CAP_META}")

if __name__ == "__main__":
    main()
