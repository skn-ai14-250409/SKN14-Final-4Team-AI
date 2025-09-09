#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build Style Matching Rules from YouTube Subtitles (KO/EN) - v2
- Copyright-safe: store ONLY (situation,item) labels + stats (no verbatim quotes)
- Input : youtube_urls.txt (one URL per line)
- Output: rules.json, synonyms.json (merged), audit_agg.csv, captions_meta.csv, diag_report.txt
- Improvements:
  * Robust transcript selection (manual -> generated -> translate)
  * Regex-based label matching (precompiled from synonyms)
  * Trigger optional (weighted), adjacency window pairing (+/-1)
  * Detailed diagnostics to avoid 'empty outputs'
"""

import re
import json
import time
import math
import csv
from pathlib import Path
from collections import defaultdict, Counter
from urllib.parse import urlparse, parse_qs

import pandas as pd

# pip install youtube-transcript-api rapidfuzz pandas
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from youtube_transcript_api.formatters import TextFormatter
from rapidfuzz import fuzz, process

# ------------------------
# Config
# ------------------------
INPUT_URLS = Path("youtube_urls.txt")
OUT_RULES_JSON = Path("rules.json")
OUT_SYNS_JSON = Path("synonyms.json")         # final merged output
OUT_AUDIT_CSV = Path("audit_agg.csv")
OUT_CAP_META = Path("captions_meta.csv")
OUT_DIAG = Path("diag_report.txt")

LANG_PRIORITIES = ["ko", "en"]   # prefer ko, then en
TRANSLATE_FALLBACK_TO = "ko"     # if only non-ko transcript exists, translate to ko

MIN_SENT_LEN = 5
MAX_SENT_LEN = 220
ADJ_WINDOW = 1                   # look at current sentence +/-1 for pairing context
FUZZ_THRESHOLD = 84              # for last-resort fuzzy collapse

REQUIRE_TRIGGER = False          # allow pairs without triggers (weight lower)
TRIGGER_WEIGHT = 0.2             # contribution to confidence
ADJ_PAIR_WEIGHT = 0.15           # adjacency (cross-sentence) contribution

# ---- Seed Synonyms (extend as needed) ----
SEED_SYNS = {
    "items": {
        "린넨 셔츠": ["마 셔츠", "linen shirt", "린넨남방", "마남방", "linen shirts", "linen"],
        "와이드 팬츠": ["와이드핏 바지", "통바지", "wide pants", "와이드 슬랙스"],
        "슬랙스": ["dress pants", "슬렉스", "슬랙", "슬랙 팬츠", "슬랙바지", "치노슬랙스"],
        "블레이저": ["자켓", "재킷", "blazer", "sport coat", "블레이져"],
        "로퍼": ["loafer", "로우퍼"],
        "샌들": ["sandal", "샌달"],
        "버뮤다 쇼츠": ["버뮤다 반바지", "bermuda shorts", "버뮤다 팬츠"],
        "린넨 팬츠": ["linen pants", "마 바지", "린넨바지"],
        "폴로 셔츠": ["피케 셔츠", "polo shirt", "피케티", "피케", "폴로티"],
        "버킷햇": ["bucket hat", "벙거지"],
        "화이트 셔츠": ["white shirt", "화이트셔츠", "흰 셔츠", "드레스 셔츠"],
        "오버셔츠": ["overshirt", "셔켓"],
        "A라인 스커트": ["a-line skirt", "에이라인 스커트", "플레어 스커트"],
        "미디 스커트": ["midi skirt", "미디움 스커트"],
        "맥시 스커트": ["maxi skirt", "롱 스커트", "롱스커트"],
        "셔츠 드레스": ["shirt dress", "셔츠원피스"],
        "스니커즈": ["운동화", "sneakers", "스니커"],
        "에스파드리유": ["espadrille", "에스파드릴"],
        "로우 힐": ["low heel", "블록 힐", "낮은굽"]
    },
    "situations": {
        "출근룩": ["오피스룩", "직장인룩", "비즈니스 캐주얼", "오피스", "회사룩"],
        "여름": ["여름철", "무더위", "한여름", "여름 시즌", "여름용", "여름 코디"],
        "봄": ["봄철", "스프링"],
        "가을": ["가을철", "가을 시즌", "폴 시즌", "F/W 초입"],
        "겨울": ["겨울철", "한겨울", "겨울 시즌"],
        "면접": ["인터뷰", "채용 면접", "인턴 면접"],
        "데이트": ["소개팅", "약속룩", "데이트룩"],
        "결혼식": ["하객룩", "하객 코디"],
        "휴가": ["바캉스룩", "리조트룩", "해변룩", "휴양지", "여행룩"],
        "장마": ["비 오는 날", "우천", "레인 시즌", "장마철"],
        "세미나": ["발표", "프레젠테이션", "컨퍼런스", "강연"],
        "캠퍼스": ["학교룩", "대학생룩"],
        "프레젠테이션": ["PT", "프레젠", "발표날"]
    }
}

TRIGGERS = [
    "추천", "좋다", "어울리", "입으면", "입기 좋", "필수", "강추", "정석", "무난", "괜찮",
    "추천한다", "어울린다", "고민 없이", "베스트", "여름에는", "겨울에는",
    "recommended", "great for", "perfect for", "works for", "go-to", "staple"
]

# ------------------------
# Basic IO
# ------------------------
def read_urls(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Create it with one YouTube URL per line.")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def extract_video_id(url: str) -> str:
    p = urlparse(url)
    if p.netloc in ("youtu.be", "www.youtu.be"):
        return p.path.strip("/")
    qs = parse_qs(p.query)
    if "v" in qs:
        return qs["v"][0]
    m = re.search(r"/shorts/([A-Za-z0-9_-]{6,})", url)
    if m: return m.group(1)
    m = re.search(r"/embed/([A-Za-z0-9_-]{6,})", url)
    if m: return m.group(1)
    raise ValueError(f"Could not parse video id from {url}")

# ------------------------
# Transcript helpers
# ------------------------
def fetch_transcript_best(video_id: str):
    """
    Try manual transcript in ko->en. If only other language, try translate to ko.
    Return (text, lang, source) or (None, None, None)
    """
    try:
        listing = YouTubeTranscriptApi.list_transcripts(video_id)
    except Exception:
        return None, None, None

    # 1) Prefer manual ko, then manual en
    for code in ["ko", "en"]:
        try:
            t = listing.find_manually_created_transcript([code])
            return " ".join([x["text"] for x in t.fetch() if x.get("text")]), code, "manual"
        except Exception:
            pass

    # 2) Generated ko, then generated en
    for code in ["ko", "en"]:
        try:
            t = listing.find_generated_transcript([code])
            return " ".join([x["text"] for x in t.fetch() if x.get("text")]), code, "generated"
        except Exception:
            pass

    # 3) Any available -> translate to ko (if possible)
    try:
        any_t = listing._manually_created_transcripts or listing._generated_transcripts
        any_t = list(any_t.values())
        if any_t:
            t0 = any_t[0]
            try:
                t_ko = t0.translate(TRANSLATE_FALLBACK_TO)
                return " ".join([x["text"] for x in t_ko.fetch() if x.get("text")]), TRANSLATE_FALLBACK_TO, "translated"
            except Exception:
                # fallback: use original lang
                return " ".join([x["text"] for x in t0.fetch() if x.get("text")]), t0.language_code, "fallback_original"
    except Exception:
        pass

    return None, None, None

# ------------------------
# Text processing
# ------------------------
def normalize_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"[“”\"\'\(\)\[\]\{\}<>\|•·…~^]+", " ", s)
    # remove low-value phrases (ads, subscribe prompts)
    s = re.sub(r"(구독|좋아요|알림설정|댓글|협찬|광고|쿠폰|링크|프로모션|sponsor(ed)?|like and subscribe)", " ", s, flags=re.I)
    s = re.sub(r"#\w+", " ", s)
    return s.strip()

def split_sentences(text: str):
    # KO/EN naive split: ?,!,., 요/니다/다 + safe length
    text = re.sub(r"([\.!?])", r"\1¶", text)
    text = re.sub(r"(요|니다|다)(\s)", r"\1¶ ", text)
    parts = [p.strip() for p in text.split("¶") if p.strip()]
    out = []
    for p in parts:
        if len(p) < MIN_SENT_LEN: continue
        if len(p) > MAX_SENT_LEN: p = p[:MAX_SENT_LEN]
        out.append(p)
    return out

# ------------------------
# Synonyms & Regex
# ------------------------
def merge_synonyms(seed: dict, user_path: Path) -> dict:
    merged = json.loads(json.dumps(seed, ensure_ascii=False))
    if user_path.exists():
        try:
            user = json.loads(user_path.read_text(encoding="utf-8"))
            for k in ["items", "situations"]:
                if k in user:
                    for norm, vars_ in user[k].items():
                        merged[k].setdefault(norm, [])
                        merged[k][norm] = sorted(set(merged[k][norm]) | set(vars_))
        except Exception:
            pass
    return merged

def make_label_regex_map(syn_map: dict):
    """
    For each normalized label, compile a regex that matches any synonym variant.
    KO doesn't respect \\b well; use negative char class boundaries.
    Also allow optional spaces within English/Korean tokens.
    """
    def altify(term):
        t = re.escape(term)
        # allow optional spaces between letters in English tokens (light)
        t = t.replace("\\ ", "\\s*")
        return t

    compiled = {}
    for norm, vars_ in syn_map.items():
        alts = [altify(norm)] + [altify(v) for v in vars_]
        pat = r"(?<![A-Za-z0-9가-힣])(?(?=)(?:%s))(?![A-Za-z0-9가-힣])" % ("|".join(alts))
        compiled[norm] = re.compile(pat, flags=re.I)
    return compiled

def find_labels(sentence: str, label_regex_map: dict):
    """
    Return dict: {norm_label: [positions]} where positions are match.start()
    """
    s = sentence
    found = {}
    for norm, rx in label_regex_map.items():
        for m in rx.finditer(s):
            found.setdefault(norm, []).append(m.start())
    return found

def contains_trigger(sent: str) -> bool:
    s = sent.lower()
    return any(t.lower() in s for t in TRIGGERS)

# ------------------------
# Pairing & Scoring
# ------------------------
def pair_in_sentence(sent, item_map, sit_map):
    items = list(item_map.keys())
    sits = list(sit_map.keys())
    pairs = []
    for si in sits:
        for it in items:
            # simple proximity: min distance between any positions
            dmin = min(abs(a - b) for a in sit_map[si] for b in item_map[it])
            if dmin <= 60:  # ~ 60 chars window
                pairs.append((si, it, "same_sentence"))
    return pairs

def pair_across_adjacent(prev_map, cur_map, next_map):
    """
    Allow (situation in prev, item in cur) or (situation in cur, item in next) etc.
    """
    pairs = []
    # prev x cur
    if prev_map and cur_map:
        for si in prev_map["sits"]:
            for it in cur_map["items"]:
                pairs.append((si, it, "prev_cur"))
        for si in cur_map["sits"]:
            for it in prev_map["items"]:
                pairs.append((si, it, "prev_cur"))
    # cur x next
    if cur_map and next_map:
        for si in cur_map["sits"]:
            for it in next_map["items"]:
                pairs.append((si, it, "cur_next"))
        for si in next_map["sits"]:
            for it in cur_map["items"]:
                pairs.append((si, it, "cur_next"))
    return pairs

# ------------------------
# Main
# ------------------------
def main():
    urls = read_urls(INPUT_URLS)

    # Merge synonyms (seed + user)
    merged_syns = merge_synonyms(SEED_SYNS, OUT_SYNS_JSON)
    # Precompile regex maps
    item_rx_map = make_label_regex_map(merged_syns["items"])
    sit_rx_map  = make_label_regex_map(merged_syns["situations"])

    agg_counter = Counter()            # (situation_norm, item_norm) -> mentions
    src_counter = defaultdict(set)     # -> set(video_id)
    trig_counter = Counter()           # -> trigger hits
    adj_counter  = Counter()           # -> adjacency-based hits
    cap_meta_rows = []
    diag_lines = []

    total_videos = 0
    videos_with_caps = 0
    total_sentences = 0
    total_sentence_pairs = 0

    for url in urls:
        try:
            vid = extract_video_id(url)
        except Exception:
            continue

        total_videos += 1
        txt, lang, src = fetch_transcript_best(vid)
        if not txt:
            cap_meta_rows.append({"video_id": vid, "lang": None, "source": None, "sentences": 0})
            diag_lines.append(f"{vid}: NO_TRANSCRIPT")
            continue

        videos_with_caps += 1
        norm = normalize_text(txt)
        sents = split_sentences(norm)
        total_sentences += len(sents)
        cap_meta_rows.append({"video_id": vid, "lang": lang, "source": src, "sentences": len(sents)})

        # Precompute label hits per sentence
        sent_hits = []
        for s in sents:
            sit_found = find_labels(s, sit_rx_map)
            item_found = find_labels(s, item_rx_map)
            sent_hits.append({
                "sits": set(sit_found.keys()),
                "items": set(item_found.keys()),
                "trigger": contains_trigger(s),
                # keep no raw text
            })

        # Same-sentence pairing
        for i, s in enumerate(sents):
            sit_found = find_labels(s, sit_rx_map)
            item_found = find_labels(s, item_rx_map)
            if not sit_found or not item_found:
                continue
            pairs = pair_in_sentence(s, item_found, sit_found)
            if not pairs:
                continue
            total_sentence_pairs += len(pairs)
            trig = contains_trigger(s)
            for (si, it, src_tag) in pairs:
                agg_counter[(si, it)] += 1
                src_counter[(si, it)].add(vid)
                if trig: trig_counter[(si, it)] += 1

        # Adjacent context pairing (+/-1)
        for i in range(len(sents)):
            prev_map = sent_hits[i-1] if i-1 >= 0 else None
            cur_map  = sent_hits[i]
            next_map = sent_hits[i+1] if i+1 < len(sents) else None

            pairs = pair_across_adjacent(prev_map, cur_map, next_map)
            for (si, it, src_tag) in pairs:
                agg_counter[(si, it)] += ADJ_WINDOW   # count softly
                src_counter[(si, it)].add(vid)
                adj_counter[(si, it)] += 1
                # trigger if any of the involved sentences had one
                trig = (cur_map and cur_map["trigger"]) or (prev_map and prev_map["trigger"]) or (next_map and next_map["trigger"])
                if trig: trig_counter[(si, it)] += 1

        # polite pacing
        time.sleep(0.15)

    # Build scored rules
    rules = []
    for (si, it), freq in agg_counter.items():
        vids = src_counter[(si, it)]
        trig_hits = trig_counter.get((si, it), 0)
        adj_hits = adj_counter.get((si, it), 0)

        # Confidence: bounded [0,1]
        base = math.tanh(freq / 6) * (0.55 - TRIGGER_WEIGHT - ADJ_PAIR_WEIGHT)
        conf = base + (TRIGGER_WEIGHT * min(1.0, trig_hits / max(1, freq))) + (ADJ_PAIR_WEIGHT * math.tanh(adj_hits / 6)) + (0.35 * math.tanh(len(vids) / 4))
        conf = round(min(0.99, conf), 2)

        rules.append({
            "situation_norm": si,
            "item_norm": it,
            "evidence": {
                "video_count": len(vids),
                "mention_count": int(freq) if isinstance(freq, (int, float)) else freq,
                "trigger_strength": round(trig_hits / max(1, (freq if isinstance(freq, (int,float)) else 1)), 2),
                "adjacency_hits": adj_hits,
                "video_ids": sorted(list(vids))[:50]
            },
            "confidence": conf,
            "notes": []
        })

    # Sort & save
    rules.sort(key=lambda x: (x["confidence"], x["evidence"]["video_count"], x["evidence"]["mention_count"]), reverse=True)

    OUT_RULES_JSON.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_SYNS_JSON.write_text(json.dumps(merge_synonyms(SEED_SYNS, OUT_SYNS_JSON), ensure_ascii=False, indent=2), encoding="utf-8")

    # Audit CSV
    audit_rows = []
    for r in rules:
        audit_rows.append({
            "situation": r["situation_norm"],
            "item": r["item_norm"],
            "video_count": r["evidence"]["video_count"],
            "mention_count": r["evidence"]["mention_count"],
            "trigger_strength": r["evidence"]["trigger_strength"],
            "adjacency_hits": r["evidence"]["adjacency_hits"],
            "confidence": r["confidence"]
        })
    pd.DataFrame(audit_rows).to_csv(OUT_AUDIT_CSV, index=False, encoding="utf-8")
    pd.DataFrame(cap_meta_rows).to_csv(OUT_CAP_META, index=False, encoding="utf-8")

    # Diagnostics
    diag_lines.insert(0, f"TOTAL_VIDEOS={total_videos}")
    diag_lines.insert(1, f"VIDEOS_WITH_TRANSCRIPT={videos_with_caps}")
    diag_lines.insert(2, f"TOTAL_SENTENCES={total_sentences}")
    diag_lines.insert(3, f"PAIR_MENTIONS={sum(int(x if isinstance(x,(int,float)) else 0) for x in agg_counter.values())}")
    OUT_DIAG.write_text("\n".join(diag_lines), encoding="utf-8")

    print(f"[DONE] Rules -> {OUT_RULES_JSON}")
    print(f"[DONE] Audit  -> {OUT_AUDIT_CSV}")
    print(f"[DONE] Meta   -> {OUT_CAP_META}")
    print(f"[DONE] Diag   -> {OUT_DIAG}")

if __name__ == "__main__":
    main()
