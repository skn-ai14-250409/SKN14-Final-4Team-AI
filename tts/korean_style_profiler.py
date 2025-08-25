#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Korean Subtitle Style Profiler
- Input: .srt / .vtt / .txt (UTF-8 recommended)
- Output:
    1) JSON style profile (metrics, distributions, derived scores)
    2) Prompt template for LLMs (style guide distilled from the profile)
No external libraries required.
"""
import re
import os
import json
import argparse
import statistics
from collections import Counter, defaultdict

def _read_text(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def _estimate_duration_minutes_from_srt_vtt(raw_text):
    # Match time ranges like 00:00:01,000 --> 00:00:03,000 (.srt) or 00:00:01.000 --> 00:00:03.000 (.vtt)
    tm = re.findall(r'(\\d{1,2}):(\\d{2}):(\\d{2})[\\.,](\\d{1,3})\\s*-->\\s*(\\d{1,2}):(\\d{2}):(\\d{2})[\\.,](\\d{1,3})', raw_text)
    if not tm:
        return None
    def to_ms(h, m, s, ms):
        # Normalize ms to 3 digits
        ms = (ms + '000')[:3]
        return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)
    starts = [to_ms(*t[:4]) for t in tm]
    ends   = [to_ms(*t[4:]) for t in tm]
    if not starts or not ends:
        return None
    duration_ms = max(ends) - min(starts)
    if duration_ms <= 0:
        return None
    return duration_ms / 60000.0

def _strip_subtitle_markup(raw_text):
    # Remove WEBVTT header, indexes, timestamps, tags, cue settings, speaker labels, music notes, etc.
    text = raw_text
    text = re.sub(r'^\\ufeff?', '', text)  # BOM
    text = re.sub(r'^WEBVTT.*$', ' ', text, flags=re.MULTILINE)  # VTT header
    # Remove timestamps
    text = re.sub(r'\\d{1,2}:\\d{2}:\\d{2}[\\.,]\\d{1,3}\\s*-->\\s*\\d{1,2}:\\d{2}:\\d{2}[\\.,]\\d{1,3}.*', ' ', text)
    # Remove SRT indices (pure number lines)
    text = re.sub(r'^\\s*\\d+\\s*$', ' ', text, flags=re.MULTILINE)
    # Remove HTML-like tags (e.g., <i>, <c.colorName>, etc.)
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove bracketed speaker notes (e.g., [웃음], (한숨), [남자])
    text = re.sub(r'[\\[\\(][^\\]\\)]{1,40}[\\)\\]]', ' ', text)
    # Replace music notes, entities
    text = text.replace('♪', ' ')
    text = (text.replace('&nbsp;', ' ')
                .replace('&amp;', '&')
                .replace('&lt;', '<')
                .replace('&gt;', '>'))
    # Normalize whitespace
    text = re.sub(r'[ \\t]+', ' ', text)
    text = re.sub(r'\\n{2,}', '\\n', text)
    return text.strip()

def _split_sentences(text):
    # Split by end-of-sentence punctuation and newlines as a fallback
    # Keep simple rules to avoid external dependencies
    # Sentence terminators: . ? ! … ~
    sents = re.split(r'(?<=[\\.\\?\\!…~])\\s+|\\n+', text)
    sents = [s.strip() for s in sents if s and s.strip()]
    return sents

def _tokenize(text):
    # Whitespace-based eojeol approximation; keep only "word-like" sequences
    # \\w in Python includes Hangul when UNICODE is on
    return re.findall(r'[\\w]+', text, flags=re.UNICODE)

def _latin_ratio(text):
    total = len(text)
    if total == 0:
        return 0.0
    latin = len(re.findall(r'[A-Za-z]', text))
    return latin / total

def _digit_ratio(text):
    total = len(text)
    if total == 0:
        return 0.0
    digits = len(re.findall(r'\\d', text))
    return digits / total

# Common Korean discourse markers / fillers / hedges (extend as needed)
DISCOURSE_MARKERS = [
    '근데', '그니까', '그러니까', '그래서', '즉', '한편', '하지만', '다만',
    '첫째', '둘째', '셋째', '결론적으로', '요컨대', '사실', '솔직히', '개인적으로',
    '혹은', '또는', '그리고', '그러면', '따라서', '때문에'
]
FILLERS = ['음', '어', '그', '저기', '음..', '어..', '그..', '뭐랄까', '그니까', '그니깐']
HEDGES = ['아마', '약간', '좀', '다소', '듯', '같', '수도', '가능', '대략', '어느정도', '어느 정도']

# Sentence-final patterns (rough heuristics)
ENDING_PATTERNS = [
    '습니다', '니다', '합니다', '합시다', '하십시오', '세요', '예요', '이에요',
    '어요', '아요', '해요', '하죠', '이죠', '죠', '네', '군요', '랍니다', '다', '야',
    '해라', '자', '일까요', '일까', '인가요', '인가', '겠죠', '겠어요', '겠니'
]

def _sent_ending(s):
    s = s.strip().strip('\"“”\'’‘')
    end_punct = None
    if s.endswith('?'):
        end_punct = '?'
    elif s.endswith('!'):
        end_punct = '!'
    elif s.endswith('…') or s.endswith('...'):
        end_punct = '…'
    else:
        end_punct = '.'
    # Remove trailing punctuation for ending extraction
    s2 = re.sub(r'[\\.?\\!…]+$', '', s)
    ending = None
    for pat in ENDING_PATTERNS:
        if s2.endswith(pat):
            ending = pat
            break
    return end_punct, ending

def _count_markers(tokens, vocab):
    c = Counter()
    for t in tokens:
        if t in vocab:
            c[t] += 1
    return c

def _estimate_imperative_tokens(tokens):
    # Very rough heuristic for imperatives/politeness
    # Look for endings like 하세요, 하십시오, 해라, 하자, 합시다
    joined = ' '.join(tokens)
    imp = len(re.findall(r'(하세요|하십시오|해라|하자|합시다|하지마|하지 마)', joined))
    return imp

def _profile(text, raw_text_for_duration=None):
    sents = _split_sentences(text)
    tokens = _tokenize(text)
    token_count = len(tokens)
    type_count = len(set(tokens))
    ttr = (type_count / token_count) if token_count else 0.0

    # Sentence length distribution (in tokens)
    sent_lens = []
    end_punct_counts = Counter()
    ending_counts = Counter()

    for s in sents:
        toks = _tokenize(s)
        if toks:
            sent_lens.append(len(toks))
        punc, ending = _sent_ending(s)
        end_punct_counts[punc] += 1
        if ending:
            ending_counts[ending] += 1

    avg_sent_len = statistics.mean(sent_lens) if sent_lens else 0.0
    std_sent_len = statistics.pstdev(sent_lens) if len(sent_lens) > 1 else 0.0

    # Markers and lexical signals
    disc = _count_markers(tokens, set(DISCOURSE_MARKERS))
    fil  = _count_markers(tokens, set(FILLERS))
    hed  = _count_markers(tokens, set(HEDGES))

    latin_char_ratio = _latin_ratio(text)
    digit_char_ratio = _digit_ratio(text)

    exclam_rate = (end_punct_counts['!'] / max(1, len(sents))) * 100
    quest_rate  = (end_punct_counts['?'] / max(1, len(sents))) * 100
    ellip_rate  = (end_punct_counts['…'] / max(1, len(sents))) * 100

    # Form/Politeness score (very rough): weight formal endings higher
    formal_endings = {'습니다','니다','합니다','합시다','하십시오','세요'}
    informal_endings = {'어요','아요','해요','다','야','해라','자','네','죠','군요','예요','이에요'}
    formal_count = sum(ending_counts[e] for e in formal_endings if e in ending_counts)
    informal_count = sum(ending_counts[e] for e in informal_endings if e in ending_counts)
    formality_score = (formal_count - informal_count) / max(1, (formal_count + informal_count))  # [-1, 1]

    # Imperatives
    imp_count = _estimate_imperative_tokens(tokens)

    # Duration & pace
    wpm = None
    if raw_text_for_duration:
        dur_min = _estimate_duration_minutes_from_srt_vtt(raw_text_for_duration)
        if dur_min and dur_min > 0:
            wpm = token_count / dur_min

    profile = {
        "counts": {
            "sentences": len(sents),
            "tokens": token_count,
            "types": type_count
        },
        "lexical": {
            "ttr": round(ttr, 4),
            "latin_char_ratio": round(latin_char_ratio, 4),
            "digit_char_ratio": round(digit_char_ratio, 4)
        },
        "sentence_length": {
            "avg_tokens_per_sentence": round(avg_sent_len, 2),
            "std_tokens_per_sentence": round(std_sent_len, 2)
        },
        "sentence_modes_pct": {
            "declarative_pct": round(((len(sents) - end_punct_counts['?'] - end_punct_counts['!'] - end_punct_counts['…']) / max(1, len(sents))) * 100, 2),
            "interrogative_pct": round(quest_rate, 2),
            "exclamatory_pct": round(exclam_rate, 2),
            "ellipses_pct": round(ellip_rate, 2)
        },
        "endings_top": ending_counts.most_common(15),
        "discourse_markers_per_1000t": {k: round(v / max(1, token_count) * 1000, 2) for k, v in disc.items()},
        "fillers_per_1000t": {k: round(v / max(1, token_count) * 1000, 2) for k, v in fil.items()},
        "hedges_per_1000t": {k: round(v / max(1, token_count) * 1000, 2) for k, v in hed.items()},
        "imperatives_estimated": imp_count,
        "formality_score": round(formality_score, 3),
        "pace_wpm_estimate": round(wpm, 1) if wpm else None
    }
    return profile

def _derive_style_guide(profile):
    # Turn metrics into succinct style guidance (heuristic)
    s_modes = profile.get("sentence_modes_pct", {})
    endings = profile.get("endings_top", [])
    avg_len = profile.get("sentence_length", {}).get("avg_tokens_per_sentence", 0)
    formality = profile.get("formality_score", 0)
    fillers = profile.get("fillers_per_1000t", {})
    hedges = profile.get("hedges_per_1000t", {})
    disc   = profile.get("discourse_markers_per_1000t", {})

    top_endings = [e for e, _ in endings[:5]]
    likely_register = "격식체 중심" if formality > 0.2 else ("반말/비격식 혼용" if formality < -0.2 else "중립체 혼합")

    guide = []
    guide.append(f"- 말끝(종결어미): 상위 분포 {top_endings}. 해당 종결을 우선 사용.")
    guide.append(f"- 문장 길이: 어절 기준 평균 {avg_len}±{profile['sentence_length'].get('std_tokens_per_sentence', 0)}. 이 범위를 유지.")
    guide.append(f"- 화법/형식: {likely_register} (formality_score={formality}).")
    guide.append(f"- 문장 유형 비율: 평서/의문/감탄 ≈ {s_modes.get('declarative_pct',0)}% / {s_modes.get('interrogative_pct',0)}% / {s_modes.get('exclamatory_pct',0)}%.")
    if disc:
        top_disc = sorted(disc.items(), key=lambda x: x[1], reverse=True)[:6]
        guide.append(f"- 담화 표지 우선 사용: {[k for k,_ in top_disc]} (빈도/1000t 기준 상위).")
    if fillers:
        top_fill = sorted(fillers.items(), key=lambda x: x[1], reverse=True)[:5]
        guide.append(f"- 구어적 말버릇(필러): {[k for k,_ in top_fill]} → 텍스트 용도에 맞게 유지/완화 결정.")
    if hedges:
        top_hed = sorted(hedges.items(), key=lambda x: x[1], reverse=True)[:5]
        guide.append(f"- 완곡/추정(hedges): {[k for k,_ in top_hed]} 적정 사용.")
    if profile.get("pace_wpm_estimate"):
        guide.append(f"- 발화 속도 추정: 약 {profile['pace_wpm_estimate']} WPM. 음성 합성 시 프로소디 기준치로 참고.")
    return guide

def _make_prompt_template(profile, guide):
    endings = [e for e,_ in profile.get("endings_top", [])[:5]]
    disc = sorted(profile.get("discourse_markers_per_1000t", {}).items(), key=lambda x: x[1], reverse=True)
    top_disc = [k for k,_ in disc[:6]]
    s_modes = profile.get("sentence_modes_pct", {})
    avg_len = profile.get("sentence_length", {}).get("avg_tokens_per_sentence", 0)

    template = []
    template.append("당신은 아래 '스타일 가이드'를 엄격히 따르는 한국어 글쓰기 보조자입니다.")
    template.append("출력은 1~2문단, 문장 수 2~5문장, 과장되지 않게.")
    template.append("")
    template.append("스타일 가이드:")
    template.append(f"- 종결어미 우선순위: {endings}")
    template.append(f"- 문장 길이 목표(어절): 평균 {avg_len} 내외")
    template.append(f"- 문장 유형 비율(대략): 평서 {s_modes.get('declarative_pct',0)}%, 의문 {s_modes.get('interrogative_pct',0)}%, 감탄 {s_modes.get('exclamatory_pct',0)}%")
    if top_disc:
        template.append(f"- 담화 표지 예시: {top_disc}")
    template.append("- 금지: 과도한 감탄사, 이모지, 지나친 반복")
    template.append("")
    template.append("아래 주제에 대해 위 가이드의 말투를 따르는 120~180자 내 글을 작성하세요.")
    template.append("주제: <여기에 주제 텍스트를 넣으세요>")
    return '\n'.join(template)

def main():
    ap = argparse.ArgumentParser(description="Korean Subtitle Style Profiler")
    ap.add_argument("input", help="Path to .srt / .vtt / .txt")
    ap.add_argument("--output", default="style_profile.json", help="Output JSON path")
    ap.add_argument("--prompt_out", default="style_prompt.txt", help="Output prompt template path")
    args = ap.parse_args()

    raw_text = _read_text(args.input)
    clean_text = _strip_subtitle_markup(raw_text)
    profile = _profile(clean_text, raw_text_for_duration=raw_text)
    guide = _derive_style_guide(profile)
    prompt = _make_prompt_template(profile, guide)

    out = {
        "profile": profile,
        "style_guide_bullets": guide
    }
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(args.prompt_out, 'w', encoding='utf-8') as f:
        f.write(prompt)

    print(f"[OK] Wrote {args.output} and {args.prompt_out}")
    print("Tip) Use the prompt template with your LLM, then feed the generated text to a TTS system for voice style.")

if __name__ == "__main__":
    main()
