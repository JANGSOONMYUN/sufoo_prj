import requests
import shutil
import json
import os
import re
import random
import time
import threading
from collections import deque
from typing import Any, Dict, Optional, Tuple

# JSON 파일에서 API_KEY 불러오기
with open('modules/search_img/config.json', 'r') as config_file:
    config = json.load(config_file)
    API_KEY = config['api_key']

BASE_URL = 'https://pixabay.com/api/'

def search_images(query, page=1, per_page=100, lang='ko'):
    params = {
        'key': API_KEY,
        'q': query,
        'image_type': 'photo',      # 일러스트/벡터/AI 이미지가 섞이지 않도록 사진만
        'orientation': 'all',
        'min_width': 640,
        'min_height': 0,
        'editors_choice': 'false',
        'safesearch': 'true',
        'order': 'popular',
        'page': page,
        'per_page': per_page,
        'lang': lang,
    }
    response = requests.get(BASE_URL, params=params, timeout=15)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# 이미지 선택 로직
#  - 예전: 항상 hits[0](인기순 1위) -> 비슷한 키워드마다 같은 사진이 반복됨
#  - 현재: 태그 일치도로 점수를 매긴 상위 후보 중에서, 최근에 쓰지 않은 사진을 랜덤 선택
# ---------------------------------------------------------------------------
_TOP_K = 8                 # 점수 상위 몇 개 중에서 랜덤 선택할지
_RECENT_MAX = 300          # 최근 사용한 이미지 id를 몇 개까지 기억할지(서버 전체, 사용자 무관)
_CACHE_TTL_SEC = 3600      # 같은 키워드 검색 결과 캐시(Pixabay 호출 한도 절약)
_NOISE_TAGS = ('wallpaper', 'background', '벽지', '배경', '4k', 'free background')

_lock = threading.Lock()
_recent_ids = deque(maxlen=_RECENT_MAX)
_search_cache: Dict[str, Tuple[float, list]] = {}


def _tokens(keyword: str):
    parts = re.split(r'[\s_/,+\-]+', (keyword or '').strip().lower())
    return [t for t in parts if t]


def _score_hit(hit: Dict[str, Any], tokens, rank: int) -> float:
    tags = str(hit.get('tags', '')).lower()
    matched = sum(1 for t in tokens if t in tags)
    score = matched * 10.0 - rank * 0.05      # 일치 토큰이 최우선, 인기 순위는 보조
    if any(n in tags for n in _NOISE_TAGS) and not any(n in ''.join(tokens) for n in _NOISE_TAGS):
        score -= 3.0                           # 키워드와 무관한 배경화면성 이미지 감점
    return score


def _fetch_hits(query: str, lang: str = 'ko') -> list:
    now = time.time()
    cache_key = f"{lang}:{query}"
    with _lock:
        cached = _search_cache.get(cache_key)
        if cached and now - cached[0] < _CACHE_TTL_SEC:
            return cached[1]
    ims = search_images(query, page=1, lang=lang)
    hits = [h for h in (ims.get('hits') or []) if isinstance(h, dict)] if isinstance(ims, dict) else []
    with _lock:
        if len(_search_cache) > 200:
            _search_cache.clear()
        _search_cache[cache_key] = (now, hits)
    return hits


# ---------------------------------------------------------------------------
# 한글 키워드 -> 영어 이미지 검색어 변환
#  - Pixabay는 영어 검색이 훨씬 정확함 ('남성영양제'->영양(antelope) 사진, 'omega-3'->3월 행사 등 오검색 방지)
#  - Vertex AI Express 키로 가벼운 모델(gemini-3.5-flash-lite)을 호출. 실패하면 None -> 원문 키워드로 검색
# ---------------------------------------------------------------------------
_TRANSLATE_MODEL = 'gemini-3.5-flash-lite'
_translate_cache: Dict[str, str] = {}
_TRANSLATE_PROMPT = (
    "Convert the following keyword (about food, nutrition, supplements or health) into a concrete English "
    "stock-photo search query of 1-4 words that names a visible, photographable object "
    "(e.g. '오메가3' -> 'fish oil capsules', '남성영양제' -> 'multivitamin pills', '고혈압관리' -> 'blood pressure monitor'). "
    "Prefer foods, ingredients, supplement pills, or medical objects. No abstract words. "
    "If the keyword is a rare ingredient that stock photos are unlikely to cover, use a broader generic "
    "visual category instead (e.g. 'herbal powder', 'green leaves'). "
    "Output only the query text.\n\nKeyword: "
)


def _load_google_api_key() -> Optional[str]:
    try:
        with open('settings/config.json', 'r', encoding='utf-8') as f:
            return json.load(f)['google']['api_key']
    except Exception:
        return None


def translate_query(keyword: str) -> Optional[str]:
    kw = (keyword or '').strip()
    if not kw:
        return None
    if kw.isascii():           # 이미 영어면 그대로 사용
        return kw
    with _lock:
        if kw in _translate_cache:
            return _translate_cache[kw]
    key = _load_google_api_key()
    if not key:
        return None
    try:
        r = requests.post(
            f'https://aiplatform.googleapis.com/v1/publishers/google/models/{_TRANSLATE_MODEL}:generateContent',
            params={'key': key},
            json={'contents': [{'role': 'user', 'parts': [{'text': _TRANSLATE_PROMPT + kw}]}],
                  'generationConfig': {'maxOutputTokens': 40}},
            timeout=8,
        )
        r.raise_for_status()
        parts = r.json()['candidates'][0]['content']['parts']
        text = ''.join(p.get('text', '') for p in parts).strip().strip('"\'`.').splitlines()[0].strip()
        if not text or len(text) > 60:
            return None
    except Exception as e:
        if os.environ.get("IMAGE_DEBUG", "0") in ("1", "true", "True", "yes", "YES"):
            print(f"[IMAGE] translate failed for '{kw}': {e}", flush=True)
        return None
    with _lock:
        if len(_translate_cache) > 500:
            _translate_cache.clear()
        _translate_cache[kw] = text
    return text


def _search_attempts(keyword: str):
    """(검색어, 언어) 후보. 영어 변환 -> 원문 -> 원문 토큰 일부 순으로 시도"""
    en = translate_query(keyword)
    if en:
        yield en, 'en'
    if not keyword.isascii() or not en:
        yield keyword, 'ko'
    toks = _tokens(keyword)
    if len(toks) > 1:
        yield toks[0], 'ko'
        yield toks[-1], 'ko'


def pick_image(keyword: str) -> Optional[Dict[str, Any]]:
    """keyword에 어울리면서 최근에 쓰지 않은 이미지 1개를 고른다. 없으면 None."""
    pool, tokens = [], _tokens(keyword)
    for q, lang in _search_attempts(keyword):
        hits = _fetch_hits(q, lang)
        if len(hits) > len(pool):
            pool, tokens = hits, _tokens(q)
        if len(hits) >= 5:      # 후보가 충분하면 더 단순화하지 않음
            break
    if not pool:
        return None

    with _lock:
        recent = set(_recent_ids)
    scored = sorted(
        ((_score_hit(h, tokens, i), h) for i, h in enumerate(pool)),
        key=lambda x: x[0], reverse=True,
    )
    fresh = [h for _, h in scored if h.get('id') not in recent]
    candidates = (fresh or [h for _, h in scored])[:_TOP_K]
    chosen = random.choice(candidates)
    with _lock:
        _recent_ids.append(chosen.get('id'))
    return chosen


def download_image(url, file_name):
    response = requests.get(url, stream=True, timeout=30)
    try:
        response.raise_for_status()
        with open(file_name, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
    finally:
        response.close()

def search_and_download_one_to_path(keyword: str, download_path: str) -> Dict[str, Any]:
    """
    keyword로 이미지를 검색해(pick_image) 어울리는 1개를 download_path로 다운로드합니다.

    - 반환:
      - ok: bool
      - source_url: 다운로드한 원본 이미지 URL (ok==True)
      - error: 에러 메시지 (ok==False)
    """
    try:
        chosen = pick_image(keyword)
        if not chosen:
            return {"ok": False, "error": "검색 결과가 없습니다."}

        source_url = chosen.get('largeImageURL') or chosen.get('webformatURL')
        if not source_url:
            return {"ok": False, "error": "다운로드 가능한 이미지 URL을 찾지 못했습니다."}

        os.makedirs(os.path.dirname(download_path), exist_ok=True)
        # 디버그 로그(환경변수 IMAGE_DEBUG=1일 때만)
        if os.environ.get("IMAGE_DEBUG", "0") in ("1", "true", "True", "yes", "YES"):
            print(f"[IMAGE] pixabay select: keyword='{keyword}' id={chosen.get('id')} tags='{chosen.get('tags')}' "
                  f"url='{source_url}' -> '{download_path}'", flush=True)
        download_image(source_url, download_path)
        return {"ok": True, "source_url": source_url}
    except Exception as e:
        return {"ok": False, "error": str(e)}
            
def search_and_download(keyword, num_imgs, download_dir):
    image_name_list = []
    j = 1
    for n in range(1, 2):
        ims = search_images(keyword, page=n)
        for i in range(len(ims['hits'])):
            payload = ims['hits'][i]['largeImageURL']
            filename = f"{keyword}_{j}.jpg"
            download_image(payload, os.path.join(download_dir, filename))
            image_name_list.append(filename)
            print(f"{j} URL of image: {payload}")
            j += 1
            
            if j > num_imgs:
                return image_name_list
    return image_name_list
    
