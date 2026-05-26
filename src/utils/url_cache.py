import json, os

CACHE_PATH = os.path.join(os.path.dirname(__file__), "url_cache.json")
AGENDA_URL_KEY = "agenda_url"

def load_cached_url() -> str | None:
    if not os.path.exists(CACHE_PATH):
        return None
    with open(CACHE_PATH) as f:
        return json.load(f).get(AGENDA_URL_KEY)

def save_cached_url(url: str) -> None:
    cache = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            cache = json.load(f)
    cache[AGENDA_URL_KEY] = url
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)

def clear_cached_url() -> None:
    if not os.path.exists(CACHE_PATH):
        return
    with open(CACHE_PATH) as f:
        cache = json.load(f)
    cache.pop(AGENDA_URL_KEY, None)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)