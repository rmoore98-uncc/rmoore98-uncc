import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import psycopg2 
import psycopg2.extras
from psycopg2 import pool
import os
import json
from geopy.geocoders import Nominatim
from functools import lru_cache
import pandas as pd
import re
import pydeck as pdk
import time
from time import monotonic
from collections import deque
from langsmith import traceable, Client as LangSmithClient
from langsmith.run_helpers import get_current_run_tree
from langsmith.wrappers import wrap_openai

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_PASSWORD = os.getenv("PASSWORD")

client = OpenAI(api_key=OPENAI_API_KEY)
langsmith_client = LangSmithClient()

# -----------------------------
# GEOCODING (Address → Lat/Lon)
# -----------------------------
geolocator = Nominatim(user_agent="foodfinder_app", timeout=10)

# -----------------------------
# ADDRESS NORMALIZATION FOR BETTER GEOCODING
# -----------------------------
def normalize_address_for_geocoding(address):
    # Remove suite/apartment info for better geocoding
    patterns = [
        r",?\s*Suite\s*\d+\w*",  # Suite 100, Suite 100A
        r",?\s*Apt\s*\d+\w*",    # Apt 5, Apt 5B
        r",?\s*Apartment\s*\d+\w*",  # Apartment 3, Apartment 3C
        r",?\s*Unit\s*\d+\w*",   # Unit 4, Unit 4D
    ]
    for pattern in patterns:
        address = re.sub(pattern, "", address, flags=re.IGNORECASE)

    # Strip trailing country name — confuses Nominatim with abbreviated streets
    address = re.sub(r',?\s*United States\s*$', '', address, flags=re.IGNORECASE)

    # Expand common street/direction abbreviations for better geocoding
    abbreviations = {
        r'\bN\b': 'North', r'\bS\b': 'South', r'\bE\b': 'East', r'\bW\b': 'West',
        r'\bNE\b': 'Northeast', r'\bNW\b': 'Northwest', r'\bSE\b': 'Southeast', r'\bSW\b': 'Southwest',
        r'\bSt\b': 'Street', r'\bAve\b': 'Avenue', r'\bBlvd\b': 'Boulevard', r'\bDr\b': 'Drive',
        r'\bLn\b': 'Lane', r'\bRd\b': 'Road', r'\bCt\b': 'Court', r'\bPl\b': 'Place',
        r'\bPkwy\b': 'Parkway', r'\bHwy\b': 'Highway', r'\bCir\b': 'Circle', r'\bTrl\b': 'Trail',
    }
    for abbr, full in abbreviations.items():
        address = re.sub(abbr, full, address, flags=re.IGNORECASE)

    return address.strip()

def strip_suite(address):
    if not address:
        return ""
    return re.sub(r',?\s*(Suite|Ste|Unit)\s+\w+', '', address, flags=re.IGNORECASE)
#-----------------------------    

@lru_cache(maxsize=1000)
def geocode_address(address):
    try:
        location = geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
    except Exception as e:
        print("Geocoding error:", e)
    return None, None


# -----------------------------
# SESSION RATE LIMITING
# -----------------------------
RATE_LIMIT_MAX_REQUESTS = 5      # allow up to 5 requests
RATE_LIMIT_WINDOW_SEC = 60       # per 60 seconds
RATE_LIMIT_COOLDOWN_SEC = 8      # minimum gap between requests

def init_rate_limit_state():
    if "request_timestamps" not in st.session_state:
        st.session_state.request_timestamps = deque()
    if "last_request_time" not in st.session_state:
        st.session_state.last_request_time = 0.0

def check_rate_limit():
    """
    Returns (allowed: bool, message: str | None)
    """
    init_rate_limit_state()

    now = monotonic()

    # Short cooldown to prevent double submits / spam clicks
    elapsed = now - st.session_state.last_request_time
    if elapsed < RATE_LIMIT_COOLDOWN_SEC:
        wait_for = int(RATE_LIMIT_COOLDOWN_SEC - elapsed) + 1
        return False, f"Please wait about {wait_for} seconds before sending another request."

    # Sliding window limit
    timestamps = st.session_state.request_timestamps
    while timestamps and (now - timestamps[0]) > RATE_LIMIT_WINDOW_SEC:
        timestamps.popleft()

    if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
        wait_for = int(RATE_LIMIT_WINDOW_SEC - (now - timestamps[0])) + 1
        return False, f"Rate limit reached. Please wait about {wait_for} seconds and try again."

    # Reserve the slot now
    timestamps.append(now)
    st.session_state.last_request_time = now
    return True, None
# -----------------------------
# JSON Formatting Helpers
import json
import re

def clean_json_text(text):
    if not text:
        return ""

    text = text.strip()

    # Remove markdown code fences if present
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    return text.strip()


def extract_json_payload(text):
    """
    Try to extract the first JSON array or object from a model response.
    Prioritize arrays since your app expects a list of recommendations.
    """
    text = clean_json_text(text)

    # Prefer JSON array
    array_match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if array_match:
        return array_match.group(0).strip()

    # Fallback to JSON object
    object_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if object_match:
        return object_match.group(0).strip()

    return text


def normalize_recommendation(rec):
    """
    Ensure every recommendation has the fields your UI expects.
    """
    return {
        "restaurant": rec.get("restaurant", "") if isinstance(rec, dict) else "",
        "dish": rec.get("dish", "") if isinstance(rec, dict) else "",
        "description": rec.get("description", "") if isinstance(rec, dict) else "",
        "review_excerpt": rec.get("review_excerpt", "") if isinstance(rec, dict) else "",
        "why_this_was_selected": rec.get("why_this_was_selected", "") if isinstance(rec, dict) else "",
        "photos": rec.get("photos", []) if isinstance(rec, dict) and isinstance(rec.get("photos", []), list) else []
    }


def parse_recommendations(answer):
    """
    Robust JSON parsing for model output.
    Returns (parsed_output, json_valid)
    """
    if not answer or not isinstance(answer, str):
        return [{
            "restaurant": "",
            "dish": "",
            "description": "I couldn't format the recommendations this time. Please try again.",
            "review_excerpt": "",
            "why_this_was_selected": "",
            "photos": []
        }], False

    # Attempt 1: direct parse
    try:
        parsed = json.loads(answer)
    except Exception:
        parsed = None

    # Attempt 2: extract likely JSON payload
    if parsed is None:
        try:
            extracted = extract_json_payload(answer)
            parsed = json.loads(extracted)
        except Exception:
            parsed = None

    # Attempt 3: if model returned a single object, wrap it in a list
    if isinstance(parsed, dict):
        parsed = [parsed]

    # Validate final structure
    if isinstance(parsed, list):
        normalized = [normalize_recommendation(rec) for rec in parsed]
        return normalized, True

    return [{
        "restaurant": "",
        "dish": "",
        "description": "I couldn't format the recommendations this time. Please try again.",
        "review_excerpt": "",
        "why_this_was_selected": "",
        "photos": []
    }], False

# -----------------------------
# MODERATION
# -----------------------------
def _to_plain_dict(obj):
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return vars(obj)
    return {}


def moderate_text(text):
    """Moderate user input or model output."""
    try:
        response = client.moderations.create(
            model="omni-moderation-latest",
            input=text
        )

        result = response.results[0]
        return {
            "flagged": bool(getattr(result, "flagged", False)),
            "categories": _to_plain_dict(getattr(result, "categories", {})),
            "category_scores": _to_plain_dict(getattr(result, "category_scores", {})),
        }
    except Exception as e:
        print("Moderation error:", e)
        return {
            "flagged": False,
            "categories": {},
            "category_scores": {},
            "error": str(e),
        }


def build_moderation_block_message(moderation_result, stage="input"):
    categories = moderation_result.get("categories", {}) or {}
    blocked_categories = [k for k, v in categories.items() if v]
    category_text = ", ".join(blocked_categories) if blocked_categories else "policy-sensitive content"

    if stage == "input":
        description = (
            f"I can’t process that request because it was flagged by moderation "
            f"({category_text}). Please rephrase and try again."
        )
    else:
        description = "I couldn’t return a safe response for that request. Please try rephrasing."

    return [{
        "restaurant": "",
        "dish": "",
        "description": description,
        "review_excerpt": "",
        "why_this_was_selected": "",
        "photos": []
    }]


# -----------------------------
# LLM as a Judge
@traceable(name="llm-judge", run_type="llm")
def evaluate_with_llm_judge(user_query, docs_for_llm, parsed_recommendations):
    """
    Uses a second LLM call to score the quality of the generated recommendations.
    Returns scores plus token usage.
    """

    review_context = build_review_context(docs_for_llm)

    judge_prompt = f"""
You are evaluating a restaurant recommendation system.

Score the recommendation output on a 1-5 scale for each category:

1. relevance_score:
How well do the recommendations match the user's request?

2. groundedness_score:
Are the recommendations supported by the retrieved review excerpts?
Do not reward made-up details.

3. helpfulness_score:
Would this answer be useful to a user?

4. overall_score:
Overall quality of the answer.

Also provide a short notes field explaining the scores.

Return ONLY valid JSON in this exact format:
{{
  "relevance_score": 1,
  "groundedness_score": 1,
  "helpfulness_score": 1,
  "overall_score": 1,
  "notes": "short explanation"
}}

User query:
{user_query}

Retrieved review context:
{review_context}

Generated recommendations:
{json.dumps(parsed_recommendations, ensure_ascii=False)}
"""

    try:
        judge_response = client.chat.completions.create(
            model="gpt-5.4-nano",
            messages=[
                {"role": "system", "content": "You are a strict evaluation judge. Return only valid JSON."},
                {"role": "user", "content": judge_prompt},
            ],
            temperature=0
        )

        judge_answer = judge_response.choices[0].message.content or ""

        judge_usage = judge_response.usage

        try:
            judge_parsed = json.loads(judge_answer)
        except Exception:
            judge_parsed = {
                "relevance_score": None,
                "groundedness_score": None,
                "helpfulness_score": None,
                "overall_score": None,
                "notes": "Judge response could not be parsed."
            }

        return {
            "judge_relevance_score": judge_parsed.get("relevance_score"),
            "judge_groundedness_score": judge_parsed.get("groundedness_score"),
            "judge_helpfulness_score": judge_parsed.get("helpfulness_score"),
            "judge_overall_score": judge_parsed.get("overall_score"),
            "judge_notes": judge_parsed.get("notes"),
        }

    except Exception as e:
        return {
            "judge_relevance_score": None,
            "judge_groundedness_score": None,
            "judge_helpfulness_score": None,
            "judge_overall_score": None,
            "judge_notes": f"Judge evaluation failed: {str(e)}",
        }
# -----------------------------
# STREAMLIT SESSION MEMORY
# -----------------------------
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = []

if "recommended_restaurants" not in st.session_state:
    st.session_state.recommended_restaurants = set()

if "last_user_query" not in st.session_state:
    st.session_state.last_user_query = None

if "last_docs_for_llm" not in st.session_state:
    st.session_state.last_docs_for_llm = None

if "last_parsed_recommendations" not in st.session_state:
    st.session_state.last_parsed_recommendations = None

if "last_metric_row_id" not in st.session_state:
    st.session_state.last_metric_row_id = None

if "seen_review_ids" not in st.session_state:
    st.session_state.seen_review_ids = set()

# -----------------------------
# DB CONNECTION
# -----------------------------
connection_pool = pool.ThreadedConnectionPool(
    minconn=2,
    maxconn=10,
    user="postgres.teyeutbzecbobotobhzc",
    password=DB_PASSWORD,
    host="aws-0-us-west-2.pooler.supabase.com",
    port=6543,
    dbname="postgres",
)

def get_connection(retries=10, delay=0.5):
    for i in range(retries):
        try:
            return connection_pool.getconn()
        except pool.PoolError:
            if i < retries - 1:
                time.sleep(delay)
    raise Exception("connection pool exhausted after retries")

def release_connection(conn):
    connection_pool.putconn(conn)

# -----------------------------
# EVALUATION METRICS LOGGING
# -----------------------------
def insert_evaluation_metric(metric_row):
    conn = None
    cur = None

    try:
        conn = get_connection()
        cur = conn.cursor()

        query = """
    INSERT INTO evaluation_metrics (
        user_query,
        retrieved_doc_count,
        avg_distance,
        retrieval_time_ms,
        generation_time_ms,
        embedding_input_tokens,
        llm_input_tokens,
        llm_output_tokens,
        llm_total_tokens,
        cached_input_tokens,
        json_valid,
        fallback_used,
        raw_output
    )
    VALUES (
        %(user_query)s,
        %(retrieved_doc_count)s,
        %(avg_distance)s,
        %(retrieval_time_ms)s,
        %(generation_time_ms)s,
        %(embedding_input_tokens)s,
        %(llm_input_tokens)s,
        %(llm_output_tokens)s,
        %(llm_total_tokens)s,
        %(cached_input_tokens)s,
        %(json_valid)s,
        %(fallback_used)s,
        %(raw_output)s::jsonb
    )
    RETURNING id
"""

        cur.execute(query, metric_row)
        inserted_id = cur.fetchone()[0]
        conn.commit()
        return inserted_id

    except Exception as e:
        print("Error inserting evaluation metric:", e)
        if conn:
            conn.rollback()
        return None

    finally:
        if cur:
            cur.close()
        if conn:
            release_connection(conn)


def update_judge_metrics(row_id, judge_results):
    if row_id is None:
        return False

    conn = None
    cur = None

    try:
        conn = get_connection()
        cur = conn.cursor()

        query = """
            UPDATE evaluation_metrics
            SET
                judge_relevance_score = %(judge_relevance_score)s,
                judge_groundedness_score = %(judge_groundedness_score)s,
                judge_helpfulness_score = %(judge_helpfulness_score)s,
                judge_overall_score = %(judge_overall_score)s,
                judge_notes = %(judge_notes)s
            WHERE id = %(id)s
        """

        payload = {"id": row_id, **judge_results}
        cur.execute(query, payload)
        conn.commit()
        return True

    except Exception as e:
        print("Error updating judge metrics:", e)
        if conn:
            conn.rollback()
        return False

    finally:
        if cur:
            cur.close()
        if conn:
            release_connection(conn)

# -----------------------------
# VECTOR SEARCH
# -----------------------------
@traceable(name="vector-search", run_type="retriever")
def similarity_search(query_text, k=20, exclude_review_ids=None):
    embedding_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query_text
    )

    embedding = embedding_response.data[0].embedding
    embedding_input_tokens = getattr(embedding_response.usage, "total_tokens", None)

    embedding_str = "[" + ",".join(map(str, embedding)) + "]"

    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    exclude_review_ids = list(exclude_review_ids or [])

    query = """
        SELECT
            rc.id,
            rc.review_id,
            rc.place_name,
            rc.chunk_text,
            rc.place_id,
            COALESCE(photo_data.photos, '[]'::json) AS photos,
            rc.embedding <=> %s::vector AS distance
        FROM review_chunks rc
        LEFT JOIN LATERAL (
            SELECT json_agg(to_jsonb(rp) - 'review_id') AS photos
            FROM review_photos rp
            WHERE rp.place_id = rc.place_id
              AND rp.review_id = rc.review_id
            LIMIT 5
        ) photo_data ON TRUE
        WHERE rc.embedding IS NOT NULL
          AND rc.embedding <=> %s::vector < 0.6
          AND (
                %s::bigint[] = '{}'
                OR rc.review_id <> ALL(%s::bigint[])
          )
        ORDER BY rc.embedding <=> %s::vector
        LIMIT %s
    """

    cur.execute(
        query,
        (
            embedding_str,
            embedding_str,
            exclude_review_ids,
            exclude_review_ids,
            embedding_str,
            k,
        )
    )
    rows = cur.fetchall()

    cur.close()
    release_connection(conn)

    return rows, embedding_input_tokens
# ------------------------------

# -----------------
# Enrich Query Results with Address
def enrich_with_location(rows):

    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    place_ids = list(set([
        row["place_id"] for row in rows if row.get("place_id")
    ]))

    address_map = {}

    if place_ids:
        cur.execute("""
            SELECT place_id, address, latitude, longitude
            FROM place_table
            WHERE place_id = ANY(%s)
        """, (place_ids,))

        address_map = {
            r["place_id"]: {
                "address": r["address"],
                "latitude": r["latitude"],
                "longitude": r["longitude"],
            }
            for r in cur.fetchall()
        }

    cur.close()
    release_connection(conn)

    # ⚠️ IMPORTANT: create a NEW list (don’t mutate original)
    enriched = []

    for row in rows:
        new_row = dict(row)  # copy

        data = address_map.get(row.get("place_id"), {})
        new_row["address"] = data.get("address")
        new_row["latitude"] = data.get("latitude")
        new_row["longitude"] = data.get("longitude")

        # Fallback: geocode if DB has no coordinates but has an address
        if new_row["latitude"] is None or new_row["longitude"] is None:
            address = new_row.get("address")
            if address:
                normalized = normalize_address_for_geocoding(address)
                lat, lon = geocode_address(normalized)
                if lat is None or lon is None:
                    fallback_addr = strip_suite(normalized)
                    lat, lon = geocode_address(fallback_addr)
                new_row["latitude"] = lat
                new_row["longitude"] = lon

        enriched.append(new_row)

    return enriched

# -----------------------------
# BUILD MEMORY CONTEXT
# -----------------------------
def build_memory_context(max_turns=3):

    memory = st.session_state.conversation_memory[-max_turns:]

    if not memory:
        return "No previous conversation."

    text = ""

    for turn in memory:
        text += f"User: {turn['user']}\n"
        text += f"Assistant: {json.dumps(turn['assistant'])}\n"

    return text


# -----------------------------
# BUILD REVIEW CONTEXT
# -----------------------------
def build_review_context(docs):

    context = ""

    for d in docs:

        photos = d.get("photos") or []
        photo_links = []

        if isinstance(photos, list):
            for photo_item in photos:
                link = photo_item.get("photo_link")
                if link:
                    photo_links.append(link)

        photos_text = ", ".join(photo_links) if photo_links else "No photos"

        context += f"""
Restaurant: {d['place_name']}
Review: {d['chunk_text']}
Similarity Score: {round(d['distance'],4)}
Photo Links: {photos_text}
"""

    return context

# -----------------------------
# Attach Addresses to Recommendations for Map/UI
def _extract_photo_urls(photos_raw):
    """Normalize photos from either list-of-dicts (DB) or list-of-strings (LLM)."""
    urls = []
    for p in (photos_raw or []):
        if isinstance(p, str) and p.startswith("http"):
            urls.append(p)
        elif isinstance(p, dict):
            link = p.get("photo_link") or p.get("url") or p.get("photo_url")
            if link and link.startswith("http"):
                urls.append(link)
    return urls


def attach_addresses_to_recommendations(recommendations, docs_for_map):
    by_place_id = {}
    by_place_name = {}

    for d in docs_for_map:
        place_id = d.get("place_id")
        place_name = d.get("place_name")
        if not place_id:
            continue

        if place_id not in by_place_id:
            by_place_id[place_id] = {
                "address": d.get("address"),
                "latitude": d.get("latitude"),
                "longitude": d.get("longitude"),
                "photos": [],
            }

        for url in _extract_photo_urls(d.get("photos") or []):
            if url not in by_place_id[place_id]["photos"]:
                by_place_id[place_id]["photos"].append(url)

        if place_name:
            by_place_name[place_name.strip().lower()] = place_id

    enriched_recs = []

    for rec in recommendations:
        new_rec = dict(rec)
        name_key = (rec.get("restaurant") or "").strip().lower()

        # No restaurant name = no address/map matching
        if not name_key:
            new_rec["address"] = None
            new_rec["latitude"] = None
            new_rec["longitude"] = None
            new_rec["photos"] = _extract_photo_urls(rec.get("photos") or [])
            enriched_recs.append(new_rec)
            continue

        place_id = by_place_name.get(name_key)
        if not place_id:
            for db_name, pid in by_place_name.items():
                if name_key in db_name or db_name in name_key:
                    place_id = pid
                    break

        data = by_place_id.get(place_id, {})
        new_rec["address"] = data.get("address")
        new_rec["latitude"] = data.get("latitude")
        new_rec["longitude"] = data.get("longitude")
        new_rec["photos"] = data.get("photos") or _extract_photo_urls(rec.get("photos") or [])

        enriched_recs.append(new_rec)

    return enriched_recs

    for rec in recommendations:
        new_rec = dict(rec)
        name_key = rec.get("restaurant", "").strip().lower()

        # Resolve place_id via exact name match, then partial match
        place_id = by_place_name.get(name_key)
        if not place_id:
            for db_name, pid in by_place_name.items():
                if name_key in db_name or db_name in name_key:
                    place_id = pid
                    break

        data = by_place_id.get(place_id, {})
        new_rec["address"]   = data.get("address")
        new_rec["latitude"]  = data.get("latitude")
        new_rec["longitude"] = data.get("longitude")
        new_rec["photos"]    = data.get("photos") or _extract_photo_urls(rec.get("photos") or [])

        enriched_recs.append(new_rec)

    return enriched_recs


# -----------------------------
# Stable prompt for OpenAI Prompt Caching
# -----------------------------
STATIC_SYSTEM_PROMPT = """
You are a professional food critic helping users choose restaurants based only on retrieved review evidence.

Your job is to generate restaurant recommendations grounded in the provided review excerpts.

Follow these rules exactly:

1. Recommend up to THREE restaurants.
2. Recommend exactly 3 only if 3 distinct clearly relevant restaurants are supported by the provided review excerpts.
3. If fewer than 3 distinct relevant restaurants are supported, return fewer than 3.
4. If no relevant restaurants are supported, return a single object with an empty restaurant name and a short explanation in the description field.
5. Use only information contained in the provided review excerpts.
6. Do not invent dishes, descriptions, quotes, or restaurant attributes.
7. review_excerpt must be verbatim or lightly trimmed from the provided review text.
8. why_this_was_selected must briefly explain why the restaurant matches the user's request.
9. Prefer recommendations that are directly relevant to the user's requested cuisine, style, or dining experience.
10. If the user asks for alternatives, avoid repeating restaurants already mentioned in prior conversation when other relevant supported options are available.

Return ONLY valid JSON using this exact structure:

[
  {
    "restaurant": "restaurant name",
    "dish": "specific dish mentioned in reviews",
    "description": "short recommendation",
    "review_excerpt": "verbatim or lightly trimmed quote from the review text",
    "why_this_was_selected": "brief explanation tying the user query to the review",
    "photos": ["photo_url1", "photo_url2"]
  }
]

Additional rules:
- No markdown
- No code fences
- No prose before or after the JSON
- No trailing commas
- Output must be a JSON array
"""


def build_memory_context(max_turns=2):
    """
    Keep memory short and lightweight so more of the prompt prefix stays stable.
    Do NOT include full assistant JSON blobs.
    """
    memory = st.session_state.conversation_memory[-max_turns:]

    if not memory:
        return "No previous conversation."

    lines = []

    for turn in memory:
        user_text = (turn.get("user") or "").strip()

        assistant_items = turn.get("assistant") or []
        restaurant_names = []

        if isinstance(assistant_items, list):
            for item in assistant_items:
                if isinstance(item, dict):
                    name = (item.get("restaurant") or "").strip()
                    if name:
                        restaurant_names.append(name)

        if restaurant_names:
            assistant_summary = "Previously recommended: " + ", ".join(restaurant_names[:5])
        else:
            assistant_summary = "Previously provided recommendations."

        if user_text:
            lines.append(f"User: {user_text}")
        lines.append(f"Assistant: {assistant_summary}")

    return "\n".join(lines)


def build_review_context(docs):
    """
    Build review context in a stable order to improve prompt prefix consistency.
    """
    context = ""

    sorted_docs = sorted(
        docs,
        key=lambda d: (
            d.get("place_name", ""),
            round(d.get("distance", 999), 4),
            d.get("review_id", 0)
        )
    )

    for d in sorted_docs:
        photos = d.get("photos") or []
        photo_links = []

        if isinstance(photos, list):
            for photo_item in photos:
                if isinstance(photo_item, dict):
                    link = photo_item.get("photo_link")
                    if link:
                        photo_links.append(link)

        photos_text = ", ".join(photo_links) if photo_links else "No photos"

        context += f"""
Restaurant: {d.get('place_name', '')}
Review: {d.get('chunk_text', '')}
Similarity Score: {round(d.get('distance', 0), 4) if d.get('distance') is not None else 'N/A'}
Photo Links: {photos_text}
"""

    return context


def build_generation_messages(user_query, memory_context, review_context):
    """
    Keep the system prompt stable.
    Put short memory before the larger changing review context.
    """
    dynamic_context = f"""Previous conversation:
{memory_context}

User request:
{user_query}

Review excerpts:
{review_context}
"""

    return [
        {"role": "system", "content": STATIC_SYSTEM_PROMPT},
        {"role": "user", "content": dynamic_context},
    ]


# -----------------------------
# EVALUATION METRICS LOGGING
# -----------------------------
def insert_evaluation_metric(metric_row):
    conn = None
    cur = None

    try:
        conn = get_connection()
        cur = conn.cursor()

        query = """
            INSERT INTO evaluation_metrics (
                user_query,
                retrieved_doc_count,
                avg_distance,
                retrieval_time_ms,
                generation_time_ms,
                embedding_input_tokens,
                llm_input_tokens,
                llm_output_tokens,
                llm_total_tokens,
                cached_input_tokens,
                json_valid,
                fallback_used,
                raw_output
            )
            VALUES (
                %(user_query)s,
                %(retrieved_doc_count)s,
                %(avg_distance)s,
                %(retrieval_time_ms)s,
                %(generation_time_ms)s,
                %(embedding_input_tokens)s,
                %(llm_input_tokens)s,
                %(llm_output_tokens)s,
                %(llm_total_tokens)s,
                %(cached_input_tokens)s,
                %(json_valid)s,
                %(fallback_used)s,
                %(raw_output)s::jsonb
            )
            RETURNING id
        """

        cur.execute(query, metric_row)
        inserted_id = cur.fetchone()[0]
        conn.commit()
        return inserted_id

    except Exception as e:
        print("Error inserting evaluation metric:", e)
        if conn:
            conn.rollback()
        return None

    finally:
        if cur:
            cur.close()
        if conn:
            release_connection(conn)

# -------------------------
# Mark review IDs used in recommendations to avoid repetition in future responses
# --------------------------    
def mark_used_review_ids(parsed_recommendations, docs):
    if "seen_review_ids" not in st.session_state:
        st.session_state.seen_review_ids = set()

    chosen_names = {
        (rec.get("restaurant") or "").strip().lower()
        for rec in parsed_recommendations
        if isinstance(rec, dict) and rec.get("restaurant")
    }

    used_review_ids = set()

    for row in docs:
        place_name = (row.get("place_name") or "").strip().lower()
        if not place_name:
            continue

        for chosen in chosen_names:
            if chosen == place_name or chosen in place_name or place_name in chosen:
                if row.get("review_id") is not None:
                    used_review_ids.add(row["review_id"])

    st.session_state.seen_review_ids.update(used_review_ids)
# -----------------------------
# MAIN RAG FUNCTION
# -----------------------------
@traceable(name="food-recommendation-pipeline", run_type="chain")
def run_rag(user_query):
    """
    Main RAG function:
    - Moderates user input before retrieval/generation
    - Performs similarity search
    - Returns recommendations or fallback if no relevant reviews
    - Inserts base evaluation metrics immediately
    - Stores data needed for optional later LLM-as-a-judge evaluation
    - Always appends the result to conversation memory
    """
    total_start = time.time()

    input_moderation = moderate_text(user_query)
    if input_moderation.get("flagged"):
        blocked = build_moderation_block_message(input_moderation, stage="input")

        metric_row = {
            "user_query": user_query,
            "retrieved_doc_count": 0,
            "avg_distance": None,
            "retrieval_time_ms": 0,
            "generation_time_ms": 0,
            "embedding_input_tokens": 0,
            "llm_input_tokens": 0,
            "llm_output_tokens": 0,
            "llm_total_tokens": 0,
            "cached_input_tokens": 0,
            "json_valid": True,
            "fallback_used": True,
            "raw_output": json.dumps({
                "type": "moderation_block",
                "stage": "input",
                "moderation": input_moderation,
                "response": blocked
            }, default=str)
        }

        metric_row_id = insert_evaluation_metric(metric_row)

        st.session_state.last_user_query = user_query
        st.session_state.last_docs_for_llm = []
        st.session_state.last_parsed_recommendations = blocked
        st.session_state.last_metric_row_id = metric_row_id

        st.session_state.conversation_memory.append({
            "user": user_query,
            "assistant": blocked
        })

        return blocked

    run_tree = get_current_run_tree()
    if run_tree:
        st.session_state.last_langsmith_run_id = str(run_tree.id)

    memory_context = build_memory_context()

    retrieval_start = time.time()

    docs, embedding_input_tokens = similarity_search(
    user_query,
    k=8,
    exclude_review_ids=st.session_state.seen_review_ids
)
   # -- If no relevant documents are found and there are previously seen review IDs, try again without excluding them. 
    if not docs and st.session_state.seen_review_ids:
        docs, embedding_input_tokens = similarity_search(
        user_query,
        k=8,
        exclude_review_ids=None
    )
        
    retrieval_time_ms = int((time.time() - retrieval_start) * 1000)

    docs_for_llm = docs
    docs_for_map = enrich_with_location(docs)

    st.session_state.last_docs = docs_for_map

    distances = [d["distance"] for d in docs if d.get("distance") is not None]
    avg_distance = sum(distances) / len(distances) if distances else None

    llm_input_tokens = 0
    llm_output_tokens = 0
    llm_total_tokens = 0
    cached_input_tokens = 0

    if not docs:
        fallback = [{
            "description": "There are no relevant reviews based on your input, try rephrasing your question or asking about something else.",
        }]

        metric_row = {
            "user_query": user_query,
            "retrieved_doc_count": 0,
            "avg_distance": None,
            "retrieval_time_ms": retrieval_time_ms,
            "generation_time_ms": 0,
            "embedding_input_tokens": embedding_input_tokens,
            "llm_input_tokens": llm_input_tokens,
            "llm_output_tokens": llm_output_tokens,
            "llm_total_tokens": llm_total_tokens,
            "cached_input_tokens": cached_input_tokens,
            "json_valid": True,
            "fallback_used": True,
            "raw_output": json.dumps(fallback, default=str)
        }

        metric_row_id = insert_evaluation_metric(metric_row)

        st.session_state.last_user_query = user_query
        st.session_state.last_docs_for_llm = docs_for_llm
        st.session_state.last_parsed_recommendations = fallback
        st.session_state.last_metric_row_id = metric_row_id

        st.session_state.conversation_memory.append({
            "user": user_query,
            "assistant": fallback
        })

        return fallback

    review_context = build_review_context(docs_for_llm)

    messages = build_generation_messages(
        user_query=user_query,
        memory_context=memory_context,
        review_context=review_context
    )

    generation_start = time.time()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3,
        prompt_cache_key="foodfinder_recommendation_v1"
    )
    generation_time_ms = int((time.time() - generation_start) * 1000)

    usage = response.usage
    llm_input_tokens = getattr(usage, "prompt_tokens", None)
    llm_output_tokens = getattr(usage, "completion_tokens", None)
    llm_total_tokens = getattr(usage, "total_tokens", None)

    prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
    if prompt_tokens_details:
        cached_input_tokens = getattr(prompt_tokens_details, "cached_tokens", 0) or 0
    else:
        cached_input_tokens = 0

    print(f"Prompt cached tokens: {cached_input_tokens}")

    answer = response.choices[0].message.content or ""

    output_moderation = moderate_text(answer)
    if output_moderation.get("flagged"):
        parsed = build_moderation_block_message(output_moderation, stage="output")
        json_valid = True
        raw_output_for_db = {
            "type": "moderation_block",
            "stage": "output",
            "moderation": output_moderation,
            "raw_model_output": answer,
            "response": parsed
        }
    else:
        parsed, json_valid = parse_recommendations(answer)
        raw_output_for_db = parsed

    parsed = attach_addresses_to_recommendations(parsed, docs_for_map)
    mark_used_review_ids(parsed, docs)

    metric_row = {
        "user_query": user_query,
        "retrieved_doc_count": len(docs),
        "avg_distance": avg_distance,
        "retrieval_time_ms": retrieval_time_ms,
        "generation_time_ms": generation_time_ms,
        "embedding_input_tokens": embedding_input_tokens,
        "llm_input_tokens": llm_input_tokens,
        "llm_output_tokens": llm_output_tokens,
        "llm_total_tokens": llm_total_tokens,
        "cached_input_tokens": cached_input_tokens,
        "json_valid": json_valid,
        "fallback_used": False,
        "raw_output": json.dumps(raw_output_for_db, default=str)
    }

    metric_row_id = insert_evaluation_metric(metric_row)

    st.session_state.last_user_query = user_query
    st.session_state.last_docs_for_llm = docs_for_llm
    st.session_state.last_parsed_recommendations = parsed
    st.session_state.last_metric_row_id = metric_row_id

    st.session_state.conversation_memory.append({
        "user": user_query,
        "assistant": parsed
    })

    return parsed
# ─────────────────────────────────────────────
# MAP RENDERER
# ─────────────────────────────────────────────
def render_small_map(lat, lon, restaurant_name="Restaurant"):
    lat = float(lat)
    lon = float(lon)
    df = pd.DataFrame([{"lat": lat, "lon": lon, "name": restaurant_name}])
    st.pydeck_chart(
        pdk.Deck(
            map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
            initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=13, pitch=0),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=df,
                    get_position='[lon, lat]',
                    get_radius=80,
                    get_fill_color=[255, 122, 53, 220],
                    stroked=True,
                    get_line_color=[240, 235, 228, 180],
                    line_width_min_pixels=1,
                    pickable=True,
                )
            ],
            tooltip={"text": "{name}"},
        ),
        height=180,
    )


# ─────────────────────────────────────────────
# GLOBAL CSS — "The Curated Hearth" Design System
# ─────────────────────────────────────────────
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Be+Vietnam+Pro:ital,wght@0,400;0,500;0,600;1,400&display=swap');

        /* ── Design Tokens — Dark Mode ── */
        :root {
            --primary:          #ff7a35;
            --primary-light:    #ffab76;
            --bg:               #1a1714;
            --surface-low:      #231f1b;
            --surface-lowest:   #2d2824;
            --gold:             #c4a560;
            --on-surface:       #f0ebe4;
            --muted:            #9a9088;
            --shadow:           rgba(0, 0, 0, 0.35);
            --ghost-border:     rgba(255, 122, 53, 0.18);
            --font-display:     'Plus Jakarta Sans', sans-serif;
            --font-body:        'Be Vietnam Pro', sans-serif;
        }

        /* ── Base ── */
        html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
            background-color: var(--bg) !important;
            font-family: var(--font-body) !important;
            color: var(--on-surface) !important;
        }

        /* ── Remove Streamlit chrome clutter ── */
        #MainMenu, footer, header { visibility: hidden; }
        .block-container { padding-top: 2rem !important; max-width: 960px; }

        /* ── Typography helpers ── */
        .headline-lg {
            font-family: var(--font-display);
            font-size: 1.75rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: var(--on-surface);
            margin: 0 0 0.25rem 0;
            line-height: 1.2;
        }
        .headline-md {
            font-family: var(--font-display);
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: var(--on-surface);
            margin: 0 0 0.15rem 0;
        }
        .label-sm-caps {
            font-family: var(--font-body);
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--gold);
            display: block;
            margin-bottom: 0.35rem;
        }
        .body-text {
            font-family: var(--font-body);
            font-size: 0.95rem;
            line-height: 1.65;
            color: var(--on-surface);
        }
        .muted-text {
            font-family: var(--font-body);
            font-size: 0.82rem;
            color: var(--muted);
            line-height: 1.5;
        }

        /* ── Wordmark ── */
        .wordmark {
            font-family: var(--font-display);
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--primary);
            letter-spacing: -0.02em;
            margin-bottom: 0.15rem;
        }
        .tagline {
            font-family: var(--font-body);
            font-size: 0.88rem;
            color: var(--muted);
        }

        /* ── Cards ── */
        .rec-card {
            background: var(--surface-lowest);
            border-radius: 1.5rem;
            overflow: hidden;
            margin-bottom: 2rem;
            box-shadow:
                0 2px 8px var(--shadow),
                0 12px 40px rgba(57, 56, 52, 0.04);
        }
        .rec-card-body { padding: 1.25rem 1.4rem 1.4rem; }
        .bleed-img {
            width: 100%;
            height: 220px;
            object-fit: cover;
            display: block;
        }
        .bleed-img-placeholder {
            width: 100%;
            height: 100px;
            background: linear-gradient(135deg, #3a2e26 0%, #2d2420 100%);
        }
        .photo-gallery {
            display: flex;
            overflow-x: auto;
            gap: 0.5rem;
            padding: 0.5rem 0;
            scrollbar-width: thin;
            scrollbar-color: var(--ghost-border) transparent;
        }
        .photo-gallery::-webkit-scrollbar {
            height: 6px;
        }
        .photo-gallery::-webkit-scrollbar-thumb {
            background: var(--ghost-border);
            border-radius: 3px;
        }
        .photo-gallery img {
            height: 160px;
            min-width: 200px;
            object-fit: cover;
            border-radius: 0.75rem;
            flex-shrink: 0;
        }

        /* ── Blockquote excerpt ── */
        .blockquote-excerpt {
            border-left: 3px solid rgba(172, 67, 0, 0.3);
            padding: 0.6rem 0 0.6rem 1rem;
            margin: 0.85rem 0;
            font-style: italic;
            font-size: 0.9rem;
            color: var(--muted);
            line-height: 1.6;
        }

        /* ── Address row ── */
        .address-row {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            font-size: 0.85rem;
            color: var(--muted);
            margin-top: 0.5rem;
        }

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            border-bottom: 1px solid var(--ghost-border) !important;
            background: transparent !important;
        }
        .stTabs [data-baseweb="tab"] {
            font-family: var(--font-display) !important;
            font-size: 0.92rem !important;
            font-weight: 500 !important;
            color: var(--muted) !important;
            background: transparent !important;
            border: none !important;
            padding: 0.6rem 1.1rem !important;
            border-radius: 0 !important;
        }
        .stTabs [aria-selected="true"] {
            color: var(--primary) !important;
            border-bottom: 2px solid var(--primary) !important;
            font-weight: 600 !important;
        }
        .stTabs [data-baseweb="tab-highlight"] { display: none !important; }
        .stTabs [data-baseweb="tab-border"]    { display: none !important; }

        /* ── Buttons ── */
        .primary-action .stButton > button {
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%) !important;
            color: #fffbff !important;
            border: none !important;
            border-radius: 9999px !important;
            padding: 0.45rem 1.2rem !important;
            font-family: var(--font-body) !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            box-shadow: 0 4px 16px rgba(172, 67, 0, 0.25) !important;
            transition: opacity 0.15s ease !important;
        }
        .primary-action .stButton > button:hover { opacity: 0.88 !important; }

        .secondary-action .stButton > button {
            background: var(--surface-lowest) !important;
            color: var(--primary) !important;
            border: none !important;
            border-radius: 9999px !important;
            padding: 0.45rem 1.2rem !important;
            font-family: var(--font-body) !important;
            font-weight: 500 !important;
            font-size: 0.85rem !important;
            box-shadow: 0 1px 4px var(--shadow) !important;
            transition: background 0.15s ease !important;
        }
        .secondary-action .stButton > button:hover { background: #3d3530 !important; }

        .danger-action .stButton > button {
            background: transparent !important;
            color: #b00020 !important;
            border: none !important;
            border-radius: 9999px !important;
            padding: 0.45rem 1rem !important;
            font-family: var(--font-body) !important;
            font-size: 0.82rem !important;
        }
        .danger-action .stButton > button:hover { background: rgba(176, 0, 32, 0.06) !important; }

        .tertiary-action .stButton > button {
            background: transparent !important;
            color: var(--muted) !important;
            border: none !important;
            font-family: var(--font-body) !important;
            font-size: 0.82rem !important;
            padding: 0.3rem 0.6rem !important;
            text-decoration: underline !important;
        }

        /* ── Fixed clear button (bottom-left of chat input) ── */
        .clear-fixed-btn .stButton > button {
            position: fixed !important;
            bottom: 1rem !important;
            left: max(1rem, calc(50% - 472px)) !important;
            z-index: 10000 !important;
            background: rgba(35, 31, 27, 0.92) !important;
            color: var(--muted) !important;
            border: 1px solid var(--ghost-border) !important;
            border-radius: 9999px !important;
            font-family: var(--font-body) !important;
            font-size: 0.78rem !important;
            padding: 0.4rem 0.9rem !important;
            white-space: nowrap !important;
            backdrop-filter: blur(8px) !important;
            transition: color 0.15s ease, border-color 0.15s ease !important;
        }
        .clear-fixed-btn .stButton > button:hover {
            color: var(--primary) !important;
            border-color: rgba(255, 122, 53, 0.4) !important;
        }

        /* ── Chat messages ── */
        .stChatMessage { background: transparent !important; border: none !important; }
        [data-testid="stChatMessageContent"] {
            background: var(--surface-low) !important;
            border-radius: 1rem !important;
            padding: 0.75rem 1rem !important;
            color: var(--on-surface) !important;
        }

        /* ── Chat input container (fixed at bottom) ── */
        [data-testid="stBottom"] {
            background-color: var(--bg) !important;
        }
        [data-testid="stBottom"] > div {
            background-color: var(--bg) !important;
            padding-left: 110px !important;
        }

        /* ── Bottom padding so last message isn't hidden behind fixed bar ── */
        .block-container {
            padding-bottom: 6rem !important;
        }
        [data-testid="stChatInput"] {
            background-color: var(--surface-low) !important;
            border-radius: 1rem !important;
            border: 1.5px solid var(--ghost-border) !important;
        }
        [data-testid="stChatInput"] textarea {
            background: var(--surface-low) !important;
            color: var(--on-surface) !important;
            border-radius: 1rem !important;
            border: none !important;
            font-family: var(--font-body) !important;
        }
        [data-testid="stChatInput"] textarea:focus {
            box-shadow: none !important;
        }
        [data-testid="stChatInput"] textarea::placeholder {
            color: var(--muted) !important;
        }

        /* ── Text inputs ── */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {
            background: var(--surface-low) !important;
            color: var(--on-surface) !important;
            border-radius: 0.75rem !important;
            border: 1.5px solid transparent !important;
            font-family: var(--font-body) !important;
        }
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {
            border-color: var(--ghost-border) !important;
            box-shadow: none !important;
        }
        .stTextInput > div > div > input::placeholder,
        .stTextArea > div > div > textarea::placeholder {
            color: var(--muted) !important;
        }

        /* ── Sidebar glassmorphism (dark) ── */
        [data-testid="stSidebar"] {
            background: rgba(35, 31, 27, 0.88) !important;
            backdrop-filter: blur(28px) !important;
            border-right: 1px solid var(--ghost-border) !important;
        }

        /* ── Progress bar ── */
        .stProgress > div > div { background-color: var(--primary) !important; }

        /* ── Score badge ── */
        .score-badge {
            display: inline-block;
            background: var(--surface-lowest);
            border-radius: 0.5rem;
            padding: 0.2rem 0.5rem;
            font-family: var(--font-display);
            font-weight: 700;
            font-size: 1rem;
            color: var(--primary);
        }

        /* ── Empty state ── */
        .empty-state {
            text-align: center;
            padding: 4rem 2rem;
            color: var(--muted);
        }
        .empty-state p {
            font-family: var(--font-body);
            font-size: 1rem;
            color: var(--muted);
            max-width: 320px;
            margin: 0 auto;
            line-height: 1.6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────
_PIN_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ff7a35" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>'


def _empty_state(message: str):
    st.markdown(
        f'<div class="empty-state"><p>{message}</p></div>',
        unsafe_allow_html=True,
    )


def _star_display(rating: int) -> str:
    filled = "&#9733;" * rating
    empty = "&#9734;" * (5 - rating)
    return f'<span style="color:var(--gold);font-size:1rem">{filled}{empty}</span>'


def _rec_card_html(rec: dict) -> str:
    restaurant = rec.get("restaurant") or "Unknown Restaurant"
    dish        = rec.get("dish", "")
    description = rec.get("description", "")
    excerpt     = rec.get("review_excerpt", "")
    why         = rec.get("why_this_was_selected", "")
    address     = rec.get("address") or "Address not available"
    photos      = rec.get("photos") or []

    if len(photos) > 1:
        gallery_imgs = ''.join(
            f'<img src="{url}" alt="{restaurant}" />'
            for url in photos
        )
        img_html = f'<div class="photo-gallery">{gallery_imgs}</div>'
    elif photos:
        img_html = f'<img class="bleed-img" src="{photos[0]}" alt="{restaurant}" />'
    else:
        img_html = '<div class="bleed-img-placeholder"></div>'

    excerpt_html = f'<blockquote class="blockquote-excerpt">"{excerpt}"</blockquote>' if excerpt else ""
    why_html     = f'<p class="muted-text" style="margin-top:0.5rem">{why}</p>' if why else ""
    dish_html    = f'<p style="font-family:var(--font-body);font-weight:600;font-size:0.95rem;margin:0.25rem 0 0.5rem">{dish}</p>' if dish else ""

    # No blank lines inside the HTML — blank lines terminate Streamlit's markdown HTML block
    parts = [
        f'<div class="rec-card">',
        img_html,
        '<div class="rec-card-body">',
        '<span class="label-sm-caps">Critic\'s Pick</span>',
        f'<p class="headline-md">{restaurant}</p>',
        dish_html,
        f'<p class="body-text">{description}</p>',
        excerpt_html,
        why_html,
        f'<div class="address-row">{_PIN_SVG}<span>{address}</span></div>',
        '</div>',
        '</div>',
    ]
    return "".join(p for p in parts if p)


# ─────────────────────────────────────────────
# RENDER FUNCTIONS
# ─────────────────────────────────────────────
def render_recommendations(recs, context="chat"):
    for i, rec in enumerate(recs):
        st.markdown(_rec_card_html(rec), unsafe_allow_html=True)

        lat = rec.get("latitude")
        lon = rec.get("longitude")
        if lat is not None and lon is not None:
            render_small_map(lat, lon, restaurant_name=rec.get("restaurant", "Restaurant"))
        else:
            st.markdown('<p class="muted-text">Map not available for this location.</p>', unsafe_allow_html=True)

        restaurant = rec.get("restaurant", "")
        if restaurant:
            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.markdown('<div class="primary-action">', unsafe_allow_html=True)
                if st.button("+ Want to Try", key=f"wtt_{context}_{i}"):
                    if not any(e.get("restaurant") == restaurant for e in st.session_state.want_to_try):
                        st.session_state.want_to_try.append(dict(rec))
                        st.toast(f"Added {restaurant} to Want to Try!")
                st.markdown("</div>", unsafe_allow_html=True)
            with col_b:
                st.markdown('<div class="secondary-action">', unsafe_allow_html=True)
                if st.button("✓ Mark as Tried", key=f"tried_{context}_{i}"):
                    entry = dict(rec)
                    entry.setdefault("rating", 3)
                    entry.setdefault("review_text", "")
                    if not any(e.get("restaurant") == restaurant for e in st.session_state.already_tried):
                        st.session_state.already_tried.append(entry)
                        st.session_state.want_to_try = [
                            e for e in st.session_state.want_to_try if e.get("restaurant") != restaurant
                        ]
                        st.toast(f"Moved {restaurant} to Already Tried!")
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)


def render_library_card(rec: dict, index: int):
    st.markdown(_rec_card_html(rec), unsafe_allow_html=True)

    lat = rec.get("latitude")
    lon = rec.get("longitude")
    if lat is not None and lon is not None:
        render_small_map(lat, lon, restaurant_name=rec.get("restaurant", "Restaurant"))

    restaurant = rec.get("restaurant", "")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown('<div class="primary-action">', unsafe_allow_html=True)
        if st.button("✓ Mark as Tried", key=f"lib_tried_{index}"):
            entry = dict(rec)
            entry.setdefault("rating", 3)
            entry.setdefault("review_text", "")
            if not any(e.get("restaurant") == restaurant for e in st.session_state.already_tried):
                st.session_state.already_tried.append(entry)
            st.session_state.want_to_try.pop(index)
            st.toast(f"Moved {restaurant} to Already Tried!")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="danger-action">', unsafe_allow_html=True)
        if st.button("Remove", key=f"lib_remove_{index}"):
            st.session_state.want_to_try.pop(index)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)


def render_tried_card(rec: dict, index: int):
    restaurant     = rec.get("restaurant", "Unknown")
    existing_review = rec.get("review_text", "")
    existing_rating = rec.get("rating", 3)

    st.markdown(_rec_card_html(rec), unsafe_allow_html=True)

    st.markdown('<span class="label-sm-caps">Your Rating</span>', unsafe_allow_html=True)
    rating = st.select_slider(
        "Rating",
        options=[1, 2, 3, 4, 5],
        value=existing_rating,
        key=f"rating_{index}",
        label_visibility="collapsed",
    )
    st.markdown(_star_display(rating), unsafe_allow_html=True)

    if existing_review:
        st.markdown(
            f'<blockquote class="blockquote-excerpt">"{existing_review}"</blockquote>',
            unsafe_allow_html=True,
        )

    btn_label = "Edit Review" if existing_review else "Write a Review"
    is_open   = st.session_state.review_open.get(index, False)

    st.markdown('<div class="secondary-action">', unsafe_allow_html=True)
    if st.button(btn_label, key=f"toggle_review_{index}"):
        st.session_state.review_open[index] = not is_open
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.review_open.get(index, False):
        review_text = st.text_area(
            "Your review",
            value=existing_review,
            placeholder=f"What did you think of {restaurant}? What did you order?",
            key=f"review_input_{index}",
            height=110,
            label_visibility="collapsed",
        )
        st.markdown('<div class="primary-action">', unsafe_allow_html=True)
        if st.button("Save Review", key=f"save_review_{index}"):
            st.session_state.already_tried[index]["review_text"] = review_text
            st.session_state.already_tried[index]["rating"] = rating
            st.session_state.review_open[index] = False
            st.toast(f"Review saved for {restaurant}!")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# STREAMLIT APP
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FoodFinder",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Session state
_defaults = {
    "conversation_memory": [],
    "recommended_restaurants": set(),
    "last_user_query": None,
    "last_docs_for_llm": None,
    "last_parsed_recommendations": None,
    "last_metric_row_id": None,
    "last_docs": None,
    "want_to_try": [],
    "already_tried": [],
    "review_open": {},
    "judge_scores": None,
    "seen_review_ids": set(),
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

inject_css()

# Wordmark
st.markdown(
    """
    <div style="margin-bottom:1.5rem">
        <p class="wordmark">FoodFinder</p>
        <p class="tagline">AI-powered restaurant recommendations, grounded in real reviews.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Fixed bottom: clear button + chat input ──
st.markdown('<div class="clear-fixed-btn">', unsafe_allow_html=True)
if st.button("Clear conversation", key="clear_history"):
    st.session_state.conversation_memory = []
    st.session_state.recommended_restaurants = set()
    st.session_state.judge_scores = None
    st.session_state.seen_review_ids = set()
    st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

user_query = st.chat_input("What kind of food or drink are you looking for?")
recs = None

if user_query:
    allowed, rate_limit_message = check_rate_limit()

    if not allowed:
        st.warning(rate_limit_message)
    else:
        with st.spinner("Analysing reviews..."):
            recs = run_rag(user_query)

tab1, tab2, tab3 = st.tabs(["Discover", "Want to Try", "Already Tried"])

# ── Tab 1: Discover ──────────────────────────
with tab1:
    with st.sidebar:
        st.markdown('<span class="label-sm-caps">Evaluation</span>', unsafe_allow_html=True)
        st.markdown('<div class="primary-action">', unsafe_allow_html=True)
        run_judge = st.button("Run Judge on Last Result", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if run_judge:
            if (
                st.session_state.last_user_query is not None
                and st.session_state.last_docs_for_llm is not None
                and st.session_state.last_parsed_recommendations is not None
                and st.session_state.last_metric_row_id is not None
            ):
                with st.spinner("Running evaluation..."):
                    judge_results = evaluate_with_llm_judge(
                        st.session_state.last_user_query,
                        st.session_state.last_docs_for_llm,
                        st.session_state.last_parsed_recommendations,
                    )
                    updated = update_judge_metrics(
                        st.session_state.last_metric_row_id,
                        judge_results,
                    )
                    if st.session_state.get("last_langsmith_run_id"):
                        try:
                            for score_key, label in [
                                ("judge_relevance_score",    "relevance"),
                                ("judge_groundedness_score", "groundedness"),
                                ("judge_helpfulness_score",  "helpfulness"),
                                ("judge_overall_score",      "overall"),
                            ]:
                                val = judge_results.get(score_key)
                                if val is not None:
                                    langsmith_client.create_feedback(
                                        run_id=st.session_state.last_langsmith_run_id,
                                        key=label,
                                        score=val / 5,
                                        comment=judge_results.get("judge_notes", ""),
                                    )
                        except Exception:
                            pass
                st.session_state.judge_scores = judge_results
                if updated:
                    st.success("Scores saved.")
                else:
                    st.error("Could not save scores.")
            else:
                st.warning("No result available to judge yet.")

        scores = st.session_state.judge_scores
        if scores:
            st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
            for label, key in [
                ("Relevance",    "judge_relevance_score"),
                ("Groundedness", "judge_groundedness_score"),
                ("Helpfulness",  "judge_helpfulness_score"),
                ("Overall",      "judge_overall_score"),
            ]:
                val = scores.get(key)
                if val is not None:
                    col_l, col_r = st.columns([3, 1])
                    with col_l:
                        st.markdown(f'<span class="muted-text">{label}</span>', unsafe_allow_html=True)
                        st.progress(int(val) / 5)
                    with col_r:
                        st.markdown(f'<span class="score-badge">{val}</span>', unsafe_allow_html=True)
            notes = scores.get("judge_notes")
            if notes:
                st.markdown(
                    f'<p class="muted-text" style="margin-top:0.75rem;font-style:italic">{notes}</p>',
                    unsafe_allow_html=True,
                )

    if not st.session_state.conversation_memory and not user_query:
        st.markdown(
            """
            <div class="empty-state" style="padding:3rem 1rem 1rem">
                <p class="headline-lg" style="text-align:center;color:var(--primary)">Where to tonight?</p>
                <p style="text-align:center;color:var(--muted);font-family:var(--font-body);margin-top:0.5rem">
                    Try asking: <em>"Best ramen in the city"</em> or <em>"Great rooftop bar with cocktails"</em>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    history = st.session_state.conversation_memory[:-1] if user_query else st.session_state.conversation_memory
    for msg in history:
        with st.chat_message("user"):
            st.markdown(f'<p class="body-text">{msg["user"]}</p>', unsafe_allow_html=True)
        with st.chat_message("assistant"):
            render_recommendations(msg["assistant"], context=f"hist_{id(msg)}")

    if user_query:
        with st.chat_message("user"):
            st.markdown(f'<p class="body-text">{user_query}</p>', unsafe_allow_html=True)
        with st.chat_message("assistant"):
            render_recommendations(recs, context="new")


# ── Tab 2: Want to Try ───────────────────────
with tab2:
    st.markdown(
        """
        <div style="margin-bottom:1.5rem">
            <span class="label-sm-caps">Your List</span>
            <p class="headline-lg">Want to Try</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    want_list = st.session_state.want_to_try
    if not want_list:
        _empty_state("Nothing saved yet — start a conversation in Discover to find somewhere new.")
    else:
        search = st.text_input(
            "Search",
            placeholder="Filter by restaurant or dish...",
            key="wtt_search",
            label_visibility="collapsed",
        )
        filtered = [
            r for r in want_list
            if not search or search.lower() in (r.get("restaurant", "") + r.get("dish", "")).lower()
        ]
        if not filtered:
            _empty_state(f'No results matching "{search}".')
        else:
            col1, col2 = st.columns(2, gap="large")
            for i, rec in enumerate(filtered):
                real_idx = want_list.index(rec)
                with (col1 if i % 2 == 0 else col2):
                    render_library_card(rec, real_idx)


# ── Tab 3: Already Tried ─────────────────────
with tab3:
    st.markdown(
        """
        <div style="margin-bottom:1.5rem">
            <span class="label-sm-caps">Your Reviews</span>
            <p class="headline-lg">Already Tried</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    tried_list = st.session_state.already_tried
    if not tried_list:
        _empty_state("No restaurants tried yet — go explore and mark something as tried!")
    else:
        for i, rec in enumerate(tried_list):
            render_tried_card(rec, i)

