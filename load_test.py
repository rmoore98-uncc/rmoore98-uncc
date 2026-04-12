"""
Load test for the FoodFinder RAG pipeline.

All functions are copied exactly from mvp1.2.py.
The only changes from the MVP are marked with # CHANGED comments:
  1. st.session_state replaced with a plain dict called `session`
  2. build_memory_context() accepts `session` as a parameter instead of reading from st
  3. @traceable name changed to "load-test-pipeline" to distinguish traces in LangSmith

Usage:
    python load_test.py
"""

import threading
import time
import random
import json
import os
import re

import psycopg2
import psycopg2.extras
from psycopg2 import pool
from openai import OpenAI
from dotenv import load_dotenv
from functools import lru_cache
from geopy.geocoders import Nominatim
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from langsmith.wrappers import wrap_openai

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_PASSWORD = os.getenv("PASSWORD")

client = wrap_openai(OpenAI(api_key=OPENAI_API_KEY))
geolocator = Nominatim(user_agent="foodfinder_app", timeout=10)


# -----------------------------
# ADDRESS NORMALIZATION FOR BETTER GEOCODING
# -----------------------------
def normalize_address_for_geocoding(address):
    patterns = [
        r",?\s*Suite\s*\d+\w*",
        r",?\s*Apt\s*\d+\w*",
        r",?\s*Apartment\s*\d+\w*",
        r",?\s*Unit\s*\d+\w*",
    ]
    for pattern in patterns:
        address = re.sub(pattern, "", address, flags=re.IGNORECASE)
    return address.strip()

def strip_suite(address):
    if not address:
        return ""
    return re.sub(r',?\s*(Suite|Ste|Unit)\s+\w+', '', address, flags=re.IGNORECASE)

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
# JSON Formatting Helpers
# -----------------------------
def clean_json_text(text):
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()

def extract_json_payload(text):
    text = clean_json_text(text)
    array_match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if array_match:
        return array_match.group(0).strip()
    object_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if object_match:
        return object_match.group(0).strip()
    return text

def normalize_recommendation(rec):
    return {
        "restaurant": rec.get("restaurant", "") if isinstance(rec, dict) else "",
        "dish": rec.get("dish", "") if isinstance(rec, dict) else "",
        "description": rec.get("description", "") if isinstance(rec, dict) else "",
        "review_excerpt": rec.get("review_excerpt", "") if isinstance(rec, dict) else "",
        "why_this_was_selected": rec.get("why_this_was_selected", "") if isinstance(rec, dict) else "",
        "photos": rec.get("photos", []) if isinstance(rec, dict) and isinstance(rec.get("photos", []), list) else []
    }

def parse_recommendations(answer):
    if not answer or not isinstance(answer, str):
        return [{
            "restaurant": "", "dish": "",
            "description": "I couldn't format the recommendations this time. Please try again.",
            "review_excerpt": "", "why_this_was_selected": "", "photos": []
        }], False
    try:
        parsed = json.loads(answer)
    except Exception:
        parsed = None
    if parsed is None:
        try:
            extracted = extract_json_payload(answer)
            parsed = json.loads(extracted)
        except Exception:
            parsed = None
    if isinstance(parsed, dict):
        parsed = [parsed]
    if isinstance(parsed, list):
        normalized = [normalize_recommendation(rec) for rec in parsed]
        return normalized, True
    return [{
        "restaurant": "", "dish": "",
        "description": "I couldn't format the recommendations this time. Please try again.",
        "review_excerpt": "", "why_this_was_selected": "", "photos": []
    }], False


# -----------------------------
# DB CONNECTION POOL
# Threads share a pool instead of each opening their own connection,
# preventing SSL drops under concurrent load.
# -----------------------------
connection_pool = pool.ThreadedConnectionPool(
    minconn=2,
    maxconn=15,
    user="postgres.teyeutbzecbobotobhzc",
    password=DB_PASSWORD,
    host="aws-0-us-west-2.pooler.supabase.com",
    port=6543,
    dbname="postgres",
)

def get_connection(retries=10, delay=0.5):
    # Retry instead of immediately failing when pool is full under concurrent load
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
                user_query, retrieved_doc_count, avg_distance,
                retrieval_time_ms, generation_time_ms, embedding_input_tokens,
                llm_input_tokens, llm_output_tokens, llm_total_tokens,
                json_valid, fallback_used, raw_output
            )
            VALUES (
                %(user_query)s, %(retrieved_doc_count)s, %(avg_distance)s,
                %(retrieval_time_ms)s, %(generation_time_ms)s, %(embedding_input_tokens)s,
                %(llm_input_tokens)s, %(llm_output_tokens)s, %(llm_total_tokens)s,
                %(json_valid)s, %(fallback_used)s, %(raw_output)s::jsonb
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
        if cur: cur.close()
        if conn: release_connection(conn)

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
        if cur: cur.close()
        if conn: release_connection(conn)


# -----------------------------
# VECTOR SEARCH
# -----------------------------
@traceable(name="vector-search", run_type="retriever")
def similarity_search(query_text, k=20):
    embedding_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query_text
    )
    embedding = embedding_response.data[0].embedding
    embedding_input_tokens = getattr(embedding_response.usage, "total_tokens", None)
    embedding_str = "[" + ",".join(map(str, embedding)) + "]"

    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    query = """
        SELECT
            rc.id, rc.review_id, rc.place_name, rc.chunk_text, rc.place_id,
            COALESCE(photo_data.photos, '[]'::json) AS photos,
            rc.embedding <=> %s::vector AS distance
        FROM review_chunks rc
        LEFT JOIN LATERAL (
            SELECT json_agg(to_jsonb(rp) - 'review_id') AS photos
            FROM review_photos rp
            WHERE rp.place_id = rc.place_id
            LIMIT 5
        ) photo_data ON TRUE
        WHERE rc.embedding IS NOT NULL
        AND rc.embedding <=> %s::vector < 0.6
        ORDER BY rc.embedding <=> %s::vector
        LIMIT %s
    """
    cur.execute(query, (embedding_str, embedding_str, embedding_str, k))
    rows = cur.fetchall()
    cur.close()
    release_connection(conn)
    return rows, embedding_input_tokens

def enrich_with_location(rows):
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    place_ids = list(set([row["place_id"] for row in rows if row.get("place_id")]))
    address_map = {}
    if place_ids:
        cur.execute("SELECT place_id, address, latitude, longitude FROM place_table WHERE place_id = ANY(%s)", (place_ids,))
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

    enriched = []
    for row in rows:
        new_row = dict(row)
        data = address_map.get(row.get("place_id"), {})
        new_row["address"] = data.get("address")
        new_row["latitude"] = data.get("latitude")
        new_row["longitude"] = data.get("longitude")
        enriched.append(new_row)
    return enriched


# -----------------------------
# BUILD MEMORY CONTEXT
# CHANGED: accepts `session` dict instead of reading from st.session_state
# -----------------------------
def build_memory_context(session):
    memory = session["conversation_memory"]  # CHANGED: was st.session_state.conversation_memory
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
# -----------------------------
def _extract_photo_urls(photos_raw):
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
        name_key = rec.get("restaurant", "").strip().lower()
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


# -----------------------------
# MAIN RAG FUNCTION
# CHANGED: name is "load-test-pipeline" instead of "food-recommendation-pipeline"
#          so load test traces are separate from real user traces in LangSmith
# CHANGED: accepts `session` dict instead of reading/writing st.session_state
# -----------------------------
@traceable(name="load-test-pipeline", run_type="chain")  # CHANGED: name only
def run_rag(user_query, session):  # CHANGED: added `session` parameter
    total_start = time.time()

    run_tree = get_current_run_tree()
    if run_tree:
        session["last_langsmith_run_id"] = str(run_tree.id)  # CHANGED: was st.session_state.last_langsmith_run_id

    memory_context = build_memory_context(session)  # CHANGED: pass session instead of reading st

    retrieval_start = time.time()
    docs, embedding_input_tokens = similarity_search(user_query, k=8)
    retrieval_time_ms = int((time.time() - retrieval_start) * 1000)

    docs_for_llm = docs
    docs_for_map = enrich_with_location(docs)

    session["last_docs"] = docs_for_map  # CHANGED: was st.session_state.last_docs

    distances = [d["distance"] for d in docs if d.get("distance") is not None]
    avg_distance = sum(distances) / len(distances) if distances else None

    llm_input_tokens = 0
    llm_output_tokens = 0
    llm_total_tokens = 0

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
            "json_valid": True,
            "fallback_used": True,
            "raw_output": json.dumps(fallback)
        }

        metric_row_id = insert_evaluation_metric(metric_row)

        session["last_user_query"] = user_query                      # CHANGED: was st.session_state.*
        session["last_docs_for_llm"] = docs_for_llm                  # CHANGED
        session["last_parsed_recommendations"] = fallback             # CHANGED
        session["last_metric_row_id"] = metric_row_id                 # CHANGED
        session["conversation_memory"].append({                       # CHANGED
            "user": user_query,
            "assistant": fallback
        })

        return fallback

    review_context = build_review_context(docs_for_llm)

    system_prompt = f"""
You are a professional food critic.

Generate up to THREE restaurant recommendations based on the review excerpts provided.

Return exactly 3 recommendations only if 3 distinct relevant restaurants are provided in the review excerpts.
If fewer than 3 distinct relevant restaurants are available, return only the relevant ones.

Return ONLY valid JSON using this structure:

[
  {{
    "restaurant": "restaurant name",
    "dish": "specific dish mentioned in reviews",
    "description": "short recommendation",
    "review_excerpt": "verbatim or lightly trimmed quote from the review text",
    "why_this_was_selected": "brief explanation tying the user query to the review",
    "photos": ["photo_url1", "photo_url2"]
  }}
]

Previous conversation:
{memory_context}

Review excerpts:
{review_context}

**Important:**
- Only recommend restaurants if the reviews are clearly relevant to the user query.
- If there are NO relevant reviews, return a single object with an empty restaurant and description indicating no recommendations.
- Use only information from the review excerpts provided. Do NOT make up any details.
- Provide your justification for selecting the review_excerpt.
"""

    generation_start = time.time()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        temperature=0.3
    )
    generation_time_ms = int((time.time() - generation_start) * 1000)

    usage = response.usage
    llm_input_tokens = getattr(usage, "prompt_tokens", None)
    llm_output_tokens = getattr(usage, "completion_tokens", None)
    llm_total_tokens = getattr(usage, "total_tokens", None)

    answer = response.choices[0].message.content or ""

    parsed, json_valid = parse_recommendations(answer)
    parsed = attach_addresses_to_recommendations(parsed, docs_for_map)

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
        "json_valid": json_valid,
        "fallback_used": False,
        "raw_output": json.dumps(parsed)
    }

    metric_row_id = insert_evaluation_metric(metric_row)

    session["last_user_query"] = user_query               # CHANGED: was st.session_state.*
    session["last_docs_for_llm"] = docs_for_llm           # CHANGED
    session["last_parsed_recommendations"] = parsed        # CHANGED
    session["last_metric_row_id"] = metric_row_id          # CHANGED
    session["conversation_memory"].append({                # CHANGED
        "user": user_query,
        "assistant": parsed
    })

    return parsed


# -----------------------------
# LOAD TEST
# -----------------------------
TEST_QUERIES = [
    "best ramen uptown charlotte",
    "gluten free brunch spots charlotte",
    "late night tacos downtown charlotte",
    "vegetarian friendly restaurants charlotte",
    "best burger in charlotte",
    "cheap lunch near noda charlotte",
    "sushi happy hour charlotte",
    "rooftop bar food charlotte",
]

results = []
lock = threading.Lock()

def run_one_user(user_id):
    query = random.choice(TEST_QUERIES)

    # Each simulated user gets their own fresh session dict,
    # mirroring how each Streamlit user gets their own st.session_state
    session = {
        "conversation_memory": [],
        "recommended_restaurants": set(),
        "last_user_query": None,
        "last_docs_for_llm": None,
        "last_parsed_recommendations": None,
        "last_metric_row_id": None,
        "last_docs": None,
        "last_langsmith_run_id": None,
    }

    print(f"[User {user_id}] Starting: '{query}'")
    t0 = time.time()
    try:
        run_rag(query, session)
        elapsed = time.time() - t0
        status = "ok"
    except Exception as e:
        elapsed = time.time() - t0
        status = f"error: {e}"

    with lock:
        results.append({"user": user_id, "query": query, "elapsed": elapsed, "status": status})
        print(f"[User {user_id}] {status} in {elapsed:.1f}s")


if __name__ == "__main__":
    N_USERS = 20  # increase to 10, 20 to stress test

    print(f"\nStarting load test with {N_USERS} concurrent users...\n")
    threads = [threading.Thread(target=run_one_user, args=(i,)) for i in range(N_USERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ok = [r for r in results if r["status"] == "ok"]
    errors = [r for r in results if r["status"] != "ok"]
    times = [r["elapsed"] for r in ok]

    print(f"\n--- Results ({N_USERS} users) ---")
    print(f"Success: {len(ok)}/{N_USERS}")
    if times:
        print(f"Avg:  {sum(times)/len(times):.1f}s")
        print(f"Min:  {min(times):.1f}s")
        print(f"Max:  {max(times):.1f}s")
    if errors:
        print(f"\nErrors:")
        for e in errors:
            print(f"  User {e['user']}: {e['status']}")
