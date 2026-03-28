import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import psycopg2
import psycopg2.extras
import os
import json
from geopy.geocoders import Nominatim
from functools import lru_cache
import pandas as pd
import re
import pydeck as pdk
import time

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_PASSWORD = os.getenv("PASSWORD")

client = OpenAI(api_key=OPENAI_API_KEY)

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
# STREAMLIT SESSION MEMORY
# -----------------------------
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = []

if "recommended_restaurants" not in st.session_state:
    st.session_state.recommended_restaurants = set()

# -----------------------------
# DB CONNECTION
# -----------------------------
def get_connection():
    return psycopg2.connect(
        user="postgres.teyeutbzecbobotobhzc",
        password=DB_PASSWORD,
        host="aws-0-us-west-2.pooler.supabase.com",
        port=6543,
        dbname="postgres",
    )

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
                %(json_valid)s,
                %(fallback_used)s,
                %(raw_output)s::jsonb
            )
        """

        cur.execute(query, metric_row)
        conn.commit()

    except Exception as e:
        print("Error inserting evaluation metric:", e)

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

# -----------------------------
# VECTOR SEARCH
# -----------------------------
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
            WHERE rp.review_id = rc.review_id
        ) photo_data ON TRUE
        WHERE rc.embedding IS NOT NULL
        AND rc.embedding <=> %s::vector < 0.6
        ORDER BY rc.embedding <=> %s::vector
        LIMIT %s
    """

    cur.execute(query, (embedding_str, embedding_str, embedding_str, k))
    rows = cur.fetchall()

    cur.close()
    conn.close()

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
            SELECT place_id, address
            FROM place_table
            WHERE place_id = ANY(%s)
        """, (place_ids,))

        address_map = {
            r["place_id"]: r["address"]
            for r in cur.fetchall()
        }

    cur.close()
    conn.close()

    # ⚠️ IMPORTANT: create a NEW list (don’t mutate original)
    enriched = []

    for row in rows:
        new_row = dict(row)  # copy

        address = address_map.get(row.get("place_id"))
        new_row["address"] = address

        # optional geocoding
        new_row["latitude"] = None
        new_row["longitude"] = None

        if address:
            normalized_address = normalize_address_for_geocoding(address)
            lat, lon = geocode_address(normalized_address)

        if lat is None or lon is None:
            fallback_address = strip_suite(normalized_address)
            lat, lon = geocode_address(fallback_address)

        new_row["latitude"] = lat
        new_row["longitude"] = lon

        enriched.append(new_row)

    return enriched

# -----------------------------
# BUILD MEMORY CONTEXT
# -----------------------------
def build_memory_context():

    memory = st.session_state.conversation_memory

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
def attach_addresses_to_recommendations(recommendations, docs_for_map):
    address_lookup = {}

    for d in docs_for_map:
        place_name = d.get("place_name")
        if place_name:
            address_lookup[place_name.strip().lower()] = {
                "address": d.get("address"),
                "latitude": d.get("latitude"),
                "longitude": d.get("longitude"),
            }

    enriched_recs = []

    for rec in recommendations:
        new_rec = dict(rec)
        restaurant = rec.get("restaurant", "")
        key = restaurant.strip().lower()

        location_data = address_lookup.get(key, {})
        new_rec["address"] = location_data.get("address")
        new_rec["latitude"] = location_data.get("latitude")
        new_rec["longitude"] = location_data.get("longitude")

        enriched_recs.append(new_rec)

    return enriched_recs


# -----------------------------
# MAIN RAG FUNCTION
# -----------------------------
def run_rag(user_query):
    """
    Main RAG function:
    - Performs similarity search
    - Returns recommendations or fallback if no relevant reviews
    - Always appends the result to conversation memory
    """
    total_start = time.time()

    memory_context = build_memory_context()

    retrieval_start = time.time()
    docs, embedding_input_tokens = similarity_search(user_query, k=20)
    retrieval_time_ms = int((time.time() - retrieval_start) * 1000)

# ✅ Use original docs for LLM (UNCHANGED)
    docs_for_llm = docs

# ✅ Use enriched docs for map/UI
    docs_for_map = enrich_with_location(docs)


    st.session_state.last_docs = docs_for_map

    distances = [d["distance"] for d in docs if d.get("distance") is not None]
    avg_distance = sum(distances) / len(distances) if distances else None

    llm_input_tokens = 0
    llm_output_tokens = 0
    llm_total_tokens = 0

    # -----------------------------
    # Fallback for irrelevant queries
    # -----------------------------
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
        insert_evaluation_metric(metric_row)

        # Append to memory
        st.session_state.conversation_memory.append({
            "user": user_query,
            "assistant": fallback
        })

        return fallback

    # -----------------------------
    # Build review context for LLM
    # -----------------------------
    review_context = build_review_context(docs)

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

    # -----------------------------
    # Call LLM
    # -----------------------------
    generation_start = time.time()
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # stable model for chat
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

    answer = response.choices[0].message.content

    # -----------------------------
    # Parse response safely
    # -----------------------------
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
    insert_evaluation_metric(metric_row)
    # -----------------------------
    # Append to conversation memory
    # -----------------------------
    st.session_state.conversation_memory.append({
        "user": user_query,
        "assistant": parsed
    })

    return parsed
# -----------------------------
# -- RENDER SMALL MAP FOR EACH RECOMMENDATION --
def render_small_map(lat, lon, restaurant_name="Restaurant"):
    lat = float(lat)
    lon = float(lon)

    df = pd.DataFrame([{
        "lat": lat,
        "lon": lon,
        "name": restaurant_name
    }])

    st.pydeck_chart(
        pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(
                latitude=lat,
                longitude=lon,
                zoom=13,
                pitch=0,
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=df,
                    get_position='[lon, lat]',
                    get_radius=80,
                    get_fill_color=[255, 0, 0, 200],
                    stroked=True,
                    get_line_color=[0, 0, 0, 200],
                    line_width_min_pixels=1,
                    pickable=True,
                )
            ],
            tooltip={"text": "{name}"},
        ),
        height=180,
    )
# -----------------------------
# RENDER RECOMMENDATIONS
# -----------------------------
def render_recommendations(recs):

    for r in recs:

        restaurant_name = r.get("restaurant")
        if not restaurant_name:
            restaurant_name = "No relevant restaurants"
        st.subheader(restaurant_name)

        dish = r.get("dish")
        if dish:
            st.write(f"**Recommended dish:** {dish}")

        st.write(r.get("description", ""))

        excerpt = r.get("review_excerpt")
        if excerpt:
            st.caption(f"🗣️ Review Excerpt: \"{excerpt}\"")

        why = r.get("why_this_was_selected")
        if why:
            st.caption(f"💡 Explanation: {why}")

        photos = r.get("photos", [])
        cols = st.columns(len(photos)) if photos else []

        for i, photo in enumerate(photos):
            with cols[i]:
                st.markdown(
                    f"""
                    <img src=\"{photo}\" style=\"width: 100%; max-width: 300px; max-height: 300px; object-fit: cover; border-radius: 8px;\" />
                    """,
                    unsafe_allow_html=True,
                )

        address = r.get("address")
        if address:
            st.write(f"**Address:** {address}")
        else:
            st.write("**Address:** Not available")

        lat = r.get("latitude")
        lon = r.get("longitude")

        if lat is not None and lon is not None:
            st.write("📍 Location")
            render_small_map(lat, lon, restaurant_name=restaurant_name)
        else:
            st.caption("Map not available for this restaurant.")

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="FoodFinder - Your friend for finding great food and drinks", layout="wide")

st.title("🍽️ FoodFinder - Your friend for finding great food and drinks!")
st.write("Ask for restaurant recommendations based on real reviews.")

if st.button("Clear history"):
    st.session_state.conversation_memory = []
    st.session_state.recommended_restaurants = set()
    st.rerun()


# -----------------------------
# DISPLAY CHAT HISTORY
# -----------------------------
for msg in st.session_state.conversation_memory:

    with st.chat_message("user"):
        st.write(msg["user"])

    with st.chat_message("assistant"):
        render_recommendations(msg["assistant"])


# -----------------------------
# USER INPUT
# -----------------------------
user_query = st.chat_input("What kind of food or drink are you looking for?")

if user_query:

    with st.chat_message("user"):
        st.write(user_query)

    with st.spinner("Analyzing reviews..."):
        recs = run_rag(user_query)

    with st.chat_message("assistant"):
        render_recommendations(recs)



