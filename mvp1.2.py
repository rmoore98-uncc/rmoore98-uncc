import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import psycopg2
import psycopg2.extras
import os
import json
from geopy.geocoders import Nominatim
from functools import lru_cache

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_PASSWORD = os.getenv("PASSWORD")

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# GEOCODING (Address → Lat/Lon)
# -----------------------------
geolocator = Nominatim(user_agent="foodfinder_app")

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
# VECTOR SEARCH
# -----------------------------
def similarity_search(query_text, k=5):

    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query_text
    ).data[0].embedding

    embedding_str = "[" + ",".join(map(str, embedding)) + "]"

    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    query = """
        WITH ranked AS (
        SELECT
            rc.id,
            rc.review_id,
            rc.place_id,
            rc.place_name,
            rc.chunk_text,
            COALESCE(photo_data.photos, '[]'::json) AS photos,
            rc.embedding <=> %s::vector AS distance,
            ROW_NUMBER() OVER (
                PARTITION BY rc.place_name
                ORDER BY rc.embedding <=> %s::vector
            ) AS rn
        FROM review_chunks rc
        LEFT JOIN LATERAL (
            SELECT json_agg(to_jsonb(rp) - 'review_id') AS photos
            FROM review_photos rp
            WHERE rp.review_id = rc.review_id
        ) photo_data ON TRUE
        WHERE rc.embedding IS NOT NULL
          AND rc.embedding <=> %s::vector < 0.8
    )
    SELECT
        id,
        review_id,
        place_id,
        place_name,
        chunk_text,
        photos,
        distance
    FROM ranked
    WHERE rn = 1
    ORDER BY distance
    LIMIT %s
    """

    cur.execute(query, (embedding_str, embedding_str, embedding_str, k))
    rows = cur.fetchall()

    cur.close()
    conn.close()

    return rows

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
        if address:
            lat, lon = geocode_address(address)
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
# MAIN RAG FUNCTION
# -----------------------------
def run_rag(user_query):
    """
    Main RAG function:
    - Performs similarity search
    - Returns recommendations or fallback if no relevant reviews
    - Always appends the result to conversation memory
    """

    memory_context = build_memory_context()
    docs = similarity_search(user_query, k=8)

# ✅ Use original docs for LLM (UNCHANGED)
    docs_for_llm = docs

# ✅ Use enriched docs for map/UI
    docs_for_map = enrich_with_location(docs)

    st.session_state.last_docs = docs_for_map

    # -----------------------------
    # Fallback for irrelevant queries
    # -----------------------------
    if not docs:
        fallback = [{
            "description": "There are no relevant reviews based on your input, try rephrasing your question or asking about something else.",
        }]

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
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # stable model for chat
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        temperature=0.3
    )

    answer = response.choices[0].message.content

    # -----------------------------
    # Parse response safely
    # -----------------------------
    try:
        parsed = json.loads(answer)
    except:
        parsed = [{
            "restaurant": "",
            "dish": "",
            "description": "Error generating recommendations. Please try again.",
            "review_excerpt": "",
            "why_this_was_selected": "",
            "photos": []
        }]

    # -----------------------------
    # Append to conversation memory
    # -----------------------------
    st.session_state.conversation_memory.append({
        "user": user_query,
        "assistant": parsed
    })

    return parsed

# -----------------------------
# RENDER RECOMMENDATIONS
# -----------------------------
def render_recommendations(recs):

    for r in recs:

        # Subheader logic for fallback
        restaurant_name = r.get("restaurant")
        if not restaurant_name:
            restaurant_name = "No relevant restaurants"
        st.subheader(restaurant_name)

        dish = r.get("dish")
        if dish:
            st.write(f"**Recommended dish:** {dish}")

        st.write(r.get("description", ""))

        #review excerpt
        excerpt = r.get("review_excerpt")
        if excerpt:
            st.caption(f"🗣️ Review Excerpt: \"{excerpt}\"")

        #explanation
        why = r.get("why_this_was_selected")
        if why:
            st.caption(f"💡 Explanation: {why}")

        photos = r.get("photos", [])

        cols = st.columns(len(photos)) if photos else []

        for i, photo in enumerate(photos):
            with cols[i]:
                st.markdown(
                    f"""
                    <img src=\"{photo}\" style=\"width: 100%; max-width: 400px; max-height: 400px; object-fit: cover; border-radius: 8px;\" />
                    """,
                    unsafe_allow_html=True,
                )


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

