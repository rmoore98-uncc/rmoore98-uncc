import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import psycopg2
import psycopg2.extras
import os
import json

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_PASSWORD = os.getenv("PASSWORD")

client = OpenAI(api_key=OPENAI_API_KEY)

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
        SELECT
            rc.id,
            rc.review_id,
            rc.place_name,
            rc.chunk_text,
            COALESCE(photo_data.photos, '[]'::json) AS photos,
            rc.embedding <=> %s::vector AS distance
        FROM review_chunks rc
        LEFT JOIN LATERAL (
            SELECT json_agg(to_jsonb(rp) - 'review_id') AS photos
            FROM review_photos rp
            WHERE rp.review_id = rc.review_id
        ) photo_data ON TRUE
        WHERE rc.embedding IS NOT NULL
        AND rc.embedding <=> %s::vector < 0.4   -- ✅ NEW THRESHOLD
        ORDER BY rc.embedding <=> %s::vector
        LIMIT %s
    """

    cur.execute(query, (embedding_str, embedding_str, embedding_str, k))
    rows = cur.fetchall()

    cur.close()
    conn.close()

    return rows

# # Food related query function --------------
# def is_food_query(query):
#     response = client.chat.completions.create(
#         model="gpt-5-nano",
#         messages=[
#             {"role": "system", "content": "Answer only yes or no. Is this about food, restaurants, or dining?"},
#             {"role": "user", "content": query},
#         ],
#         temperature=0
#     )
#     return "yes" in response.choices[0].message.content.lower()
# #-----------------------------

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
Review Text: {d['chunk_text']}
Similarity Score: {round(d['distance'],4)}
Photo Links: {photos_text}
"""

    return context


# -----------------------------
# MAIN RAG FUNCTION
# -----------------------------
def run_rag(user_query):

    memory_context = build_memory_context()

    # ✅ STEP 1: FILTER NON-FOOD QUERIES
    # #if not is_food_query(user_query):
      #  return [{
      #      "restaurant": "",
      #      "dish": "",
      #      "description": "That question isn't related to food or restaurants. Try asking about dining recommendations.",
      #      "review_excerpt": "",
      #      "why_this_was_selected": "",
      #      "photos": []
      #  }]

    docs = similarity_search(user_query, k=8)

    # ✅ STEP 2: DISTANCE GUARD
    if docs:
        avg_distance = sum(d["distance"] for d in docs) / len(docs)
        if avg_distance > 0.45:
            docs = []

    # ✅ STEP 3: FALLBACK (STRUCTURED)
    if not docs:
        return [{
            "restaurant": "",
            "dish": "",
            "description": "There are no relevant reviews based on your input, try rephrasing your question or asking about something else.",
            "review_excerpt": "",
            "why_this_was_selected": "",
            "photos": []
        }]

    review_context = build_review_context(docs)

    system_prompt = f"""
You are a professional food critic.

Using the review excerpts, generate THREE restaurant recommendations.

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

Rules:
- Only recommend restaurants if clearly relevant to the query
- review_excerpt must come directly from the review text (max 25 words)
- why_this_was_selected must explain relevance (max 30 words)

Previous conversation:
{memory_context}

Review excerpts:
{review_context}

Use only the provided reviews. Do NOT make up information.
"""

    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        temperature=0.3
    )

    answer = response.choices[0].message.content

    try:
        parsed = json.loads(answer)
    except:
        parsed = []

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

        st.subheader(r.get("restaurant", "Restaurant"))

        dish = r.get("dish")
        if dish:
            st.write(f"**Recommended dish:** {dish}")

        st.write(r.get("description", ""))

        # ✅ NEW: review excerpt
        excerpt = r.get("review_excerpt")
        if excerpt:
            st.caption(f"🗣️ Review Excerpt: \"{excerpt}\"")

        # ✅ NEW: explanation
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