import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import psycopg2
import psycopg2.extras
import os
import json
import uuid

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DB_PASSWORD = os.getenv("PASSWORD")

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
# CHAT SESSION FUNCTIONS
# -----------------------------
def create_new_session():
    session_id = str(uuid.uuid4())
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO chat_sessions (session_id, title) VALUES (%s, %s)",
        (session_id, "New Chat")
    )
    conn.commit()

    cur.close()
    conn.close()
    return session_id


def get_all_sessions():
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute("""
        SELECT session_id, title
        FROM chat_sessions
        ORDER BY created_at DESC
    """)

    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def load_conversation(session_id):
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute("""
        SELECT user_message, assistant_response
        FROM conversations
        WHERE session_id = %s
        ORDER BY created_at
    """, (session_id,))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [{"user": r["user_message"], "assistant": r["assistant_response"]} for r in rows]


def save_message(session_id, user_msg, assistant_msg):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO conversations (session_id, user_message, assistant_response)
        VALUES (%s, %s, %s)
    """, (session_id, user_msg, json.dumps(assistant_msg)))

    conn.commit()
    cur.close()
    conn.close()


def update_session_title(session_id, user_query):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        UPDATE chat_sessions
        SET title = %s
        WHERE session_id = %s
        AND title = 'New Chat'
    """, (user_query[:40], session_id))

    conn.commit()
    cur.close()
    conn.close()

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
        SELECT rc.place_name, rc.chunk_text,
               COALESCE(photo_data.photos, '[]'::json) AS photos,
               rc.embedding <=> %s::vector AS distance
        FROM review_chunks rc
        LEFT JOIN LATERAL (
            SELECT json_agg(to_jsonb(rp) - 'review_id') AS photos
            FROM review_photos rp
            WHERE rp.review_id = rc.review_id
        ) photo_data ON TRUE
        WHERE rc.embedding IS NOT NULL
        AND rc.embedding <=> %s::vector < 0.4
        ORDER BY rc.embedding <=> %s::vector
        LIMIT %s
    """

    cur.execute(query, (embedding_str, embedding_str, embedding_str, k))
    rows = cur.fetchall()

    cur.close()
    conn.close()
    return rows

# -----------------------------
# BUILD CONTEXT
# -----------------------------
def build_review_context(docs):
    context = ""
    for d in docs:
        photos = d.get("photos") or []
        links = [p.get("photo_link") for p in photos if p.get("photo_link")]
        context += f"""
Restaurant: {d['place_name']}
Review Text: {d['chunk_text']}
Photo Links: {", ".join(links) if links else "None"}
"""
    return context

# -----------------------------
# RAG FUNCTION
# -----------------------------
def run_rag(user_query):
    docs = similarity_search(user_query, k=8)

    # Fallback
    if not docs:
        fallback = [{
            "restaurant": "",
            "dish": "",
            "description": "No relevant reviews found.",
            "review_excerpt": "",
            "why_this_was_selected": "",
            "photos": []
        }]

        save_message(st.session_state.session_id, user_query, fallback)

        st.session_state.conversation_memory.append({
            "user": user_query,
            "assistant": fallback
        })

        return fallback

    review_context = build_review_context(docs)

    system_prompt = f"""
Return JSON:
[
  {{
    "restaurant": "...",
    "dish": "...",
    "description": "...",
    "review_excerpt": "...",
    "why_this_was_selected": "...",
    "photos": []
  }}
]

{review_context}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0.3
    )

    try:
        parsed = json.loads(response.choices[0].message.content)
    except:
        parsed = [{
            "restaurant": "",
            "dish": "",
            "description": "Error generating recommendations.",
            "review_excerpt": "",
            "why_this_was_selected": "",
            "photos": []
        }]

    save_message(st.session_state.session_id, user_query, parsed)
    update_session_title(st.session_state.session_id, user_query)

    st.session_state.conversation_memory.append({
        "user": user_query,
        "assistant": parsed
    })

    return parsed

# -----------------------------
# STREAMLIT STATE INIT
# -----------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = create_new_session()

if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = []

# -----------------------------
# SIDEBAR (ChatGPT-style)
# -----------------------------
st.sidebar.title("💬 Conversations")

if st.sidebar.button("➕ New Chat"):
    st.session_state.session_id = create_new_session()
    st.session_state.conversation_memory = []
    st.rerun()

sessions = get_all_sessions()

for s in sessions:
    label = s["title"]
    if s["session_id"] == st.session_state.session_id:
        label = f"👉 {label}"

    if st.sidebar.button(label, key=s["session_id"]):
        st.session_state.session_id = s["session_id"]
        st.session_state.conversation_memory = load_conversation(s["session_id"])
        st.rerun()

# -----------------------------
# MAIN UI
# -----------------------------
st.title("🍽️ FoodFinder")

for msg in st.session_state.conversation_memory:
    with st.chat_message("user"):
        st.write(msg["user"])
    with st.chat_message("assistant"):
        for r in msg["assistant"]:
            name = r.get("restaurant") or "No relevant restaurants"
            st.subheader(name)
            st.write(r.get("description", ""))

user_query = st.chat_input("Ask for food recommendations")

if user_query:
    with st.chat_message("user"):
        st.write(user_query)

    recs = run_rag(user_query)

    with st.chat_message("assistant"):
        for r in recs:
            name = r.get("restaurant") or "No relevant restaurants"
            st.subheader(name)
            st.write(r.get("description", ""))