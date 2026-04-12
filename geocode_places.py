"""
One-time script to pre-populate latitude and longitude in place_table.
Run this locally once — it updates the same Supabase DB the deployed app uses.

Usage:
    python geocode_places.py
"""

import time
import re
import os
import psycopg2
from geopy.geocoders import Nominatim
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

DB_PASSWORD = os.getenv("PASSWORD")

geolocator = Nominatim(user_agent="foodfinder_geocode_script", timeout=10)

def get_connection():
    return psycopg2.connect(
        user="postgres.teyeutbzecbobotobhzc",
        password=DB_PASSWORD,
        host="aws-0-us-west-2.pooler.supabase.com",
        port=6543,
        dbname="postgres",
    )

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
        print(f"  Geocoding error: {e}")
    return None, None


conn = get_connection()
cur = conn.cursor()

cur.execute("""
    SELECT place_id, address
    FROM place_table
    WHERE address IS NOT NULL AND latitude IS NULL
""")
rows = cur.fetchall()
print(f"Found {len(rows)} places to geocode.\n")

for i, (place_id, address) in enumerate(rows):
    normalized = normalize_address_for_geocoding(address)
    lat, lon = geocode_address(normalized)

    if lat is None or lon is None:
        lat, lon = geocode_address(strip_suite(normalized))

    cur.execute(
        "UPDATE place_table SET latitude = %s, longitude = %s WHERE place_id = %s",
        (lat, lon, place_id)
    )
    conn.commit()
    print(f"[{i+1}/{len(rows)}] {place_id}: {lat}, {lon}")

    time.sleep(1.1)  # Nominatim rate limit: 1 request/second

cur.close()
conn.close()
print("\nDone.")
