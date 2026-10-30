"""
Database layer for AgriSchedule.

Defaults to SQLite so the whole app runs with zero external services —
good for an academic MVP/demo.

Postgres-ready: set the DATABASE_URL environment variable, e.g.

    postgresql://user:password@host:5432/dbname

...and every function below talks to Postgres instead — nothing else
in the app needs to change. Requires `psycopg2-binary` to be installed
when Postgres mode is used (not needed for the default SQLite mode):

    pip install psycopg2-binary
"""

import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "agrischedule.db")

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
USE_POSTGRES = bool(DATABASE_URL)

if USE_POSTGRES:
    import psycopg2
    import psycopg2.extras
else:
    import sqlite3


# ---------------------------------------------------------------------
# Low-level connection / query helpers (backend-agnostic)
# ---------------------------------------------------------------------

def get_conn():
    """Return a live connection. Callers are responsible for closing it."""
    if USE_POSTGRES:
        return psycopg2.connect(DATABASE_URL)
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)  # ensure "data" folder exists
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _cursor(conn):
    if USE_POSTGRES:
        return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    return conn.cursor()


def _q(sql):
    """Translate SQLite-style '?' placeholders to Postgres '%s' when needed.
    Everywhere else in this file, write queries using '?' only."""
    return sql.replace("?", "%s") if USE_POSTGRES else sql


def fetchall(sql, params=()):
    conn = get_conn()
    try:
        cur = _cursor(conn)
        cur.execute(_q(sql), params)
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def fetchone(sql, params=()):
    rows = fetchall(sql, params)
    return rows[0] if rows else None


def execute_write(sql, params=(), return_id=False):
    """Run an INSERT/UPDATE/DELETE. If return_id, returns the new row's id
    (SQLite's lastrowid, or Postgres via an appended RETURNING id)."""
    conn = get_conn()
    try:
        cur = _cursor(conn)
        if return_id and USE_POSTGRES:
            sql = sql.rstrip().rstrip(";") + " RETURNING id"
        cur.execute(_q(sql), params)
        new_id = None
        if return_id:
            new_id = cur.fetchone()["id"] if USE_POSTGRES else cur.lastrowid
        conn.commit()
        return new_id
    finally:
        conn.close()


# ---------------------------------------------------------------------
# Schema / seed data
# ---------------------------------------------------------------------

def init_db():
    conn = get_conn()
    cur = _cursor(conn)

    pk = "SERIAL PRIMARY KEY" if USE_POSTGRES else "INTEGER PRIMARY KEY AUTOINCREMENT"

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS crop_stages (
            id {pk},
            crop TEXT NOT NULL,
            variety TEXT NOT NULL,
            day_offset INTEGER NOT NULL,
            stage_name TEXT NOT NULL,
            category TEXT NOT NULL,        -- irrigation | fertilizer | pesticide | weeding | observation | harvest
            instructions TEXT NOT NULL,
            weather_sensitive INTEGER NOT NULL DEFAULT 0  -- 1 if rain/wind should delay this operation
        )
    """)

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS farmers (
            id {pk},
            name TEXT NOT NULL,
            phone TEXT,
            location TEXT NOT NULL,
            latitude REAL,
            longitude REAL,
            crop TEXT NOT NULL,
            variety TEXT NOT NULL,
            field_area REAL NOT NULL,
            sowing_date TEXT NOT NULL,
            language TEXT NOT NULL DEFAULT 'en',
            created_at TEXT NOT NULL
        )
    """)

    # Existing databases created before the phone field was added won't have
    # this column yet — add it if missing, ignore the error if it's already there.
    try:
        cur.execute(_q("ALTER TABLE farmers ADD COLUMN phone TEXT"))
        conn.commit()
    except Exception:
        conn.rollback()

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS pest_reports (
            id {pk},
            farmer_id INTEGER,
            image_name TEXT,
            diagnosis TEXT,
            confidence REAL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (farmer_id) REFERENCES farmers (id)
        )
    """)

    conn.commit()

    # Seed default crop schedules only if table is empty
    cur.execute("SELECT COUNT(*) as c FROM crop_stages")
    row = cur.fetchone()
    count = row["c"]
    if count == 0:
        seed_default_schedules(conn)

    conn.close()


def seed_default_schedules(conn):
    """Seed a few realistic crop calendars so the app is usable out of the box."""
    rows = []

    # --- RICE (generic, applies to common transplanted varieties) ---
    rice_stages = [
        (0, "Nursery seed sowing", "sowing", "Soak seeds 24 hrs, sow in prepared nursery bed.", 0),
        (14, "Transplanting", "transplanting", "Transplant 25-30 day old... (use 14-21 day seedlings for short duration varieties).", 1),
        (21, "First weeding", "weeding", "Hand weed or apply pre-emergence herbicide as per local recommendation.", 1),
        (25, "First fertilizer top dressing (Urea)", "fertilizer", "Apply nitrogen top dressing at active tillering stage.", 1),
        (35, "First pest scouting", "observation", "Scout for stem borer and leaf folder; check for egg masses.", 0),
        (45, "Second weeding", "weeding", "Remove weeds before panicle initiation stage begins.", 1),
        (55, "Second fertilizer top dressing", "fertilizer", "Apply nitrogen at panicle initiation stage for grain filling.", 1),
        (60, "Pesticide spray if pest threshold crossed", "pesticide", "Spray recommended pesticide ONLY if pest scouting shows threshold levels.", 1),
        (75, "Flowering stage water management", "irrigation", "Maintain 5cm standing water; avoid moisture stress during flowering.", 1),
        (100, "Maturity check", "observation", "Check grain hardness and field yellowing (~80% grains golden).", 0),
        (110, "Harvest", "harvest", "Harvest when 80-85% of grains have turned golden yellow.", 1),
    ]
    for day, stage, cat, instr, ws in rice_stages:
        rows.append(("Rice", "Generic (Short Duration)", day, stage, cat, instr, ws))

    # --- WHEAT ---
    wheat_stages = [
        (0, "Sowing", "sowing", "Sow seeds at 4-5 cm depth in well-prepared field.", 0),
        (20, "First irrigation (Crown Root Initiation)", "irrigation", "Critical irrigation stage — do not delay even if soil looks moist.", 1),
        (25, "First weeding", "weeding", "Hand weed or apply post-emergence herbicide.", 1),
        (40, "First fertilizer top dressing", "fertilizer", "Apply nitrogen top dressing after first irrigation.", 1),
        (60, "Second irrigation (Tillering)", "irrigation", "Irrigate at tillering stage for better yield.", 1),
        (80, "Second fertilizer top dressing", "fertilizer", "Apply nitrogen at jointing stage.", 1),
        (90, "Third irrigation (Flowering)", "irrigation", "Critical stage; avoid moisture stress during flowering.", 1),
        (110, "Pest & disease scouting", "observation", "Check for rust and aphid infestation.", 0),
        (130, "Maturity check", "observation", "Grains should be hard, straw turning golden.", 0),
        (140, "Harvest", "harvest", "Harvest when moisture content is around 20-25%.", 1),
    ]
    for day, stage, cat, instr, ws in wheat_stages:
        rows.append(("Wheat", "Generic", day, stage, cat, instr, ws))

    # --- TOMATO ---
    tomato_stages = [
        (0, "Nursery sowing", "sowing", "Sow seeds in trays/nursery bed with fine, well-drained soil.", 0),
        (25, "Transplanting", "transplanting", "Transplant 25 day old seedlings to main field, 45x60 cm spacing.", 1),
        (35, "First fertilizer dose", "fertilizer", "Apply NPK dose after establishment.", 1),
        (40, "Staking", "observation", "Provide stakes/support as plants grow.", 0),
        (45, "Pest scouting (fruit borer, whitefly)", "observation", "Scout for fruit borer and whitefly-transmitted viral symptoms.", 0),
        (50, "Pesticide/IPM spray if needed", "pesticide", "Spray only if pest scouting crosses threshold; prefer IPM methods.", 1),
        (60, "Second fertilizer dose", "fertilizer", "Apply second dose at flowering stage.", 1),
        (65, "Flowering stage irrigation", "irrigation", "Maintain consistent soil moisture during flowering/fruit set.", 1),
        (80, "First harvest window begins", "harvest", "Begin harvesting fruits at breaker/turning stage.", 1),
    ]
    for day, stage, cat, instr, ws in tomato_stages:
        rows.append(("Tomato", "Generic", day, stage, cat, instr, ws))

    cur = _cursor(conn)
    cur.executemany(
        _q("""INSERT INTO crop_stages
              (crop, variety, day_offset, stage_name, category, instructions, weather_sensitive)
              VALUES (?, ?, ?, ?, ?, ?, ?)"""),
        rows,
    )
    conn.commit()


# ---------------------------------------------------------------------
# Crop schedule helpers
# ---------------------------------------------------------------------

def get_crops():
    rows = fetchall("SELECT DISTINCT crop FROM crop_stages ORDER BY crop")
    return [r["crop"] for r in rows]


def get_varieties(crop):
    rows = fetchall(
        "SELECT DISTINCT variety FROM crop_stages WHERE crop = ? ORDER BY variety", (crop,)
    )
    return [r["variety"] for r in rows]


def get_schedule(crop, variety):
    return fetchall(
        """SELECT * FROM crop_stages WHERE crop = ? AND variety = ?
           ORDER BY day_offset ASC""",
        (crop, variety),
    )


def add_crop_stage(crop, variety, day_offset, stage_name, category, instructions, weather_sensitive):
    execute_write(
        """INSERT INTO crop_stages
           (crop, variety, day_offset, stage_name, category, instructions, weather_sensitive)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (crop, variety, day_offset, stage_name, category, instructions, int(weather_sensitive)),
    )


def delete_crop_stage(stage_id):
    execute_write("DELETE FROM crop_stages WHERE id = ?", (stage_id,))


# ---------------------------------------------------------------------
# Farmer helpers
# ---------------------------------------------------------------------

def add_farmer(name, phone, location, latitude, longitude, crop, variety, field_area, sowing_date, language="en"):
    farmer_id = execute_write(
        """INSERT INTO farmers
           (name, phone, location, latitude, longitude, crop, variety, field_area, sowing_date, language, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (name, phone, location, latitude, longitude, crop, variety, field_area,
         sowing_date, language, datetime.now().isoformat()),
        return_id=True,
    )
    return farmer_id


def get_all_farmers():
    return fetchall("SELECT * FROM farmers ORDER BY created_at DESC")


def get_farmer(farmer_id):
    return fetchone("SELECT * FROM farmers WHERE id = ?", (farmer_id,))


def delete_farmer(farmer_id):
    execute_write("DELETE FROM farmers WHERE id = ?", (farmer_id,))


def get_all_crop_stages():
    return fetchall("SELECT * FROM crop_stages ORDER BY crop, variety, day_offset ASC")


def log_pest_report(farmer_id, image_name, diagnosis, confidence):
    execute_write(
        """INSERT INTO pest_reports (farmer_id, image_name, diagnosis, confidence, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (farmer_id, image_name, diagnosis, confidence, datetime.now().isoformat()),
    )


def get_all_pest_reports():
    return fetchall("""
        SELECT p.*, f.name as farmer_name, f.crop as farmer_crop
        FROM pest_reports p
        LEFT JOIN farmers f ON p.farmer_id = f.id
        ORDER BY p.created_at DESC
    """)
