from sqlalchemy import Column, Integer, String, DateTime, Text, Float, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime
import json
import os

Base = declarative_base()

class AccessLog(Base):
    __tablename__ = 'access_logs'
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.now)
    name = Column(String(255), index=True)
    status = Column(String(255))
    confidence = Column(Float, nullable=True)  # Match confidence 0-100%
    image_path = Column(String(512), nullable=True)  # Path to captured face snapshot
    department = Column(String(255), nullable=True)  # Snapshot of the user's department at scan time
    check_type = Column(String(20), nullable=True, default="DETECT")  # IN / OUT / DETECT (passive dashboard sighting)
    latitude = Column(Float, nullable=True)   # GPS captured by the browser at check-in/out
    longitude = Column(Float, nullable=True)

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, index=True)
    image_path = Column(String(512), nullable=True)
    registered_at = Column(DateTime, default=datetime.datetime.now)
    face_encodings = Column(Text, nullable=True)  # Stored as JSON string
    department = Column(String(255), nullable=True)

from backend.paths import APP_DIR as PROJECT_ROOT
# Absolute path so the same database is used no matter which directory the app
# is launched from (a shortcut on the Desktop has a different working dir).
SQLITE_URL = "sqlite:///" + os.path.join(PROJECT_ROOT, "faceguard.db").replace("\\", "/")


def get_db_url():
    """MySQL only when config.json explicitly asks for it, otherwise SQLite.

    Previously this always built a MySQL URL from defaults, so every start-up
    spent seconds failing to reach a server that was never configured.
    """
    config_path = os.path.join(PROJECT_ROOT, "config.json")
    config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            config = {}

    if not config.get("use_mysql") and not config.get("db_host"):
        return SQLITE_URL

    host = config.get("db_host", "localhost")
    port = config.get("db_port", 3306)
    user = config.get("db_user", "root")
    password = config.get("db_password", "root")
    db_name = config.get("db_name", "faceguard_db")
    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}"


DB_URL = get_db_url()
try:
    if DB_URL.startswith("sqlite"):
        engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
    else:
        engine = create_engine(DB_URL, pool_pre_ping=True)
    engine.connect().close()
    print(f"[INFO] Database: {DB_URL.split('@')[-1]}")
except Exception as e:
    print(f"[WARN] Database connection failed: {e}. Falling back to SQLite.")
    engine = create_engine(SQLITE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def _add_missing_columns():
    """Lightweight migration: `create_all` only creates missing *tables*, so a
    database created before the department/check-in columns existed needs
    them added by hand or the app crashes reading its own schema."""
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    wanted = {
        "users": [("department", "VARCHAR(255)")],
        "access_logs": [
            ("department", "VARCHAR(255)"),
            ("check_type", "VARCHAR(20)"),
            ("latitude", "FLOAT"),
            ("longitude", "FLOAT"),
        ],
    }
    with engine.begin() as conn:
        for table, columns in wanted.items():
            if table not in existing_tables:
                continue
            existing_cols = {c["name"] for c in inspector.get_columns(table)}
            for col_name, col_type in columns:
                if col_name in existing_cols:
                    continue
                try:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}"))
                    print(f"[INFO] Migrated: added {table}.{col_name}")
                except Exception as e:
                    print(f"[WARN] Could not add {table}.{col_name}: {e}")


def init_db():
    Base.metadata.create_all(bind=engine)
    _add_missing_columns()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
