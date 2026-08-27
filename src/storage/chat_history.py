import json, sqlite3, uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / ".storage" / "chat_history.db"
MAX_THREAD_TITLE_LENGTH = 80
def _now(): return datetime.now(timezone.utc).isoformat()
@contextmanager
def _connect(path=DEFAULT_DB_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True); conn=sqlite3.connect(path, timeout=5); conn.execute("PRAGMA foreign_keys=ON")
    try: yield conn; conn.commit()
    except BaseException: conn.rollback(); raise
    finally: conn.close()
def initialize_database(path=DEFAULT_DB_PATH):
    with _connect(path) as c:
        c.executescript("CREATE TABLE IF NOT EXISTS chat_threads(id TEXT PRIMARY KEY,title TEXT NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL);CREATE TABLE IF NOT EXISTS chat_messages(id TEXT PRIMARY KEY,thread_id TEXT NOT NULL,sequence INTEGER NOT NULL,role TEXT NOT NULL,content TEXT NOT NULL,route TEXT,reason TEXT,duration_seconds REAL,sources_json TEXT,created_at TEXT NOT NULL,FOREIGN KEY(thread_id) REFERENCES chat_threads(id) ON DELETE CASCADE,UNIQUE(thread_id,sequence));")
def title_from_prompt(prompt, limit=56):
    title=" ".join(prompt.split()); return title if len(title)<=limit else title[:limit-3].rstrip()+"..."
def normalize_thread_title(title, limit=MAX_THREAD_TITLE_LENGTH):
    normalized=" ".join(str(title).replace("\r", " ").replace("\n", " ").split())
    if not normalized: raise ValueError("Thread title cannot be empty.")
    return normalized[:limit].rstrip()
def create_thread(prompt, path=DEFAULT_DB_PATH):
    initialize_database(path); ident=str(uuid.uuid4()); now=_now()
    with _connect(path) as c: c.execute("INSERT INTO chat_threads VALUES(?,?,?,?)",(ident,title_from_prompt(prompt),now,now))
    return ident
def list_threads(path=DEFAULT_DB_PATH, limit=20):
    initialize_database(path)
    with _connect(path) as c: return [{"id":r[0],"title":r[1],"updated_at":r[2]} for r in c.execute("SELECT id,title,updated_at FROM chat_threads ORDER BY updated_at DESC LIMIT ?",(limit,))]
def rename_thread(thread_id, new_title, path=DEFAULT_DB_PATH):
    title=normalize_thread_title(new_title); now=_now(); initialize_database(path)
    with _connect(path) as c:
        changed=c.execute("UPDATE chat_threads SET title=?,updated_at=? WHERE id=?",(title,now,thread_id)).rowcount
        if changed != 1: raise KeyError("Chat thread was not found.")
    return title
def add_message(thread_id, message, path=DEFAULT_DB_PATH):
    initialize_database(path); now=_now(); sources=json.dumps(message.get("sources", "")); duration=message.get("elapsed", "").removesuffix("s") or None
    with _connect(path) as c:
        sequence=c.execute("SELECT COALESCE(MAX(sequence),-1)+1 FROM chat_messages WHERE thread_id=?",(thread_id,)).fetchone()[0]
        c.execute("INSERT INTO chat_messages VALUES(?,?,?,?,?,?,?,?,?,?)",(str(uuid.uuid4()),thread_id,sequence,message["role"],message["content"],message.get("route"),message.get("reason"),duration,sources,now))
        c.execute("UPDATE chat_threads SET updated_at=? WHERE id=?",(now,thread_id))
def load_messages(thread_id, path=DEFAULT_DB_PATH):
    initialize_database(path)
    with _connect(path) as c: rows=c.execute("SELECT role,content,route,reason,duration_seconds,sources_json FROM chat_messages WHERE thread_id=? ORDER BY sequence",(thread_id,)).fetchall()
    result=[]
    for role,content,route,reason,duration,sources in rows:
        try: sources=json.loads(sources or '""')
        except json.JSONDecodeError: sources=""
        result.append({"role":role,"content":content,"route":route or "","reason":reason or "","elapsed":f"{duration:g}s" if duration is not None else "","sources":sources if isinstance(sources,str) else ""})
    return result
def delete_thread(thread_id, path=DEFAULT_DB_PATH):
    initialize_database(path)
    with _connect(path) as c:
        changed=c.execute("DELETE FROM chat_threads WHERE id=?",(thread_id,)).rowcount
        if changed != 1: raise KeyError("Chat thread was not found.")
