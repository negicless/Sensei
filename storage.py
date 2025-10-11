import sqlite3, os
from typing import List

class WatchlistDB:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS wl(uid TEXT, code TEXT, PRIMARY KEY(uid,code));")
    def add_code(self, uid: str, code: str):
        self.conn.execute("INSERT OR IGNORE INTO wl(uid,code) VALUES(?,?)", (str(uid), code)); self.conn.commit()
    def remove_code(self, uid: str, code: str):
        self.conn.execute("DELETE FROM wl WHERE uid=? AND code=?", (str(uid), code)); self.conn.commit()
    def list_codes(self, uid: str) -> List[str]:
        cur = self.conn.execute("SELECT code FROM wl WHERE uid=? ORDER BY code", (str(uid),))
        return [r[0] for r in cur.fetchall()]
