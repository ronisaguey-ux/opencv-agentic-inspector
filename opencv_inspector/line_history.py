"""Line history — the memory the agent reasons over.

A detector sees one frame. The decision needs the line's recent past: how many of this
defect, how recently, is it getting worse. This store holds that context and answers the
retrieval the agent asks for. SQLite so the state survives a restart (a real line runs for
weeks); a demo can seed it in memory.

`similar()` ranks priors by class match, then recency — the two signals a supervisor
actually uses when they say "this looks like the one from an hour ago".
"""
import sqlite3
import time
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS defects (
  id        TEXT PRIMARY KEY,
  station   TEXT NOT NULL,
  class     TEXT NOT NULL,
  severity  TEXT,
  confidence REAL,
  ts        REAL NOT NULL
);
"""


class LineHistory:
    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute(SCHEMA)
        self.db.commit()

    # -- writes ---------------------------------------------------------------
    def add(self, defect_id: str, station: str, cls: str, severity: str = "low",
            confidence: float = 1.0, ts: float = None):
        self.db.execute("INSERT OR REPLACE INTO defects VALUES (?,?,?,?,?,?)",
                        (defect_id, station, cls, severity, confidence,
                         ts if ts is not None else time.time()))
        self.db.commit()

    def seed(self, rows):
        for r in rows:
            self.add(r["id"], r["station"], r["class"], r.get("severity", "low"),
                     r.get("confidence", 1.0),
                     time.time() - r.get("minutes_ago", 0) * 60)

    # -- reads ----------------------------------------------------------------
    def recent(self, station: str, limit: int = 20) -> list:
        cur = self.db.execute(
            "SELECT id, station, class, severity, confidence, ts FROM defects "
            "WHERE station=? ORDER BY ts DESC LIMIT ?", (station, limit))
        return [self._row(r) for r in cur.fetchall()]

    def similar(self, station: str, cls: str, limit: int = 5) -> list:
        cur = self.db.execute(
            "SELECT id, station, class, severity, confidence, ts FROM defects "
            "WHERE station=? ORDER BY (class=?) DESC, ts DESC LIMIT ?",
            (station, cls, limit))
        return [self._row(r) for r in cur.fetchall()]

    def stats(self, station: str, window_min: int = 480) -> dict:
        since = time.time() - window_min * 60
        cur = self.db.execute(
            "SELECT class, COUNT(*) FROM defects WHERE station=? AND ts>=? GROUP BY class "
            "ORDER BY COUNT(*) DESC", (station, since))
        by_class = dict(cur.fetchall())
        cur = self.db.execute("SELECT COUNT(*) FROM defects WHERE station=? AND ts>=?",
                              (station, since))
        return {"window_minutes": window_min, "total": cur.fetchone()[0], "by_class": by_class}

    @staticmethod
    def _row(r) -> dict:
        defect_id, station, cls, sev, conf, ts = r
        return {
            "id": defect_id, "station": station, "class": cls, "severity": sev,
            "confidence": conf,
            "minutes_ago": round((time.time() - ts) / 60, 1),
            "at": datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds"),
        }

    # -- the tool the agent may call -----------------------------------------
    def __call__(self, station: str, limit: int = 20):
        return self.recent(station, limit)
