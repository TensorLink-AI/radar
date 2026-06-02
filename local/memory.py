"""Optional bitemporal memory layer for agents.

Disabled by default; set ``RADAR_MEMORY_ENABLED=1`` to switch on. When
off, every method on ``Memory`` is a silent no-op so callers can wire it
in unconditionally without changing current behavior.

One SQLite file (``RADAR_MEMORY_DB``, default ``local/radar_memory.db``)
with three tables: ``mem_nodes``, ``mem_edges``, ``mem_embeds``. Nodes
and edges are bitemporal — ``recall`` always filters by an ``as_of``
cutoff so "what did the agent know last Tuesday?" is a free query.
Embed/extract are pluggable; the defaults are a feature-hashing sketch
and a regex triple extractor so the module has zero runtime deps beyond
numpy + stdlib.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

EmbedFn = Callable[[str], np.ndarray]
ExtractFn = Callable[[str], list[dict]]

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_TRIPLE_RE = re.compile(r"\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*,\s*([^()]+?)\s*\)")
_DEFAULT_DIM = 256


def _hash_embed(text: str, dim: int = _DEFAULT_DIM) -> np.ndarray:
    """Normalized feature-hashing sketch — deterministic, no deps."""
    vec = np.zeros(dim, dtype=np.float32)
    for tok in _TOKEN_RE.findall(text.lower()):
        h = int.from_bytes(
            hashlib.blake2s(tok.encode(), digest_size=4).digest(), "little",
        )
        vec[(h >> 1) % dim] += 1.0 if (h & 1) else -1.0
    n = float(np.linalg.norm(vec))
    if n > 0:
        vec /= n
    return vec


def _regex_extract(text: str) -> list[dict]:
    """Pull ``(subj, pred, obj)`` triples written in parentheses."""
    out: list[dict] = []
    for m in _TRIPLE_RE.finditer(text):
        s, p, o = (x.strip() for x in m.groups())
        if s and p and o:
            out.append({"subj": s, "pred": p, "obj": o})
    return out


@dataclass
class MemoryHit:
    node_id: int
    kind: str
    text: str
    score: float
    props: dict = field(default_factory=dict)
    edges: list[dict] = field(default_factory=list)


@dataclass
class Memory:
    """Bitemporal triple + vector store with a hard on/off switch.

    Either fully enabled (writes + reads against SQLite) or fully no-op
    (every method returns empty / ``None`` / ``False``). Callers should
    not special-case the disabled path.
    """

    db_path: str = ""
    enabled: bool = False
    embed: EmbedFn = field(default=_hash_embed)
    extract: ExtractFn = field(default=_regex_extract)
    dim: int = _DEFAULT_DIM
    _conn: Optional[sqlite3.Connection] = field(
        default=None, init=False, repr=False,
    )

    @classmethod
    def from_env(cls, db_path: Optional[str] = None,
                 embed: Optional[EmbedFn] = None,
                 extract: Optional[ExtractFn] = None) -> "Memory":
        on = os.getenv("RADAR_MEMORY_ENABLED", "").lower() in (
            "1", "true", "yes", "on",
        )
        path = db_path or os.getenv("RADAR_MEMORY_DB") or "local/radar_memory.db"
        m = cls(
            db_path=path,
            enabled=on,
            embed=embed or _hash_embed,
            extract=extract or _regex_extract,
        )
        if on:
            m._open()
            logger.info("memory enabled at %s", path)
        return m

    def _open(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            self.db_path, isolation_level=None, check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS mem_nodes(
                id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL,
                text TEXT NOT NULL, props_json TEXT NOT NULL DEFAULT '{}',
                valid_from REAL NOT NULL, valid_to REAL);
            CREATE INDEX IF NOT EXISTS mem_nodes_kind ON mem_nodes(kind);
            CREATE INDEX IF NOT EXISTS mem_nodes_valid
                ON mem_nodes(valid_from, valid_to);
            CREATE TABLE IF NOT EXISTS mem_edges(
                id INTEGER PRIMARY KEY AUTOINCREMENT, src INTEGER NOT NULL,
                dst INTEGER NOT NULL, rel TEXT NOT NULL,
                props_json TEXT NOT NULL DEFAULT '{}',
                valid_from REAL NOT NULL, valid_to REAL);
            CREATE INDEX IF NOT EXISTS mem_edges_src ON mem_edges(src, rel);
            CREATE INDEX IF NOT EXISTS mem_edges_dst ON mem_edges(dst, rel);
            CREATE TABLE IF NOT EXISTS mem_embeds(
                node_id INTEGER PRIMARY KEY, dim INTEGER NOT NULL,
                vec BLOB NOT NULL);
        """)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ── public API ────────────────────────────────────────────

    def remember(self, text: str, *, actor: str = "",
                 ts: Optional[float] = None,
                 props: Optional[dict] = None,
                 kind: str = "memory") -> Optional[int]:
        """Store a memory node + any triples found in ``text``.

        Returns the memory node id, or ``None`` if memory is disabled.
        """
        if not self.enabled or self._conn is None or not text:
            return None
        ts = float(ts if ts is not None else time.time())
        all_props = dict(props or {})
        if actor:
            all_props["actor"] = actor
        node_id = self._upsert_node(kind, text, all_props, ts)
        for trip in self.extract(text):
            subj, pred, obj = trip.get("subj"), trip.get("pred"), trip.get("obj")
            if not (subj and pred and obj):
                continue
            self._add_fact(subj, pred, obj, ts, trip.get("props") or {})
        return node_id

    def add_fact(self, subj: str, pred: str, obj: str, *,
                 ts: Optional[float] = None,
                 props: Optional[dict] = None) -> Optional[int]:
        """Insert a ``(subj, pred, obj)`` edge, superseding any prior live one."""
        if not self.enabled or self._conn is None:
            return None
        ts = float(ts if ts is not None else time.time())
        return self._add_fact(subj, pred, obj, ts, props or {})

    def recall(self, query: str, *, k: int = 5,
               as_of: Optional[float] = None,
               hops: int = 1) -> list[MemoryHit]:
        """Hybrid query: vector top-k over live nodes + graph expansion."""
        if not self.enabled or self._conn is None or not query:
            return []
        as_of = float(as_of if as_of is not None else time.time())
        q_vec = self.embed(query).astype(np.float32)
        # Full scan of live embeddings — fine up to ~100k nodes; swap for
        # sqlite-vec / a real ANN index past that.
        rows = list(self._conn.execute(
            "SELECT n.id, n.kind, n.text, n.props_json, e.vec "
            "FROM mem_embeds e JOIN mem_nodes n ON n.id = e.node_id "
            "WHERE n.valid_from <= ? "
            "  AND (n.valid_to IS NULL OR n.valid_to > ?)",
            (as_of, as_of),
        ))
        if not rows:
            return []
        scored = []
        for nid, kind, text, props_json, blob in rows:
            v = np.frombuffer(blob, dtype=np.float32)
            if v.shape != q_vec.shape:
                continue
            scored.append((float(q_vec @ v), nid, kind, text, props_json))
        scored.sort(reverse=True)
        hits: list[MemoryHit] = []
        for score, nid, kind, text, props_json in scored[:k]:
            edges = self._edges_for(nid, as_of, hops=hops) if hops > 0 else []
            hits.append(MemoryHit(
                node_id=nid, kind=kind, text=text, score=score,
                props=json.loads(props_json or "{}"), edges=edges,
            ))
        return hits

    # ── internals ─────────────────────────────────────────────

    def _upsert_node(self, kind: str, text: str, props: dict, ts: float) -> int:
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT id FROM mem_nodes "
            "WHERE kind=? AND text=? AND valid_to IS NULL",
            (kind, text),
        ).fetchone()
        if row:
            return int(row[0])
        cur = self._conn.execute(
            "INSERT INTO mem_nodes(kind, text, props_json, valid_from) "
            "VALUES (?, ?, ?, ?)",
            (kind, text, json.dumps(props), ts),
        )
        node_id = int(cur.lastrowid)
        # Embed every new node — including entity endpoints created by
        # add_fact — so recall can locate the graph via vector top-k.
        self._write_embed(node_id, text)
        return node_id

    def _write_embed(self, node_id: int, text: str) -> None:
        assert self._conn is not None
        v = self.embed(text).astype(np.float32)
        self._conn.execute(
            "INSERT OR REPLACE INTO mem_embeds(node_id, dim, vec) "
            "VALUES (?, ?, ?)",
            (node_id, int(v.shape[0]), v.tobytes()),
        )

    def _add_fact(self, subj: str, pred: str, obj: str, ts: float,
                  props: dict) -> int:
        assert self._conn is not None
        s_id = self._upsert_node("entity", subj, {}, ts)
        o_id = self._upsert_node("entity", obj, {}, ts)
        # Supersede any live edge with the same (src, rel, dst) — the
        # bitemporal "update" — instead of mutating in place.
        self._conn.execute(
            "UPDATE mem_edges SET valid_to=? "
            "WHERE src=? AND dst=? AND rel=? AND valid_to IS NULL",
            (ts, s_id, o_id, pred),
        )
        cur = self._conn.execute(
            "INSERT INTO mem_edges(src, dst, rel, props_json, valid_from) "
            "VALUES (?, ?, ?, ?, ?)",
            (s_id, o_id, pred, json.dumps(props), ts),
        )
        return int(cur.lastrowid)

    def _edges_for(self, node_id: int, as_of: float,
                   hops: int = 1) -> list[dict]:
        assert self._conn is not None
        frontier = {node_id}
        visited = {node_id}
        seen_edges: set[int] = set()
        out: list[dict] = []
        for _ in range(hops):
            if not frontier:
                break
            placeholders = ",".join("?" for _ in frontier)
            params = list(frontier) + list(frontier) + [as_of, as_of]
            rows = self._conn.execute(
                f"SELECT id, src, dst, rel, props_json FROM mem_edges "
                f"WHERE (src IN ({placeholders}) "
                f"   OR dst IN ({placeholders})) "
                f"  AND valid_from <= ? "
                f"  AND (valid_to IS NULL OR valid_to > ?)",
                params,
            ).fetchall()
            nxt: set[int] = set()
            for eid, src, dst, rel, props_json in rows:
                if eid in seen_edges:
                    continue
                seen_edges.add(eid)
                out.append({"src": src, "dst": dst, "rel": rel,
                            "props": json.loads(props_json or "{}")})
                for n in (src, dst):
                    if n not in visited:
                        nxt.add(n)
            visited |= nxt
            frontier = nxt
        return out
