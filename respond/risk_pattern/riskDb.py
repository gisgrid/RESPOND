from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import md5
from json import dumps as _json_dumps, loads as _json_loads
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from chromadb import Client
from chromadb.config import Settings

from datetime import datetime, timezone, timedelta

__all__ = [
    "MemoryItem",
    "RiskMemory",
]

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class MemoryItem:
    vec: List[float]               # 15‑dim embedding
    risk_vec4: List[float]          # front, behind, left, right risk values
    risk_pattern: List[int]
    ego_state: Tuple[float, float, int] # (speed, acceleration, lane_id)
    drive_intension: str
    action_id: int                   # 0: turn left, 1: idle, 2: turn right, 3: accelerate, 4: decelerate 
    outcome_cnt: List[int] = field(default_factory=lambda: [0, 0, 0, 0])  # SAFE, UNSAFE, CORRECTED, COLLISION
    reflection_text: str | None = None
    style_tag: str | None = None
    env: str | None = None

    count: int = 1
    action_cnt: List[int] = field(default_factory=lambda: [0, 0, 0, 0, 0]) # 0: turn left, 1: idle, 2: turn right, 3: accelerate, 4: decelerate
    ts_first: float | None = None
    ts_last: float | None = None
    confidence: float = 0.0  # Confidence level, default is 0.0; only two values are used: 0.0 and 1.0

    def to_metadata(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("vec")
        return _sanitize_meta(d)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class RiskMemory:
    _COLL_OK = "patch_ok"
    _COLL_BAD = "patch_bad"

    def __init__(self, db_path: str):
        self._lock = RLock()
        Path(db_path).mkdir(parents=True, exist_ok=True)
        self._client = Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=db_path))
        self._ok = self._client.get_or_create_collection(self._COLL_OK)
        self._bad = self._client.get_or_create_collection(self._COLL_BAD)

    # ------------------------------------------------------------------
    # Insert / merge
    # ------------------------------------------------------------------

    def add_risk_mem(
        self,
        *,
        risk_vec4: Sequence[float],
        risk_pattern: Sequence[int],
        ego_state: Tuple[float, float, int],
        drive_intension: str,
        action_id: int,
        outcome: str,
        reflection_text: str | None = None,
        style_tag: str | None = None,
        env: str | None = None,
    ) -> str:
        assert len(risk_pattern) == 15, "risk_pattern must have 15 ints"
        assert len(risk_vec4) == 4, "risk_vec4 must have 4 floats"

        vec_int = np.asarray(risk_pattern, dtype=np.int32)
        mem_id = _hash_vec(vec_int)

        # 初始化 outcome_cnt（先假设库里没有）
        outcome_idx = {"SAFE": 0, "UNSAFE": 1, "CORRECTED": 2, "COLLISION": 3}
        outcome_cnt = [0, 0, 0, 0]
        if outcome in outcome_idx:
            outcome_cnt[outcome_idx[outcome]] += 1

        update_confidence = 0.0  

        coll = self._ok if outcome in {"SAFE", "UNSAFE","CORRECTED"} else self._bad
        if outcome not in {"SAFE", "UNSAFE", "CORRECTED", "COLLISION"}:
            print(f"Warning: unknown outcome '{outcome}', using 'bad' collection.")
            coll = self._bad
        elif outcome == "COLLISION":
            print(f"Warning: outcome 'COLLISION' detected, using 'bad' collection.")
            coll = self._bad
        elif outcome == "CORRECTED":
            print(f"Outcome 'CORRECTED' detected, using 'ok' collection, set confidence to 1.0.")
            coll = self._ok
            update_confidence = 1.0  

        item = MemoryItem(
            vec=vec_int.astype("float32").tolist(),
            risk_vec4=list(risk_vec4),
            risk_pattern=list(vec_int.tolist()),
            ego_state=ego_state,
            drive_intension=drive_intension,
            action_id=action_id,                        
            outcome_cnt=outcome_cnt,
            reflection_text=reflection_text,
            style_tag=style_tag,
            env=env,
            count=1, 
            action_cnt=[1 if i == action_id else 0 for i in range(5)], 
            ts_first=(datetime.now(timezone.utc) + timedelta(hours=8)).isoformat(timespec="seconds"),
            ts_last=(datetime.now(timezone.utc) + timedelta(hours=8)).isoformat(timespec="seconds"),
            confidence=update_confidence,
        )

        with self._lock:
            
            exists = coll.get(ids=[mem_id])
            if exists["ids"] and exists["ids"][0]:
               
                _merge_metadata(coll, mem_id, action_id, outcome)
            else:
                
                coll.add(
                    ids=[mem_id],
                    embeddings=[item.vec],      
                    metadatas=[item.to_metadata()],
                )

            coll._client.persist()          
    
        print(f"Current total records in {'ok' if coll == self._ok else 'bad'} collection: {coll.count()}")

        return mem_id

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve_risk_mem(
        self,
        *,
        risk_pattern: Sequence[int],
        top_k: int = 3,
        search_bad: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return up to *top_k* nearest memories for the given pattern."""
        vec = np.asarray(risk_pattern, dtype=np.float32).tolist()
        coll = self._bad if search_bad else self._ok

        with self._lock:
            index_size = coll.count()
            if index_size == 0:
                print("Warning: empty collection, search_bad flag: ", search_bad)
                return []
            n_results = min(max(top_k, 1), index_size)
            res = coll.query(query_embeddings=[vec], n_results=n_results)
            
            print(f"Queried {n_results} results, returned {len(res['ids'][0])} results.")

        out: List[Dict[str, Any]] = []
        for _id, _dist, _meta in zip(
            res["ids"][0],
            res["distances"][0],
            res["metadatas"][0],
        ):
            meta = dict(_meta)
            # ---- decode action_cnt -----------------------------------
            cnts = meta.get("action_cnt")
            if isinstance(cnts, str):
                try:
                    cnts = _json_loads(cnts)
                except Exception:
                    print("Warning: failed to decode action_cnt")
                    # cnts = [0] * 5
            elif cnts is None:
                print("Warning: action_cnt is None")
                # cnts = [0] * 5
            meta["action_cnt"] = cnts

            # ---- return the most met action -----------------------------------------
            try:
                count_val = int(meta.get("count", 1))
            except (TypeError, ValueError):
                print("Warning: failed to decode count")
                count_val = 1
            meta["max_action_stat"] = (max(cnts) if cnts else 0) / max(count_val, 1)

            meta.update(id=_id, distance=_dist)
            out.append(meta)
        return out

    # ------------------------------------------------------------------
    # Crud helpers
    # ------------------------------------------------------------------

    def delete_risk_mem(self, mem_id: str) -> None:
        with self._lock:
            self._ok.delete(ids=[mem_id], allow_missing=True)
            self._bad.delete(ids=[mem_id], allow_missing=True)

    def persist(self):
        self._client.persist()

    def size(self, collection: str = "ok") -> int:
        coll = self._ok if collection == "ok" else self._bad
        with self._lock:
            count = coll.count()
        # print(f"Current total records in '{collection}' collection: {count}")
        return count


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hash_vec(vec_int: np.ndarray) -> str:
    return md5(vec_int.tobytes()).hexdigest()


def _now() -> float:
    import time

    return time.time()


def _sanitize_meta(d: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in list(d.items()):
        if v is None:
            d.pop(k)        
        elif not isinstance(v, (str, int, float, bool)):
            d[k] = _json_dumps(v, separators=(",", ":")) 
    return d

def _merge_metadata(coll, mem_id: str, action_id: int, outcome: str) -> None:
    """
    合并已有 memory：
      • count      += 1
      • action_cnt[action_id] += 1  (action_id ∈ [0,4])
      • outcome_cnt[outcome_idx] += 1 (SAFE: 0, UNSAFE: 1, CORRECTED: 2, COLLISION: 3)
      • ts_last     = ISO-8601 时间戳（UTC）
    """
    prev = coll.get(ids=[mem_id])
    if not prev["metadatas"]:
        print(f"ERROR: no metadata to be merged found for id {mem_id}")
        return                                        

    meta = dict(prev["metadatas"][0])

    
    previous_action_id = meta.get("action_id", -1)  # 默认值为-1，表示未设置
    
    meta["action_id"] = action_id



    raw_action_cnt = meta.get("action_cnt")
    if isinstance(raw_action_cnt, str):
        try:
            action_cnt = _json_loads(raw_action_cnt)
        except Exception:
            print("Warning: failed to decode action_cnt, resetting to default [0,0,0,0,0]")
            action_cnt = [0] * 5
    elif isinstance(raw_action_cnt, list):     
        action_cnt = raw_action_cnt
    else:
        print("Warning: action_cnt is not a valid type, resetting to default [0,0,0,0,0]")
        action_cnt = [0] * 5

    action_cnt = (action_cnt + [0] * 5)[:5]            

    idx = max(0, min(4, action_id))         
    action_cnt[idx] += 1
    meta["action_cnt"] = _json_dumps(action_cnt, separators=(",", ":"))
    # print(" successfully updated action_cnt: ", cnts)

    raw_outcome_cnt = meta.get("outcome_cnt")
    if isinstance(raw_outcome_cnt, str):
        try:
            outcome_cnt = _json_loads(raw_outcome_cnt)
        except Exception:
            print("Warning: failed to decode outcome_cnt, resetting to default [0,0,0,0]")
            outcome_cnt = [0, 0, 0, 0]
    elif isinstance(raw_outcome_cnt, list):
        outcome_cnt = raw_outcome_cnt
    else:
        print("Warning: outcome_cnt is not a valid type, resetting to default [0,0,0,0]")
        outcome_cnt = [0, 0, 0, 0]

    outcome_idx = {"SAFE": 0, "UNSAFE": 1, "CORRECTED": 2, "COLLISION": 3}

    if outcome in outcome_idx:
        outcome_cnt[outcome_idx[outcome]] += 1
    else:
        print(f"Warning: unknown outcome '{outcome}', outcome_cnt not updated.")
    meta["outcome_cnt"] = _json_dumps(outcome_cnt, separators=(",", ":"))


    if outcome == "CORRECTED":
        
        print("Outcome is 'CORRECTED', the previous action_id is:", previous_action_id)
        print("Outcome is 'CORRECTED', the previous confidence levle is:", meta.get("confidence", "None"))
        meta["confidence"] = 1.0

        print("Updated confidence to 1.0")

    # ---------- count & timestamp --------------------------------------
    try:
        meta["count"] = int(meta.get("count", 1)) + 1
        # print(" successfully updated count: ", meta["count"])
    except (TypeError, ValueError):
        print( "exception while updating count, the hash id is: ", mem_id)
        print(" the count is: ", meta["count"])
        # meta["count"] = 1
        # print(" failed to update count, reset to 1")

    now_iso = (datetime.now(timezone.utc) + timedelta(hours=8)).isoformat(timespec="seconds")
    if "ts_first" not in meta:                           
        meta["ts_first"] = now_iso
        print( "should not be here, the ts_first not exist, the hash id is: ", mem_id)
    meta["ts_last"] = now_iso


    coll.update(ids=[mem_id], metadatas=[_sanitize_meta(meta)])


