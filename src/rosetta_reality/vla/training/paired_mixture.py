"""Create-only shared 3D+1 auxiliary exposure schedules; no ML imports."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _samples(rows, name):
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{name} requires real nonempty eligible inputs")
    result = []
    for row in rows:
        if not isinstance(row, dict) or row.get("positive_supervision") is not True:
            raise ValueError(f"{name} input has no positive supervision audit")
        identity = (row.get("episode"), row.get("frame"))
        if any(type(x) is not int or x < 0 for x in identity):
            raise ValueError("Invalid episode/frame identity")
        if row.get("chunk_crosses_unqualified_action") is not False:
            raise ValueError("Chunk crosses unknown or deliberately incorrect action")
        result.append(dict(row))
    if len({(r["episode"], r["frame"]) for r in result}) != len(result):
        raise ValueError("Input inventory contains repeated frames")
    return result


def _balanced(rows, count, rng):
    groups = {}
    for row in rows:
        groups.setdefault(row["episode"], []).append(row)
    episodes = sorted(groups)
    selected = []
    cursors = {ep: 0 for ep in episodes}
    for values in groups.values():
        rng.shuffle(values)
    while len(selected) < count:
        order = list(episodes)
        rng.shuffle(order)
        for ep in order:
            values = groups[ep]
            selected.append(values[cursors[ep] % len(values)])
            cursors[ep] += 1
            if len(selected) == count:
                break
    return selected


def build_schedules(original, recovery=None, success=None, *, seed=20260809, updates=5000):
    """Freeze common original slots and episode-balanced auxiliary order.

    Missing auxiliary cohorts produce an explicit unavailable arm, never D
    substitution. Inventory frames are unique; planned repeated exposures are
    recorded individually and do not inflate the unique-frame count.
    """
    if type(updates) is not int or updates <= 0:
        raise ValueError("Updates must be positive")
    d = _samples(original, "D")
    common = _balanced(d, updates * 3, random.Random(seed))
    auxiliaries = {"A": _balanced(d, updates, random.Random(seed + 1))}
    for name, rows in (("B", recovery), ("C", success)):
        auxiliaries[name] = None if not rows else _balanced(_samples(rows, name), updates, random.Random(seed + 2))
    schedules = {}
    for arm, auxiliary in auxiliaries.items():
        if auxiliary is None:
            schedules[arm] = {"status": "input_unavailable", "planned_updates": updates, "actual_updates": 0}
            continue
        entries = []
        for update in range(updates):
            for slot, row in enumerate(common[update * 3:update * 3 + 3] + [auxiliary[update]]):
                source = "D" if slot < 3 or arm == "A" else ("R" if arm == "B" else "S")
                entries.append({"update": update + 1, "slot": slot, "source": source, "episode": row["episode"], "frame": row["frame"]})
        schedules[arm] = {"status": "sealed", "seed": seed, "updates": updates, "batch_size": 4, "common_D_exposures": updates * 3, "auxiliary_exposures": updates, "total_exposures": updates * 4, "entries": entries, "entries_sha256": digest(entries), "unique_inputs": len({(r["source"], r["episode"], r["frame"]) for r in entries})}
    schedules["common_D_sha256"] = digest([{"episode": r["episode"], "frame": r["frame"]} for r in common])
    return schedules


EXPLORATORY_SCOPE = "fresh_lineage_fixed_development_exploration_v1"


def verify_matched_cohort(pairs, *, tolerance=0.05, research_scope="strict_scene_disjoint_v1"):
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("N=0 cannot provide an S/R comparison")
    parents = set()
    for pair in pairs:
        parent = pair.get("parent_id")
        if not isinstance(parent, str) or not parent or parent in parents:
            raise ValueError("Matched parents must be unique")
        parents.add(parent)
        if research_scope == EXPLORATORY_SCOPE:
            if (pair.get("research_scope") != EXPLORATORY_SCOPE
                    or pair.get("fresh_lineage_verified") is not True
                    or pair.get("new_scene_disjoint_verified") is not True
                    or pair.get("legacy_object_scene_nonoverlap") != "not_established"
                    or pair.get("N_strict") != 0):
                raise ValueError("Exploratory fresh lineage or disclosed legacy boundary is unverified")
            if pair.get("scene_disjoint_verified") is True:
                raise ValueError("Exploratory evidence cannot claim legacy strict scene independence")
        elif research_scope == "strict_scene_disjoint_v1":
            if pair.get("scene_disjoint_verified") is not True:
                raise ValueError("Scene/split independence is unverified")
        else:
            raise ValueError("Unknown research scope")
        for key in ("effective_frames", "valid_target_slots"):
            s, r = pair["S"][key], pair["R"][key]
            if any(type(x) is not int or x <= 0 for x in (s, r)) or abs(s-r)/max(s,r) > tolerance:
                raise ValueError("S/R effective supervision mismatch")
        for name in ("S", "R"):
            branch = pair[name]
            if branch.get("task_success") is not True or branch.get("labels_independently_verified") is not True:
                raise ValueError("Auxiliary success or labels are unverified")
        if pair["R"].get("recovery_chain_verified") is not True:
            raise ValueError("R lacks deviation/correction/recovery/success evidence")
        if pair["S"].get("ordinary_success_verified") is not True:
            raise ValueError("S is not verified ordinary success")
    return len(pairs)


def save_bundle(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
