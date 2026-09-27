"""Read-only checks of recorded identity claims against independent inputs."""

import hashlib

from .analysis import pointer
from .io import child


def check_identity(store, sources, run_id, claim_pointer, *, expected=None, source=None, path=None):
    """A matching claim is not byte verification unless source bytes were hashed."""
    if not claim_pointer.startswith("/"):
        raise ValueError("Claim requires a JSON Pointer beginning with /")
    if (source is None) != (path is None):
        raise ValueError("source and path must be supplied together")
    if expected is None and source is None:
        raise ValueError("Supply expected and/or a configured source file")
    if source is not None and source not in sources:
        raise ValueError("Source alias was not configured by the operator")

    record = store.get(run_id)[0]  # Verify the immutable Basin copy before using a claim.
    try:
        claim = pointer(record, claim_pointer)
    except (KeyError, IndexError, TypeError):
        claim = None
    if claim is not None and not isinstance(claim, str):
        raise ValueError("Identity claim must be a string")

    checks = {}
    if expected is not None:
        checks["expected"] = {"value": expected,
                              "status": "missing" if claim is None else
                              "match" if claim == expected else "mismatch"}
    if source is not None:
        digest = hashlib.sha256()
        size = 0
        with child(sources[source], path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(block)
                if size > 4 * 1024 ** 3:
                    raise ValueError("Source file exceeds 4 GiB identity-check limit")
                digest.update(block)
        actual = digest.hexdigest()
        checks["source_file"] = {"source": source, "path": path,
                                 "sha256": actual, "bytes": size,
                                 "status": "missing" if claim is None else
                                 "match" if claim == actual else "mismatch"}
    statuses = {item["status"] for item in checks.values()}
    status = "missing" if "missing" in statuses else "mismatch" if "mismatch" in statuses else "match"
    return {"run_id": run_id, "pointer": claim_pointer, "claim": claim,
            "status": status, "checks": checks,
            "verification": "source_bytes_hashed" if source is not None else "claim_value_compared",
            "limitations": ["A matching claim does not establish the measurement or training history.",
                            "Only the named source file is byte-verified when source_file is present."]}
