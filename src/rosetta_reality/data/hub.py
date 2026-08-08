"""Small requests-based Hub client for immutable public dataset snapshots."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote


@dataclass(frozen=True, slots=True)
class HubFile:
    """One file in an immutable Hub dataset tree."""

    path: str
    size: int
    digest: str | None = None


def _session(retries: int = 5) -> Any:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session


def resolve_dataset_revision(
    repo_id: str,
    revision: str,
    *,
    session: Any | None = None,
    timeout_seconds: float = 60,
) -> str:
    """Resolve a public dataset revision through the Hub REST API."""

    client = _session() if session is None else session
    url = (
        "https://huggingface.co/api/datasets/"
        f"{quote(repo_id, safe='/')}/revision/{quote(revision, safe='')}"
    )
    response = client.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    sha = response.json().get("sha")
    if not sha:
        raise RuntimeError(f"Hub did not return a commit SHA for {repo_id}@{revision}.")
    return str(sha)


def _digest_from_tree_row(row: dict[str, Any]) -> str | None:
    lfs = row.get("lfs")
    if isinstance(lfs, dict) and lfs.get("oid"):
        return str(lfs["oid"]).removeprefix("sha256:")
    xet_hash = row.get("xetHash")
    if xet_hash:
        return str(xet_hash)
    return None


def list_dataset_files(
    repo_id: str,
    revision: str,
    *,
    prefixes: tuple[str, ...],
    session: Any | None = None,
    timeout_seconds: float = 60,
) -> list[HubFile]:
    """List files beneath approved prefixes at one immutable revision."""

    client = _session() if session is None else session
    url: str | None = (
        "https://huggingface.co/api/datasets/"
        f"{quote(repo_id, safe='/')}/tree/{quote(revision, safe='')}"
    )
    params: dict[str, str] | None = {"recursive": "true", "expand": "true"}
    files: list[HubFile] = []
    while url is not None:
        response = client.get(url, params=params, timeout=timeout_seconds)
        response.raise_for_status()
        for row in response.json():
            path = str(row.get("path", ""))
            if row.get("type") == "file" and path.startswith(prefixes):
                files.append(
                    HubFile(
                        path=path,
                        size=int(row["size"]),
                        digest=_digest_from_tree_row(row),
                    )
                )
        next_link = response.links.get("next")
        url = next_link["url"] if next_link else None
        params = None
    if not files:
        raise RuntimeError(f"No dataset files matched {prefixes!r} at {repo_id}@{revision}.")
    return sorted(files, key=lambda file: file.path)


def _safe_target(root: Path, remote_path: str) -> Path:
    relative = PurePosixPath(remote_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe Hub dataset path: {remote_path!r}.")
    return root.joinpath(*relative.parts)


def _file_hashes(path: Path) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1()
    git_sha1.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)
            git_sha1.update(block)
    return sha256.hexdigest(), git_sha1.hexdigest()


def _validate_file(path: Path, remote: HubFile, etag: str | None = None) -> None:
    if path.stat().st_size != remote.size:
        raise ValueError(
            f"Cached file size mismatch for {path}: expected {remote.size}, "
            f"received {path.stat().st_size}."
        )
    digest = remote.digest
    if digest is None and etag is not None:
        digest = etag.removeprefix("W/").strip('"')
    if digest is None or len(digest) not in (40, 64):
        return
    sha256, git_sha1 = _file_hashes(path)
    actual = sha256 if len(digest) == 64 else git_sha1
    if actual != digest:
        raise ValueError(f"Cached file digest mismatch for {path}.")


def _download_file(
    *,
    session: Any,
    repo_id: str,
    revision: str,
    remote: HubFile,
    root: Path,
    timeout_seconds: float,
) -> Path:
    target = _safe_target(root, remote.path)
    url = (
        "https://huggingface.co/datasets/"
        f"{quote(repo_id, safe='/')}/resolve/{quote(revision, safe='')}/"
        f"{quote(remote.path, safe='/')}"
    )
    if target.exists():
        head = session.head(url, allow_redirects=True, timeout=timeout_seconds)
        head.raise_for_status()
        _validate_file(target, remote, head.headers.get("etag"))
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.partial")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > remote.size:
        raise ValueError(f"Partial download is larger than the expected file: {partial}.")
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    response = session.get(
        url,
        headers=headers,
        allow_redirects=True,
        stream=True,
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    if offset and response.status_code != 206:
        raise RuntimeError(f"Server did not honor Range resume for {partial}.")
    mode = "ab" if offset else "xb"
    with partial.open(mode) as file:
        for block in response.iter_content(chunk_size=1024 * 1024):
            if block:
                file.write(block)
    _validate_file(partial, remote, response.headers.get("etag"))
    if target.exists():
        raise FileExistsError(f"Refusing to replace a concurrently created file: {target}.")
    partial.rename(target)
    return target


def download_dataset_snapshot(
    *,
    repo_id: str,
    revision: str,
    root: Path,
    prefixes: tuple[str, ...],
    timeout_seconds: float = 120,
) -> list[Path]:
    """Download or validate an approved immutable snapshot without overwrites."""

    client = _session()
    remote_files = list_dataset_files(
        repo_id,
        revision,
        prefixes=prefixes,
        session=client,
        timeout_seconds=timeout_seconds,
    )
    return [
        _download_file(
            session=client,
            repo_id=repo_id,
            revision=revision,
            remote=remote,
            root=root,
            timeout_seconds=timeout_seconds,
        )
        for remote in remote_files
    ]
