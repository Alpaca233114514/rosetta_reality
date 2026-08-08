"""Offline tests for bounded Hub snapshot downloads."""

import hashlib
from pathlib import Path

from requests.exceptions import ChunkedEncodingError

from rosetta_reality.data.hub import HubFile, _digest_from_tree_row, _download_file


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int,
        blocks: tuple[bytes | Exception, ...],
    ) -> None:
        self.status_code = status_code
        self.blocks = blocks
        self.headers = {}

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, *, chunk_size: int):
        assert chunk_size == 1024 * 1024
        for block in self.blocks:
            if isinstance(block, Exception):
                raise block
            yield block


class InterruptedSession:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def get(self, url: str, *, headers: dict[str, str], **kwargs: object) -> FakeResponse:
        self.calls.append(headers)
        if len(self.calls) == 1:
            return FakeResponse(
                status_code=200,
                blocks=(b"abc", ChunkedEncodingError("interrupted")),
            )
        return FakeResponse(status_code=206, blocks=(b"def",))


class NoNetworkSession:
    def get(self, *args: object, **kwargs: object) -> FakeResponse:
        raise AssertionError("A valid cached file must not access the network.")

    def head(self, *args: object, **kwargs: object) -> FakeResponse:
        raise AssertionError("A valid cached file must not access the network.")


def test_stream_interruption_resumes_from_partial_file(tmp_path: Path) -> None:
    session = InterruptedSession()
    remote = HubFile(path="videos/example.mp4", size=6)

    path = _download_file(
        session=session,
        repo_id="lerobot/example",
        revision="a" * 40,
        remote=remote,
        root=tmp_path,
        timeout_seconds=60,
    )

    assert path.read_bytes() == b"abcdef"
    assert session.calls == [{}, {"Range": "bytes=3-"}]


def test_existing_file_validates_from_immutable_tree_metadata(tmp_path: Path) -> None:
    payload = b"cached"
    target = tmp_path / "data" / "file.parquet"
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)

    remote = HubFile(
        path="data/file.parquet",
        size=len(payload),
        digest=hashlib.sha256(payload).hexdigest(),
    )

    path = _download_file(
        session=NoNetworkSession(),
        repo_id="lerobot/example",
        revision="a" * 40,
        remote=remote,
        root=tmp_path,
        timeout_seconds=60,
    )

    assert path == target


def test_tree_digest_prefers_large_file_identity_and_falls_back_to_git_oid() -> None:
    assert _digest_from_tree_row(
        {"lfs": {"oid": "sha256:lfs"}, "xetHash": "xet", "oid": "git"}
    ) == "lfs"
    assert _digest_from_tree_row({"xetHash": "xet", "oid": "git"}) == "xet"
    assert _digest_from_tree_row({"oid": "git"}) == "git"
