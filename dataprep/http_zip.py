"""Read individual members out of a remote zip over HTTP, without downloading
the whole archive.

HTTPRangeFile is a seekable, read-only file object whose reads become HTTP
byte-range requests, so Python's own zipfile can open a remote archive
directly: listing members costs a few small requests for the central directory
at the tail, and reading a member fetches only that member's bytes. zipfile
handles ZIP64, so this works on AWA2's ~14GB zip as well as xlsa17.zip.
"""
import io
import zipfile

import requests

READ_BUFFER_SIZE = 1 << 20  # coalesce zipfile's many small reads into 1MB range requests


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url: str):
        self.url = url
        self.session = requests.Session()
        response = self.session.head(url, timeout=30, allow_redirects=True)
        response.raise_for_status()
        if response.headers.get("Accept-Ranges", "").lower() != "bytes":
            raise ValueError(f"{url} does not support HTTP range requests")
        self.size = int(response.headers["Content-Length"])
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = base + offset
        return self.position

    def readinto(self, buffer) -> int:
        if self.position >= self.size:
            return 0
        end_inclusive = min(self.position + len(buffer), self.size) - 1
        headers = {"Range": f"bytes={self.position}-{end_inclusive}"}
        response = self.session.get(self.url, headers=headers, timeout=60)
        response.raise_for_status()
        if response.status_code != 206:
            raise ValueError(f"{self.url} ignored the range request (HTTP {response.status_code})")
        data = response.content
        buffer[:len(data)] = data
        self.position += len(data)
        return len(data)


def open_remote_zip(url: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(HTTPRangeFile(url), buffer_size=READ_BUFFER_SIZE))
