"""Read specific members out of a remote zip over HTTP, without downloading the
whole archive.

Used to pull the handful of tiny class-list .txt files out of Xian et al.'s
~900MB xlsa17.zip release (whose bulk is pre-extracted ResNet101 features we
don't need) via byte-range requests: one small request for the tail (central
directory), then one small request per wanted member. Assumes a plain (non
ZIP64) zip, which xlsa17.zip is -- it has only 70 entries.
"""
import struct
import zlib
from io import BytesIO

import requests

TAIL_SIZE = 2_000_000  # generous margin over xlsa17.zip's actual ~30KB central directory
LOCAL_HEADER_FIXED_SIZE = 30


def _content_length(url: str) -> int:
    response = requests.head(url, timeout=30, allow_redirects=True)
    response.raise_for_status()
    return int(response.headers["Content-Length"])


def _get_range(url: str, start: int, end_inclusive: int) -> bytes:
    headers = {"Range": f"bytes={start}-{end_inclusive}"}
    response = requests.get(url, headers=headers, timeout=60, allow_redirects=True)
    response.raise_for_status()
    return response.content


def list_members(url: str) -> dict[str, tuple[int, int, int]]:
    """Returns {name: (header_offset, compress_size, compress_type)} via zipfile,
    fed a buffer that's zero everywhere except the tail (where the central
    directory lives) -- zipfile only needs the tail to list names/offsets.
    """
    import zipfile

    total = _content_length(url)
    tail_start = max(0, total - TAIL_SIZE)
    tail = _get_range(url, tail_start, total - 1)

    buf = bytearray(total)
    buf[tail_start:tail_start + len(tail)] = tail

    zf = zipfile.ZipFile(BytesIO(bytes(buf)))
    return {
        info.filename: (info.header_offset, info.compress_size, info.compress_type)
        for info in zf.infolist()
    }


def fetch_member(url: str, header_offset: int, compress_size: int, compress_type: int) -> bytes:
    """Range-fetches just one member's local header + compressed data and
    decompresses it -- no knowledge of the rest of the archive needed.
    """
    window = LOCAL_HEADER_FIXED_SIZE + 512 + compress_size  # +512 slack for filename/extra fields
    raw = _get_range(url, header_offset, header_offset + window - 1)

    sig, _ver, _flags, method, _mtime, _mdate, _crc, _csize, _usize, fname_len, extra_len = (
        struct.unpack_from("<IHHHHHIIIHH", raw, 0)
    )
    if sig != 0x04034B50:
        raise ValueError(f"no local file header at offset {header_offset}")

    data_start = LOCAL_HEADER_FIXED_SIZE + fname_len + extra_len
    comp_data = raw[data_start:data_start + compress_size]

    if compress_type == 0:
        return comp_data
    if compress_type == 8:
        return zlib.decompress(comp_data, -15)
    raise ValueError(f"unsupported compression method {compress_type}")
