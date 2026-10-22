"""Collection of real P&ID drawings: archive scanning and source downloads.

``scan_archive(root, out_index=None)`` walks a directory tree of PDFs and records, per file:
SHA-256 (duplicates are flagged, not re-analysed), page sizes (with ISO paper guess),
readability (text characters, vector path count and raster image coverage per page; only
*readable* vector PDFs are in scope for the extractor), and a system guess from the
title-block region text (bottom-right of the sheet as displayed, honouring ``/Rotate``) using
:func:`shippid.catalog.detect_system`, falling back to the whole first-page text. Results go
to ``index.jsonl``.

``fetch_sources(sources_yaml, dest)`` downloads the URLs listed in a YAML file::

    sources:
      - url: https://example.org/pid/ballast.pdf
        name: ballast_pid.pdf      # optional target file name
        sha256: 3f5c...            # optional checksum to verify
        system: ballast            # optional hint, copied to the result record
        license: CC-BY-4.0         # optional, copied to the result record

with httpx (redirects followed), a size limit, a content-type / ``%PDF`` magic check and
resume of partial downloads (HTTP ``Range``). Failures are recorded and skipped.

Target files are unique per source: without an explicit ``name`` the file is called
``<url stem>_<sha1(url)[:8]>.pdf`` (the query string is part of the hash, so
``download.php?id=1`` and ``?id=2`` do not collide). Explicit names that collide
(case-insensitively) with another source's target raise ``ValueError`` before anything is
downloaded; the same URL listed twice is fetched once (the repeat is reported as
``duplicate``). Every completed download gets a provenance sidecar ``<file>.source.json``
(url, sha256, bytes); an existing file is only reported as ``exists`` when its sidecar names
the same URL and checksum, or when its content matches the ``sha256`` given for the source.
Partial downloads are kept per URL (``<file>.<sha1(url)[:8]>.part``) so a resume never mixes
bytes from different URLs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote, urlparse

import httpx
import pymupdf
import yaml

from ..catalog import detect_system

log = logging.getLogger(__name__)

PAPER_SIZES_MM = {
    "A0": (1189, 841), "A1": (841, 594), "A2": (594, 420), "A3": (420, 297), "A4": (297, 210),
    "ANSI B": (432, 279), "ANSI C": (559, 432), "ANSI D": (864, 559), "ANSI E": (1118, 864),
}
PDF_CONTENT_TYPES = ("application/pdf", "application/x-pdf", "application/octet-stream", "binary/octet-stream",
                     "application/download", "application/force-download")


def paper_name(width_pt: float, height_pt: float, tol_mm: float = 8.0) -> Optional[str]:
    """ISO / ANSI sheet name of a page size in points (orientation independent)."""
    w, h = sorted((width_pt * 25.4 / 72, height_pt * 25.4 / 72), reverse=True)
    for name, (a, b) in PAPER_SIZES_MM.items():
        if abs(w - a) <= tol_mm and abs(h - b) <= tol_mm:
            return name
    return None


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- scanning


@dataclass
class PageStats:
    index: int
    width: float
    height: float
    paper: Optional[str]
    text_chars: int
    vector_paths: int
    image_ratio: float  # fraction of the page area covered by raster images (clipped, summed, <= 1)
    readable: bool


@dataclass
class ScanRecord:
    path: str  # relative to the scanned root
    sha256: str
    bytes: int
    status: str  # readable | unreadable | duplicate | encrypted | error
    reason: str = ""
    duplicate_of: Optional[str] = None
    pages: int = 0
    pages_scanned: int = 0
    readable_pages: int = 0
    page_sizes: list[list[Any]] = field(default_factory=list)  # [width, height, paper]
    page_stats: list[PageStats] = field(default_factory=list)
    title_text: str = ""
    system_guess: Optional[str] = None
    system_source: Optional[str] = None  # "title_block" | "page_text" | None
    pdf_title: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _page_stats(page: pymupdf.Page, min_text_chars: int, min_paths: int, max_image_ratio: float) -> PageStats:
    rect = page.rect  # displayed (rotated) page
    area = max(rect.width * rect.height, 1e-9)
    text = page.get_text("text")
    chars = sum(1 for c in text if not c.isspace())
    try:
        paths = len(page.get_cdrawings())
    except Exception:  # pragma: no cover - very old PyMuPDF
        paths = len(page.get_drawings())
    covered = 0.0
    rot = page.rotation_matrix  # image bboxes are in unrotated page space
    for info in page.get_image_info():
        r = (pymupdf.Rect(info["bbox"]) * rot) & rect
        if not r.is_empty:
            covered += r.width * r.height
    ratio = min(1.0, covered / area)
    readable = chars >= min_text_chars and paths >= min_paths and ratio <= max_image_ratio
    return PageStats(page.number, round(rect.width, 2), round(rect.height, 2), paper_name(rect.width, rect.height),
                     chars, paths, round(ratio, 4), readable)


def _title_text(page: pymupdf.Page) -> str:
    """Text of the bottom-right title-block region of the sheet *as displayed* (``/Rotate`` aware:
    landscape CAD sheets are often plotted on portrait media with a 90/270 degree rotation)."""
    r = page.rect
    clip = pymupdf.Rect(r.x0 + r.width * 0.55, r.y0 + r.height * 0.6, r.x1, r.y1)
    clip = clip * page.derotation_matrix  # get_text clips in unrotated page space
    return " ".join(page.get_text("text", clip=clip).split())


def analyze_pdf(path: Path, rel: str, digest: str, *, max_pages: int = 20, min_text_chars: int = 30,
                min_paths: int = 50, max_image_ratio: float = 0.5) -> ScanRecord:
    rec = ScanRecord(path=rel, sha256=digest, bytes=path.stat().st_size, status="error")
    try:
        doc = pymupdf.open(str(path))
    except Exception as exc:
        rec.reason = f"cannot open: {type(exc).__name__}: {exc}"[:300]
        return rec
    try:
        if doc.needs_pass:
            rec.status, rec.reason = "encrypted", "password protected"
            return rec
        if not doc.is_pdf:
            rec.reason = "not a PDF"
            return rec
        rec.pages = doc.page_count
        rec.pdf_title = (doc.metadata or {}).get("title", "") or ""
        texts = []
        title_region = ""
        for page in doc:
            if page.number >= max_pages:
                break
            st = _page_stats(page, min_text_chars, min_paths, max_image_ratio)
            rec.page_stats.append(st)
            rec.page_sizes.append([st.width, st.height, st.paper])
            if page.number == 0:
                title_region = _title_text(page)
                rec.title_text = title_region[-1500:]  # title blocks sit bottom-right: keep the tail
                texts.append(page.get_text("text"))
        rec.pages_scanned = len(rec.page_stats)
        rec.readable_pages = sum(1 for s in rec.page_stats if s.readable)
        if rec.readable_pages:
            rec.status = "readable"
        else:
            rec.status = "unreadable"
            s0 = rec.page_stats[0] if rec.page_stats else None
            if s0 is None:
                rec.reason = "no pages"
            elif s0.image_ratio > max_image_ratio:
                rec.reason = f"raster image covers {s0.image_ratio:.0%} of the page (scanned drawing)"
            elif s0.text_chars < min_text_chars:
                rec.reason = "no text layer (outlined or scanned text)"
            else:
                rec.reason = f"only {s0.vector_paths} vector paths (not a drawing)"
        guess = detect_system(title_region) if title_region else None
        if guess is not None:
            rec.system_guess, rec.system_source = guess.value, "title_block"
        else:
            guess = detect_system(" ".join(texts) + " " + rec.pdf_title)
            if guess is not None:
                rec.system_guess, rec.system_source = guess.value, "page_text"
        return rec
    except Exception as exc:
        rec.status, rec.reason = "error", f"{type(exc).__name__}: {exc}"[:300]
        return rec
    finally:
        doc.close()


def scan_archive(root: str | Path, out_index: Optional[str | Path] = None, *, max_pages: int = 20,
                 min_text_chars: int = 30, min_paths: int = 50, max_image_ratio: float = 0.5) -> list[dict[str, Any]]:
    """Scan every ``*.pdf`` under ``root`` (sorted, case-insensitive suffix) and optionally write
    ``out_index`` (JSON lines, default ``<root>/index.jsonl`` when ``out_index is True``)."""
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(root)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")
    seen: dict[str, str] = {}
    out: list[dict[str, Any]] = []
    for p in files:
        rel = p.relative_to(root).as_posix()
        try:
            digest = sha256_file(p)
        except OSError as exc:
            out.append(ScanRecord(path=rel, sha256="", bytes=0, status="error", reason=str(exc)).to_dict())
            continue
        if digest in seen:
            rec = ScanRecord(path=rel, sha256=digest, bytes=p.stat().st_size, status="duplicate",
                             duplicate_of=seen[digest], reason="identical content")
        else:
            seen[digest] = rel
            rec = analyze_pdf(p, rel, digest, max_pages=max_pages, min_text_chars=min_text_chars,
                              min_paths=min_paths, max_image_ratio=max_image_ratio)
        out.append(rec.to_dict())
    if out_index is not None:
        target = root / "index.jsonl" if out_index is True else Path(out_index)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            for r in out:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


# --------------------------------------------------------------------------- downloading

SIDECAR_SUFFIX = ".source.json"
_FETCH_ERRORS = (httpx.HTTPError, httpx.InvalidURL, httpx.StreamError, OSError, ValueError)


def _url_hash(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]


def _want_sha(entry: dict[str, Any]) -> Optional[str]:
    raw = entry.get("sha256")
    return str(raw).strip().lower() if raw not in (None, "") else None


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")


def target_name(entry: dict[str, Any]) -> str:
    """File name of a source inside ``dest``: the explicit ``name`` (sanitised, ``.pdf`` added) or
    ``<url stem>_<sha1(url)[:8]>.pdf``, unique per URL including its query string."""
    name = entry.get("name")
    if name:
        out = _safe_name(str(name)) or f"source_{_url_hash(str(entry['url']))}"
        return out if out.lower().endswith(".pdf") else out + ".pdf"
    url = str(entry["url"])
    base = Path(unquote(urlparse(url).path)).name
    stem = base[:-4] if base.lower().endswith(".pdf") else Path(base).stem
    stem = _safe_name(stem)[:80] or "source"
    return f"{stem}_{_url_hash(url)}.pdf"


def load_sources(sources_yaml: str | Path) -> list[dict[str, Any]]:
    """Source entries of a YAML file (a list, or a mapping with a ``sources`` list). Raises
    ``ValueError`` for malformed entries and for different sources that would be saved under the
    same file name (compared case-insensitively) or the same URL listed with conflicting checksums."""
    data = yaml.safe_load(Path(sources_yaml).read_text(encoding="utf-8")) or []
    if isinstance(data, dict):
        data = data.get("sources", [])
    if not isinstance(data, list):
        raise ValueError("sources file must be a list or a mapping with a 'sources' list")
    out = []
    for entry in data:
        if isinstance(entry, str):
            entry = {"url": entry}
        if not isinstance(entry, dict) or not entry.get("url"):
            raise ValueError(f"invalid source entry: {entry!r}")
        out.append(dict(entry))
    _plan_targets(out)
    return out


def _plan_targets(sources: list[dict[str, Any]]) -> list[tuple[str, Optional[int]]]:
    """(target name, index of the earlier identical source or None) per source."""
    first: dict[str, int] = {}
    plan: list[tuple[str, Optional[int]]] = []
    conflicts = []
    for i, entry in enumerate(sources):
        name = target_name(entry)
        j = first.setdefault(name.lower(), i)
        if j == i:
            plan.append((name, None))
            continue
        other = sources[j]
        wa, wb = _want_sha(other), _want_sha(entry)
        if str(other["url"]) != str(entry["url"]):
            conflicts.append(f"entries {j} and {i} ({other['url']} / {entry['url']}) share the file name {name!r}")
        elif wa and wb and wa != wb:
            conflicts.append(f"entries {j} and {i} list {entry['url']} with different sha256")
        plan.append((name, j))
    if conflicts:
        raise ValueError("conflicting sources: " + "; ".join(conflicts))
    return plan


def _sidecar(target: Path) -> Path:
    return target.with_name(target.name + SIDECAR_SUFFIX)


def _read_sidecar(target: Path) -> dict[str, Any]:
    try:
        data = json.loads(_sidecar(target).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_sidecar(target: Path, url: str, digest: str) -> None:
    tmp = target.with_name(target.name + SIDECAR_SUFFIX + ".tmp")
    tmp.write_text(json.dumps({"url": url, "sha256": digest, "bytes": target.stat().st_size}, ensure_ascii=False),
                   encoding="utf-8")
    tmp.replace(_sidecar(target))


def part_path(target: Path, url: str) -> Path:
    """Partial-download file of ``url`` for ``target`` (one per URL, so resumes never mix sources)."""
    return target.with_name(f"{target.name}.{_url_hash(url)}.part")


def fetch_sources(sources_yaml: str | Path, dest: str | Path, *, client: Optional[httpx.Client] = None,
                  max_bytes: int = 200 * 1024 * 1024, timeout: float = 60.0,
                  user_agent: str = "shippid-collector/0.1 (+P&ID research dataset)") -> list[dict[str, Any]]:
    """Download the PDFs listed in ``sources_yaml`` into ``dest``; returns one result record per
    source (status: downloaded | exists | duplicate | error | skipped) and writes
    ``dest/fetch_results.jsonl`` (one line per source as it completes)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    sources = load_sources(sources_yaml)
    plan = _plan_targets(sources)
    own = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=timeout, headers={"User-Agent": user_agent})
    results: list[dict[str, Any]] = []
    try:
        with (dest / "fetch_results.jsonl").open("w", encoding="utf-8") as log_fh:
            for entry, (name, dup) in zip(sources, plan):
                if dup is not None:
                    prev = results[dup]
                    res = {"url": str(entry["url"]), "path": name, "status": "duplicate", "bytes": prev.get("bytes", 0),
                           "duplicate_of": dup, "reason": f"same URL as entry {dup} ({prev['status']})"}
                    if prev.get("sha256"):
                        res["sha256"] = prev["sha256"]
                else:
                    res = _fetch_one(client, entry, dest / name, max_bytes)
                for k in ("system", "license", "notes"):
                    if k in entry:
                        res[k] = entry[k]
                results.append(res)
                log_fh.write(json.dumps(res, ensure_ascii=False) + "\n")
                log_fh.flush()
    finally:
        if own:
            client.close()
    return results


def _fetch_one(client: httpx.Client, entry: dict[str, Any], target: Path, max_bytes: int) -> dict[str, Any]:
    url = str(entry["url"])
    want = _want_sha(entry)
    res: dict[str, Any] = {"url": url, "path": target.name, "status": "error", "bytes": 0}
    if target.exists():
        digest = sha256_file(target)
        meta = _read_sidecar(target)
        same_source = meta.get("url") == url and meta.get("sha256") == digest
        if (want is not None and digest == want) or (want is None and same_source):
            if not same_source:  # content verified by the configured checksum: adopt the file
                _write_sidecar(target, url, digest)
            res.update(status="exists", bytes=target.stat().st_size, sha256=digest)
            return res
        if want is not None:  # known wrong content
            target.unlink()
            _sidecar(target).unlink(missing_ok=True)
        # unknown provenance without a checksum: download again, replace only on success
    part = part_path(target, url)
    for stale in target.parent.glob(f"{target.name}.*.part"):  # partials of other URLs for this name
        if stale != part:
            stale.unlink(missing_ok=True)
    offset = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    try:
        with client.stream("GET", url, headers=headers) as resp:
            if resp.status_code == 416 and offset:  # nothing left to fetch: the partial file is complete
                res["resumed_from"] = offset
            elif resp.status_code >= 400:
                res["reason"] = f"HTTP {resp.status_code}"
                return res
            else:
                ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                res["content_type"] = ctype
                if ctype and ctype not in PDF_CONTENT_TYPES:
                    res.update(status="skipped", reason=f"content-type {ctype!r} is not a PDF")
                    return res
                resumed = resp.status_code == 206 and offset > 0
                if resumed:
                    m = re.match(r"bytes\s+(\d+)-", resp.headers.get("content-range", ""))
                    if m is not None and int(m.group(1)) != offset:
                        part.unlink(missing_ok=True)
                        res["reason"] = f"server resumed at byte {m.group(1)} instead of {offset}; partial file discarded"
                        return res
                else:
                    offset = 0
                try:
                    length: Optional[int] = int(resp.headers.get("content-length", ""))
                except ValueError:
                    length = None  # absent or malformed: enforced while streaming
                if length is not None and offset + length > max_bytes:
                    res.update(status="skipped", reason=f"too large ({offset + length} bytes > {max_bytes})")
                    return res
                written = offset
                with part.open("ab" if resumed else "wb") as fh:
                    for chunk in resp.iter_bytes():
                        written += len(chunk)
                        if written > max_bytes:
                            fh.close()
                            part.unlink(missing_ok=True)
                            res.update(status="skipped", reason=f"exceeded size limit of {max_bytes} bytes")
                            return res
                        fh.write(chunk)
                res["resumed_from"] = offset if resumed else 0
    except _FETCH_ERRORS as exc:
        res["reason"] = f"{type(exc).__name__}: {exc}"[:300]  # a partial file is kept for resume
        return res
    with part.open("rb") as fh:
        magic = fh.read(5)
    if magic != b"%PDF-":
        part.unlink(missing_ok=True)
        res.update(status="skipped", reason="downloaded content is not a PDF")
        return res
    digest = sha256_file(part)
    if want is not None and digest != want:
        part.unlink(missing_ok=True)
        res.update(reason=f"sha256 mismatch ({digest} != {want})")
        return res
    part.replace(target)
    _write_sidecar(target, url, digest)
    res.update(status="downloaded", bytes=target.stat().st_size, sha256=digest)
    return res
