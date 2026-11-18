"""Stage 00 - verify that every raw archive exists and matches its recorded checksum.

Writes data/manifests/raw_sources_verified.csv (adds bytes, md5, sha256, status).
"""
import csv
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nailrev.config import load_config  # noqa: E402


def digests(path: Path) -> tuple[str, str]:
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            md5.update(block)
            sha.update(block)
    return md5.hexdigest(), sha.hexdigest()


def main():
    cfg = load_config()
    raw, man = cfg.path("raw"), cfg.path("manifests")
    rows = list(csv.DictReader(open(man / "raw_sources.csv", encoding="utf-8")))
    ok = True
    for r in rows:
        p = raw / r["folder"] / r["file"]
        if not p.exists():
            r.update(bytes="", md5="", sha256="", status="MISSING")
            ok = False
        else:
            md5, sha = digests(p)
            status = "ok" if not r["md5_expected"] or r["md5_expected"] == md5 else "MD5 MISMATCH"
            ok &= status == "ok"
            r.update(bytes=p.stat().st_size, md5=md5, sha256=sha, status=status)
        print(f"{r['status']:>12}  {r['source_id']:<16} {r['folder']}/{r['file']}")
    with open(man / "raw_sources_verified.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
