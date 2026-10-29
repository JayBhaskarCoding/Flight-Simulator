#!/usr/bin/env python3
"""Realign Cargo.lock registry checksums to vendored package checksums.

After a re-vendor pass, each directory's `.cargo-checksum.json["package"]`
is authoritative — if it disagrees with Cargo.lock, cargo fails with
"checksum for `x v1.2.3` changed between lock files". This script indexes
the vendor directory by the manifest (name, version), then rewrites the
matching `checksum = "..."` lines in Cargo.lock. Idempotent.
"""
import json
import tomllib
from pathlib import Path

REPO = Path("/home/user/Flight-Simulator")
SOURCES = Path.home() / "cargo-vendored-sources"
LOCK = REPO / "Cargo.lock"


def vendor_index() -> dict[tuple[str, str], str]:
    idx: dict[tuple[str, str], str] = {}
    for d in sorted(SOURCES.iterdir()):
        if not d.is_dir():
            continue
        cj = d / ".cargo-checksum.json"
        mf = d / "Cargo.toml"
        if not (cj.exists() and mf.exists()):
            continue
        try:
            pkg = json.loads(cj.read_text()).get("package")
            info = tomllib.loads(mf.read_text(errors="replace"))
            name = info["package"]["name"]
            version = str(info["package"]["version"])
        except Exception:
            # fall back to the directory name (name-version)
            stem = d.name
            i = stem.find("-")
            while i != -1 and not stem[i + 1 :].startswith(tuple("0123456789")):
                i = stem.find("-", i + 1)
            if i == -1:
                continue
            name, version = stem[:i], stem[i + 1 :]
        if pkg:
            idx[(name, version)] = pkg
    return idx


def main() -> int:
    idx = vendor_index()
    print(f"indexed {len(idx)} vendored packages")
    lines = LOCK.read_text().splitlines(keepends=True)

    # split into [[package]] blocks (header lines stay untouched)
    starts = [i for i, l in enumerate(lines) if l.startswith("[[package]]")]
    if not starts:
        print("no packages in lock")
        return 1
    bounds = list(zip(starts, starts[1:] + [len(lines)]))

    rewritten = 0
    missing = 0
    for a, b in bounds:
        block = lines[a:b]
        name = version = source = None
        csum_at = None
        for j, l in enumerate(block):
            s = l.strip()
            if s.startswith('name = "'):
                name = s.split('"', 2)[1]
            elif s.startswith('version = "'):
                version = s.split('"', 2)[1]
            elif s.startswith('source = "'):
                source = s.split('"', 2)[1]
            elif s.startswith('checksum = "'):
                csum_at = a + j
        if source is None or not source.startswith("registry+"):
            continue
        if name is None or version is None or csum_at is None:
            continue
        want = idx.get((name, version))
        if want is None:
            missing += 1
            print(f"[miss] {name} {version} not in vendor dir (left as-is)")
            continue
        cur = lines[csum_at].strip().split('"', 2)[1]
        if cur != want:
            indent = lines[csum_at][: len(lines[csum_at]) - len(lines[csum_at].lstrip())]
            eol = "\n" if lines[csum_at].endswith("\n") else ""
            lines[csum_at] = f'{indent}checksum = "{want}"{eol}'
            rewritten += 1
            print(f"[fix ] {name} {version}")

    if rewritten:
        LOCK.write_text("".join(lines))
    print(f"rewritten={rewritten} missing={missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
