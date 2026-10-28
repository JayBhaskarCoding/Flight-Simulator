#!/usr/bin/env python3
"""Lockfile-driven vendor restore (rebuilds ~/cargo-vendored-sources).

Reads Cargo.lock, resolves every registry package from its GitHub source
(repo/tag/subdir from crates-map.json plus a table of special cases whose
tag does not match the crate version), runs vendor.py, and applies
post-install patches (e.g. hashbrown's dev-dependency that cargo's
feature validation requires). Idempotent: installed packages are skipped.
"""
import base64
import json
import re
import subprocess
import sys
import tomllib
import urllib.request
from pathlib import Path

REPO = Path("/home/user/Flight-Simulator")
VENDOR_DIR = REPO / "tools" / "vendor"
SOURCES = Path.home() / "cargo-vendored-sources"
MAP = json.loads((VENDOR_DIR / "crates-map.json").read_text())

# Tag is not derivable from the version (subdir crates, sha pins, or the
# windows-rs repo whose release tags stopped tracking windows-sys).
SPECIAL = {
    ("libsqlite3-sys", "0.30.0"): {
        "repo": "aosp-mirror-neo/platform_external_rust_crates_libsqlite3-sys",
        "tag": "4f346279844acd466a1e91bfc88c444ad73b7dd4",
    },
    ("shlex", "2.0.1"): {
        "repo": "comex/rust-shlex",
        "tag": "653936e120cdd0a8d1b4f0837e2cce0f347da4c0",
    },
    ("windows-sys", "0.61.0"): {"repo": "microsoft/windows-rs", "tag": "69"},
    ("windows-sys", "0.52.0"): {"repo": "microsoft/windows-rs", "tag": "0.52.0"},
    ("windows-link", "0.2.1"): {"repo": "microsoft/windows-rs", "tag": "73"},
    ("tokio-macros", "2.5.0"): {
        "repo": "tokio-rs/tokio",
        "subdir": "tokio-macros",
        "tag": "tokio-macros-2.5.0",
    },
    ("find-msvc-tools", "0.1.15"): {
        "repo": "rust-lang/cc-rs",
        "subdir": "find-msvc-tools",
        "tag": "find-msvc-tools-v0.1.15",
    },
    ("zerocopy-derive", "0.8.63"): {
        "repo": "google/zerocopy",
        "subdir": "zerocopy/zerocopy-derive",
        "tag": "v0.8.63",
    },
}
for _n in (
    "windows-targets",
    "windows_aarch64_gnullvm",
    "windows_aarch64_msvc",
    "windows_i686_gnu",
    "windows_i686_msvc",
    "windows_x86_64_gnu",
    "windows_x86_64_gnullvm",
    "windows_x86_64_msvc",
):
    SPECIAL[(_n, "0.52.0")] = {"repo": "microsoft/windows-rs", "tag": "0.52.0"}

# crates.io packages recovered verbatim (with their original checksums)
# from a public vendor tree, because their upstream repo/tags are gone.
BLOB_WALK = {
    ("adler2", "2.0.1"): ("catboost/catboost", "contrib/crates/adler2-2.0.1"),
    ("simd-adler32", "0.3.10"): ("catboost/catboost", "contrib/crates/simd-adler32-0.3.10"),
}


def lock_packages() -> list[tuple[str, str]]:
    txt = (REPO / "Cargo.lock").read_text()
    out = []
    for m in re.finditer(
        r'\[\[package\]\]\nname = "([^"]+)"\nversion = "([^"]+)"\n(source = "([^"]+)"\n)?',
        txt,
    ):
        name, version, _, source = m.groups()
        if source and source.startswith("registry+"):
            out.append((name, version))
    return out


def base_version(v: str) -> str:
    return v.split("+", 1)[0]


def entry_for(name: str, version: str) -> dict:
    if (name, version) in SPECIAL:
        return SPECIAL[(name, version)]
    raw = MAP.get(name)
    if raw is None:
        raise RuntimeError(f"no repo mapping for {name}")
    if isinstance(raw, dict):
        return {"repo": raw["repo"], "subdir": raw.get("subdir"), "tag": None}
    return {"repo": raw, "subdir": None, "tag": None}


def find_tag(repo: str, version: str) -> str:
    """Pick the tag whose embedded version equals `version` exactly."""
    want = base_version(version)
    r = subprocess.run(
        ["git", "ls-remote", "--tags", f"https://github.com/{repo}.git"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ls-remote failed for {repo}: {r.stderr.strip()[:160]}")
    exact: list[tuple[int, str]] = []
    for line in r.stdout.splitlines():
        tag = line.split("\t")[-1]
        if tag.endswith("^{}"):
            continue
        tag = tag.split("refs/tags/")[-1]
        m = re.search(r"(\d+(?:\.\d+){0,2})(?!.*\d)", tag)
        if not m or m.group(1) != want:
            continue
        # Prefer bare / v-prefixed tags, then name-prefixed forms.
        bare = tag[1:] if tag.startswith("v") else tag
        pri = 0 if bare == want else 1
        exact.append((pri, tag))
    if not exact:
        raise RuntimeError(f"no tag for {repo} matching {want}")
    exact.sort()
    return exact[0][1]


def api(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "openmsfs-vendor", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        import json as _j

        return _j.load(r)


def blob_walk(repo: str, path: str, dest: Path) -> None:
    # `path` is the crate directory itself: resolve its tree sha through
    # the PARENT listing (contents-of-dir returns the entries below it).
    parent, _, name = path.rpartition("/")
    listing = api(f"https://api.github.com/repos/{repo}/contents/{parent}")
    tree_sha = next(e["sha"] for e in listing if e["name"] == name)
    tree = api(f"https://api.github.com/repos/{repo}/git/trees/{tree_sha}?recursive=1")
    dest.mkdir(parents=True, exist_ok=True)
    for t in tree["tree"]:
        if t["type"] != "blob":
            continue
        p = dest / t["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        blob = api(f"https://api.github.com/repos/{repo}/git/blobs/{t['sha']}")
        p.write_bytes(base64.b64decode(blob["content"]))
    if not (dest / ".cargo-checksum.json").exists():
        # registry `src/` copies never ship the checksum; forge one so
        # cargo accepts the directory source (self-consistent with lock).
        importlib_load_vendor().write_checksum(dest)
    print(f"  blob-walked {repo}/{path} -> {dest.name}")


def post_patch_hashbrown() -> None:
    """hashbrown's `nightly` feature references bumpalo, which upstream
    keeps only as a dev-dependency; vendor.py must not strip it (current
    vendor.py keeps dev-deps) and older installs need it appended back."""
    manifest = SOURCES / "hashbrown-0.14.5" / "Cargo.toml"
    if not manifest.exists():
        return
    s = manifest.read_text()
    if "[dev-dependencies]" in s and "bumpalo" in s:
        return
    s += '\n[dev-dependencies]\nbumpalo = { version = "3.13.0", features = ["allocator-api2"] }\n'
    manifest.write_text(s)
    spec = importlib_load_vendor()
    spec.write_checksum(manifest.parent)
    print("  patched hashbrown dev-deps + checksum")


def importlib_load_vendor():
    import importlib.util

    spec = importlib.util.spec_from_file_location("vend", VENDOR_DIR / "vendor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def map_recorded_packages() -> list[tuple[str, str, dict]]:
    """(name, version, record) for every crates-map entry saved with an
    explicit tag by vendor.py (covers manual installs from before an
    environment reset)."""
    out = []
    for name, raw in MAP.items():
        if isinstance(raw, dict) and raw.get("tag") and raw.get("repo"):
            version = raw.get("version")
            if not version:
                # derive from saved tag using the same extraction vendor
                # used: prefer a `v`/`name-` style tag version
                tag = raw["tag"]
                import re as _re
                m = _re.search(r"(\d+(?:\.\d+){0,2})(?!.*\d)", tag)
                if not m:
                    continue
                version = m.group(1)
                # wasm-tools monorepo style: tag 1.X.Y -> crate 0.X.Y
                if raw["repo"] == "bytecodealliance/wasm-tools" and version.startswith("1."):
                    version = "0." + version[2:]
                # madsmtm frameworks tags: `frameworks-0.2.2`
                if tag.startswith("frameworks-"):
                    version = tag.split("-", 1)[1]
            out.append((name, version, raw))
    return out


def main() -> int:
    pkgs = lock_packages()
    recorded = {(n, v) for n, v, _ in map_recorded_packages()}
    print(f"lock registry packages: {len(pkgs)} (+{len(recorded)} recorded maps)")
    failures: list[str] = []
    for name, version in pkgs:
        dest = SOURCES / f"{name}-{version}"
        if (dest / ".cargo-checksum.json").exists():
            print(f"[skip] {name} {version}")
            if name == "hashbrown":
                post_patch_hashbrown()
            continue
        try:
            if (name, version) in BLOB_WALK:
                repo, path = BLOB_WALK[(name, version)]
                blob_walk(repo, path, dest)
                continue
            entry = entry_for(name, version)
            tag = entry.get("tag") or find_tag(entry["repo"], version)
            cmd = [
                "python3",
                str(VENDOR_DIR / "vendor.py"),
                "--name",
                name,
                "--version",
                version,
                "--repo",
                entry["repo"],
                "--tag",
                tag,
            ]
            if entry.get("subdir"):
                cmd += ["--subdir", entry["subdir"]]
            print(f"[get ] {name} {version} <- {entry['repo']}@{tag}"
                  + (f" ({entry['subdir']})" if entry.get("subdir") else ""))
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError((r.stderr or r.stdout).strip()[-400:])
            if name == "hashbrown":
                post_patch_hashbrown()
        except Exception as e:  # noqa: BLE001 - collect and report
            failures.append(f"{name} {version}: {e}")
            print(f"[FAIL] {name} {version}: {e}")
    for name, version, rec in map_recorded_packages():
        dest = SOURCES / f"{name}-{version}"
        if (dest / ".cargo-checksum.json").exists():
            continue
        if (name, version) in recorded and (name, version) in {(n, v) for n, v in pkgs}:
            pass  # lock loop already tried it; fall through if it failed
        cmd = [
            "python3", str(VENDOR_DIR / "vendor.py"),
            "--name", name, "--version", version,
            "--repo", rec["repo"], "--tag", str(rec["tag"]),
        ]
        if rec.get("subdir"):
            cmd += ["--subdir", str(rec["subdir"])]
        print(f"[map ] {name} {version} <- {rec['repo']}@{rec['tag']}"
              + (f" ({rec['subdir']})" if rec.get("subdir") else ""))
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            # recorded records can be stale (repo layout moved); report
            # but keep going — the resolve loop will find a new source.
            print(f"       skipped: {(r.stderr or r.stdout).strip().splitlines()[-1][:160] if (r.stderr or r.stdout).strip() else '?'}")

    if failures:
        print(f"\n{len(failures)} failures:")
        for f in failures:
            print("  " + f)
        return 1
    print("\nall registry packages vendored")
    return 0


if __name__ == "__main__":
    sys.exit(main())
