#!/usr/bin/env python3
"""Drive vendor.py until `cargo check -p <pkg>` resolves (helper, not shipped).

Reads missing-package errors, looks up requirements from already-vendored
manifests, picks the newest satisfying semver tag from the mapped GitHub
repo, and vendors it. Repeat until the check passes.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path.home() / "Flight-Simulator" if (Path.home() / "Flight-Simulator").exists() else Path.cwd()
if not (REPO_DIR / "Cargo.toml").exists():
    REPO_DIR = Path("/home/user/Flight-Simulator")
VENDOR = REPO_DIR / "tools" / "vendor"
SOURCES = Path.home() / "cargo-vendored-sources"
_raw_map = json.loads((VENDOR / "crates-map.json").read_text())
# values are "owner/repo" strings or {"repo","subdir"} objects
MAP: dict = {}
SUBDIR: dict = {}
for _k, _v in _raw_map.items():
    if isinstance(_v, dict):
        MAP[_k] = _v["repo"]
        if _v.get("subdir"):
            SUBDIR[_k] = _v["subdir"]
    else:
        MAP[_k] = _v


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def requirement_for(name: str):
    """Find a version requirement for `name` in vendored manifests, the
    workspace manifest and workspace member manifests."""
    pat_sec = re.compile(rf"^\[dependencies\.{re.escape(name)}\]", re.M)
    pat_inline = re.compile(rf'^{re.escape(name)}\s*=\s*"([^"]+)"', re.M)
    best = None
    manifests = list(SOURCES.glob("*/Cargo.toml"))
    manifests.append(REPO_DIR / "Cargo.toml")
    manifests.extend(sorted(REPO_DIR.glob("crates/*/Cargo.toml")))
    for manifest in manifests:
        if not manifest.exists():
            continue
        text = manifest.read_text(errors="replace")
        m = pat_sec.search(text)
        if m:
            # requirement inside the section: first `version = "..."`
            end = text.find("\n[", m.end())
            section = text[m.end():end if end != -1 else None]
            v = re.search(r'version\s*=\s*"([^"]+)"', section)
            if v:
                return v.group(1)
        m = pat_inline.search(text)
        if m and not m.group(1).startswith("{"):
            best = m.group(1)
    if best:
        return best
    # inline table form: name = { version = "..." }
    pat_tbl = re.compile(rf'^{re.escape(name)}\s*=\s*{{[^}}]*version\s*=\s*"([^"]+)"', re.M)
    # renamed package: alias = { package = "name", version = "..." }
    pat_rename = re.compile(
        rf'^[\w-]+\s*=\s*{{[^}}]*package\s*=\s*"{re.escape(name)}"[^}}]*version\s*=\s*"([^"]+)"',
        re.M,
    )
    for manifest in manifests:
        if not manifest.exists():
            continue
        text = manifest.read_text(errors="replace")
        m = pat_tbl.search(text) or pat_rename.search(text)
        if m:
            return m.group(1)
    return None


def parse_req(req: str):
    """Return (op, major, minor, patch) for caret/equals/range-free reqs."""
    req = req.strip()
    if req.startswith("^"):
        return ("caret", *version_triple(req[1:]))
    if req.startswith("="):
        return ("eq", *version_triple(req[1:]))
    if req[0].isdigit():
        return ("caret", *version_triple(req))
    # tilde etc: fall back to caret on the base
    if req.startswith("~"):
        return ("tilde", *version_triple(req[1:]))
    raise ValueError(f"unsupported req {req!r}")


def version_triple(v: str):
    parts = re.split(r"[.\-+]", v.strip())
    nums = []
    for p in parts[:3]:
        nums.append(int(p) if p.isdigit() else 0)
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums)


def satisfies(v: str, req) -> bool:
    op, major, minor, patch = req
    try:
        t = version_triple(v)
    except Exception:
        return False
    if any(ch.isalpha() for ch in v.split("-")[0] if not ch.isdigit() and ch != "."):
        return False  # prerelease-ish
    if "-" in v:
        return False  # skip prereleases entirely
    if op == "eq":
        return t == (major, minor, patch)
    if op == "tilde":
        # ~X.Y.Z -> >=X.Y.Z <X.(Y+1).0 ; ~X.Y -> >=X.Y.0 <X.(Y+1).0
        return t[0] == major and t[1] == minor and t >= (major, minor, patch)
    # caret: ^major.minor.patch
    #   major > 0 -> same major, at least this version
    #   major = 0, minor > 0 -> 0.minor.*, at least this version
    #   0.0.patch -> same 0.0.* series, patch >= requested
    #   ("^0.0" as written by cargo means >=0.0.0 <0.1.0, i.e. any 0.0.z)
    if major > 0:
        return t[0] == major and t >= (major, minor, patch)
    if minor > 0:
        return t[0] == 0 and t[1] == minor and t >= (0, minor, patch)
    return t[0] == 0 and t[1] == 0 and t[2] >= patch


def repo_base(repo: str) -> str:
    if repo.startswith("http://") or repo.startswith("https://"):
        return repo.rstrip("/")
    return f"https://github.com/{repo}"


def tag_score(name: str, tag: str, raw: str) -> int:
    """Prefer tags that belong to THIS crate when a repo carries
    several crates (e.g. rust-mobile/ndk: `ndk-0.9.0` must win over
    `cargo-apk-0.9.6`, both of which match `^0.9.0`)."""
    if tag in (f"{name}-{raw}", f"{name}-v{raw}", f"{name}_v{raw}"):
        return 0
    if tag.startswith(f"{name}-"):
        return 1
    if tag in (raw, f"v{raw}"):
        return 2
    return 3


def best_tag(repo: str, name: str, req) -> str | None:
    r = run(["git", "ls-remote", "--tags", f"{repo_base(repo)}.git"])
    if r.returncode != 0:
        print(f"    ls-remote failed for {repo}: {r.stderr.strip()[:120]}")
        return None
    matches: list[tuple[int, tuple[int, int, int], str]] = []
    for line in r.stdout.splitlines():
        ref = line.split("\t")[-1]
        if ref.endswith("^{}"):
            continue
        tag = ref.split("refs/tags/")[-1]
        # tag styles: v1.2.3, 1.2.3, tokio-1.44.2, name-v0.2.15 ...
        mver = re.search(r"(\d+(?:\.\d+){0,2})(?!.*\d)", tag)
        if not mver:
            continue
        raw = mver.group(1)
        if satisfies(raw, req):
            t = version_triple(raw)
            matches.append((tag_score(name, tag, raw), tuple(-x for x in t), tag))
    if not matches and req[0] == "caret" and req[1] == 0:
        # wasm-tools monorepo: crate 0.X.Y ships at tag v1.X.Y — only
        # tags whose minor equals the requested minor qualify.
        want_minor, want_patch = req[2], req[3]
        for line in r.stdout.splitlines():
            ref = line.split("\t")[-1]
            if ref.endswith("^{}"):
                continue
            tag = ref.split("refs/tags/")[-1]
            mver = re.search(r"(\d+(?:\.\d+){0,2})(?!.*\d)", tag)
            if not mver:
                continue
            t = version_triple(mver.group(1))
            if t[0] == 1 and t[1] == want_minor and t[2] >= want_patch:
                matches.append(
                    (tag_score(name, tag, mver.group(1)), tuple(-x for x in t), tag)
                )
    if not matches and req[0] == "caret" and req[1] >= 100:
        # wasm-tools wast/wat: crate M.0.P ships at repo tag v1.M.P
        # (crate version major == repo tag minor).
        want_major, want_patch = req[1], req[3]
        for line in r.stdout.splitlines():
            ref = line.split("\t")[-1]
            if ref.endswith("^{}"):
                continue
            tag = ref.split("refs/tags/")[-1]
            mver = re.search(r"(\d+(?:\.\d+){0,2})(?!.*\d)", tag)
            if not mver:
                continue
            t = version_triple(mver.group(1))
            if t[0] == 1 and t[1] == want_major and t[2] == want_patch:
                matches.append(
                    (tag_score(name, tag, mver.group(1)), tuple(-x for x in t), tag)
                )
    if not matches:
        return None
    matches.sort()
    return matches[0][2]


def main() -> int:
    pkg = sys.argv[1] if len(sys.argv) > 1 else "sim-world"
    attempts: dict[str, int] = {}
    for round_no in range(1, 60):
        r = run(["cargo", "check", "-p", pkg], cwd=REPO_DIR, timeout=600)
        out = r.stdout + r.stderr
        if r.returncode == 0:
            print("CHECK OK")
            return 0
        m = re.search(r"no matching package named `([^`]+)` found", out)
        if not m:
            m = re.search(r"searched package name: `([^`]+)`", out)
        if not m:
            m2 = re.search(r"failed to select a version for the requirement `([^`]+)`", out)
            if not m2:
                print("UNPARSED ERROR:\n" + out[-3000:])
                return 2
            spec = m2.group(1).strip()
            # spec forms: "name req" or just "req" (then the name came from elsewhere)
            parts = spec.split()
            if parts[0][0].isdigit() or parts[0][0] in "^~=":
                name = pkg if False else name if False else parts[0]
                # requirement-only spec: fall back to the searched package name
                ms = re.search(r"searched package name: `([^`]+)`", out)
                name = ms.group(1) if ms else ""
                req_s = parts[0]
            else:
                name = parts[0]
                # cargo may print "name = 1.2.3" with spaces around the op
                req_s = "".join(parts[1:]) if len(parts) > 1 else ""
            req_s = req_s.strip().strip('"').strip("'")
            if req_s.startswith("="):
                req_s = req_s[1:].lstrip("=")
            req_s = req_s.strip("^")
        else:
            name = m.group(1)
            req_s = requirement_for(name)
        print(f"[round {round_no}] missing: {name} req={req_s}")
        if name not in MAP:
            print(f"    NO REPO MAPPING for {name}; add to crates-map.json")
            return 3
        # Shared guard: the same package must not be retried forever
        # (a bad map tag or a lying directory used to spin all 60 rounds).
        attempts[name] = attempts.get(name, 0) + 1
        if attempts[name] > 3:
            print(f"    {name} still missing after 3 install attempts; "
                  f"fix the crates-map entry or the vendored directory")
            return 3
        # Map entries that pin an exact repo tag are handled below, once
        # the requirement is known (the pinned tag must satisfy it — one
        # crate can need several major versions, e.g. redox_syscall
        # 0.5.18 AND 0.9.x from the same repo).
        repo = MAP[name]
        # harden: strip cargo quoting/formatting noise from any branch
        req_s = (req_s or "").strip().strip('"').strip("'")
        while req_s and req_s[0] in "=^~ ":
            req_s = req_s[1:]
        req = None
        if req_s:
            try:
                req = parse_req(req_s)
            except ValueError:
                mv = re.search(r"\d+(?:\.\d+){0,2}", req_s)
                if mv:
                    req = ("caret", *version_triple(mv.group(0)))
        if req is None:
            print(f"    cannot determine requirement for {name} (got {req_s!r})")
            return 3
        # Map entries that pin an exact repo tag: install straight from
        # that tag (and let vendor.py derive the crate version from the
        # manifest) — but only when that pinned version actually
        # satisfies the current requirement.
        entry = _raw_map.get(name)
        if isinstance(entry, dict) and entry.get("repo") and entry.get("tag"):
            # Raw commit SHAs and subdir members are trusted (member
            # crate versions routinely diverge from the repo tag, e.g.
            # windows-rs tag 0.58.0 -> windows-targets 0.52.6): vendor.py
            # bakes the manifest version into the directory name.
            tag = entry["tag"]
            is_sha = bool(re.fullmatch(r"[0-9a-f]{7,40}", tag))
            mver = re.search(r"(\d+(?:\.\d+){0,2})(?!.*\d)", tag) if not is_sha else None
            if is_sha or entry.get("subdir") or (mver and satisfies(mver.group(1), req)):
                cmd = ["python3", str(VENDOR / "vendor.py"), "--name", name,
                       "--repo", entry["repo"], "--tag", entry["tag"]]
                if entry.get("subdir"):
                    cmd += ["--subdir", entry["subdir"]]
                print(f"    -> {entry['repo']} @ {entry['tag']} "
                      f"(version from manifest)")
                v = run(cmd, timeout=600)
                print(v.stdout.strip()[-400:] or v.stderr.strip()[-400:])
                if v.returncode != 0:
                    return 4
                continue
        tag = best_tag(repo, name, req)
        # Subdir crates inside the wasmtime monorepo (cranelift/*) are
        # versioned independently of the repo tags: use the wasmtime
        # release tag matching the installed wasmtime package.
        if tag is None and SUBDIR.get(name) and repo == "bytecodealliance/wasmtime":
            wm = sorted(SOURCES.glob("wasmtime-[0-9]*"))
            if wm:
                cand = "v" + wm[-1].name[len("wasmtime-"):]
                chk = run(["git", "ls-remote", "--tags",
                           f"{repo_base(repo)}.git", f"refs/tags/{cand}"])
                if chk.returncode == 0 and chk.stdout.strip():
                    tag = cand
        if tag is None:
            print(f"    no satisfying tag for {name} {req_s} in {repo}")
            return 3
        print(f"    -> {repo} @ {tag}")
        mver = re.search(r"(\d+(?:\.\d+){0,2})(?!.*\d)", tag)
        version = mver.group(1)
        # wasm-tools monorepo: tag 1.X.Y carries crate version 0.X.Y
        # (and for wast/wat: tag 1.M.P carries crate version M.0.P).
        if req[0] == "caret" and req[1] == 0 and version.startswith("1."):
            version = "0." + version[2:]
        if req[0] == "caret" and req[1] >= 100 and version.startswith("1."):
            parts = version.split(".")
            version = f"{req[1]}.0.{parts[2] if len(parts) > 2 else 0}"
        # Subdir crate version differs from the repo tag's version: the
        # requirement is the crate's real version.
        if not satisfies(version, req) and req[0] in ("caret", "eq", "tilde"):
            version = f"{req[1]}.{req[2]}.{req[3]}"
        cmd = ["python3", str(VENDOR / "vendor.py"), "--name", name,
               "--version", version, "--repo", repo, "--tag", tag]
        if name in SUBDIR:
            cmd += ["--subdir", SUBDIR[name]]
        v = run(cmd, timeout=600)
        print(v.stdout.strip()[-400:] or v.stderr.strip()[-400:])
        if v.returncode != 0:
            return 4
    print("round limit reached")
    return 1


if __name__ == "__main__":
    sys.exit(main())
