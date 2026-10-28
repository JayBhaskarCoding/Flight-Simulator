#!/usr/bin/env python3
"""Air-gapped crate vendoring for OpenMSFS.

Fetches a crate's source from its GitHub repository at the matching release
tag, normalizes the manifest (workspace inheritance, path dependencies),
strips dev-dependencies, and installs it into a Cargo *directory source*
compatible with `cargo vendor` (.cargo-checksum.json included).

Typical use:

    python3 tools/vendor/vendor.py --name glam --version 0.30.0

Environment:
    VENDOR_DIR   target directory source (default: ~/cargo-vendored-sources)
    CRATES_MAP   JSON file with crate -> repo overrides
                 (default: tools/vendor/crates-map.json)

The crates vendored this way are MIT/Apache-2.0 licensed open-source
projects published by their authors on crates.io; this tool only mirrors
public source releases and performs no reverse engineering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:  # pragma: no cover
    sys.exit("python 3.11+ required (tomllib)")


# ---------------------------------------------------------------------------
# TOML serialization (subset used by Cargo manifests)
# ---------------------------------------------------------------------------

_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")

_DEP_SPEC_KEYS = {
    "version", "path", "git", "branch", "tag", "rev", "registry",
    "features", "default-features", "optional", "package", "public",
    "workspace", "artifact", "target",
}


def _key(k: str) -> str:
    return k if _BARE_KEY.match(k) else json.dumps(k, ensure_ascii=False)


def _path_keys(path: tuple[str, ...]) -> str:
    return ".".join(_key(k) for k in path)


def _inline(v: Any) -> str:
    """Strictly single-line value rendering (valid inside inline tables)."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        return "[" + ", ".join(_inline(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{_key(k)} = {_inline(x)}" for k, x in v.items()) + " }"
    raise TypeError(f"cannot serialize {type(v)}")


def _is_aot(v: Any) -> bool:
    return isinstance(v, list) and bool(v) and all(isinstance(x, dict) for x in v)


def _is_inline_table(v: Any) -> bool:
    """A dict rendered as `{ ... }` instead of a `[section]`.

    Contextual rule: dependency-spec style dicts are inlined. Everything
    else becomes a section. Both forms carry identical data, so the choice
    is syntactic; we only need Cargo to parse the result.
    """
    return isinstance(v, dict) and bool(v) and all(k in _DEP_SPEC_KEYS for k in v)


def _render(v: Any, indent: str = "") -> str:
    """Top-level (section child) rendering; allows multi-line arrays."""
    if isinstance(v, list) and not _is_aot(v):
        parts = [_inline(x) for x in v]
        one = "[" + ", ".join(parts) + "]"
        if len(one) <= 100 or not parts:
            return one
        return "[\n" + "".join(f"{indent}    {p},\n" for p in parts) + f"{indent}]"
    return _inline(v)


def _scalar_lines(data: dict) -> list[str]:
    out: list[str] = []
    for k, v in data.items():
        if _is_aot(v):
            continue
        if isinstance(v, dict) and not _is_inline_table(v):
            continue
        out.append(f"{_key(k)} = {_render(v)}")
    return out


def _emit_children(lines: list[str], data: dict, path: tuple[str, ...]) -> None:
    for k, v in data.items():
        if _is_aot(v):
            header = _path_keys(path + (k,))
            for item in v:
                lines.append(f"[[{header}]]")
                lines.extend(_scalar_lines(item))
                lines.append("")
                _emit_children(lines, item, path + (k,))
        elif isinstance(v, dict) and not _is_inline_table(v):
            if v:
                _emit_table(lines, v, path + (k,))
            else:
                lines.append(f"[{_path_keys(path + (k,))}]")
                lines.append("")


def _emit_table(lines: list[str], data: dict, path: tuple[str, ...]) -> None:
    lines.append(f"[{_path_keys(path)}]")
    lines.extend(_scalar_lines(data))
    lines.append("")
    _emit_children(lines, data, path)


def toml_dumps(data: dict) -> str:
    lines: list[str] = []
    lines.extend(_scalar_lines(data))
    if lines:
        lines.append("")
    _emit_children(lines, data, ())
    text = "\n".join(lines)
    while text.endswith("\n\n"):
        text = text[:-1]
    if not text.endswith("\n"):
        text += "\n"
    return text


# ---------------------------------------------------------------------------
# process helpers
# ---------------------------------------------------------------------------

def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def run(cmd: list[str], check: bool = True) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if check and p.returncode != 0:
        die(f"command failed: {' '.join(cmd)}\n{p.stderr.strip()}")
    return p.stdout


def _map_path() -> Path:
    env = os.environ.get("CRATES_MAP")
    return Path(env) if env else Path(__file__).with_name("crates-map.json")


def load_map() -> dict:
    p = _map_path()
    if p.exists():
        return json.loads(p.read_text())
    return {}


def save_map(m: dict) -> None:
    p = _map_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")


def gh_json(args: list[str]) -> Any:
    return json.loads(run(["gh", "api"] + args))


def discover_repo(name: str) -> str | None:
    try:
        items = gh_json([
            "-X", "GET", "search/repositories",
            "-f", f"q={name} in:name",
            "-f", "per_page=10",
        ]).get("items", [])
    except SystemExit:
        return None
    aliases = {name, f"{name}-rs", f"rust-{name}", name.replace("_", "-")}
    for it in items:
        if it["name"] in aliases:
            return it["full_name"]
    for it in items:
        desc = (it.get("description") or "").lower()
        if name.replace("_", "-") in desc.replace("_", "-") and ("crate" in desc or "rust" in desc):
            return it["full_name"]
    return items[0]["full_name"] if items else None


def list_tags(repo: str) -> list[str]:
    out = run(["git", "ls-remote", "--tags", f"{repo_base(repo)}.git"])
    tags: list[str] = []
    for line in out.splitlines():
        if "\t" not in line:
            continue
        _, ref = line.split("\t", 1)
        if ref.startswith("refs/tags/") and not ref.endswith("^{}"):
            tags.append(ref[len("refs/tags/"):])
    return tags


def pick_tag(tags: list[str], name: str, version: str) -> str | None:
    ver = re.escape(version)
    exact_candidates = [
        version,
        f"v{version}",
        f"{name}-{version}",
        f"{name}-v{version}",
        f"release-{version}",
        f"release-v{version}",
    ]
    for c in exact_candidates:
        if c in tags:
            return c
    # bounded suffix match, e.g. "cargo-1.88.0" for version 1.88.0,
    # but never "foo-11.0.0" for 1.0.0 (digit boundary required).
    pat = re.compile(rf"(^|[^0-9]){ver}$")
    matches = [t for t in tags if pat.search(t)]
    if matches:
        matches.sort(key=len)
        return matches[0]
    return None


def repo_base(repo: str) -> str:
    """Clone URL base: full http(s) URLs pass through, `owner/name`
    shorthand resolves to GitHub."""
    if repo.startswith("http://") or repo.startswith("https://"):
        return repo.rstrip("/")
    return f"https://github.com/{repo}"


def download_tarball(repo: str, ref: str, dest_dir: Path) -> Path:
    tarball = dest_dir / "src.tar.gz"
    # a 40-hex ref is a commit sha (codeload serves it directly); otherwise
    # try a tag first and fall back to a branch name.
    if len(ref) == 40 and all(c in "0123456789abcdef" for c in ref.lower()):
        candidates = [ref]
    else:
        candidates = [f"refs/tags/{ref}", f"refs/heads/{ref}"]
    data = None
    last_err = None
    for cand in candidates:
        if repo.startswith("http://") or repo.startswith("https://"):
            # Non-GitHub git hosts use the archive endpoint
            # ({base}/-/archive/{ref}/{name}-{ref}.tar.gz for GitLab).
            raw = cand.removeprefix("refs/tags/").removeprefix("refs/heads/")
            raw_q = urllib.request.quote(raw, safe="")
            name = repo.rstrip("/").rsplit("/", 1)[-1]
            url = (
                f"{repo.rstrip('/')}/-/archive/{raw_q}/"
                f"{urllib.request.quote(f'{name}-{raw}', safe='')}.tar.gz"
            )
        else:
            url = f"https://codeload.github.com/{repo}/tar.gz/{urllib.request.quote(cand)}"
        req = urllib.request.Request(url, headers={"User-Agent": "openmsfs-vendor"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                data = r.read()
            last_err = None
            break
        except Exception as e:  # noqa: BLE001 - try the next ref form
            last_err = e
    if last_err is not None:
        raise last_err
    with open(tarball, "wb") as f:
        f.write(data)
    extract_dir = dest_dir / "src"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:gz") as tf:
        # python 3.11.2 lacks the extraction filter kwarg; guard manually
        base = extract_dir.resolve()
        for m in tf.getmembers():
            dest = (extract_dir / m.name).resolve()
            if not str(dest).startswith(str(base)):
                die(f"unsafe path in tarball: {m.name}")
        tf.extractall(extract_dir)
    entries = list(extract_dir.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return extract_dir


def find_package_dirs(root: Path, name: str) -> list[Path]:
    hits: list[Path] = []
    for p in root.rglob("Cargo.toml"):
        if "target" in p.parts or ".git" in p.parts:
            continue
        try:
            data = tomllib.loads(p.read_text(errors="replace"))
        except Exception:
            continue
        pkg = data.get("package")
        if isinstance(pkg, dict) and pkg.get("name") == name:
            hits.append(p.parent)
    return hits


def find_workspace_root(pkg_dir: Path, stop: Path) -> Path | None:
    cur = pkg_dir
    found: Path | None = None
    while True:
        m = cur / "Cargo.toml"
        if m.exists():
            try:
                data = tomllib.loads(m.read_text(errors="replace"))
            except Exception:
                data = {}
            if isinstance(data.get("workspace"), dict):
                found = cur
        if cur == stop or cur == cur.parent:
            break
        cur = cur.parent
    return found


# ---------------------------------------------------------------------------
# manifest normalization
# ---------------------------------------------------------------------------

def normalize_manifest(pkg_dir: Path, name: str, src_root: Path) -> None:
    manifest_path = pkg_dir / "Cargo.toml"
    data = tomllib.loads(manifest_path.read_text(errors="replace"))

    pkg = data.get("package")
    if not isinstance(pkg, dict):
        die(f"{pkg_dir}: no [package] section")

    ws_root = find_workspace_root(pkg_dir, src_root)
    ws: dict = {}
    if ws_root is not None:
        try:
            ws = tomllib.loads((ws_root / "Cargo.toml").read_text(errors="replace"))
        except Exception:
            ws = {}
    ws_pkg = (ws.get("workspace") or {}).get("package") or {}
    ws_deps = (ws.get("workspace") or {}).get("dependencies") or ws.get("dependencies") or {}
    ws_lints = (ws.get("workspace") or {}).get("lints")

    # [package] workspace inheritance
    for key in list(pkg.keys()):
        val = pkg[key]
        if isinstance(val, dict) and val.get("workspace") is True:
            if key in ws_pkg:
                pkg[key] = ws_pkg[key]
            else:
                pkg.pop(key)
    pkg.pop("workspace", None)
    # cargo defaults a missing edition to 2015; keep old manifests buildable
    if pkg.get("edition") is None:
        pkg["edition"] = "2015"
    if not isinstance(pkg.get("version"), str) or not isinstance(pkg.get("edition"), str):
        die(f"{pkg_dir}: unresolved version/edition after inheritance "
            f"(version={pkg.get('version')!r}, edition={pkg.get('edition')!r})")

    def resolve_dep(dep_key: str, spec: Any, ctx: str) -> Any:
        if isinstance(spec, dict) and spec.get("workspace") is True:
            base = ws_deps.get(dep_key)
            if base is None:
                die(f"{pkg_dir}: workspace dep '{dep_key}' ({ctx}) missing in {ws_root}")
            base = {"version": base} if isinstance(base, str) else dict(base)
            merged = dict(base)
            for lk, lv in spec.items():
                if lk == "workspace":
                    continue
                if lk == "features":
                    merged["features"] = sorted(set(base.get("features", [])) | set(lv))
                else:
                    merged[lk] = lv
            spec = merged
        if isinstance(spec, str):
            return spec
        if isinstance(spec, dict):
            spec = dict(spec)
            if "path" in spec:
                rel = spec.pop("path")
                dep_manifest = (pkg_dir / rel).resolve()
                dep_ver = None
                if (dep_manifest / "Cargo.toml").exists():
                    dep_data = tomllib.loads(
                        (dep_manifest / "Cargo.toml").read_text(errors="replace"))
                    dp = dep_data.get("package") or {}
                    dv = dp.get("version")
                    if isinstance(dv, dict) and dv.get("workspace"):
                        droot = find_workspace_root(dep_manifest, src_root)
                        if droot:
                            dws = tomllib.loads((droot / "Cargo.toml").read_text(errors="replace"))
                            dv = ((dws.get("workspace") or {}).get("package") or dws.get("package") or {}).get("version")
                    dep_ver = dv if isinstance(dv, str) else None
                if "version" not in spec:
                    if dep_ver is None:
                        die(f"{pkg_dir}: path dep '{dep_key}' ({ctx}) lacks version "
                            f"and sibling version is unknown")
                    spec["version"] = dep_ver
            if not spec:
                spec = {"version": "*"}
        return spec

    for top_key in list(data.keys()):
        sec = data[top_key]
        if top_key in ("dependencies", "build-dependencies", "dev-dependencies") and isinstance(sec, dict):
            # dev-deps are kept: some upstream features reference them and
            # cargo validates the reference against the manifest
            data[top_key] = {k: resolve_dep(k, v, top_key) for k, v in sec.items()}
        elif top_key == "target" and isinstance(sec, dict):
            for tgt, tsec in list(sec.items()):
                if not isinstance(tsec, dict):
                    continue
                for dk in ("dependencies", "build-dependencies", "dev-dependencies"):
                    if isinstance(tsec.get(dk), dict):
                        tsec[dk] = {k: resolve_dep(k, v, dk) for k, v in tsec[dk].items()}
                if not tsec:
                    sec.pop(tgt)

    # workspace lints inheritance
    if isinstance(data.get("lints"), dict) and data["lints"].get("workspace"):
        if ws_lints and not (isinstance(ws_lints, dict) and ws_lints.get("workspace")):
            data["lints"] = ws_lints
        else:
            data.pop("lints")

    # crate may double as its own workspace root
    data.pop("workspace", None)

    # drop metadata that points outside the vendored subtree
    for key in ("readme", "license-file"):
        v = data.get("package", {}).get(key)
        if isinstance(v, str) and v and not (pkg_dir / v).exists():
            data["package"].pop(key)

    for sect in ("dependencies", "build-dependencies"):
        for k, spec in (data.get(sect) or {}).items():
            if isinstance(spec, dict) and "path" in spec:
                die(f"{pkg_dir}: unresolved path dep in {sect}: {k}")

    # sanity: result must round-trip
    text = toml_dumps(data)
    try:
        tomllib.loads(text)
    except Exception as e:
        die(f"{pkg_dir}: normalized manifest failed to re-parse: {e}\n---\n{text[:2000]}")
    manifest_path.write_text(text)


# ---------------------------------------------------------------------------
# checksum + install
# ---------------------------------------------------------------------------

def write_checksum(pkg_dir: Path) -> None:
    files: dict[str, str] = {}
    for p in sorted(pkg_dir.rglob("*")):
        if p.is_file() and p.name != ".cargo-checksum.json":
            rel = p.relative_to(pkg_dir).as_posix()
            files[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    blob = json.dumps(files, sort_keys=True).encode()
    package_hash = hashlib.sha256(blob).hexdigest()
    (pkg_dir / ".cargo-checksum.json").write_text(
        json.dumps({"files": files, "package": package_hash}, indent=1))


def install(pkg_src: Path, name: str, version: str, vendor_dir: Path) -> Path:
    target = vendor_dir / f"{name}-{version}"
    if target.exists():
        shutil.rmtree(target)
    vendor_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(pkg_src, target, symlinks=True)
    return target


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="crates.io package name")
    ap.add_argument("--version",
                    help="exact package version to vendor "
                         "(derived from the manifest when omitted)")
    ap.add_argument("--repo", help="github owner/name (skips discovery)")
    ap.add_argument("--tag", help="git tag (skips tag selection)")
    ap.add_argument("--subdir", help="path of the package inside the repository")
    ap.add_argument("--map-only", action="store_true",
                    help="only record the discovered repo/tag in crates-map.json")
    args = ap.parse_args()

    vendor_dir = Path(os.environ.get("VENDOR_DIR")
                      or os.path.expanduser("~/cargo-vendored-sources"))
    if args.version:
        existing = vendor_dir / f"{args.name}-{args.version}"
        if existing.exists() and (existing / ".cargo-checksum.json").exists():
            print(f"already vendored: {existing}")
            return

    cmap = load_map()
    entry = cmap.get(args.name)
    repo, subdir, tag0 = args.repo, args.subdir, args.tag
    if isinstance(entry, dict):
        repo = repo or entry.get("repo")
        subdir = subdir or entry.get("subdir")
        tag0 = tag0 or entry.get("tag")
    elif isinstance(entry, str):
        repo = repo or entry

    if not repo:
        repo = discover_repo(args.name)
        if not repo:
            die(f"could not discover a GitHub repository for '{args.name}'")
        print(f"discovered repo: {repo}")

    tag = tag0
    if not tag:
        if not args.version:
            die("--tag (or a crates-map tag entry) is required when "
                "--version is omitted")
        tags = list_tags(repo)
        tag = pick_tag(tags, args.name, args.version)
        if not tag:
            sample = ", ".join(sorted(tags)[-15:])
            die(f"no tag found for {args.name} {args.version} in {repo} "
                f"(recent tags: {sample}); pass --tag manually")
    if args.map_only:
        cmap[args.name] = {"repo": repo, "tag": tag, **({"subdir": subdir} if subdir else {})}
        save_map(cmap)
        print(f"map updated: {args.name} -> {repo}@{tag}")
        return

    print(f"vendoring {args.name} {args.version or '(manifest version)'} "
          f"<- {repo}@{tag}"
          + (f" (subdir {subdir})" if subdir else ""))
    with tempfile.TemporaryDirectory(prefix="vendor-") as td:
        tdp = Path(td)
        root = download_tarball(repo, tag, tdp)
        if subdir:
            pkg_dir = root / subdir
            if not (pkg_dir / "Cargo.toml").exists():
                die(f"subdir {subdir} has no Cargo.toml under {root}")
        else:
            hits = find_package_dirs(root, args.name)
            if not hits:
                roots = [str(p.parent.relative_to(root))
                         for p in root.rglob("Cargo.toml") if "target" not in p.parts]
                die(f"could not locate package '{args.name}' in {repo}@{tag}; "
                    f"candidates: {roots[:12]} (use --subdir)")
            if len(hits) > 1:
                hits.sort(key=lambda p: len(p.parts))
                print(f"multiple package dirs: {[str(h) for h in hits]}; using {hits[0]}")
            pkg_dir = hits[0]

        normalize_manifest(pkg_dir, args.name, root)
        if not args.version:
            # Derive the exact package version from the (normalized)
            # manifest so the directory name always matches what cargo
            # will see inside — tags in monorepos often differ from the
            # member crate's version (e.g. glutin v0.32.3 -> wgl_sys 0.6.1).
            import tomllib as _tl
            _pkg = _tl.loads((pkg_dir / "Cargo.toml").read_text())["package"]
            args.version = str(_pkg["version"])
            existing = vendor_dir / f"{args.name}-{args.version}"
            if existing.exists() and (existing / ".cargo-checksum.json").exists():
                print(f"already vendored: {existing}")
                return
        stage = tdp / "pkg"
        shutil.copytree(pkg_dir, stage, symlinks=True)
        for junk in ("target", ".git"):
            shutil.rmtree(stage / junk, ignore_errors=True)
        # Repos sometimes commit *broken* symlinks (generated code that
        # is gitignored, e.g. madsmtm/objc2's src/generated -> empty
        # root dir). Keep the path but make it an empty directory so the
        # directory source stays readable — such crates are only
        # resolved (never compiled) on this target anyway.
        for path in sorted(stage.rglob("*"), reverse=True):
            if path.is_symlink() and not path.exists():
                path.unlink()
                path.mkdir()
        write_checksum(stage)
        out = install(stage, args.name, args.version, vendor_dir)

    cmap = load_map()
    cmap[args.name] = {"repo": repo, "tag": tag, **({"subdir": subdir} if subdir else {})}
    save_map(cmap)
    print(f"installed -> {out}")


if __name__ == "__main__":
    main()
