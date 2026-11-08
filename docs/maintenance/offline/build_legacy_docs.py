#!/usr/bin/env python3
"""Build an offline manual with the published legacy URLs. Python 3.9+."""
import argparse
import html
from html.parser import HTMLParser
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import shutil
import sys
from urllib.parse import unquote, urlsplit

PROJECTS = ("Audio_ALG", "AEC", "NR", "audiolib")
MANIFEST = Path(__file__).with_name("aliases.tsv")
ANCHOR = re.compile(r'<a\b([^>]*?\bhref\s*=\s*)(["\'])(.*?)\2([^>]*)>(.*?)</a\s*>',
                    re.I | re.S)


def inside(path, root):
    return path == root or root in path.parents


def safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or name != str(path) or "\\" in name:
        raise ValueError(f"Invalid relative path in manifest: {name}")
    return path


def load_aliases():
    result = {project: {} for project in PROJECTS}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        project, old, current, *extras = line.split("\t")
        href, _, fragment = current.partition("#")
        src, dst = safe_name(old), safe_name(href)
        if (project not in result or src in result[project] or src.suffix != ".html"
                or not str(dst).startswith("docs/html/") or dst.suffix != ".html"):
            raise ValueError(f"Invalid legacy mapping: {line}")
        result[project][src] = (dst, fragment, extras[0].split(",") if extras else [])
    return result


def canonical_files(root, aliases):
    for folder in [root, root / "docs", root / "docs" / "html"]:
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError(f"Expected a local documentation directory: {folder}")
    files = {}
    for base, dirs, names in os.walk(root / "docs/html", followlinks=False):
        dirs[:] = sorted(name for name in dirs if not name.startswith("."))
        for name in dirs + sorted(names):
            if name.startswith("."):
                continue
            f = Path(base, name)
            if f.is_symlink():
                raise ValueError(f"Refusing to copy or follow a symlink: {f}")
            if name in dirs:
                continue
            if not f.is_file():
                raise ValueError(f"Expected a regular file: {f}")
            path = PurePosixPath(f.relative_to(root).as_posix())
            if path not in aliases:
                files[path] = f
    if PurePosixPath("docs/html/index.html") not in files:
        raise ValueError(f"Missing manual homepage in {root}")
    return files


def redirect(relative, default, extra_ids):
    base = json.dumps(relative)
    fragment = json.dumps("#" + default if default else "")
    fallback = html.escape(relative + ("#" + default if default else ""), quote=True)
    sections = "".join(f'<p><a id="{html.escape(name, quote=True)}" '
                       f'href="{html.escape(relative + "#" + name, quote=True)}">'
                       f'{html.escape(name)}</a></p>\n' for name in extra_ids)
    return f"""<!doctype html>
<html lang="zh-Hant">
<head><meta charset="utf-8"><title>文件位置</title>
<script>location.replace({base} + location.search + (location.hash || {fragment}));</script>
</head>
<body><p><a href="{fallback}">開啟文件</a></p>
{sections}</body></html>
"""


class Identifiers(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key == "id" or (tag == "a" and key == "name"):
                self.ids.add(value)


def docs_only_links(markup, current, available):
    """Keep checkout-only references legible without copying headers, source or submodules."""
    def render(m):
        href = html.unescape(m[3])
        u = urlsplit(href)
        if u.scheme or u.netloc or not u.path:
            return m[0]
        key = posixpath.normpath(posixpath.join(str(current.parent), unquote(u.path)))
        if key in available:
            return m[0]
        if key.startswith(current.parts[0] + "/docs/html/"):
            raise ValueError(f"Missing document linked from {current}: {href}")
        label = m[5]
        return f'<span title="{html.escape(key, quote=True)}">{label}</span>'
    return ANCHOR.sub(render, markup)


def build(selection, source, output):
    aliases = load_aliases()
    projects = PROJECTS if selection == "all" else (selection,)
    roots = {p: (source / p if selection == "all" else source) for p in projects}
    if any(inside(output, r) or inside(r, output) for r in roots.values()):
        raise ValueError("Source checkout and output must be disjoint")
    canonical = {p: canonical_files(r, aliases[p]) for p, r in roots.items()}
    available = {str(PurePosixPath(p) / rel) for p in projects
                 for rel in set(canonical[p]) | set(aliases[p])}
    # Only accept a new directory, also when two exporters race for the same destination.
    output.mkdir()
    try:
        for p in projects:
            out_root = output / p if selection == "all" else output
            ids = {}
            for rel, f in canonical[p].items():
                out = out_root / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                if f.suffix.lower() != ".html":
                    shutil.copyfile(f, out)
                    continue
                markup = docs_only_links(f.read_text(encoding="utf-8"), PurePosixPath(p) / rel, available)
                tags = Identifiers()
                tags.feed(markup)
                ids[rel] = tags.ids
                out.write_text(markup, encoding="utf-8")
            for old, (dst, fragment, extras) in aliases[p].items():
                if dst not in ids or not set(filter(None, [fragment, *extras])).issubset(ids[dst]):
                    raise ValueError(f"Missing target or section in {p}: {old} -> {dst}#{fragment}")
                out = out_root / old
                out.parent.mkdir(parents=True, exist_ok=True)
                url = posixpath.relpath(str(dst), str(old.parent))
                out.write_text(redirect(url, fragment, extras), encoding="utf-8")
    except BaseException:
        shutil.rmtree(output)
        raise
    for p in projects:
        print(f"{p}: {len(canonical[p])} manual files, {len(aliases[p])} legacy URLs")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--repo", choices=(*PROJECTS, "all"), required=True)
    cli.add_argument("--source", type=Path, required=True,
                     help="project checkout, or its parent for --repo all")
    cli.add_argument("--output", type=Path, required=True,
                     help="new directory under an existing parent, outside the checkout")
    args = cli.parse_args()
    try:
        # Resolve the parent (including a macOS /tmp alias), never dereference an output leaf.
        raw = args.output.absolute()
        if raw.is_symlink():
            raise ValueError(f"Output may not be a symlink: {raw}")
        build(args.repo, args.source.resolve(strict=True), raw.parent.resolve(strict=True) / raw.name)
    except (OSError, ValueError) as exc:
        print(f"Cannot build the documentation: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
