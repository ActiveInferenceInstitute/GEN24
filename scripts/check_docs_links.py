#!/usr/bin/env python3
"""Check relative links in the repository's Markdown and the docs index.

Two checks, both offline:

1. Every relative link target in a tracked ``*.md`` file resolves on disk.
   Absolute URLs, bare ``#fragment`` targets and links inside fenced code
   blocks are not checked, and neither are fragments on a resolved file.
2. ``docs/README.md`` links every page directly under ``docs/``, and
   ``docs/documentation-index.md`` links every page under ``docs/`` at any
   depth. ``AGENTS.md`` is excluded from both: it is agent guidance, not a
   documentation page.

Exit codes: ``0`` when both checks pass, ``1`` when anything is broken.

    python scripts/check_docs_links.py
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HUB = ROOT / "docs" / "README.md"
FLAT_INDEX = ROOT / "docs" / "documentation-index.md"
EXCLUDED_FROM_INDEX = {"AGENTS.md", "README.md", "documentation-index.md"}

# [text](target) and ![alt](target); the leading "!" is not part of the match.
INLINE_LINK = re.compile(r"\[[^\]]*\]\(\s*(<[^>]*>|[^)\s]+)(?:\s+[\"'][^\"']*[\"'])?\s*\)")
REFERENCE_LINK = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*(\S+)")
ABSOLUTE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*:")


def tracked_markdown():
    """Return tracked and untracked-but-not-ignored Markdown paths.

    Untracked files are included so a new page is checked before it is
    committed; ignored files stay out so build output is never scanned. Falls
    back to a filesystem walk when git is unavailable.
    """
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "*.md"],
        cwd=ROOT,
        capture_output=True,
        stdin=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        return sorted(p for p in ROOT.rglob("*.md") if ".git" not in p.parts)
    return sorted(ROOT / name for name in result.stdout.decode().split("\0") if name)


def iter_links(text):
    """Yield (line number, raw target) for links outside fenced code blocks."""
    in_fence = False
    for number, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in INLINE_LINK.finditer(line):
            yield number, match.group(1)
        reference = REFERENCE_LINK.match(line)
        if reference:
            yield number, reference.group(1)


def resolve(source, raw):
    """Return the on-disk target for a relative link, or None when it is skipped."""
    target = raw.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1].strip()
    if not target or target.startswith("#") or ABSOLUTE.match(target):
        return None
    target = target.split("#", 1)[0].split("?", 1)[0]
    if not target:
        return None
    return (source.parent / target).resolve()


def check_links(files):
    """Return a list of (path, line, target) for every unresolvable link."""
    broken = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        for line, raw in iter_links(text):
            resolved = resolve(path, raw)
            if resolved is not None and not resolved.exists():
                broken.append((path, line, raw))
    return broken


def linked_targets(path):
    """Return the set of on-disk paths a document links to."""
    text = path.read_text(encoding="utf-8", errors="replace")
    targets = set()
    for _, raw in iter_links(text):
        resolved = resolve(path, raw)
        if resolved is not None:
            targets.add(resolved)
    return targets


def check_index():
    """Return a list of (index, missing page) for pages no index links."""
    missing = []
    for index, pattern in ((HUB, "docs/*.md"), (FLAT_INDEX, "docs/**/*.md")):
        if not index.exists():
            missing.append((index, "the index file itself"))
            continue
        linked = linked_targets(index)
        for page in sorted(ROOT.glob(pattern)):
            if page.name in EXCLUDED_FROM_INDEX or not page.is_file():
                continue
            if page.resolve() not in linked:
                missing.append((index, page.relative_to(ROOT).as_posix()))
    return missing


def main():
    files = tracked_markdown()
    broken = check_links(files)
    missing = check_index()

    for path, line, target in broken:
        print(f"broken link: {path.relative_to(ROOT).as_posix()}:{line} -> {target}")
    for index, page in missing:
        print(f"missing from {index.relative_to(ROOT).as_posix()}: {page}")

    if broken or missing:
        print(
            f"FAIL: {len(broken)} broken link(s), {len(missing)} unindexed page(s) "
            f"across {len(files)} Markdown file(s)"
        )
        return 1

    print(f"OK: {len(files)} Markdown files, relative links resolve, docs index complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
