"""Check the role-based public docs, BSU research roadmap, and local links."""
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"README.md", "THIRD_PARTY_NOTICES.md", "docs/MODEL_ARCHITECTURE.md",
    "docs/RUN_GUIDE.md", "docs/DATA.md", "docs/METRICS.md", "docs/RESULTS.md",
    "docs/research/BSU_RATE_QUALITY_IDEAS_2026-10-11.md",
    "docs/research/BSU_GEN_IMPROVEMENT_IDEAS_2026-10-19.md",
    "docs/research/BSU_JOINT_ALLOCATION_PREDICTOR.md"}


def release_files():
    result = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, capture_output=True)
    if result.returncode == 0:
        return {p for p in result.stdout.decode().split("\0") if p and (ROOT / p).is_file()}
    # Source archives also need to be checkable without a .git directory.
    return {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()
        and not any(part in {".local", ".git", "outputs", "data", "__pycache__", ".pytest_cache"}
                    for part in p.relative_to(ROOT).parts)}


def anchors(text):
    result = set(re.findall(r'<a\s+id=["\']([^"\']+)', text))
    counts = {}
    fenced = False
    for line in text.splitlines():
        if line.startswith("```"):
            fenced = not fenced
        if fenced or not re.match(r"^#{1,6} ", line):
            continue
        title = re.sub(r"<[^>]+>", "", line.lstrip("# ")).lower()
        title = re.sub(r"[^\w\- ]", "", title).replace(" ", "-")
        ordinal = counts.get(title, 0)
        result.add(title + (f"-{ordinal}" if ordinal else ""))
        counts[title] = ordinal + 1
    return result


def main():
    files = release_files()
    markdown = {p for p in files if p.lower().endswith(".md")}
    errors = []
    if markdown != EXPECTED:
        errors.append(f"public Markdown differs: extra={sorted(markdown-EXPECTED)}, missing={sorted(EXPECTED-markdown)}")
    links = 0
    for name in sorted(EXPECTED & markdown):
        path = ROOT / name
        text = path.read_text()
        if text.count("```") % 2:
            errors.append(f"{name}: unmatched code fence")
        for url in re.findall(r"\[[^\]\n]+\]\(([^)\n]+)\)", text):
            if re.match(r"^(https?://|mailto:|data:)", url):
                continue
            target, _, fragment = unquote(url).partition("#")
            dest = (path.parent / target).resolve() if target else path
            try:
                relative = dest.relative_to(ROOT).as_posix()
            except ValueError:
                errors.append(f"{name}: link outside release: {url}")
                continue
            if relative not in files and not any(p.startswith(relative.rstrip("/") + "/") for p in files):
                errors.append(f"{name}: missing release target: {url}")
            elif fragment and dest.suffix == ".md" and fragment not in anchors(dest.read_text()):
                errors.append(f"{name}: missing anchor: {url}")
            links += 1
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"PASS: {len(markdown)} public Markdown files, {links} local links/anchors")


if __name__ == "__main__":
    main()
