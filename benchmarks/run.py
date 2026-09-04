#!/usr/bin/env python3
"""Public, reproducible measurements of folio.

Two corpora, pinned by commit, fetched from GitHub and indexed through the
`folio` command on PATH. Nothing here reads folio's source: the harness sees
what any caller sees, which is the same surface the decision layer's fences use.

    ./run.py                 both corpora
    ./run.py rust-book       one of them
    ./run.py --keep          leave the fetched corpora in place for a re-run

Requires an OpenAI-compatible embeddings endpoint. See ../README.md; set
FOLIO_ENDPOINT, and pass --model so the report says what produced the numbers.

What this measures and what it does not is in README.md. In short: cost is
measured, contamination is measured, and ranking quality is not.
"""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Pinned so that two runs measure the same bytes. Bump deliberately, and say so
# in the change that bumps them.
CORPORA = {
    "rust-book": {
        "repo": "rust-lang/book",
        "commit": "1500248d8f230566e4ec9f27fcbb8fe9e2898ab1",
        "subpath": "src",
        "note": "no frontmatter at all, so it exercises heading sections alone",
    },
    "mdn": {
        "repo": "mdn/content",
        "commit": "f4c14731a1a157fc8d8f7357ac4d74d14a7d7fb5",
        "subpath": "files/en-us",
        "note": "frontmatter with a `status` list carrying `deprecated`",
    },
}

QUERY_COUNT = 40
TOP_K = 10
SEED = 20260905  # Fixed, so the sampled queries are the same on every run.


def die(message):
    print(f"error: {message}", file=sys.stderr)
    sys.exit(1)


def folio(*args, cwd=None):
    proc = subprocess.run(
        ["folio", *args], cwd=cwd, capture_output=True, text=True
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        hint = tail[0] if tail else "no output"
        if "embedding request to" in (proc.stderr or ""):
            die(f"no embeddings endpoint reachable: {hint}")
        die(f"folio {' '.join(args)} failed: {hint}")
    return proc.stdout


def fetch(name, spec, keep):
    root = HERE / "corpora" / name
    if root.exists():
        if keep:
            return root
        shutil.rmtree(root)
    root.mkdir(parents=True)

    url = f"https://codeload.github.com/{spec['repo']}/tar.gz/{spec['commit']}"
    print(f"  fetching {spec['repo']}@{spec['commit'][:12]}")
    with urllib.request.urlopen(url) as response:
        archive = HERE / f"{name}.tar.gz"
        archive.write_bytes(response.read())

    prefix = f"{spec['repo'].split('/')[1]}-{spec['commit']}/{spec['subpath']}/"
    with tarfile.open(archive) as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".md"):
                continue
            if not member.name.startswith(prefix):
                continue
            target = root / member.name[len(prefix):]
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is not None:
                target.write_bytes(source.read())
    archive.unlink()
    return root


def shape(root):
    files = list(root.rglob("*.md"))
    return len(files), sum(f.stat().st_size for f in files)


def records(root):
    return [
        json.loads(line)
        for line in (root / ".folio" / "sections.jsonl").read_text().splitlines()
    ]


def timed(fn):
    started = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - started


def hits(root, query, extra=()):
    """The (path, start, end) of each hit, in rank order."""
    out = folio("query", query, "-l", str(TOP_K), *extra, cwd=root)
    found = []
    for line in out.splitlines():
        if not line.startswith("#"):
            continue
        where = line.split()[-1]
        path, _, span = where.rpartition(":")
        start, _, end = span.partition("-")
        found.append((path, int(start), int(end)))
    return found


def deprecated_keys(rows):
    """Records whose `status` carries `deprecated`, keyed the way a hit is."""
    out = set()
    for r in rows:
        status = r.get("fm", {}).get("status")
        values = status if isinstance(status, list) else [status]
        if "deprecated" in [str(v) for v in values]:
            out.add((r["path"], r["start"], r["end"]))
    return out


def sample_queries(rows, count):
    """Queries drawn from the corpus without reference to `status`.

    Taking them from the titles of randomly sampled sections keeps the sample
    unbiased with respect to what is being measured: a query set chosen from
    deprecated pages would guarantee contamination rather than estimate it.
    """
    import random

    titles = []
    for r in rows:
        title = r.get("fm", {}).get("title") or r.get("heading")
        if isinstance(title, str) and len(title.split()) >= 2:
            titles.append(title)
    random.Random(SEED).shuffle(titles)
    seen, picked = set(), []
    for t in titles:
        if t not in seen:
            seen.add(t)
            picked.append(t)
        if len(picked) == count:
            break
    return picked


def measure(name, spec, model, keep):
    print(f"{name}: {spec['repo']}")
    root = fetch(name, spec, keep)
    files, bytes_ = shape(root)
    print(f"  {files} files, {bytes_ / 1e6:.1f} MB")

    args = ["index"] + (["--model", model] if model else [])
    _, index_seconds = timed(lambda: folio(*args, cwd=root))
    rows = records(root)
    dim = int(
        next(
            line.split()[-1]
            for line in folio("status", cwd=root).splitlines()
            if line.strip().startswith("dim")
        )
    )
    vectors = (root / ".folio" / "vectors.f32").stat().st_size
    print(f"  indexed in {index_seconds:.1f}s: {len(rows)} sections, dim {dim}")

    queries = sample_queries(rows, QUERY_COUNT)
    if not queries:
        die(f"{name} yielded no usable queries")
    latencies = []
    for q in queries:
        _, seconds = timed(lambda q=q: hits(root, q))
        latencies.append(seconds)

    # One file changed, so the whole file is re-embedded and nothing else is.
    victim = next(root.rglob("*.md"))
    victim.write_text(victim.read_text() + "\n<!-- benchmark -->\n")
    incremental_out, incremental_seconds = timed(lambda: folio(*args, cwd=root))
    reembedded = next(
        (
            line.strip().split()[0]
            for line in incremental_out.splitlines()
            if "re-embedded" in line
        ),
        "?",
    )

    report = {
        "corpus": spec["repo"],
        "commit": spec["commit"],
        "subpath": spec["subpath"],
        "note": spec["note"],
        "files": files,
        "bytes": bytes_,
        "sections": len(rows),
        "dim": dim,
        "index_seconds": round(index_seconds, 1),
        "vectors_bytes": vectors,
        "vectors_exact": vectors == dim * 4 * len(rows),
        "query_median_ms": round(statistics.median(latencies) * 1000),
        "query_max_ms": round(max(latencies) * 1000),
        "queries": len(queries),
        "top_k": TOP_K,
        "incremental_seconds": round(incremental_seconds, 2),
        "incremental_reembedded": reembedded,
        "model": model or "(endpoint default)",
        "endpoint": os.environ.get("FOLIO_ENDPOINT", "(folio default)"),
    }

    retired = deprecated_keys(rows)
    if retired:
        total = surfaced = 0
        for q in queries:
            got = hits(root, q)
            total += len(got)
            surfaced += sum(1 for h in got if h in retired)
        filtered_total = filtered_surfaced = 0
        for q in queries:
            got = hits(root, q, ("--where", "status!=deprecated"))
            filtered_total += len(got)
            filtered_surfaced += sum(1 for h in got if h in retired)
        report["deprecated_sections"] = len(retired)
        report["deprecated_share_of_corpus"] = round(100 * len(retired) / len(rows), 2)
        report["contamination_pct"] = round(100 * surfaced / max(total, 1), 2)
        report["contamination_filtered_pct"] = round(
            100 * filtered_surfaced / max(filtered_total, 1), 2
        )
        print(
            f"  deprecated: {report['deprecated_share_of_corpus']}% of the corpus, "
            f"{report['contamination_pct']}% of hits, "
            f"{report['contamination_filtered_pct']}% once filtered"
        )
    return report


def write_report(reports):
    out = HERE / "reports"
    out.mkdir(exist_ok=True)
    (out / "latest.json").write_text(json.dumps(reports, indent=2) + "\n")

    lines = [
        "# Benchmark report",
        "",
        "Generated by `run.py`. Read README.md for what these numbers are and are not.",
        "",
    ]
    for r in reports:
        lines += [
            f"## {r['corpus']}",
            "",
            f"`{r['commit']}`, `{r['subpath']}/`. {r['note']}.",
            "",
            f"Model `{r['model']}` at `{r['endpoint']}`.",
            "",
            "| | |",
            "|---|---|",
            f"| Corpus | {r['files']} files, {r['bytes'] / 1e6:.1f} MB |",
            f"| Sections | {r['sections']} |",
            f"| Index | {r['index_seconds']} s |",
            f"| Vectors | {r['vectors_bytes'] / 1e6:.1f} MB at dim {r['dim']}"
            f"{', exactly dim x 4 x sections' if r['vectors_exact'] else ', NOT the flat-matrix size'} |",
            f"| Query | {r['query_median_ms']} ms median, {r['query_max_ms']} ms worst, "
            f"over {r['queries']} queries at top {r['top_k']} |",
            f"| Re-index after one changed file | {r['incremental_seconds']} s, "
            f"{r['incremental_reembedded']} file re-embedded |",
        ]
        if "contamination_pct" in r:
            lines += [
                f"| Deprecated sections | {r['deprecated_sections']} "
                f"({r['deprecated_share_of_corpus']}% of the corpus) |",
                f"| Deprecated share of hits | **{r['contamination_pct']}%** |",
                f"| Deprecated share of hits, `--where status!=deprecated` | "
                f"**{r['contamination_filtered_pct']}%** |",
            ]
        lines.append("")
    (out / "latest.md").write_text("\n".join(lines))
    print(f"\nwrote {out / 'latest.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpora", nargs="*", choices=list(CORPORA), default=None)
    parser.add_argument("--model", help="passed to `folio index`, and reported")
    parser.add_argument("--keep", action="store_true", help="reuse a fetched corpus")
    args = parser.parse_args()

    if shutil.which("folio") is None:
        die("folio is not on PATH; `cargo install --path ..` first")

    chosen = args.corpora or list(CORPORA)
    write_report([measure(n, CORPORA[n], args.model, args.keep) for n in chosen])


if __name__ == "__main__":
    main()
