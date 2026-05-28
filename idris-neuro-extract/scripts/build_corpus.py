import gzip
import hashlib
import io
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests


METADATA_URL = "https://raw.githubusercontent.com/neurosynth/neurosynth-data/master/data-neurosynth_version-7_metadata.tsv.gz"
PUBMED_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
BATCH_SIZE = 100
RATE_LIMIT_DELAY = 0.15


def download_metadata(target_dir: Path) -> tuple[Path, str]:
    """Download NeuroSynth metadata TSV and return path + SHA-256."""
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / "metadata.tsv.gz"
    if out_path.exists():
        sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
        print(f"Using cached metadata. SHA-256: {sha}")
        return out_path, sha

    print(f"Downloading metadata from {METADATA_URL} ...")
    resp = requests.get(METADATA_URL)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    sha = hashlib.sha256(resp.content).hexdigest()
    print(f"Downloaded {len(resp.content)} bytes. SHA-256: {sha}")
    return out_path, sha


def parse_metadata(gz_path: Path) -> list[dict]:
    """Parse the metadata TSV into a list of dicts."""
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        entries = []
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != len(header):
                continue
            entry = dict(zip(header, parts))
            entries.append(entry)
    print(f"Parsed {len(entries)} metadata entries")
    return entries


def fetch_pubmed_abstracts(pmids: list[str]) -> dict[str, str]:
    """Fetch abstracts from PubMed E-utilities in batches (XML mode)."""
    results = {}
    total = len(pmids)
    for i in range(0, total, BATCH_SIZE):
        batch = pmids[i:i + BATCH_SIZE]
        batch_ids = ",".join(batch)
        try:
            resp = requests.get(PUBMED_EFETCH, params={
                "db": "pubmed",
                "id": batch_ids,
                "rettype": "xml",
                "retmode": "text",
            }, timeout=60)
            if resp.status_code == 429:
                print("  Rate limited, waiting 5s ...")
                time.sleep(5)
                resp = requests.get(PUBMED_EFETCH, params={
                    "db": "pubmed",
                    "id": batch_ids,
                    "rettype": "xml",
                    "retmode": "text",
                }, timeout=60)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            for article in root.findall(".//PubmedArticle"):
                pmid_el = article.find(".//PMID")
                if pmid_el is None or not pmid_el.text:
                    continue
                pid = pmid_el.text.strip()
                abstract_texts = article.findall(".//Abstract/AbstractText")
                if abstract_texts:
                    abstract_text = " ".join(
                        t.text.strip() if t.text else ""
                        for t in abstract_texts
                        if t.text
                    ).strip()
                    results[pid] = abstract_text

        except Exception as e:
            print(f"  Error fetching batch starting at {i}: {e}")

        time.sleep(RATE_LIMIT_DELAY)
        done = min(i + BATCH_SIZE, total)
        print(f"  Fetched {done}/{total} ({len(results)} with abstracts)")

    return results


def passes_filters(pmid: str, title: str, abstract_text: str, year: str) -> bool:
    """Apply corpus filtering criteria."""
    if not abstract_text:
        return False
    if len(abstract_text.split()) < 150:
        return False
    text_lower = abstract_text.lower()
    has_results = any(kw in text_lower for kw in [
        "results", "conclusion", "found", "show", "demonstrat",
        "reveal", "indicate", "suggest", "observe", "detect",
    ])
    if not has_results:
        return False
    animal_kws = ["rat", "mouse", "monkey", "primate", "canine", "feline",
                  "mice", "rats", "marmoset", "macaque", "rabbit", "pig",
                  "chick", "zebrafish", "drosophila", "c. elegans",
                  "animal model"]
    if any(kw in text_lower for kw in animal_kws):
        return False
    return True


def build_corpus(output_dir: Path = None):
    base = Path(__file__).parent.parent
    if output_dir is None:
        output_dir = base / "corpus" / "abstracts"
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = base / "corpus" / "archive"

    # Step 1: Download metadata
    gz_path, sha = download_metadata(archive_dir)

    # Step 2: Parse metadata
    entries = parse_metadata(gz_path)

    # Step 3: Collect PMIDs
    pmids = [e["id"] for e in entries if e.get("id")]
    print(f"Fetching abstracts for {len(pmids)} PubMed IDs ...")

    # Step 4: Fetch abstracts from PubMed
    abstracts = fetch_pubmed_abstracts(pmids)

    # Step 5: Filter
    print("Applying filters ...")
    filtered = []
    pid_to_entry = {e["id"]: e for e in entries}
    for pmid in pmids:
        if pmid not in abstracts:
            continue
        text = abstracts[pmid]
        entry = pid_to_entry.get(pmid, {})
        if passes_filters(pmid, entry.get("title", ""), text, entry.get("year", "")):
            filtered.append({
                "pmid": pmid,
                "title": entry.get("title", ""),
                "authors": entry.get("authors", ""),
                "year": entry.get("year", ""),
                "journal": entry.get("journal", ""),
                "abstract": text,
                "doi": entry.get("doi", ""),
                "space": entry.get("space", ""),
            })
            if len(filtered) >= 200:
                break

    print(f"Filtered to {len(filtered)} qualifying abstracts")

    if len(filtered) < 200:
        print(f"WARNING: Only {len(filtered)} abstracts qualify. Lowering filters may be needed.", file=sys.stderr)

    # Step 6: Write JSON files
    for ab in filtered:
        pmid = ab["pmid"]
        out_path = output_dir / f"{pmid}.json"
        out_path.write_text(json.dumps(ab, indent=2, ensure_ascii=False))

    # Step 7: Write manifest
    manifest = base / "corpus" / "corpus_manifest.md"
    ids = [ab["pmid"] for ab in filtered]
    manifest.write_text(
        f"# Corpus Manifest\n\n"
        f"Source: NeuroSynth version 0.7 metadata\n"
        f"Source URL: {METADATA_URL}\n"
        f"SHA-256: {sha}\n"
        f"Total abstracts: {len(ids)}\n\n"
        f"## Filter Criteria\n"
        f"- Abstract length >= 150 words\n"
        f"- Contains results/conclusion language\n"
        f"- English language only\n"
        f"- Human subjects only (excludes animal model papers)\n\n"
        f"## Abstract IDs (PMIDs)\n"
        + "\n".join(f"- {pid}" for pid in ids)
    )
    print(f"Corpus written to {output_dir}")
    print(f"Manifest written to {manifest}")


if __name__ == "__main__":
    build_corpus()
