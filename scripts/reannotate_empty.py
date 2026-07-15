import json
import os
import time
from pathlib import Path

import openai


def get_client():
    base_url = os.environ.get("ANNOTATION_API_BASE",
                   os.environ.get("OPENAI_API_BASE",
                   os.environ.get("LLAMA_CPP_BASE_URL", "")))
    api_key = os.environ.get("ANNOTATION_API_KEY",
                   os.environ.get("OPENAI_API_KEY",
                   os.environ.get("LLAMA_CPP_API_KEY", "no-key")))
    model = os.environ.get("ANNOTATION_MODEL",
                   os.environ.get("LLAMA_CPP_MODEL_ID", "qwen3-27b-mtp-6bit"))

    if base_url:
        client = openai.OpenAI(base_url=base_url, api_key=api_key)
    else:
        client = openai.OpenAI(api_key=api_key)
    return client, model


def load_prompt(base_dir: Path) -> str:
    prompt_path = base_dir / "corpus" / "annotation_prompt.md"
    return prompt_path.read_text()


def annotate_abstract(client, model: str, prompt: str, abstract_id: str, abstract_text: str) -> list[str]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Abstract ID: {abstract_id}\n\n{abstract_text}"},
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    data = json.loads(content)
    return data.get("claims", [])


def reannotate_empty(gt_path: Path = None, abstracts_dir: Path = None):
    base = Path(__file__).parent.parent
    if gt_path is None:
        gt_path = base / "corpus" / "ground_truth.jsonl"
    if abstracts_dir is None:
        abstracts_dir = base / "corpus" / "abstracts"

    prompt = load_prompt(base)
    client, model = get_client()

    existing = {}
    for line in gt_path.read_text(encoding="utf-8-sig").strip().splitlines():
        entry = json.loads(line)
        existing[entry["abstract_id"]] = entry

    empty_ids = [aid for aid, entry in existing.items() if not entry.get("claims")]
    print(f"Found {len(empty_ids)} entries with empty claims: {empty_ids}")

    updated = {}
    for aid in empty_ids:
        af = abstracts_dir / f"{aid}.json"
        if not af.exists():
            print(f"  WARNING: Abstract file missing for {aid}, skipping")
            continue

        abstract_data = json.loads(af.read_text(encoding="utf-8"))
        abstract_text = abstract_data.get("abstract", abstract_data.get("text", ""))
        print(f"Re-annotating {aid} ...")
        claims = []
        for attempt in range(3):
            try:
                claims = annotate_abstract(client, model, prompt, aid, abstract_text)
                break
            except json.JSONDecodeError:
                if attempt < 2:
                    print(f"  Retry {attempt+1}/3...")
                    time.sleep(2)
                else:
                    print(f"  Failed after 3 retries")
            except Exception as e:
                print(f"  Error: {e}")
                break
        updated[aid] = claims
        print(f"  Got {len(claims)} claims")

    all_entries = {}
    for aid, entry in existing.items():
        if aid in updated:
            all_entries[aid] = {"abstract_id": aid, "claims": updated[aid]}
        else:
            all_entries[aid] = entry

    gt_path.write_text(
        "\n".join(json.dumps(all_entries[aid]) for aid in sorted(all_entries.keys())) + "\n",
        encoding="utf-8"
    )

    filled = sum(1 for claims in updated.values() if claims)
    print(f"Re-annotation complete. Filled {filled}/{len(empty_ids)} entries.")
    print(f"Ground truth written to {gt_path}")


if __name__ == "__main__":
    reannotate_empty()
