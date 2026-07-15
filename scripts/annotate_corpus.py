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


def annotate_corpus(output_path: Path = None):
    base = Path(__file__).parent.parent
    abstracts_dir = base / "corpus" / "abstracts"
    if output_path is None:
        output_path = base / "corpus" / "ground_truth.jsonl"

    prompt = load_prompt(base)
    client, model = get_client()

    lines = []
    for af in sorted(abstracts_dir.glob("*.json")):
        abstract_id = af.stem
        abstract_data = json.loads(af.read_text())
        abstract_text = abstract_data.get("abstract", abstract_data.get("text", ""))
        print(f"Annotating {abstract_id} ...")
        claims = []
        for attempt in range(3):
            try:
                claims = annotate_abstract(client, model, prompt, abstract_id, abstract_text)
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
        entry = {"abstract_id": abstract_id, "claims": claims}
        lines.append(json.dumps(entry))

    output_path.write_text("\n".join(lines) + "\n")
    print(f"Ground truth written to {output_path} ({len(lines)} entries)")


if __name__ == "__main__":
    annotate_corpus()
