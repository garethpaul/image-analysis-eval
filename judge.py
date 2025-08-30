import os, json, time, argparse
from openai import OpenAI

def load_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def save_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_existing_ids(path):
    ids = set()
    if not os.path.exists(path):
        return ids
    try:
        for row in load_jsonl(path):
            row_id = row.get("id") or row.get("example_id") or row.get("exampleId")
            if row_id is not None:
                ids.add(row_id)
    except Exception:
        # If file is partially written/corrupted, best-effort resume
        pass
    return ids

def count_lines(path):
    count = 0
    with open(path, "r") as f:
        for _ in f:
            count += 1
    return count

def get_tqdm_progress(total):
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm(total=total)
    except Exception:
        return None

JUDGE_SYSTEM = (
    "You are an evaluator. Compare the model's response to the rubric and decide correctness.\n"
    "Return a strict JSON object: {\"score\": 0 or 1, \"explanation\": \"...\"}.\n"
    "Score 1 if the response satisfies the rubric; otherwise 0. Keep the explanation concise."
)

def build_user_prompt(example, response_text):
    prompt = example.get("prompt", "")
    rubric = example.get("rubric") or example.get("reference") or ""
    category = example.get("category", "")
    return (
        f"Category: {category}\n"
        f"Prompt: {prompt}\n\n"
        f"Rubric/Reference (ground truth/requirements): {rubric}\n\n"
        f"Model response:\n{response_text}\n\n"
        "Decide 0/1 and explain briefly."
    )

def parse_json_reply(text):
    try:
        return json.loads(text)
    except Exception:
        # Try to extract JSON object heuristically
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                pass
        raise

def judge_with_poe(client, judge_model, example, model_response, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=judge_model,
                temperature=0,
                messages=[
                    {"role":"system", "content": JUDGE_SYSTEM},
                    {"role":"user", "content": build_user_prompt(example, model_response)},
                ],
            )
            content = resp.choices[0].message.content
            data = parse_json_reply(content)
            score_raw = data.get("score", 0)
            try:
                score = float(score_raw)
            except Exception:
                score = 1.0 if str(score_raw).strip() in {"1", "true", "True"} else 0.0
            score = 1.0 if score >= 0.5 else 0.0
            explanation = str(data.get("explanation", "")).strip()
            return score, explanation
        except Exception as e:
            if attempt == max_retries - 1:
                return 0.0, f"Judging failed: {e}"
            time.sleep(1.5 * (attempt + 1))
    return 0.0, "Judging failed."

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--generations_in", required=True)
    ap.add_argument("--generations_out", required=True)
    ap.add_argument("--judge_model", default="GPT-5")
    ap.add_argument("--poe_api_key", default=os.getenv("POE_API_KEY"))
    ap.add_argument("--progress", action="store_true", help="Show progress updates while judging")
    ap.add_argument("--progress_every", type=int, default=10, help="If tqdm is unavailable, print every N examples")
    ap.add_argument("--append", action="store_true", help="Append to generations_out and resume (skip already judged ids)")
    ap.add_argument("--stream_write", action="store_true", help="Write each result line immediately instead of batching")
    args = ap.parse_args()

    if not args.poe_api_key:
        raise SystemExit("Set POE_API_KEY env var or pass --poe_api_key")

    client = OpenAI(api_key=args.poe_api_key, base_url="https://api.poe.com/v1")

    # Index dataset by id/example_id for rubric/category/etc.
    data_by_id = {}
    for ex in load_jsonl(args.dataset):
        ex_id = ex.get("id") or ex.get("example_id") or ex.get("exampleId")
        if not ex_id:
            continue
        data_by_id[ex_id] = ex

    total_examples = count_lines(args.generations_in)
    use_progress = bool(args.progress)
    progress_bar = get_tqdm_progress(total_examples) if use_progress else None
    printed = 0
    start_time = time.time()

    existing_ids = load_existing_ids(args.generations_out) if args.append else set()
    if use_progress and existing_ids:
        print(f"Resuming: found {len(existing_ids)} existing judged rows in output")

    out_file = None
    if args.append:
        out_file = open(args.generations_out, "a")
    elif args.stream_write:
        out_file = open(args.generations_out, "w")

    out_rows = []
    processed = 0
    skipped = 0
    for row in load_jsonl(args.generations_in):
        row_id = row.get("id") or row.get("example_id") or row.get("exampleId")
        if row_id in existing_ids:
            skipped += 1
            processed += 1
            if progress_bar is not None:
                progress_bar.update(1)
            elif use_progress and (processed == 1 or processed % max(1, args.progress_every) == 0 or processed == total_examples):
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0.0
                print(f"Judging progress: {processed}/{total_examples} ({processed*100//max(1,total_examples)}%), ~{rate:.2f} ex/s (skipped {skipped})")
            continue
        ex = data_by_id.get(row_id, {})
        response_text = row.get("response") or row.get("generation") or ""
        score, explanation = judge_with_poe(client, args.judge_model, ex, response_text)
        row["score"] = score
        row["explanation"] = explanation
        # Optional: also include image URL from dataset if not present on the row
        if "image" not in row:
            if "media_url" in ex:
                row["image"] = ex["media_url"]
            elif "image" in ex:
                row["image"] = ex["image"]
        if out_file is not None:
            out_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            if not progress_bar and use_progress and (processed % max(1, args.progress_every) == 0):
                out_file.flush()
        else:
            out_rows.append(row)
        processed += 1
        if progress_bar is not None:
            progress_bar.update(1)
        elif use_progress and (processed == 1 or processed % max(1, args.progress_every) == 0 or processed == total_examples):
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0.0
            print(f"Judging progress: {processed}/{total_examples} ({processed*100//max(1,total_examples)}%), ~{rate:.2f} ex/s (skipped {skipped})")

    if out_file is not None:
        out_file.flush()
        out_file.close()
    else:
        save_jsonl(args.generations_out, out_rows)
    if progress_bar is not None:
        progress_bar.close()
    if use_progress:
        elapsed = time.time() - start_time
        total_written = processed - skipped if out_rows is None else len(out_rows)
        rate = (processed) / elapsed if elapsed > 0 else 0.0
        print(f"Done. Processed {processed} (judged {processed - skipped}, skipped {skipped}) in {elapsed:.1f}s (~{rate:.2f} ex/s)")

if __name__ == "__main__":
    main()