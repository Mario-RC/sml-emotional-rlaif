import argparse
import json
import os
import re

EMO_RE = re.compile(r"\(.*?\)")
DEFAULT_MODELS = [
    "gemma-2-2b-it",
    "Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct",
]
SFT_DATA_PATH = "./data/sft_demonstration_dataset_test.json"
DPR_DATA_PATH = "./data/ppo_unlabeled_prompts_dataset_test.json"
RLAIF_RUNS = ("dpo_1ep", "ppo_1ep", "sft_dpr_3ep")
GEMMA_OPTIONAL_RLAIF_RUNS = ("dpo_3ep",)


def _parse_args():
    parser = argparse.ArgumentParser(description="Aggregate emotional evaluation results for SML RLAIF models.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--verbose", action="store_true", help="Print labelled metrics instead of bare values.")
    parser.add_argument(
        "--regenerate-from-predictions",
        action="store_true",
        help="Accepted for CLI compatibility; this script already rebuilds results from current prediction files.",
    )
    return parser.parse_args()


def pct(count, total):
    return round(count / total * 100, 2) if total else 0.0


def build_metric_row(model, dataset, prediction_model, total, count_user_emo, count_chatbot_emo, count_neutral_emo, results_file):
    return {
        "model": model,
        "dataset": dataset,
        "prediction_model": prediction_model,
        "total_examples": total,
        "results_file": results_file,
        "user_emotion": {
            "correct": count_user_emo,
            "percentage": pct(count_user_emo, total),
        },
        "chatbot_emotion": {
            "correct": count_chatbot_emo,
            "percentage": pct(count_chatbot_emo, total),
        },
        "neutral_emotion": {
            "correct": count_neutral_emo,
            "percentage": pct(count_neutral_emo, total),
        },
    }


def _read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _extract_three(text):
    found = EMO_RE.findall(text)
    found += [""] * (3 - len(found))
    return found[:3]


def _score_predictions(targets, predictions):
    count_user_emo, count_chatbot_emo, count_neutral_emo = 0, 0, 0

    for target, prediction in zip(targets, predictions):
        target_user_emo, target_chatbot_emo, _ = _extract_three(target)
        predict_user_emo, predict_chatbot_emo, predict_neutral_emo = _extract_three(prediction)

        if predict_user_emo == target_user_emo:
            count_user_emo += 1
        if predict_chatbot_emo == target_chatbot_emo:
            count_chatbot_emo += 1
        if predict_neutral_emo == "(NEUTRAL)":
            count_neutral_emo += 1

    return count_user_emo, count_chatbot_emo, count_neutral_emo


def _print_chatbot_metric(run_name, chatbot_hits, total, only_values):
    if only_values:
        print(f"{pct(chatbot_hits, total):0.2f}%")
    else:
        print("\nPREDICTION MODEL " + run_name)
        print(f"CHATBOT EMOTION: {pct(chatbot_hits, total):0.2f}%")


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.loads(f.read())


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _rlaif_runs_for_model(model):
    runs = [(run_name, False) for run_name in RLAIF_RUNS]
    if "gemma" in model.lower():
        runs.extend((run_name, True) for run_name in GEMMA_OPTIONAL_RLAIF_RUNS)
    return runs


def _prepare_sft_records(model):
    prediction_path = f"./saves/{model}/predict/sft_d_3ep/generated_predictions.jsonl"
    if not os.path.exists(prediction_path):
        print("\n[WARN] Missing SFT predictions for " + model + ": " + prediction_path)
        return None, None

    results_sft_d_3ep = _read_jsonl(prediction_path)
    d_data = _load_json(SFT_DATA_PATH)

    for idx, result_sft_d_3ep in enumerate(results_sft_d_3ep[:len(d_data)]):
        d_data[idx].pop("input", None)
        d_data[idx]["prompt"] = d_data[idx].pop("instruction")
        d_data[idx]["instruction"] = d_data[idx].pop("system")
        d_data[idx]["history"] = d_data[idx].pop("history")
        d_data[idx]["prompt"] = d_data[idx].pop("prompt")
        d_data[idx]["target"] = d_data[idx].pop("output")
        d_data[idx]["predict_sft_d_3ep"] = result_sft_d_3ep["predict"].replace("\n", "").strip()
        d_data[idx]["model"] = model
        d_data[idx]["did"] = d_data[idx].pop("did", d_data[idx].pop("dialogue_id", str(idx)))

    results_file = f"./saves/{model}/emotional_balanced/sft_demonstration_dataset_test_results.json"
    _write_json(results_file, d_data)
    return d_data, results_file


def _prepare_rlaif_records(model):
    run_results = {}
    for run_name, optional in _rlaif_runs_for_model(model):
        prediction_path = f"./saves/{model}/predict/{run_name}/generated_predictions.jsonl"
        if not os.path.exists(prediction_path):
            if not optional:
                print("\n[WARN] Missing prediction file for " + model + " " + run_name + ": " + prediction_path)
            continue
        run_results[run_name] = _read_jsonl(prediction_path)

    if not run_results:
        return None, None, ()

    dpr_data = _load_json(DPR_DATA_PATH)
    for idx, entry in enumerate(dpr_data):
        entry.pop("input", None)
        entry["prompt"] = entry.pop("instruction")
        entry["instruction"] = entry.pop("system")
        entry["history"] = entry.pop("history")
        entry["prompt"] = entry.pop("prompt")
        entry["target"] = entry.pop("output")
        for run_name, results in run_results.items():
            if idx < len(results):
                entry["predict_" + run_name] = results[idx]["predict"].replace("\n", "").strip()
        entry["model"] = model
        entry["did"] = entry.pop("did", entry.pop("dialogue_id", str(idx)))

    results_file = f"./saves/{model}/emotional_balanced/ppo_unlabeled_prompts_dataset_test_results.json"
    _write_json(results_file, dpr_data)
    return dpr_data, results_file, tuple(run_results.keys())


def run(models, only_values=True):
    for model in models:
        print("\n---------------------------------\nMODEL: " + model + "\n---------------------------------")
        emotional_summary = []

        sft_records, sft_results_file = _prepare_sft_records(model)
        if sft_records is None:
            continue

        sft_targets = [entry["target"] for entry in sft_records]
        sft_predictions = [entry["predict_sft_d_3ep"] for entry in sft_records]
        count_user_emo, count_chatbot_emo, count_neutral_emo = _score_predictions(sft_targets, sft_predictions)
        _print_chatbot_metric("sft_d_3ep", count_chatbot_emo, len(sft_targets), only_values)
        emotional_summary.append(
            build_metric_row(
                model,
                "sft_demonstration_dataset_test",
                "sft_d_3ep",
                len(sft_targets),
                count_user_emo,
                count_chatbot_emo,
                count_neutral_emo,
                sft_results_file,
            )
        )

        dpr_records, dpr_results_file, available_runs = _prepare_rlaif_records(model)
        if dpr_records is not None:
            for run_name in available_runs:
                targets = []
                predictions = []
                for entry in dpr_records:
                    predict_key = "predict_" + run_name
                    if predict_key not in entry:
                        continue
                    targets.append(entry["target"])
                    predictions.append(entry[predict_key])

                count_user_emo, count_chatbot_emo, count_neutral_emo = _score_predictions(targets, predictions)
                _print_chatbot_metric(run_name, count_chatbot_emo, len(targets), only_values)
                emotional_summary.append(
                    build_metric_row(
                        model,
                        "ppo_unlabeled_prompts_dataset_test",
                        run_name,
                        len(targets),
                        count_user_emo,
                        count_chatbot_emo,
                        count_neutral_emo,
                        dpr_results_file,
                    )
                )

        summary_file = f"./saves/{model}/emotional_balanced/emotional_results_summary.json"
        _write_json(summary_file, {"model": model, "metrics": emotional_summary})
        if not only_values:
            print("\nSaved emotional summary: " + summary_file)


if __name__ == "__main__":
    args = _parse_args()
    run(models=args.models, only_values=not args.verbose)
