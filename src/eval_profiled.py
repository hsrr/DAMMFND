import argparse
import json
import os
import random
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import seaborn as sns
except Exception:
    sns = None

try:
    from fvcore.nn import FlopCountAnalysis
except Exception:
    FlopCountAnalysis = None

try:
    from thop import profile as thop_profile
except Exception:
    thop_profile = None

from model.dammfnd import DAMMFNDMODEL
from utils.clip_dataloader import bert_data as weibo_data
from utils.sixclass_dataloader import bert_data as sixclass_data
from utils.weibo21_clip_dataloader import bert_data as weibo21_data


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

DATASET_META = {
    "sixclass": {
        "num_classes": 6,
        "class_names": [
            "real_news",
            "image_forgery",
            "entity_inconsist",
            "event_inconsist",
            "time_inconsist",
            "invalid_visual",
        ],
        "category_dict": {
            "real_news": 0,
            "image_forgery": 1,
            "entity_inconsist": 2,
            "event_inconsist": 3,
            "time_inconsist": 4,
            "invalid_visual": 5,
        },
    },
    "weibo": {
        "num_classes": 2,
        "class_names": ["real", "fake"],
        "category_dict": {
            "经济": 0,
            "健康": 1,
            "军事": 2,
            "科学": 3,
            "政治": 4,
            "国际": 5,
            "教育": 6,
            "娱乐": 7,
            "社会": 8,
        },
    },
    "weibo21": {
        "num_classes": 2,
        "class_names": ["real", "fake"],
        "category_dict": {
            "科技": 0,
            "军事": 1,
            "教育考试": 2,
            "灾难事故": 3,
            "政治": 4,
            "医药健康": 5,
            "财经商业": 6,
            "文体娱乐": 7,
            "社会生活": 8,
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile DAMMFND inference latency and classification metrics."
    )
    parser.add_argument("--dataset", choices=["sixclass", "weibo", "weibo21"], default="sixclass")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pkl/.pth).")
    parser.add_argument(
        "--bert",
        default="./pretrained_model/chinese_roberta_wwm_base_ext_pytorch",
        help="BERT model directory, interpreted relative to src/ after path normalization.",
    )
    parser.add_argument(
        "--bert-vocab-file",
        default="./pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt",
        help="Tokenizer vocab path, interpreted relative to src/ after path normalization.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=197)
    parser.add_argument("--emb-dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--mlp-dims", default="384", help="Comma-separated MLP dims, e.g. 384 or 384,192")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument(
        "--test-path",
        default=None,
        help="Optional explicit test annotation path. If unset, dataset default is used.",
    )
    parser.add_argument(
        "--data-root",
        default="/map-vepfs/liniuniu/hesirui/datasets",
        help="Root for sixclass jsonl files.",
    )
    parser.add_argument(
        "--image-root",
        default=None,
        help="Root for sixclass images. Defaults to --data-root.",
    )
    parser.add_argument(
        "--root-path",
        default=None,
        help="Root path for weibo/weibo21 annotations and cached tensors.",
    )
    parser.add_argument(
        "--confusion-matrix-dir",
        default="./profile_results",
        help="Directory to save confusion matrices and text dumps.",
    )
    parser.add_argument(
        "--results-txt-path",
        default="./profile_results/results_profiled.txt",
        help="Append human-readable evaluation summary to this file.",
    )
    parser.add_argument(
        "--profile-output-json",
        default=None,
        help="Optional JSON file to save structured metrics.",
    )
    parser.add_argument("--latency-warmup-iters", type=int, default=20)
    parser.add_argument("--latency-measure-iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3074)
    parser.add_argument("--report-weighted-f1", action="store_true")
    parser.add_argument("--disable-plots", action="store_true")
    return parser.parse_args()


def resolve_user_path(path: Optional[str], base_dir: str) -> Optional[str]:
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def resolve_src_relative_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(SCRIPT_DIR, path))


def parse_mlp_dims(text: str) -> List[int]:
    dims = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not dims:
        raise ValueError("--mlp-dims produced an empty list.")
    return dims


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sync_if_needed(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "params_total": int(total),
        "params_trainable": int(trainable),
    }


def format_big_number(x):
    if x is None:
        return "N/A"
    x = float(x)
    if x >= 1e12:
        return f"{x / 1e12:.4f} T"
    if x >= 1e9:
        return f"{x / 1e9:.4f} G"
    if x >= 1e6:
        return f"{x / 1e6:.4f} M"
    if x >= 1e3:
        return f"{x / 1e3:.4f} K"
    return f"{x:.4f}"


def build_dataset_paths(args):
    if args.dataset == "sixclass":
        data_root = args.data_root
        image_root = args.image_root or data_root
        test_path = args.test_path or os.path.join(data_root, "test.jsonl")
        return {
            "test_path": test_path,
            "image_root": image_root,
        }

    if args.dataset == "weibo":
        root_path = args.root_path or os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))
        test_path = args.test_path or os.path.join(root_path, "test_2_domain.csv")
        return {
            "test_path": test_path,
            "image_tensor_path": os.path.join(root_path, "test_loader.pkl"),
            "clip_image_tensor_path": os.path.join(root_path, "test_clip_loader.pkl"),
        }

    root_path = args.root_path or os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Weibo_21"))
    test_path = args.test_path or os.path.join(root_path, "test_2_domain.xlsx")
    return {
        "test_path": test_path,
        "image_tensor_path": os.path.join(root_path, "test_loader.pkl"),
        "clip_image_tensor_path": os.path.join(root_path, "test_clip_loader.pkl"),
    }


def get_test_loader(args):
    meta = DATASET_META[args.dataset]
    paths = build_dataset_paths(args)

    if args.dataset == "sixclass":
        loader = sixclass_data(
            max_len=args.max_len,
            batch_size=args.batch_size,
            vocab_file=args.bert_vocab_file,
            num_workers=args.num_workers,
        )
        test_loader = loader.load_data(paths["test_path"], paths["image_root"], False)
    elif args.dataset == "weibo":
        loader = weibo_data(
            max_len=args.max_len,
            batch_size=args.batch_size,
            vocab_file=args.bert_vocab_file,
            category_dict=meta["category_dict"],
            num_workers=args.num_workers,
        )
        test_loader = loader.load_data(
            paths["test_path"],
            paths["image_tensor_path"],
            paths["clip_image_tensor_path"],
            False,
        )
    else:
        loader = weibo21_data(
            max_len=args.max_len,
            batch_size=args.batch_size,
            vocab_file=args.bert_vocab_file,
            category_dict=meta["category_dict"],
            num_workers=args.num_workers,
        )
        test_loader = loader.load_data(
            paths["test_path"],
            paths["image_tensor_path"],
            paths["clip_image_tensor_path"],
            False,
        )

    return test_loader, paths


def batch_to_device(batch, device: torch.device) -> Dict[str, torch.Tensor]:
    batch_data = {
        "content": batch[0].to(device, non_blocking=True),
        "content_masks": batch[1].to(device, non_blocking=True),
        "label": batch[2].to(device, non_blocking=True),
        "category": batch[3].to(device, non_blocking=True),
        "image": batch[4].to(device, non_blocking=True),
        "clip_image": batch[5].to(device, non_blocking=True),
        "clip_text": batch[6].to(device, non_blocking=True),
    }
    if len(batch) > 7:
        batch_data["multi_category"] = batch[7].to(device, non_blocking=True)
    return batch_data


def slice_single_sample(batch_data: Dict[str, torch.Tensor], index: int = 0):
    single = {}
    for key, value in batch_data.items():
        if torch.is_tensor(value) and value.ndim > 0:
            single[key] = value[index:index + 1].contiguous()
        else:
            single[key] = value
    return single


class ForwardProfileWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, content, content_masks, image, clip_image, clip_text):
        final_logits, _, _, _, _, _, _ = self.model(
            content=content,
            content_masks=content_masks,
            image=image,
            clip_image=clip_image,
            clip_text=clip_text,
        )
        return final_logits


def estimate_compute_cost(model, single_batch, device):
    wrapper = ForwardProfileWrapper(model).to(device)
    wrapper.eval()
    profile_inputs = (
        single_batch["content"],
        single_batch["content_masks"],
        single_batch["image"],
        single_batch["clip_image"],
        single_batch["clip_text"],
    )

    result = {
        "compute_cost_backend": None,
        "forward_macs": None,
        "forward_flops_approx": None,
        "compute_cost_note": "single-sample forward only",
    }

    with torch.no_grad():
        try:
            sync_if_needed(device)
            _ = wrapper(*profile_inputs)
            sync_if_needed(device)
        except Exception as exc:
            result["compute_cost_note"] = f"warmup forward failed: {exc}"
            return result

    if FlopCountAnalysis is not None:
        try:
            analysis = FlopCountAnalysis(wrapper, profile_inputs)
            flops = float(analysis.total())
            result["compute_cost_backend"] = "fvcore"
            result["forward_flops_approx"] = flops
            result["forward_macs"] = flops / 2.0
            return result
        except Exception as exc:
            result["compute_cost_note"] = f"fvcore failed: {exc}"

    if thop_profile is not None:
        try:
            macs, params = thop_profile(wrapper, inputs=profile_inputs, verbose=False)
            result["compute_cost_backend"] = "thop"
            result["forward_macs"] = float(macs)
            result["forward_flops_approx"] = float(macs) * 2.0
            result["thop_params_from_wrapper"] = float(params)
            return result
        except Exception as exc:
            prev = result.get("compute_cost_note", "")
            result["compute_cost_note"] = (prev + f" | thop failed: {exc}").strip(" |")

    result["compute_cost_note"] = (
        result["compute_cost_note"] + " | No supported FLOPs backend available."
    ).strip(" |")
    return result


def summarize_latency(latencies_ms: List[float]):
    arr = np.asarray(latencies_ms, dtype=np.float64)
    if arr.size == 0:
        return {
            "mean": None,
            "std": None,
            "p50": None,
            "p90": None,
            "p95": None,
        }
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def measure_single_sample_latency(model, single_batch, device, warmup_iters=20, measure_iters=100):
    latencies_ms = []
    with torch.no_grad():
        for _ in range(max(0, int(warmup_iters))):
            sync_if_needed(device)
            _ = model(**single_batch)
            sync_if_needed(device)

        for _ in range(max(1, int(measure_iters))):
            sync_if_needed(device)
            t0 = time.perf_counter()
            _ = model(**single_batch)
            sync_if_needed(device)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    stats = summarize_latency(latencies_ms)
    out = {
        "single_forward_latency_ms_mean": stats["mean"],
        "single_forward_latency_ms_std": stats["std"],
        "single_forward_latency_ms_p50": stats["p50"],
        "single_forward_latency_ms_p90": stats["p90"],
        "single_forward_latency_ms_p95": stats["p95"],
        "single_forward_samples_per_s": None,
        "latency_measure_iters": int(measure_iters),
        "latency_warmup_iters": int(warmup_iters),
        "ttft_ms_mean": None,
        "tpot_ms_mean": None,
        "generation_note": "TTFT/TPOT are not applicable to classifier-style forward inference.",
    }
    if stats["mean"] is not None and stats["mean"] > 0:
        out["single_forward_samples_per_s"] = float(1000.0 / stats["mean"])
    return out


def maybe_plot_confusion_matrix(y_true, y_pred, class_names, title, save_path, disable_plots=False):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.savetxt(save_path.replace(".png", ".txt"), cm, fmt="%d")

    if disable_plots or plt is None or sns is None:
        print(f"Confusion matrix text saved: {save_path.replace('.png', '.txt')}")
        return

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title(title, fontsize=14)
    plt.xlabel("pred label", fontsize=12)
    plt.ylabel("true label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved: {save_path}")


def maybe_write_profile_json(profile_payload: dict, out_path: Optional[str]):
    if not out_path:
        return
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profile_payload, f, ensure_ascii=False, indent=2)
    print(f"Profile summary written: {out_path}")


def load_checkpoint(model: nn.Module, checkpoint_path: str):
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[load] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[load] Unexpected keys: {len(unexpected)}")
    if missing[:10]:
        print(f"[load] Missing sample: {missing[:10]}")
    if unexpected[:10]:
        print(f"[load] Unexpected sample: {unexpected[:10]}")


def build_model(args, device):
    num_classes = args.num_classes or DATASET_META[args.dataset]["num_classes"]
    model = DAMMFNDMODEL(
        emb_dim=args.emb_dim,
        mlp_dims=parse_mlp_dims(args.mlp_dims),
        bert=args.bert,
        out_channels=320,
        dropout=args.dropout,
        num_classes=num_classes,
    )
    load_checkpoint(model, args.checkpoint)
    model = model.to(device)
    model.eval()
    return model, num_classes


def main():
    args = parse_args()

    launch_cwd = os.getcwd()
    args.checkpoint = resolve_user_path(args.checkpoint, launch_cwd)
    args.bert = resolve_src_relative_path(args.bert)
    args.bert_vocab_file = resolve_src_relative_path(args.bert_vocab_file)
    args.test_path = resolve_user_path(args.test_path, launch_cwd)
    args.data_root = resolve_user_path(args.data_root, launch_cwd)
    args.image_root = resolve_user_path(args.image_root, launch_cwd)
    args.root_path = resolve_user_path(args.root_path, launch_cwd)
    args.confusion_matrix_dir = resolve_user_path(args.confusion_matrix_dir, launch_cwd)
    args.results_txt_path = resolve_user_path(args.results_txt_path, launch_cwd)
    args.profile_output_json = resolve_user_path(args.profile_output_json, launch_cwd)

    os.chdir(SCRIPT_DIR)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "DAMMFNDMODEL currently assumes CUDA-only components inside model initialization; "
            "please run this profiling script on a CUDA machine."
        )

    setup_seed(args.seed)
    device = torch.device("cuda")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {args.checkpoint}")

    print("Loading test loader...")
    test_loader, dataset_paths = get_test_loader(args)

    print("Loading model...")
    model, num_classes = build_model(args, device)

    profile_summary = {
        "dataset": args.dataset,
        "device": str(device),
        "checkpoint": args.checkpoint,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "max_len": int(args.max_len),
        "test_path": dataset_paths["test_path"],
        "num_classes": int(num_classes),
    }
    profile_summary.update(count_parameters(model))

    try:
        profile_batch_raw = next(iter(test_loader))
    except StopIteration:
        raise RuntimeError("Test loader is empty.")

    profile_batch = batch_to_device(profile_batch_raw, device)
    single_batch = slice_single_sample(profile_batch, index=0)
    profile_summary["profile_text_token_count"] = int(single_batch["content"].shape[-1])
    profile_summary["profile_batch_size_for_loader"] = int(profile_batch["content"].shape[0])

    compute_cost = estimate_compute_cost(model, single_batch, device)
    profile_summary.update(compute_cost)

    print("\n[Profile] Computation cost")
    print(f"  Params (total):      {format_big_number(profile_summary['params_total'])}")
    print(f"  Params (trainable):  {format_big_number(profile_summary['params_trainable'])}")
    print(f"  Forward MACs:        {format_big_number(profile_summary.get('forward_macs'))}")
    print(f"  Forward FLOPs~:      {format_big_number(profile_summary.get('forward_flops_approx'))}")
    print(f"  Backend:             {profile_summary.get('compute_cost_backend')}")
    print(f"  Note:                {profile_summary.get('compute_cost_note')}")

    latency_stats = measure_single_sample_latency(
        model,
        single_batch,
        device,
        warmup_iters=args.latency_warmup_iters,
        measure_iters=args.latency_measure_iters,
    )
    profile_summary.update(latency_stats)
    print("\n[Profile] Single-sample forward latency")
    print(f"  Mean latency:        {latency_stats['single_forward_latency_ms_mean']:.4f} ms")
    print(f"  Std latency:         {latency_stats['single_forward_latency_ms_std']:.4f} ms")
    print(
        f"  P50 / P90 / P95:     {latency_stats['single_forward_latency_ms_p50']:.4f} / "
        f"{latency_stats['single_forward_latency_ms_p90']:.4f} / "
        f"{latency_stats['single_forward_latency_ms_p95']:.4f} ms"
    )
    if latency_stats.get("single_forward_samples_per_s") is not None:
        print(f"  Samples / second:    {latency_stats['single_forward_samples_per_s']:.4f}")
    print("  TTFT / TPOT:         N/A for non-generative classifier forward")

    y_true, y_pred = [], []
    forward_batch_latencies_ms = []
    e2e_batch_latencies_ms = []
    observed_batch_sizes = []
    total_eval_samples = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    print("Running evaluation...")
    for batch in test_loader:
        batch_t0 = time.perf_counter()
        batch_data = batch_to_device(batch, device)
        labels = batch_data["label"]
        observed_batch_sizes.append(int(labels.shape[0]))
        total_eval_samples += int(labels.shape[0])

        with torch.no_grad():
            sync_if_needed(device)
            fwd_t0 = time.perf_counter()
            final_logits, _, _, _, _, _, _ = model(**batch_data)
            sync_if_needed(device)
            fwd_t1 = time.perf_counter()

        preds = torch.argmax(final_logits, dim=-1)
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())

        batch_t1 = time.perf_counter()
        forward_batch_latencies_ms.append((fwd_t1 - fwd_t0) * 1000.0)
        e2e_batch_latencies_ms.append((batch_t1 - batch_t0) * 1000.0)

    print("Computing metrics...")
    class_names = DATASET_META[args.dataset]["class_names"][:num_classes]
    labels_range = list(range(num_classes))

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = None
    if args.report_weighted_f1:
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    if weighted_f1 is not None:
        print(f"Weighted F1: {weighted_f1:.4f}")

    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels_range,
        target_names=class_names,
        zero_division=0,
        digits=4,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels_range,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    print("\n--- F1 Per Class (FPC) ---")
    print(report_text)

    profile_summary["accuracy"] = float(acc)
    profile_summary["macro_f1"] = float(macro_f1)
    profile_summary["macro_precision"] = float(macro_precision)
    profile_summary["macro_recall"] = float(macro_recall)
    profile_summary["weighted_f1"] = float(weighted_f1) if weighted_f1 is not None else None
    profile_summary["classification_report"] = report_dict
    profile_summary["total_eval_samples"] = int(total_eval_samples)

    batch_latency_stats = summarize_latency(forward_batch_latencies_ms)
    e2e_latency_stats = summarize_latency(e2e_batch_latencies_ms)
    profile_summary["eval_forward_batch_latency_ms_mean"] = batch_latency_stats["mean"]
    profile_summary["eval_forward_batch_latency_ms_p50"] = batch_latency_stats["p50"]
    profile_summary["eval_forward_batch_latency_ms_p90"] = batch_latency_stats["p90"]
    profile_summary["eval_forward_batch_latency_ms_p95"] = batch_latency_stats["p95"]
    profile_summary["eval_e2e_batch_latency_ms_mean"] = e2e_latency_stats["mean"]
    profile_summary["eval_e2e_batch_latency_ms_p50"] = e2e_latency_stats["p50"]
    profile_summary["eval_e2e_batch_latency_ms_p90"] = e2e_latency_stats["p90"]
    profile_summary["eval_e2e_batch_latency_ms_p95"] = e2e_latency_stats["p95"]

    mean_batch_size = float(np.mean(observed_batch_sizes)) if observed_batch_sizes else None
    if batch_latency_stats["mean"] is not None and batch_latency_stats["mean"] > 0 and mean_batch_size is not None:
        profile_summary["eval_forward_sample_latency_ms_mean"] = float(
            batch_latency_stats["mean"] / max(1.0, mean_batch_size)
        )
        profile_summary["eval_forward_samples_per_s"] = float(
            mean_batch_size / (batch_latency_stats["mean"] / 1000.0)
        )
    else:
        profile_summary["eval_forward_sample_latency_ms_mean"] = None
        profile_summary["eval_forward_samples_per_s"] = None

    print("\n[Profile] Full evaluation forward latency")
    print(f"  Mean batch latency:  {profile_summary['eval_forward_batch_latency_ms_mean']:.4f} ms")
    print(
        f"  P50 / P90 / P95:     {profile_summary['eval_forward_batch_latency_ms_p50']:.4f} / "
        f"{profile_summary['eval_forward_batch_latency_ms_p90']:.4f} / "
        f"{profile_summary['eval_forward_batch_latency_ms_p95']:.4f} ms"
    )
    if profile_summary["eval_forward_sample_latency_ms_mean"] is not None:
        print(f"  Mean/sample latency: {profile_summary['eval_forward_sample_latency_ms_mean']:.4f} ms")
    if profile_summary["eval_forward_samples_per_s"] is not None:
        print(f"  Samples / second:    {profile_summary['eval_forward_samples_per_s']:.4f}")

    print("\n[Profile] Full evaluation end-to-end batch latency")
    print(f"  Mean batch latency:  {profile_summary['eval_e2e_batch_latency_ms_mean']:.4f} ms")
    print(
        f"  P50 / P90 / P95:     {profile_summary['eval_e2e_batch_latency_ms_p50']:.4f} / "
        f"{profile_summary['eval_e2e_batch_latency_ms_p90']:.4f} / "
        f"{profile_summary['eval_e2e_batch_latency_ms_p95']:.4f} ms"
    )

    if device.type == "cuda":
        profile_summary["cuda_max_memory_allocated_mb"] = float(
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        )
        profile_summary["cuda_max_memory_reserved_mb"] = float(
            torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        )
        print("\n[Profile] CUDA memory peak")
        print(f"  Max allocated:       {profile_summary['cuda_max_memory_allocated_mb']:.4f} MB")
        print(f"  Max reserved:        {profile_summary['cuda_max_memory_reserved_mb']:.4f} MB")

    maybe_plot_confusion_matrix(
        y_true,
        y_pred,
        class_names=class_names,
        title=f"{args.dataset} confusion matrix",
        save_path=os.path.join(args.confusion_matrix_dir, f"{args.dataset}_cm.png"),
        disable_plots=args.disable_plots,
    )

    maybe_write_profile_json(profile_summary, args.profile_output_json)

    results_dir = os.path.dirname(args.results_txt_path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    txt = f"""
================================ Profiled Eval Result - {ts} ================================
Dataset:
- dataset: {args.dataset}
- checkpoint: {args.checkpoint}
- test_path: {dataset_paths['test_path']}

Classification:
- Accuracy: {acc:.4f}
- Macro F1: {macro_f1:.4f}
- Macro Precision: {macro_precision:.4f}
- Macro Recall: {macro_recall:.4f}

--- F1 Per Class (FPC) ---
{report_text}

Profiling:
- Params(total): {profile_summary.get('params_total')}
- Params(trainable): {profile_summary.get('params_trainable')}
- Forward MACs(single-sample): {profile_summary.get('forward_macs')}
- Forward FLOPs~(single-sample): {profile_summary.get('forward_flops_approx')}
- Compute backend: {profile_summary.get('compute_cost_backend')}
- Single forward latency mean(ms): {profile_summary.get('single_forward_latency_ms_mean')}
- Single forward latency p95(ms): {profile_summary.get('single_forward_latency_ms_p95')}
- TTFT mean(ms): {profile_summary.get('ttft_ms_mean')}
- TPOT mean(ms): {profile_summary.get('tpot_ms_mean')}
- Eval forward batch latency mean(ms): {profile_summary.get('eval_forward_batch_latency_ms_mean')}
- Eval forward sample latency mean(ms): {profile_summary.get('eval_forward_sample_latency_ms_mean')}
- Eval forward samples/sec: {profile_summary.get('eval_forward_samples_per_s')}
- Eval e2e batch latency mean(ms): {profile_summary.get('eval_e2e_batch_latency_ms_mean')}
- CUDA max allocated(MB): {profile_summary.get('cuda_max_memory_allocated_mb')}
- CUDA max reserved(MB): {profile_summary.get('cuda_max_memory_reserved_mb')}

Args:
- batch_size: {args.batch_size}
- num_workers: {args.num_workers}
- max_len: {args.max_len}
- bert: {args.bert}
- bert_vocab_file: {args.bert_vocab_file}
- latency_warmup_iters: {args.latency_warmup_iters}
- latency_measure_iters: {args.latency_measure_iters}
==============================================================================================
"""
    if weighted_f1 is not None:
        txt += f"\nExtra weighted F1:\n- Weighted F1: {weighted_f1:.4f}\n"
    with open(args.results_txt_path, "a", encoding="utf-8") as f:
        f.write(txt.strip() + "\n")
    print(f"Metrics written: {args.results_txt_path}")


if __name__ == "__main__":
    main()
