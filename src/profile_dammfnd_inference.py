import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from thop import profile as thop_profile
except ImportError:
    thop_profile = None

try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    FlopCountAnalysis = None

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model.dammfnd import DAMMFNDMODEL  # noqa: E402
from utils.custom_dataloader import bert_data as custom_data  # noqa: E402


CLASS_NAMES = [
    "real_news",
    "image_forgery",
    "entity_inconsistency",
    "event_inconsistency",
    "temporal_inconsistency",
    "invalid_visual",
]


class ProfilingWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, input_names: Sequence[str]) -> None:
        super().__init__()
        self.model = model
        self.input_names = list(input_names)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        kwargs = {name: value for name, value in zip(self.input_names, inputs)}
        return self.model(**kwargs)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DAMMFND checkpoints and benchmark inference performance on the custom JSONL dataset."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the DAMMFND checkpoint (parameter_dammfnd.pkl).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/map-vepfs/liniuniu/hesirui/datasets",
        help="Directory containing train/val/test jsonl files and AMG_MEDIA/.",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=None,
        help="Explicit path to the test jsonl file. Defaults to <data-dir>/test.jsonl.",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default=None,
        help="Single image root for all splits. Used when --test-image-root is not provided.",
    )
    parser.add_argument(
        "--test-image-root",
        type=str,
        default=None,
        help="Image root for test split. Defaults to <data-dir>/AMG_MEDIA/test_imagesN.",
    )
    parser.add_argument(
        "--bert",
        type=str,
        default="/map-vepfs/liniuniu/hesirui/bert-base-uncased",
        help="Local path to the BERT backbone.",
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default="/map-vepfs/liniuniu/hesirui/clip-vit-base-patch16",
        help="HuggingFace CLIP model path or name.",
    )
    parser.add_argument(
        "--use-cn-clip",
        action="store_true",
        default=False,
        help="Use CN-CLIP instead of HuggingFace CLIP.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=197)
    parser.add_argument("--emb-dim", type=int, default=768)
    parser.add_argument("--mlp-dims", type=str, default="384", help="Comma-separated MLP hidden dims.")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--single-sample-steps", type=int, default=50)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Only evaluate the first N batches. 0 means full test set.",
    )
    parser.add_argument(
        "--flops-backend",
        type=str,
        default="auto",
        choices=("auto", "thop", "fvcore", "none"),
        help="Backend used to estimate MACs/FLOPs.",
    )
    parser.add_argument(
        "--print-classification-metrics",
        action="store_true",
        default=False,
        help="If set, print binary and 6-class classification metrics in addition to performance metrics.",
    )
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save the report as JSON.")
    return parser.parse_args()


def parse_mlp_dims(raw_dims: str) -> List[int]:
    dims = [segment.strip() for segment in raw_dims.split(",") if segment.strip()]
    if not dims:
        raise ValueError("--mlp-dims must contain at least one integer.")
    return [int(dim) for dim in dims]


def discover_test_path(data_dir: Path) -> Optional[Path]:
    exact_candidates = [
        data_dir / "test.jsonl",
        data_dir / "test.json",
        data_dir / "test.ndjson",
    ]
    for candidate in exact_candidates:
        if candidate.exists():
            return candidate.resolve()

    pattern_priority = [
        "test*.jsonl",
        "test*.json",
        "*test*.jsonl",
        "*test*.json",
    ]
    ranked_candidates: List[Tuple[int, str, Path]] = []
    for pattern_rank, pattern in enumerate(pattern_priority):
        for candidate in data_dir.rglob(pattern):
            if not candidate.is_file():
                continue
            rel_path = str(candidate.relative_to(data_dir))
            score = (
                pattern_rank,
                0 if candidate.parent == data_dir else 1,
                len(candidate.parts),
                len(rel_path),
            )
            ranked_candidates.append((score[0] * 1000 + score[1] * 100 + score[2] * 10 + score[3], rel_path, candidate))

    if not ranked_candidates:
        return None

    ranked_candidates.sort(key=lambda item: (item[0], item[1]))
    return ranked_candidates[0][2].resolve()


def summarize_annotation_candidates(data_dir: Path, limit: int = 10) -> List[str]:
    candidates: List[str] = []
    for pattern in ("*.jsonl", "*.json", "*.ndjson"):
        for candidate in data_dir.rglob(pattern):
            if candidate.is_file():
                candidates.append(str(candidate.relative_to(data_dir)))
    candidates = sorted(set(candidates))
    return candidates[:limit]


def resolve_input_paths(args: argparse.Namespace) -> Dict[str, Path]:
    data_dir = Path(args.data_dir).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if args.test_path:
        test_path = Path(args.test_path).expanduser().resolve()
    else:
        discovered_test_path = discover_test_path(data_dir)
        test_path = discovered_test_path if discovered_test_path is not None else data_dir / "test.jsonl"
    if args.test_image_root:
        test_image_root = Path(args.test_image_root).expanduser().resolve()
    elif args.image_root:
        test_image_root = Path(args.image_root).expanduser().resolve()
    else:
        test_image_root = data_dir / "AMG_MEDIA" / "test_imagesN"

    output_json = Path(args.output_json).expanduser().resolve() if args.output_json else None
    return {
        "checkpoint": checkpoint,
        "data_dir": data_dir,
        "test_path": test_path,
        "test_image_root": test_image_root,
        "output_json": output_json,
    }


def validate_paths(paths: Dict[str, Path], args: argparse.Namespace) -> None:
    for key in ("checkpoint", "data_dir", "test_image_root"):
        if not paths[key].exists():
            raise FileNotFoundError(f"{key} not found: {paths[key]}")
    if not paths["test_path"].exists():
        candidates = summarize_annotation_candidates(paths["data_dir"])
        message = f"test_path not found: {paths['test_path']}"
        if args.test_path:
            message += ". Please pass a valid --test-path."
        else:
            message += ". Could not auto-discover a test annotation file under --data-dir."
        if candidates:
            message += " Candidate annotation files: " + ", ".join(candidates)
        raise FileNotFoundError(message)
    bert_path = Path(args.bert).expanduser()
    if bert_path.exists():
        pass
    elif os.path.sep in args.bert:
        raise FileNotFoundError(f"bert not found: {args.bert}")
    clip_model_path = Path(args.clip_model).expanduser()
    if clip_model_path.exists():
        return
    if os.path.sep in args.clip_model:
        raise FileNotFoundError(f"clip_model not found: {args.clip_model}")


def strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key.replace("module.", "", 1) if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def is_tensor_state_dict(candidate: Any) -> bool:
    return isinstance(candidate, dict) and candidate and all(torch.is_tensor(v) for v in candidate.values())


def extract_state_dict(checkpoint_obj: Any) -> Dict[str, Any]:
    if is_tensor_state_dict(checkpoint_obj):
        return strip_module_prefix(checkpoint_obj)

    if isinstance(checkpoint_obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "network", "module"):
            candidate = checkpoint_obj.get(key)
            if is_tensor_state_dict(candidate):
                return strip_module_prefix(candidate)

    raise ValueError("Unable to locate a valid state_dict in the checkpoint.")


def build_test_loader(args: argparse.Namespace, paths: Dict[str, Path]) -> Any:
    category_dict = {name: index for index, name in enumerate(CLASS_NAMES)}
    loader_builder = custom_data(
        max_len=args.max_len,
        batch_size=args.batch_size,
        bert=args.bert,
        category_dict=category_dict,
        num_workers=args.num_workers,
        root_dir=str(paths["test_image_root"]),
        clip_model=args.clip_model,
    )
    return loader_builder.load_data(str(paths["test_path"]), shuffle=False)


def build_model(args: argparse.Namespace, device: torch.device) -> DAMMFNDMODEL:
    model = DAMMFNDMODEL(
        emb_dim=args.emb_dim,
        mlp_dims=parse_mlp_dims(args.mlp_dims),
        bert=args.bert,
        out_channels=320,
        dropout=args.dropout,
        num_classes=args.num_classes,
        use_cn_clip=args.use_cn_clip,
        clip_model=args.clip_model,
    )
    state_dict = extract_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint incompatibility detected. "
            f"missing_keys={list(incompatible.missing_keys)}, "
            f"unexpected_keys={list(incompatible.unexpected_keys)}"
        )
    model.to(device)
    model.eval()
    return model


def maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_ms(values_ms: Sequence[float]) -> Dict[str, float]:
    values = list(values_ms)
    if not values:
        return {
            "mean_ms": float("nan"),
            "std_ms": float("nan"),
            "p50_ms": float("nan"),
            "p90_ms": float("nan"),
            "p95_ms": float("nan"),
        }
    return {
        "mean_ms": float(statistics.mean(values)),
        "std_ms": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "p50_ms": percentile(values, 50),
        "p90_ms": percentile(values, 90),
        "p95_ms": percentile(values, 95),
    }


def current_cuda_memory(device: torch.device) -> Dict[str, Optional[float]]:
    if device.type != "cuda":
        return {
            "cuda_max_memory_allocated_mb": None,
            "cuda_max_memory_reserved_mb": None,
        }
    return {
        "cuda_max_memory_allocated_mb": float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)),
        "cuda_max_memory_reserved_mb": float(torch.cuda.max_memory_reserved(device) / (1024 ** 2)),
    }


def batch_to_device(batch: Tuple[Any, ...], device: torch.device) -> Dict[str, Any]:
    batch_data = {
        "content": batch[0].to(device, non_blocking=True),
        "content_masks": batch[1].to(device, non_blocking=True),
        "label": batch[2].to(device, non_blocking=True),
        "category": batch[3].to(device, non_blocking=True),
        "image": batch[4].to(device, non_blocking=True),
        "clip_image": batch[5].to(device, non_blocking=True),
        "clip_text": batch[6].to(device, non_blocking=True),
        "multi_category": batch[7].to(device, non_blocking=True),
    }
    if len(batch) > 8:
        batch_data["clip_attention_mask"] = batch[8].to(device, non_blocking=True)
    return batch_data


def make_forward_kwargs(batch_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    forward_keys = [
        "content",
        "content_masks",
        "image",
        "clip_image",
        "clip_text",
        "multi_category",
    ]
    kwargs = {key: batch_data[key] for key in forward_keys}
    if "clip_attention_mask" in batch_data:
        kwargs["clip_attention_mask"] = batch_data["clip_attention_mask"]
    return kwargs


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "params_total": int(total),
        "params_trainable": int(trainable),
    }


def build_profile_inputs(batch_data: Dict[str, Any]) -> Tuple[List[str], Tuple[torch.Tensor, ...]]:
    input_names = [
        "content",
        "content_masks",
        "image",
        "clip_image",
        "clip_text",
        "multi_category",
    ]
    if "clip_attention_mask" in batch_data:
        input_names.append("clip_attention_mask")
    sample_inputs = tuple(batch_data[name] for name in input_names)
    return input_names, sample_inputs


def _profile_with_backend(
    model: torch.nn.Module,
    sample_inputs: Tuple[torch.Tensor, ...],
    input_names: Sequence[str],
    backend: str,
) -> Dict[str, Any]:
    wrapper = ProfilingWrapper(model, input_names)
    report = {
        "backend": backend,
        "forward_macs": None,
        "forward_flops": None,
        "notes": [],
    }

    with torch.inference_mode():
        if backend == "thop":
            macs, _ = thop_profile(wrapper, inputs=sample_inputs, verbose=False)
            report["forward_macs"] = float(macs)
            report["forward_flops"] = float(macs * 2.0)
            report["notes"].append("FLOPs are reported as 2 x MACs under the thop convention.")
            return report

        if backend == "fvcore":
            analysis = FlopCountAnalysis(wrapper, sample_inputs)
            flops = analysis.total()
            report["forward_macs"] = float(flops)
            report["forward_flops"] = float(flops * 2.0)
            unsupported = analysis.unsupported_ops()
            if unsupported:
                report["notes"].append(
                    "Unsupported ops skipped by fvcore: "
                    + ", ".join(f"{name} x {count}" for name, count in unsupported.items())
                )
            report["notes"].append("FLOPs are approximated as 2 x fvcore-reported ops.")
            return report

    raise ValueError(f"Unsupported backend: {backend}")


def profile_compute_cost(
    model: torch.nn.Module,
    sample_batch: Dict[str, Any],
    backend: str,
) -> Dict[str, Any]:
    input_names, sample_inputs = build_profile_inputs(sample_batch)
    report = {
        "backend": "none",
        "forward_macs": None,
        "forward_flops": None,
        "notes": [],
    }

    candidate_backends: List[str] = []
    if backend == "auto":
        if thop_profile is not None:
            candidate_backends.append("thop")
        if FlopCountAnalysis is not None:
            candidate_backends.append("fvcore")
    elif backend != "none":
        candidate_backends.append(backend)

    if not candidate_backends:
        report["notes"].append("Install thop or fvcore to enable static compute analysis.")
        return report

    for candidate_backend in candidate_backends:
        if candidate_backend == "thop" and thop_profile is None:
            report["notes"].append("thop is not installed.")
            continue
        if candidate_backend == "fvcore" and FlopCountAnalysis is None:
            report["notes"].append("fvcore is not installed.")
            continue
        try:
            return _profile_with_backend(model, sample_inputs, input_names, candidate_backend)
        except Exception as exc:
            report["backend"] = candidate_backend
            report["notes"].append(f"{candidate_backend} profiling failed: {exc}")

    return report


def safe_metric(
    func: Any,
    *metric_args: Any,
    **metric_kwargs: Any,
) -> Optional[float]:
    try:
        return float(func(*metric_args, **metric_kwargs))
    except ValueError:
        return None


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: np.ndarray,
    num_classes: int,
) -> Dict[str, Any]:
    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    labels = list(range(num_classes))
    per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(
        y_true_np,
        y_pred_np,
        labels=labels,
        zero_division=0,
    )

    multiclass_metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "macro_precision": float(precision_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true_np, y_pred_np, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true_np, y_pred_np, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_np, y_pred_np, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_np, y_pred_np)),
        "confusion_matrix": confusion_matrix(y_true_np, y_pred_np, labels=labels).tolist(),
        "per_class": {},
    }

    if y_prob.ndim == 2 and y_prob.shape[1] == num_classes:
        multiclass_metrics["roc_auc_ovr_macro"] = safe_metric(
            roc_auc_score,
            y_true_np,
            y_prob,
            multi_class="ovr",
            average="macro",
            labels=labels,
        )
        multiclass_metrics["roc_auc_ovo_macro"] = safe_metric(
            roc_auc_score,
            y_true_np,
            y_prob,
            multi_class="ovo",
            average="macro",
            labels=labels,
        )
    else:
        multiclass_metrics["roc_auc_ovr_macro"] = None
        multiclass_metrics["roc_auc_ovo_macro"] = None

    for index in labels:
        class_name = CLASS_NAMES[index] if index < len(CLASS_NAMES) else f"class_{index}"
        multiclass_metrics["per_class"][class_name] = {
            "precision": float(per_precision[index]),
            "recall": float(per_recall[index]),
            "f1": float(per_f1[index]),
            "support": int(per_support[index]),
        }

    binary_true = (y_true_np > 0).astype(np.int64)
    binary_pred = (y_pred_np > 0).astype(np.int64)
    if y_prob.ndim == 2 and y_prob.shape[1] == num_classes:
        binary_fake_prob = 1.0 - y_prob[:, 0]
    else:
        binary_fake_prob = binary_pred.astype(np.float64)

    bin_precision, bin_recall, bin_f1, bin_support = precision_recall_fscore_support(
        binary_true,
        binary_pred,
        labels=[0, 1],
        zero_division=0,
    )
    binary_metrics = {
        "accuracy": float(accuracy_score(binary_true, binary_pred)),
        "macro_precision": float(precision_score(binary_true, binary_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(binary_true, binary_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(binary_true, binary_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(binary_true, binary_pred)),
        "roc_auc": safe_metric(roc_auc_score, binary_true, binary_fake_prob),
        "pr_auc": safe_metric(average_precision_score, binary_true, binary_fake_prob),
        "confusion_matrix": confusion_matrix(binary_true, binary_pred, labels=[0, 1]).tolist(),
        "per_class": {
            "real": {
                "precision": float(bin_precision[0]),
                "recall": float(bin_recall[0]),
                "f1": float(bin_f1[0]),
                "support": int(bin_support[0]),
            },
            "fake": {
                "precision": float(bin_precision[1]),
                "recall": float(bin_recall[1]),
                "f1": float(bin_f1[1]),
                "support": int(bin_support[1]),
            },
        },
    }

    return {
        "multiclass": multiclass_metrics,
        "binary_real_vs_fake": binary_metrics,
    }


def benchmark_single_sample_latency(
    model: torch.nn.Module,
    single_batch: Dict[str, Any],
    device: torch.device,
    warmup_steps: int,
    measure_steps: int,
) -> Dict[str, Any]:
    sample_batch: Dict[str, Any] = {}
    for key, value in single_batch.items():
        if torch.is_tensor(value):
            sample_batch[key] = value[:1]
        else:
            sample_batch[key] = value
    forward_kwargs = make_forward_kwargs(sample_batch)
    timings_ms: List[float] = []
    ttfp_ms: List[float] = []

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for _ in range(max(warmup_steps, 0)):
            _ = model(**forward_kwargs)
        maybe_sync(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for _ in range(max(measure_steps, 1)):
            maybe_sync(device)
            start = time.perf_counter()
            _ = model(**forward_kwargs)
            maybe_sync(device)
            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000.0
            timings_ms.append(elapsed_ms)
            ttfp_ms.append(elapsed_ms)

    metrics = summarize_ms(timings_ms)
    metrics["ttfp_mean_ms"] = float(statistics.mean(ttfp_ms)) if ttfp_ms else float("nan")
    metrics["ttfp_p95_ms"] = percentile(ttfp_ms, 95)
    metrics["tpot_mean_ms"] = None
    metrics["tpot_p95_ms"] = None
    metrics["mean_output_seq_len"] = 1.0
    metrics["tokens_per_second"] = float(1000.0 / metrics["mean_ms"]) if metrics["mean_ms"] > 0 else float("nan")
    metrics["notes"] = [
        "DAMMFND is a non-autoregressive classifier: TTFP equals one forward pass, TPOT is not applicable, and sequence length is fixed to 1."
    ]
    metrics.update(current_cuda_memory(device))
    return metrics


def evaluate_and_profile(
    model: torch.nn.Module,
    dataloader: Any,
    device: torch.device,
    max_batches: int,
) -> Dict[str, Any]:
    logits_list: List[np.ndarray] = []
    labels_list: List[int] = []
    preds_list: List[int] = []
    forward_batch_latency_ms: List[float] = []
    e2e_batch_latency_ms: List[float] = []
    forward_sample_latency_ms: List[float] = []
    total_samples = 0
    total_e2e_seconds = 0.0
    batches_measured = 0
    iterator = iter(dataloader)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        while True:
            if max_batches and batches_measured >= max_batches:
                break

            e2e_start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break

            batch_data = batch_to_device(batch, device)
            forward_kwargs = make_forward_kwargs(batch_data)
            maybe_sync(device)
            forward_start = time.perf_counter()
            outputs = model(**forward_kwargs)[0]
            maybe_sync(device)
            forward_end = time.perf_counter()
            e2e_end = time.perf_counter()

            probs = torch.softmax(outputs, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            labels = batch_data["label"]

            logits_list.append(probs.detach().cpu().numpy())
            preds_list.extend(preds.detach().cpu().numpy().tolist())
            labels_list.extend(labels.detach().cpu().numpy().tolist())

            batch_size = int(labels.shape[0])
            total_samples += batch_size
            total_e2e_seconds += max(e2e_end - e2e_start, 0.0)
            forward_elapsed_ms = (forward_end - forward_start) * 1000.0
            forward_batch_latency_ms.append(forward_elapsed_ms)
            e2e_batch_latency_ms.append((e2e_end - e2e_start) * 1000.0)
            forward_sample_latency_ms.append(forward_elapsed_ms / max(batch_size, 1))
            batches_measured += 1

    probabilities = np.concatenate(logits_list, axis=0) if logits_list else np.empty((0, model.num_classes))
    classification = compute_classification_metrics(
        y_true=labels_list,
        y_pred=preds_list,
        y_prob=probabilities,
        num_classes=model.num_classes,
    )
    return {
        "num_batches_measured": batches_measured,
        "num_samples_measured": total_samples,
        "classification": classification,
        "forward_batch_latency_ms": summarize_ms(forward_batch_latency_ms),
        "e2e_batch_latency_ms": summarize_ms(e2e_batch_latency_ms),
        "forward_sample_latency_mean_ms": float(statistics.mean(forward_sample_latency_ms))
        if forward_sample_latency_ms
        else float("nan"),
        "eval_samples_per_sec": float(total_samples / total_e2e_seconds) if total_e2e_seconds > 0 else float("nan"),
        "cuda_memory_peak": current_cuda_memory(device),
    }


def safe_format(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "N/A"
    return f"{value:.4f}"


def humanize_count(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "N/A"
    units = ["", "K", "M", "G", "T", "P"]
    scaled = float(value)
    unit_index = 0
    while abs(scaled) >= 1000.0 and unit_index < len(units) - 1:
        scaled /= 1000.0
        unit_index += 1
    return f"{scaled:.4f} {units[unit_index]}".rstrip()


def print_report(report: Dict[str, Any]) -> None:
    print("=== Setup ===")
    for key, value in report["setup"].items():
        print(f"{key}: {value}")
    print("")

    compute_cost = report["compute_cost"]
    print("[Profile] Computation cost")
    print(f"  Params (total):      {humanize_count(report['parameters']['params_total'])}")
    print(f"  Params (trainable):  {humanize_count(report['parameters']['params_trainable'])}")
    print(f"  Forward MACs:        {humanize_count(compute_cost['forward_macs'])}")
    print(f"  Forward FLOPs~:      {humanize_count(compute_cost['forward_flops'])}")
    print(f"  Backend:             {compute_cost['backend']}")
    if compute_cost["notes"]:
        for note in compute_cost["notes"]:
            print(f"  Note:                {note}")
    print("")

    print("[Profile] Generate latency (single sample)")
    single = report["single_sample_forward_latency_ms"]
    print(f"  Mean latency:        {safe_format(single['mean_ms'])} ms")
    print(f"  Std latency:         {safe_format(single['std_ms'])} ms")
    print(
        "  P50 / P90 / P95:     "
        f"{safe_format(single['p50_ms'])} / {safe_format(single['p90_ms'])} / {safe_format(single['p95_ms'])} ms"
    )
    print(
        "  TTFP (首个预测延迟):  "
        f"{safe_format(single['ttfp_mean_ms'])} ms (P95: {safe_format(single['ttfp_p95_ms'])} ms)"
    )
    print(
        "  TPOT (单步增量延迟):   "
        f"{safe_format(single['tpot_mean_ms'])} ms (P95: {safe_format(single['tpot_p95_ms'])} ms)"
    )
    print(f"  Mean gen seq len:    {safe_format(single['mean_output_seq_len'])}")
    print(f"  Tokens / second:     {safe_format(single['tokens_per_second'])}")
    if single["notes"]:
        for note in single["notes"]:
            print(f"  Note:                {note}")
    print("")

    perf = report["dataset_eval"]
    print("[Profile] Full evaluation generate latency")
    print(f"  Mean batch latency:  {safe_format(perf['forward_batch_latency_ms']['mean_ms'])} ms")
    print(
        "  P50 / P90 / P95:     "
        f"{safe_format(perf['forward_batch_latency_ms']['p50_ms'])} / "
        f"{safe_format(perf['forward_batch_latency_ms']['p90_ms'])} / "
        f"{safe_format(perf['forward_batch_latency_ms']['p95_ms'])} ms"
    )
    print(f"  Mean/sample latency: {safe_format(perf['forward_sample_latency_mean_ms'])} ms")
    print(f"  Samples / second:    {safe_format(perf['eval_samples_per_sec'])}")
    print("")

    print("[Profile] Full evaluation end-to-end batch latency")
    print(f"  Mean batch latency:  {safe_format(perf['e2e_batch_latency_ms']['mean_ms'])} ms")
    print(
        "  P50 / P90 / P95:     "
        f"{safe_format(perf['e2e_batch_latency_ms']['p50_ms'])} / "
        f"{safe_format(perf['e2e_batch_latency_ms']['p90_ms'])} / "
        f"{safe_format(perf['e2e_batch_latency_ms']['p95_ms'])} ms"
    )
    print("")

    memory = perf["cuda_memory_peak"]
    print("[Profile] CUDA memory peak")
    print(f"  Max allocated:       {safe_format(memory['cuda_max_memory_allocated_mb'])} MB")
    print(f"  Max reserved:        {safe_format(memory['cuda_max_memory_reserved_mb'])} MB")

    if report["setup"]["print_classification_metrics"]:
        print("")
        multiclass = perf["classification"]["multiclass"]
        print("=== Classification: Multi-class (6-way) ===")
        print(f"accuracy: {safe_format(multiclass['accuracy'])}")
        print(f"macro_precision: {safe_format(multiclass['macro_precision'])}")
        print(f"macro_recall: {safe_format(multiclass['macro_recall'])}")
        print(f"macro_f1: {safe_format(multiclass['macro_f1'])}")
        print(f"weighted_f1: {safe_format(multiclass['weighted_f1'])}")
        print(f"balanced_accuracy: {safe_format(multiclass['balanced_accuracy'])}")
        print(f"roc_auc_ovr_macro: {safe_format(multiclass['roc_auc_ovr_macro'])}")
        print(f"roc_auc_ovo_macro: {safe_format(multiclass['roc_auc_ovo_macro'])}")
        print("confusion_matrix:")
        for row in multiclass["confusion_matrix"]:
            print(f"  {row}")
        print("per_class:")
        for class_name, metrics in multiclass["per_class"].items():
            print(
                f"  {class_name:25s} -> "
                f"P: {safe_format(metrics['precision'])}  "
                f"R: {safe_format(metrics['recall'])}  "
                f"F1: {safe_format(metrics['f1'])}  "
                f"support: {metrics['support']}"
            )

        print("")
        binary = perf["classification"]["binary_real_vs_fake"]
        print("=== Classification: Binary (real vs fake) ===")
        print(f"accuracy: {safe_format(binary['accuracy'])}")
        print(f"macro_precision: {safe_format(binary['macro_precision'])}")
        print(f"macro_recall: {safe_format(binary['macro_recall'])}")
        print(f"macro_f1: {safe_format(binary['macro_f1'])}")
        print(f"balanced_accuracy: {safe_format(binary['balanced_accuracy'])}")
        print(f"roc_auc: {safe_format(binary['roc_auc'])}")
        print(f"pr_auc: {safe_format(binary['pr_auc'])}")
        print("confusion_matrix:")
        for row in binary["confusion_matrix"]:
            print(f"  {row}")
        print(
            "  real -> "
            f"P: {safe_format(binary['per_class']['real']['precision'])}  "
            f"R: {safe_format(binary['per_class']['real']['recall'])}  "
            f"F1: {safe_format(binary['per_class']['real']['f1'])}  "
            f"support: {binary['per_class']['real']['support']}"
        )
        print(
            "  fake -> "
            f"P: {safe_format(binary['per_class']['fake']['precision'])}  "
            f"R: {safe_format(binary['per_class']['fake']['recall'])}  "
            f"F1: {safe_format(binary['per_class']['fake']['f1'])}  "
            f"support: {binary['per_class']['fake']['support']}"
        )


def main() -> None:
    args = parse_args()
    paths = resolve_input_paths(args)
    validate_paths(paths, args)
    args.checkpoint = str(paths["checkpoint"])

    if str(SRC_DIR) != os.getcwd():
        os.chdir(SRC_DIR)

    device = torch.device(args.device)
    dataloader = build_test_loader(args, paths)
    try:
        first_batch_raw = next(iter(dataloader))
    except StopIteration as exc:
        raise ValueError(f"Test dataloader is empty: {paths['test_path']}") from exc
    first_batch = batch_to_device(first_batch_raw, device)

    model = build_model(args, device)
    report = {
        "setup": {
            "device": str(device),
            "checkpoint": str(paths["checkpoint"]),
            "test_path": str(paths["test_path"]),
            "test_image_root": str(paths["test_image_root"]),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "warmup_steps": args.warmup_steps,
            "single_sample_steps": args.single_sample_steps,
            "max_batches": args.max_batches,
            "use_cn_clip": args.use_cn_clip,
            "clip_model": args.clip_model,
            "bert": args.bert,
            "print_classification_metrics": args.print_classification_metrics,
        },
        "parameters": count_parameters(model),
        "compute_cost": profile_compute_cost(
            model=model,
            sample_batch=first_batch,
            backend=args.flops_backend,
        ),
        "single_sample_forward_latency_ms": benchmark_single_sample_latency(
            model=model,
            single_batch=first_batch,
            device=device,
            warmup_steps=args.warmup_steps,
            measure_steps=args.single_sample_steps,
        ),
        "dataset_eval": evaluate_and_profile(
            model=model,
            dataloader=dataloader,
            device=device,
            max_batches=args.max_batches,
        ),
    }

    print_report(report)

    output_json = paths["output_json"]
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report to: {output_json}")


if __name__ == "__main__":
    main()
