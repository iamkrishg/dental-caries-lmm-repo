"""
Dental caries screening evaluation harness
==========================================
Evaluates a general-purpose large multimodal model (Claude on Microsoft
Foundry) on zero-shot dental caries detection from intraoral photographs,
replicating the methodology of Moharrami et al. (International Dental Journal,
2026), which benchmarked Google Gemini.

The model receives one photograph plus a structured "caries scanner" prompt and
returns detections in a fixed format:

    DETECTED: [confidence, ymin, xmin, ymax, xmax]

with coordinates on a normalised 0-1000 scale. Predictions are scored against
expert dentist annotations at two levels:

    image level  - does this photograph contain caries?
    tooth level  - is each lesion correctly localised? (IoU 0.3 / 0.4 / 0.5)

Dataset
-------
Ahmed et al., "Annotated intraoral image dataset for dental caries detection",
Scientific Data 12:1297 (2025). CC BY 4.0.
DOI: 10.5281/zenodo.14827784

The dataset is NOT redistributed with this repository. Download it separately
and point --split-dir at one of its split folders (test / valid / train).

Credentials
-----------
Set as environment variables; never hardcode keys.

    FOUNDRY_RESOURCE   e.g. my-foundry-resource   (subdomain, or full URL)
    FOUNDRY_API_KEY    your Foundry API key
    FOUNDRY_MODEL      deployment name, default "claude-sonnet-4-5"

Usage
-----
    pip install -r requirements.txt

    export FOUNDRY_RESOURCE=...        # Windows: set FOUNDRY_RESOURCE=...
    export FOUNDRY_API_KEY=...

    python src/evaluate_caries.py \
        --split-dir "/path/to/Benchmarking Dataset/test" \
        --out-dir results/ \
        --samples 0                     # 0 = all labelled images in the split

Licence
-------
MIT (see LICENSE). This is research code, not a medical device. It must not be
used for clinical diagnosis or claim adjudication.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
from pathlib import Path

IMG_EXTS = (".jpg", ".jpeg", ".png")

PROMPT_TEMPLATE = """You are a dental caries screening scanner, not a conversational assistant.
The image is {W} pixels wide and {H} pixels tall.
Task: examine this intraoral photograph tooth by tooth, scanning left to right.
Criteria: a tooth is carious if it shows visible decay per the simplified ICDAS
3-stage rubric (initial, moderate, extensive). Sound teeth (ICDAS 0), plaque,
calculus, stains, shadows and soft tissue are NOT caries.
For each carious lesion, first screen with high sensitivity, then immediately
re-check the same tooth with high precision before confirming.
Report every confirmed lesion on its own line, exactly as:
DETECTED: [confidence, ymin, xmin, ymax, xmax]
Coordinates MUST be integers between 0 and 1000 inclusive on a normalised scale
(top-left origin), where 0 = top/left edge and 1000 = bottom/right edge of THIS
image. Never output a coordinate above 1000 or below 0.
Confidence is a float 0-1. If no caries: output NONE. Then output [End Scan].
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clamp(v: float, lo: int = 0, hi: int = 1000) -> int:
    return max(lo, min(hi, int(round(v))))


def find_pairs(split_dir: Path, images_subdir: str, labels_subdir: str,
               limit: int = 0):
    """Pair each label file with its image, matched by filename stem.

    Walking the labels first guarantees every returned sample has ground truth
    and is therefore scoreable.
    """
    images_dir = split_dir / images_subdir
    labels_dir = split_dir / labels_subdir

    if not images_dir.is_dir():
        sys.exit(f"images folder not found: {images_dir}")
    if not labels_dir.is_dir():
        sys.exit(f"labels folder not found: {labels_dir}")

    images_by_stem = {p.stem: p for p in images_dir.rglob("*")
                      if p.suffix.lower() in IMG_EXTS}
    label_files = sorted(labels_dir.rglob("*.txt"))
    if not label_files:
        sys.exit(f"no .txt label files found in {labels_dir}")

    pairs, orphaned = [], 0
    for lbl in label_files:
        img = images_by_stem.get(lbl.stem)
        if img is None:
            orphaned += 1
            continue
        pairs.append((img, lbl))
        if limit and len(pairs) >= limit:
            break

    if not pairs:
        sys.exit(f"found {len(label_files)} labels but none matched an image "
                 f"by filename stem")

    print(f"Selected {len(pairs)} image/label pairs "
          f"(from {len(label_files)} label files)")
    if orphaned:
        print(f"  {orphaned} labels had no matching image and were skipped")
    print(f"  images: {images_dir}")
    print(f"  labels: {labels_dir}\n")
    return pairs


def yolo_to_norm1000(label_path: Path):
    """Convert YOLO (cx cy w h, normalised 0-1) to [ymin, xmin, ymax, xmax]
    on a 0-1000 scale."""
    boxes = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, w, h = map(float, parts[:5])
        boxes.append([
            clamp((cy - h / 2) * 1000), clamp((cx - w / 2) * 1000),
            clamp((cy + h / 2) * 1000), clamp((cx + w / 2) * 1000),
        ])
    return boxes


def prepare_image(image_path: Path, max_side: int):
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        w, h = int(w * scale), int(h * scale)
        img = img.resize((w, h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode(), w, h


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_client():
    """Foundry client from environment variables. Falls back to Entra ID auth
    when no API key is set."""
    from anthropic import AnthropicFoundry

    resource = os.environ.get("FOUNDRY_RESOURCE", "").strip()
    api_key = os.environ.get("FOUNDRY_API_KEY", "").strip()
    if not resource:
        sys.exit("Set FOUNDRY_RESOURCE (your Foundry resource subdomain).")

    resource = resource.replace("https://", "").replace("http://", "")
    resource = resource.split(".services.ai.azure.com")[0].strip("/")

    if api_key:
        return AnthropicFoundry(api_key=api_key, resource=resource)

    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(), "https://ai.azure.com/.default")
    return AnthropicFoundry(azure_ad_token_provider=token_provider,
                            resource=resource)


def call_model(client, model: str, image_path: Path, max_side: int):
    b64, w, h = prepare_image(image_path, max_side)
    resp = client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
                                         "media_type": "image/jpeg",
                                         "data": b64}},
            {"type": "text", "text": PROMPT_TEMPLATE.format(W=w, H=h)},
        ]}],
    )
    text = "".join(b.text for b in resp.content
                   if getattr(b, "type", None) == "text")
    return text, ("" if text else "empty response")


# ---------------------------------------------------------------------------
# Parsing and scoring
# ---------------------------------------------------------------------------
def parse_detections(text: str):
    """Extract DETECTED lines; coordinates are clamped to the valid range."""
    boxes = []
    for match in re.findall(r"\[([0-9.,\s]+)\]", text):
        vals = [float(x) for x in match.split(",") if x.strip()]
        if len(vals) == 5:
            boxes.append({"score": vals[0],
                          "coords": [clamp(v) for v in vals[1:]]})
        elif len(vals) == 4:
            boxes.append({"score": None, "coords": [clamp(v) for v in vals]})
    return boxes


def iou(a, b) -> float:
    yi1, xi1 = max(a[0], b[0]), max(a[1], b[1])
    yi2, xi2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, yi2 - yi1) * max(0, xi2 - xi1)
    area = lambda z: max(0, z[2] - z[0]) * max(0, z[3] - z[1])
    union = area(a) + area(b) - inter
    return inter / union if union else 0.0


def count_true_positives(gt_boxes, pred_boxes, threshold: float) -> int:
    """Greedy one-to-one matching: each ground-truth box may claim at most one
    prediction, and vice versa."""
    used, tp = set(), 0
    for g in gt_boxes:
        best_j, best_iou = -1, 0.0
        for j, p in enumerate(pred_boxes):
            if j in used:
                continue
            v = iou(g, p["coords"])
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= threshold:
            used.add(best_j)
            tp += 1
    return tp


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def save_overlay(image_path: Path, pred_boxes, gt_boxes, out_path: Path):
    try:
        import cv2
    except ImportError:
        return
    img = cv2.imread(str(image_path))
    if img is None:
        return
    h, w = img.shape[:2]
    to_px = lambda c: (int(c[1] / 1000 * w), int(c[0] / 1000 * h),
                       int(c[3] / 1000 * w), int(c[2] / 1000 * h))
    for g in gt_boxes:                                  # ground truth: green
        x1, y1, x2, y2 = to_px(g)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for p in pred_boxes:                                # predictions: red
        x1, y1, x2, y2 = to_px(p["coords"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if p["score"] is not None:
            cv2.putText(img, f"{p['score']:.2f}", (x1 + 2, max(12, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Evaluate zero-shot LMM caries detection on intraoral photographs.")
    ap.add_argument("--split-dir", required=True, type=Path,
                    help="dataset split folder containing images/ and yolo/")
    ap.add_argument("--out-dir", default=Path("results"), type=Path,
                    help="where to write results JSON and overlays")
    ap.add_argument("--samples", type=int, default=0,
                    help="max images to evaluate (0 = all labelled images)")
    ap.add_argument("--images-subdir", default="images")
    ap.add_argument("--labels-subdir", default="yolo")
    ap.add_argument("--max-side", type=int, default=2048,
                    help="pre-scale longest image edge to this many pixels")
    ap.add_argument("--iou", type=float, nargs="+", default=[0.3, 0.4, 0.5],
                    help="IoU thresholds for tooth-level scoring")
    ap.add_argument("--no-overlays", action="store_true")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress per-image raw model output")
    args = ap.parse_args()

    model = os.environ.get("FOUNDRY_MODEL", "claude-sonnet-4-5")
    print(f"Model: {model} on Microsoft Foundry | max_side={args.max_side}\n")

    client = build_client()
    pairs = find_pairs(args.split_dir, args.images_subdir, args.labels_subdir,
                       args.samples)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.out_dir / "overlays"

    results = []
    img_tp = img_tn = img_fp = img_fn = 0
    tooth_tp = {t: 0 for t in args.iou}
    total_gt = total_pred = 0

    for image_path, label_path in pairs:
        gt = yolo_to_norm1000(label_path)
        try:
            raw, note = call_model(client, model, image_path, args.max_side)
        except Exception as exc:                      # keep going on API errors
            print(f"{image_path.name}: API error -> {exc}\n")
            results.append({"image": image_path.name, "gt": len(gt),
                            "pred": 0, "note": f"error: {exc}", "raw": ""})
            continue

        preds = parse_detections(raw)
        total_gt += len(gt)
        total_pred += len(preds)

        tps = {t: count_true_positives(gt, preds, t) for t in args.iou}
        for t in args.iou:
            tooth_tp[t] += tps[t]

        pred_has, gt_has = bool(preds), bool(gt)
        if gt_has and pred_has:
            img_tp += 1; correct = True
        elif not gt_has and not pred_has:
            img_tn += 1; correct = True
        elif not gt_has and pred_has:
            img_fp += 1; correct = False
        else:
            img_fn += 1; correct = False

        tp_str = " ".join(f"IoU{t}:{tps[t]}" for t in args.iou)
        print(f"[{'OK ' if correct else 'XX '}] {image_path.name}")
        print(f"      predicted={'CAVITY' if pred_has else 'NO-CAVITY':9s} "
              f"actual={'CAVITY' if gt_has else 'NO-CAVITY':9s} "
              f"| GT={len(gt)} pred={len(preds)} | tooth-TP {tp_str}")
        if note:
            print(f"      <-- {note}")
        if not args.quiet:
            print(f"      raw: {raw!r}")
        print()

        results.append({
            "image": image_path.name,
            "gt": len(gt), "pred": len(preds),
            "pred_label": "CAVITY" if pred_has else "NO-CAVITY",
            "true_label": "CAVITY" if gt_has else "NO-CAVITY",
            "image_correct": correct,
            "tooth_tp": {str(k): v for k, v in tps.items()},
            "note": note, "raw": raw,
        })

        if not args.no_overlays:
            save_overlay(image_path, preds, gt,
                         overlay_dir / f"{image_path.stem}_overlay.jpg")

    # ---- summary -----------------------------------------------------------
    out_json = args.out_dir / "results.json"
    out_json.write_text(json.dumps(results, indent=2))

    n = len(results)
    accuracy = (img_tp + img_tn) / n if n else 0.0
    sensitivity = img_tp / (img_tp + img_fn) if (img_tp + img_fn) else 0.0
    specificity = img_tn / (img_tn + img_fp) if (img_tn + img_fp) else 0.0
    precision = img_tp / (img_tp + img_fp) if (img_tp + img_fp) else 0.0
    f1 = (2 * precision * sensitivity / (precision + sensitivity)
          if (precision + sensitivity) else 0.0)

    summary = {
        "model": model,
        "images_evaluated": n,
        "image_level": {
            "accuracy": round(accuracy, 4),
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
            "confusion": {"tp": img_tp, "tn": img_tn,
                          "fp": img_fp, "fn": img_fn},
        },
        "tooth_level": {
            "ground_truth_lesions": total_gt,
            "predicted_lesions": total_pred,
            "by_iou": {
                str(t): {
                    "true_positives": tooth_tp[t],
                    "recall": round(tooth_tp[t] / total_gt, 4) if total_gt else 0.0,
                    "precision": round(tooth_tp[t] / total_pred, 4) if total_pred else 0.0,
                } for t in args.iou
            },
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("=" * 62)
    print("IMAGE-LEVEL (does the photograph contain caries?)")
    print(f"  accuracy    : {accuracy:.2f}  ({img_tp + img_tn}/{n})")
    print(f"  sensitivity : {sensitivity:.2f}  ({img_tp} of {img_tp + img_fn} positives)")
    print(f"  precision   : {precision:.2f}")
    print(f"  F1          : {f1:.2f}")
    print(f"  specificity : {specificity:.2f}  ({img_tn} of {img_tn + img_fp} negatives)")
    print(f"  confusion   : TP={img_tp} TN={img_tn} FP={img_fp} FN={img_fn}")
    if (img_tn + img_fp) < 30:
        print("  NOTE: too few caries-free images for a reliable specificity estimate.")
    print()
    print("TOOTH-LEVEL (is each lesion correctly localised?)")
    print(f"  ground-truth lesions : {total_gt}")
    print(f"  predicted lesions    : {total_pred}")
    for t in args.iou:
        rec = tooth_tp[t] / total_gt if total_gt else 0.0
        prec = tooth_tp[t] / total_pred if total_pred else 0.0
        print(f"  IoU>={t}: TP={tooth_tp[t]:4d}  recall={rec:.2f}  precision={prec:.2f}")
    print("=" * 62)
    print(f"\nPer-image results : {out_json}")
    print(f"Summary metrics   : {args.out_dir / 'summary.json'}")
    if not args.no_overlays:
        print(f"Overlays          : {overlay_dir} (green = ground truth, red = model)")


if __name__ == "__main__":
    main()
