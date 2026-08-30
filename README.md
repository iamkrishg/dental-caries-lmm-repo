# Dental Caries Screening with a General-Purpose Large Multimodal Model

Zero-shot evaluation of **Claude (Sonnet 4.5) on Microsoft Foundry** for dental
caries detection from intraoral photographs, replicating the methodology of
Moharrami et al. (*International Dental Journal*, 2026), which benchmarked
Google Gemini.

> **This is research code, not a medical device.** It must not be used for
> clinical diagnosis, treatment planning, or claim adjudication. See
> [Limitations](#limitations).

---

## Summary of findings

Evaluated on **232 expert-annotated intraoral photographs** (benchmark test
split), zero-shot, single inference pass.

### Image level — does the photograph contain caries?

| Metric | Result |
|---|---|
| Accuracy | 0.84 (196/232) |
| **Sensitivity (recall)** | **0.86** (194/226) |
| **Precision (PPV)** | **0.98** |
| **F1** | **0.92** |
| Specificity | 0.33 (2/6) — not interpretable, see below |
| Confusion | TP=194, TN=2, FP=4, FN=32 |

### Tooth level — is each lesion correctly localised?

| IoU threshold | True positives | Recall | Precision |
|---|---|---|---|
| ≥ 0.3 | 105 | 0.15 | 0.14 |
| ≥ 0.4 | 57 | 0.08 | 0.08 |
| ≥ 0.5 | 28 | 0.04 | 0.04 |

### The central finding

The model produced **727 detections against 690 annotated lesions** (3.1 vs 3.0
per image) — it is well calibrated on *how many* lesions are present, while
performing poorly on *where* they are. Strong screening, weak localisation.

This replicates, on a different model family and cloud platform, the
conclusion of the prior Gemini study: general-purpose LMMs suit preliminary
**screening and triage**, not lesion-level diagnosis.

---

## Repository contents

```
src/evaluate_caries.py     evaluation harness (single script)
results/                   summary metrics and per-image results
docs/                      write-up (IEEE conference format)
requirements.txt           Python dependencies
```

---

## Getting started

### 1. Install

```bash
git clone https://github.com/iamkrishg/dental-caries-lmm-repo.git
cd <your-repo>
pip install -r requirements.txt
```

### 2. Download the dataset

The dataset is **not redistributed here**. Download it from Zenodo:

> Ahmed et al., *Annotated intraoral image dataset for dental caries detection*,
> Scientific Data 12:1297 (2025). CC BY 4.0.
> DOI: [10.5281/zenodo.14827784](https://doi.org/10.5281/zenodo.14827784)

Each split (`test/`, `valid/`, `train/`) contains parallel `images/` and
`yolo/` folders.

### 3. Set credentials

Credentials are read from environment variables. **Never commit keys.**

```bash
export FOUNDRY_RESOURCE="your-foundry-resource"   # subdomain or full URL
export FOUNDRY_API_KEY="your-api-key"
export FOUNDRY_MODEL="claude-sonnet-4-5"          # optional, this is default
```

On Windows PowerShell:

```powershell
$env:FOUNDRY_RESOURCE="your-foundry-resource"
$env:FOUNDRY_API_KEY="your-api-key"
```

If `FOUNDRY_API_KEY` is unset, the script falls back to Entra ID
(`DefaultAzureCredential`).

### 4. Run

```bash
python src/evaluate_caries.py \
    --split-dir "/path/to/Benchmarking Dataset/test" \
    --out-dir results/
```

Useful flags:

| Flag | Purpose |
|---|---|
| `--samples N` | cap the number of images (0 = all) |
| `--iou 0.3 0.4 0.5` | IoU thresholds for tooth-level scoring |
| `--no-overlays` | skip writing annotated images |
| `--quiet` | suppress per-image raw model output |
| `--max-side N` | pre-scale longest image edge (default 2048) |

### Output

```
results/results.json      per-image predictions and raw model output
results/summary.json      aggregate metrics
results/overlays/         images with ground truth (green) and predictions (red)
```

---

## Method

1. Each photograph is submitted with a structured "caries scanner" prompt
   specifying tooth-by-tooth scanning, a simplified ICDAS three-stage rubric,
   and a screen-then-verify procedure.
2. The model returns `DETECTED: [confidence, ymin, xmin, ymax, xmax]` lines on
   a normalised 0–1000 coordinate scale.
3. Predictions are matched to expert annotations by greedy Intersection-over-
   Union assignment and scored at image and tooth level.

No fine-tuning or task-specific adaptation is performed at any stage.

---

## Limitations

- **Single inference run.** No multi-run averaging or confidence intervals;
  run-to-run variability is uncharacterised.
- **Specificity is not interpretable.** Only 6 of 232 images (2.6%) were
  caries-free. The denominator is far too small to support an estimate.
- **Views are pooled.** Occlusal, mandibular, frontal and lateral views are
  combined. Prior work reports materially different performance by view, so
  these figures likely understate occlusal and overstate frontal/lateral
  performance.
- **Zero-shot only.** Few-shot prompting, which prior work reports improves
  precision, was not evaluated.
- **Single population.** The benchmark derives from one population and excludes
  low-quality images; generalisation is unestablished.
- **32 false negatives.** In a screening application a missed case carries more
  clinical consequence than a false alarm. This error mode requires
  characterisation before any real-world use.

---

## Reproducing the reported results

```bash
python src/evaluate_caries.py \
    --split-dir "/path/to/Benchmarking Dataset/test" \
    --out-dir results/ \
    --quiet
```

Model outputs are non-deterministic, so exact figures will vary between runs.

---

## Citing

If you use this code, please cite the underlying study and dataset:

```bibtex
@article{moharrami2026lmm,
  title   = {Detecting dental caries using general-purpose large multimodal
             models from oral photographs},
  author  = {Moharrami, M. and Asadi, S. and Farooqi, O. and Amin, M. and
             Glogauer, M. and Quinonez, C. and Schwendicke, F.},
  journal = {International Dental Journal},
  year    = {2026}
}

@article{ahmed2025dataset,
  title   = {Annotated intraoral image dataset for dental caries detection},
  author  = {Ahmed, S. M. F. and Ghori, M. H. and others},
  journal = {Scientific Data},
  volume  = {12},
  pages   = {1297},
  year    = {2025},
  doi     = {10.5281/zenodo.14827784}
}
```

---

## Acknowledgements

The intraoral photographs and expert annotations were created and openly
published by Ahmed et al. under CC BY 4.0. All credit for the dataset belongs
to them; this work is an independent secondary use of their open data. We also
acknowledge Moharrami et al., whose published methodology this work replicates.

---

## Licence

Code is released under the [MIT Licence](LICENSE). The dataset is governed by
its own CC BY 4.0 licence and is not redistributed here.
