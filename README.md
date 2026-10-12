# TALC-AWR

Implementation of **Task-Aware Lightweight Continual Learning for All-in-One Adverse Weather Image Restoration**.

TALC-AWR learns rain, fog, snow and low-light restoration sequentially. A shared encoder–decoder routes each image to at most two residual experts. Later stages freeze old experts and constrain the remaining trainable parameters with paired replay, output and feature distillation, and an orthogonality loss. Inference takes an RGB image without a task label.

The repository includes weather-data preparation, continual baselines, component and capacity controls, restoration and routing metrics, frozen perception evaluation, and plots built from saved results.

## Installation

Use Python 3.11. Install the PyTorch build for your hardware before installing the package. The default configuration uses PyTorch 2.7.1 and Torchvision 0.22.1.

```bash
git clone https://github.com/Hqa77880011/TALC-AWR.git
cd TALC-AWR
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux/macOS, or `.venv\Scripts\Activate.ps1` in Windows PowerShell.

For an NVIDIA GPU with a compatible CUDA driver:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e ".[metrics,perception]"
```

For CPU:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[metrics,perception]"
```

Pass `--device cpu` to commands that use models. LPIPS and the perception models download their feature weights on first use. Jetson users should install the PyTorch/Torchvision builds provided for their JetPack version and install this package with `--no-deps` after installing its remaining dependencies.

All commands below run from the repository root. `python -m talc_awr` is equivalent to `talc`; `talc COMMAND --help` lists the arguments for each entry point.

## 1. Prepare data

### Cityscapes-Weather-CL

Download `leftImg8bit_trainvaltest`, `gtFine_trainvaltest`, `disparity_trainvaltest`, and `camera_trainvaltest` from [Cityscapes](https://www.cityscapes-dataset.com/downloads/). Extract them into one directory:

```text
data/cityscapes/
  leftImg8bit/train/<city>/*_leftImg8bit.png
  leftImg8bit/val/<city>/*_leftImg8bit.png
  gtFine/train/<city>/*_gtFine_labelIds.png
  gtFine/train/<city>/*_gtFine_instanceIds.png
  gtFine/val/<city>/...
  disparity/train/<city>/*_disparity.png
  disparity/val/<city>/...
  camera/train/<city>/*_camera.json
  camera/val/<city>/...
```

```bash
talc prepare-cityscapes --root data/cityscapes --output data/weather --seed 17
```

Preparation reserves 500 official training scenes for internal validation using a city-stratified split. The remaining 2,475 scenes supply four deterministic variants per weather, giving 9,900 training pairs per task. The official 500 validation scenes serve as the test partition. Each validation/test scene supplies one variant per task.

The output contains `train_TASK.jsonl`, `val_TASK.jsonl`, `test_TASK.jsonl`, `splits.json`, and disparity-derived depth arrays. It also includes the four `_high` tasks, `test_mixed.jsonl`, `test_mixed_shift.jsonl`, and `test_blur.jsonl`. Each record stores its clean image, degradation parameters and seed. Weather images are generated from these records when read. Mixed weather uses rain→fog on half the test scenes and snow→low light on the other half; no mixtures enter training.

The default degradation sets are:

| Task | Parameters |
| --- | --- |
| Rain | Streak length 15/25/35 px; angle −15/0/15°; opacity 0.12/0.20/0.28; three frequency bands |
| Fog | Depth-based atmospheric scattering; β 0.008/0.016/0.024; airlight 0.75/0.85/0.95 |
| Snow | 400/800/1,200 particles; radius 1/2/3 px; opacity 0.30/0.50/0.70 |
| Low light | Exposure 0.40/0.55/0.70; gamma 1.8/2.2/2.6; read-noise σ 0.005/0.010/0.015 |

Selections within each set are uniform. High severity uses rain length 30/opacity 0.32, fog β 0.028/airlight 0.90, 1,400 snow particles/opacity 0.75, and low-light exposure 0.35/gamma 2.8/σ 0.018. Blur uses Gaussian σ 1.5 px.

### RealRain-1k

Obtain the L/H archives from [RealRain-1k](https://github.com/hiker-lw/RealRain-1k). Index the official test input and clean-reference folders separately for each density:

```bash
talc prepare-pairs --inputs data/realrain/L/test/rainy --targets data/realrain/L/test/gt --task RealRain-L --output data/realrain_L.jsonl
talc prepare-pairs --inputs data/realrain/H/test/rainy --targets data/realrain/H/test/gt --task RealRain-H --output data/realrain_H.jsonl
```

Set `--inputs` and `--targets` to the folders in your downloaded archive. Pairing uses matching relative stems. If filenames carry different suffixes, set `--input-suffix` and `--target-suffix`. The official test sets contain 224 pairs per density. Their training and validation images are not used for TALC model selection.

### ACDC

Download the RGB and labelled train/val archives from [ACDC](https://acdc.vision.ee.ethz.ch/), keeping their `rgb_anon` and `gt` directories under `data/acdc`.

```bash
talc prepare-acdc --root data/acdc --output data/acdc.jsonl
```

The manifest covers rain, fog, snow and night from the labelled train/val archive. Semantic evaluation uses `*_gt_labelTrainIds.png`, 19 classes and ignore ID 255. ACDC's normal-condition correspondences are not used as pixel-aligned restoration targets.

## 2. Build a common replay plan

All replay methods use the same ordered patches selected with a frozen stage-one shared probe. Train this reference once, then select up to 64 patches per task for the memory sweep:

```bash
talc train --variant talc --stage-limit 1 --pairs-per-task 0 --seed 17 --output runs/embedding
talc select-memory --reference runs/embedding/stage1.pt --max-pairs 64 --output data/replay_plan.json
```

Herding first picks the normalized embedding nearest the task mean, then greedily matches the mean with the average of selected embeddings. The plan stores image IDs, crop seeds and selection order. This fixed order also defines smaller memory budgets and later truncation. The reference serves patch selection only.

For the eight-stage stream, select a plan containing all eight tasks:

```bash
talc select-memory --reference runs/embedding/stage1.pt --tasks rain,fog,snow,low_light,rain_high,fog_high,snow_high,low_light_high --max-pairs 64 --output data/replay_plan8.json
```

## 3. Train TALC and the baselines

```bash
talc train --variant talc --replay-plan data/replay_plan.json --seed 17 --output runs/talc/seed17
talc train --variant derpp --replay-plan data/replay_plan.json --seed 17 --output runs/derpp/seed17
```

`configs/default.yaml` defines the architecture, losses and optimization settings. Defaults are 60,000 steps per task, validation every 2,000 steps, separate batches of eight current 256×256 patches and eight replay 128×128 patches, AdamW with cosine learning rate 2e−4→2e−6, weight decay 1e−4, and gradient clipping at 1.0. The stage checkpoint with the highest mean validation PSNR on observed tasks is retained. Hyperparameters stay fixed across stages and suites.

| `--variant` | Update rule |
| --- | --- |
| `talc` | Selective shared update, frozen old experts, replay, output/feature distillation and orthogonality |
| `finetune` | Current-task restoration loss only |
| `ewc` | Current loss plus online diagonal Fisher penalty, coefficient 100 |
| `lwf` | Previous-model output L1 on current inputs, coefficient 1; no memory |
| `replay` | Current and supervised replay restoration losses |
| `derpp` | Current loss + 0.5 supervised replay + 0.5 previous-model output L1 on replay |
| `derpp_capacity` | DER++-style rule with the capacity-matched plain backbone |
| `joint` | All tasks accessible together; total steps equal the sum of sequential stage budgets |
| `separate` | One independently trained plain backbone per task; evaluation selects it with a task label |

The plain backbone adds one deepest encoder block and two 64-channel decoder blocks to the shared architecture. The capacity reference widens the deepest two scales to 144/280 channels.

Useful arguments:

| Argument | Meaning |
| --- | --- |
| `--config` | YAML configuration path |
| `--data` | Directory containing weather manifests |
| `--tasks` | Comma-separated task order; labels follow that order |
| `--seed` | Training seed; paper protocol uses 17, 29 and 43 |
| `--pairs-per-task` | Retained pairs per observed task; default 32 |
| `--total-pairs` | Upper bound on total pairs; balance across observed tasks when the cap is reached |
| `--steps`, `--batch-size`, `--validation-interval` | Override the corresponding training settings |
| `--stage-limit` | Stop after the specified completed stage |
| `--resume` | Continue from a completed stage checkpoint into a new output directory |

Each run writes `config.json`, `stageN.pt`, `stageN_loss.jsonl`, and `stageN_summary.json`. Checkpoints include the model, task order, seed, selected step, uint8 paired replay patches, and any EWC state. Historical full training sets are not read by later stages. Resume works at stage boundaries; matrix evaluation should receive all stage checkpoints in one run directory.

## 4. Baseline, ablation and stress-test suites

`sweep` runs each experiment and seed sequentially. `--dry-run` prints the commands. Use a separate output root for a new experiment series.

```bash
talc sweep --suite main --replay-plan data/replay_plan.json --seeds 17 29 43 --output runs
talc sweep --suite ablations --replay-plan data/replay_plan.json --seeds 17 29 43 --output runs
talc sweep --suite capacity --replay-plan data/replay_plan.json --seeds 17 29 43 --output runs
talc sweep --suite memory --replay-plan data/replay_plan.json --seeds 17 29 43 --output runs
talc sweep --suite orders --replay-plan data/replay_plan.json --seeds 17 29 43 --output runs
talc sweep --suite eight --replay-plan data/replay_plan8.json --seeds 17 29 43 --output runs
```

| Suite | Experiments |
| --- | --- |
| `main` | All eight methods listed above, except the separate capacity reference |
| `ablations` | Full TALC, uniform all-expert routing, task-shared adapters, no replay/distillation, no orthogonality, learned top-1 |
| `capacity` | Full TALC, matched DER++-style, unfreeze old experts, disable distillation while keeping replay, both changes |
| `memory` | 0/8/16/32/64 paired patches per task |
| `orders` | rain→fog→snow→low light; fog→low light→rain→snow; snow→rain→low light→fog |
| `eight` | TALC and DER++-style with a 128-pair total cap; TALC with 32 pairs per task |

The corresponding TALC variants are `talc_uniform`, `talc_shared`, `talc_no_replay`, `talc_no_orth`, `talc_top1`, `talc_unfreeze`, `talc_no_distill`, and `talc_unfreeze_no_distill`. Uniform routing removes the MLP and its classification loss. Shared adapters retain the MLP but use one adapter set for every task and omit inter-expert orthogonality. The no-replay ablation removes both replay supervision and replay distillation; `talc_no_distill` retains supervised replay.

In the eight-stage stream, original weather tasks are followed by their `_high` counterparts. The total-cap runs retain 32 pairs per task through stage four and then allocate 26/26/26/25/25, 22/22/21/21/21/21, 19/19/18/18/18/18/18 and finally 16 per task. Only prefixes of saved old memory are retained; old full datasets are not reopened.

## 5. Restoration and continual metrics

Evaluate every stage on its observed test tasks:

```bash
talc evaluate-matrix --run runs/talc/seed17 --output runs/talc/seed17/evaluation
```

For a single checkpoint, mixtures or external paired images:

```bash
talc evaluate --checkpoint runs/talc/seed17/stage4.pt --manifests data/weather/test_mixed.jsonl --save-images --output runs/talc/seed17/mixed
talc evaluate --checkpoint runs/talc/seed17/stage4.pt --manifests data/realrain_L.jsonl data/realrain_H.jsonl --save-images --output runs/talc/seed17/realrain
```

`per_image.csv` records restored and input PSNR/SSIM/LPIPS, their differences, task identity and available routing probabilities. `summary.json` gives task means and their equal-weight mean. `evaluate-matrix` also writes the lower-triangular PSNR task matrix and separate per-image files for each stage. `--no-lpips` omits LPIPS explicitly.

For a matrix R with stage t and test task j:

- **AP** is the mean final-row PSNR over tasks.
- **F** averages `max(R[j:T−1,j]) − R[T,j]` over earlier tasks. The final stage is excluded from the maximum.
- **BWT** averages `R[T,j] − R[j,j]` over earlier tasks.
- Eight-stage reports additionally give final PSNR on the original four tasks, their loss relative to stage four, and mean PSNR on the new-severity tasks.

PSNR is computed per image on RGB [0,1], with no border crop. SSIM uses an 11×11 Gaussian window, σ 1.5, and valid-window averaging. LPIPS uses AlexNet on RGB [−1,1]. Positive ΔPSNR/ΔSSIM and negative ΔLPIPS indicate improvement over the degraded input. RealRain L/H summaries use equal density weights.

## 6. Routing diagnostics

```bash
talc diagnostics --checkpoint runs/talc/seed17/stage4.pt --output runs/talc/seed17/routing
talc plot-diagnostics --diagnostics runs/talc/seed17/routing --output runs/figures/routing
```

Diagnostics use one fixed learned four-task router and report:

- Single-weather macro accuracy, full confusion counts, row-normalized confusion and 10-bin top-1 ECE on all single-weather test images.
- Learned top-1, learned top-2, true-pair and complementary wrong-pair restoration scores on familiar and severity-shifted mixtures.
- Both-component top-2 recall from the learned probabilities, irrespective of a forced policy used for restoration.
- Normalized entropy AUROC, with blur as positive and 125 known images per weather as negative; blur restoration gain relative to input.

Forced-pair policies keep the router's probabilities and renormalize within the chosen pair. `--known-per-task` changes the balanced known-weather control size. The CSV files preserve each policy's individual image scores.

## 7. Frozen perception evaluation

### Semantic segmentation

SegFormer-B2 uses the released [Cityscapes checkpoint](https://huggingface.co/nvidia/segformer-b2-finetuned-cityscapes-1024-1024):

```bash
talc segmentation --manifest data/acdc.jsonl --checkpoint runs/talc/seed17/stage4.pt --segmenter segformer --output runs/talc/seed17/segformer
talc plot-perception --results runs/talc/seed17/segformer/segmentation.json --output runs/figures/segformer
```

For DeepLabV3+ ResNet-50, obtain the Cityscapes weights from [DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch) and keep its code in `external/DeepLabV3Plus-Pytorch`:

```bash
git clone https://github.com/VainF/DeepLabV3Plus-Pytorch.git external/DeepLabV3Plus-Pytorch
talc segmentation --manifest data/acdc.jsonl --checkpoint runs/talc/seed17/stage4.pt --segmenter deeplab --deeplab-source external/DeepLabV3Plus-Pytorch --segmenter-weights checkpoints/best_deeplabv3plus_resnet50_cityscapes_os16.pth --output runs/talc/seed17/deeplab
```

Both segmenters stay frozen. Each image and its restored version use the same aspect-preserving resize (`--short-side 1024`), ImageNet normalization and padding to a multiple of 32. Logits are unpadded and resized to the label dimensions. For each condition, intersections/unions are pooled over images before computing 19-class mIoU. The main average gives each condition weight 1/4. Pooled-archive class IoUs and their differences are reported separately in `segmentation.json`.

For any external restoration method, replace `--checkpoint` with `--predictions DIRECTORY` containing `ID.png` files. This also enables the paper's offline front-end comparisons with a common frozen evaluator.

### Object detection

The detection entry point covers the appendix's Faster R-CNN ResNet-50-FPN protocol. Convert Cityscapes instance maps to COCO boxes; the eight classes are person, rider, car, truck, bus, train, motorcycle and bicycle. Group instances are COCO crowd regions.

```bash
talc prepare-detection --manifest data/weather/train_rain.jsonl --output data/detector_train.json
talc train-detector --manifest data/weather/train_rain.jsonl --annotations data/detector_train.json --epochs 12 --output checkpoints/cityscapes_detector.pt
talc prepare-detection --manifest data/weather/test_rain.jsonl --output data/detector_test_rain.json
talc detection --manifest data/weather/test_rain.jsonl --annotations data/detector_test_rain.json --detector-weights checkpoints/cityscapes_detector.pt --checkpoint runs/talc/seed17/stage4.pt --output runs/talc/seed17/detection_rain
```

Detector training reads the clean targets and collapses repeated weather variants to one record per clean scene. It fine-tunes COCO-initialized Faster R-CNN on clean Cityscapes and saves the nine-label state (background plus eight classes). Existing weights in this format can be supplied directly. Evaluation freezes the detector, uses score threshold 0.05 and NMS 0.5 for both front ends, and reports AP50:95, AP50, APsmall, person AP and car AP, plus differences in percentage points. Repeat the two test commands for fog, snow and low light. A metric with no supported ground-truth instances is reported as null.

## 8. External restoration references

Run [TransWeather](https://github.com/jeya-maria-jose/TransWeather), [WGWS](https://github.com/zhuyr97/WGWS-Net), [LDR](https://github.com/noxsine/LDR), [Histoformer](https://github.com/sunshangquan/Histoformer), [SemiDDM-Weather](https://github.com/longfafffa/SemiDDM-Weather), or [DRR-Weather](https://github.com/GZHU-DVL/DRR-Weather) with their released code and weights. Preserve the manifest image ID as the prediction filename, RGB color order and original geometry.

Export synthetic or external inputs for those models:

```bash
talc export-data --manifest data/weather/test_mixed.jsonl --output data/export/mixed
talc export-data --manifest data/realrain_L.jsonl --output data/export/realrain_L
```

The exported `inputs/ID.png` and, for paired data, `targets/ID.png` share the scoring manifest's IDs. PNG export quantizes inputs to 8-bit RGB; use the same exported inputs when comparing front ends through file-based interfaces. Score the real prediction files after running each external model:

```bash
talc score-predictions --name Histoformer --manifest data/realrain_L.jsonl --predictions predictions/histoformer/L --output runs/references/histoformer/L
talc score-predictions --name Histoformer --manifest data/realrain_H.jsonl --predictions predictions/histoformer/H --output runs/references/histoformer/H
```

The same command scores mixtures and any other aligned manifest. External references retain their own architectures and data access; use their per-image files to compare restoration scores on the same scenes. `segmentation --predictions` and `detection --predictions` provide the corresponding downstream comparisons. External implementations and weights remain outside this package.

## 9. Inference, complexity and plots

Restore an image, an image tree, or a manifest:

```bash
talc infer --checkpoint runs/talc/seed17/stage4.pt --input data/example.png --output predictions/talc
talc infer --checkpoint runs/talc/seed17/stage4.pt --manifest data/weather/test_rain.jsonl --output predictions/talc_rain
```

Measure the selected checkpoint on the target device:

```bash
talc profile --checkpoint runs/talc/seed17/stage4.pt --height 512 --width 1024 --precision fp16 --warmup 100 --frames 500 --device-setup "AC power; fixed performance profile" --output runs/talc/seed17/profile
```

`profile.json` contains stored parameters, FP32 parameter storage in decimal MB, paired replay in MiB, active convolution/linear FLOPs, per-frame latency, mean/median/p95 latency, FPS, peak CUDA allocation and runtime/device metadata. Routing probe work and the full sparse restoration pass are included. One MAC counts as two FLOPs; normalization, activation and other elementwise operations are excluded from this FLOP count. Timing uses a preallocated batch-one tensor and CUDA synchronization, excluding image decoding, data transfer and downstream models. Set Orin's power mode and clocks on the device before measuring and record them with `--device-setup`.

After evaluating each run, aggregate real results:

```bash
talc evaluate-matrix --run runs/talc/seed29 --output runs/talc/seed29/evaluation
talc evaluate-matrix --run runs/talc/seed43 --output runs/talc/seed43/evaluation
talc report --evaluations runs/talc/seed17/evaluation runs/talc/seed29/evaluation runs/talc/seed43/evaluation --output runs/figures/main
talc qualitative --manifest data/weather/test_mixed.jsonl --predictions runs/talc/seed17/mixed/images --names TALC-AWR --limit 4 --output runs/figures/qualitative
```

Train seeds 29/43 before evaluating them. Add baseline evaluation directories to `--evaluations` for comparison. AP/F/BWT are calculated within each run before means and sample standard deviations are computed. A single run has null standard deviation. `comparison.csv` and `comparison.json` contain the aggregates; plots include stage curves, task matrices and final comparisons in PNG and editable SVG.

Use `evaluate-matrix --name memory0` or another experiment name when comparing runs of the same variant with different memory budgets or task orders. Reports group by that name and reject duplicate seeds within a group.

## Implementation conventions

The shared widths are 32/64/128/256, encoder depths 3/3/5/8, and shallow-to-deep decoder depths 3/2/5. Independent adapters have ranks 8/16/32/64 and 4/3/7/8 sites per scale. The router uses a shared-only two-level probe followed by a 96→1664→T MLP; the restoration pass recomputes the prefix with routed adapters. The four-task model has 6,487,012 parameters under this configuration.

After stage one, trainable shared parameters are channel normalization affine parameters, decoder pointwise projections and the RGB reconstruction head. Router parameters and the new expert also update. Stem, encoder convolution weights, scale changes, skip fusion, depthwise kernels, residual scales and old experts stay frozen unless an intervention explicitly unfreezes old experts. New adapter up-projections start at zero; adding a router logit can still change the whole model's output.

The restoration objective is elementwise Charbonnier with ε=1e−3 plus 0.2×(1−SSIM). Replay is a separate 128×128 forward with its own mean loss. Distillation is replay-output L1 plus 0.05 times the sum of four feature-scale L1 means, multiplied by 0.5. Router cross-entropy is one mean over current and replay labels, weighted by 0.1. Orthogonality uses normalized residuals at the last adapter site of each scale, removes the Gram diagonal, detaches old columns and averages over ordered expert pairs; its coefficient is 0.01. These terms are omitted at their empty-memory or single-expert boundaries.

Fog depth uses `(disparity_uint16−1)/256` and camera focal length/baseline, clipped to 0.1–200 m. Missing pixels use the valid median of their connected semantic region, then the image median if that region has no valid depth. Rain uses three sparse streak bands; snow combines blurred particles and a sparse streak layer. Low light uses `(RGB×exposure)^gamma`, 255-level Poisson sampling and Gaussian read noise. Synthetic inputs are float RGB; replay is quantized to uint8 paired patches. Padding is on the right/bottom with edge replication and is removed before restoration scoring. Perfect PSNR is capped at 120 dB for finite result serialization.

EWC uses squared per-sample gradients of the restoration loss as its empirical diagonal Fisher, accumulated online at the selected checkpoint. The DER++-style restoration baseline uses the previous stage's output on replay rather than storing historical logits. Shared replay plans, deterministic current-crop streams and independent replay streams keep those choices consistent across methods. These conventions, frozen-model weights, preprocessing resolution and the stated FLOP definition determine how to interpret newly generated scores.

## Citation

```bibtex
@misc{hu_talc_awr,
  author = {Hu, Qinao},
  title = {Task-Aware Lightweight Continual Learning for All-in-One Adverse Weather Image Restoration},
  year = {2026},
  howpublished = {\url{https://github.com/Hqa77880011/TALC-AWR}}
}
```
