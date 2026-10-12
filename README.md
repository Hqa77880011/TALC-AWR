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

Pass `--device cpu` to commands that use models; use `--precision fp32` for CPU profiling. LPIPS and the perception models download their feature weights on first use. The package also supports Python 3.10 on Jetson. Jetson users should install the PyTorch/Torchvision builds provided for their JetPack version and install this package with `--no-deps` after installing its remaining dependencies.

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
talc prepare-cityscapes --root data/cityscapes --output data/weather --seed 2025 --degradation-root 41001
```

Preparation reserves 500 official training scenes for internal validation using a city-stratified split. The remaining 2,475 scenes supply four deterministic variants per weather, giving 9,900 training pairs per task. The official 500 validation scenes serve as the test partition. Each validation/test scene supplies one variant per task.

The output contains `train_TASK.jsonl`, `val_TASK.jsonl`, `test_TASK.jsonl`, `splits.json`, `preparation.json`, and disparity-derived depth arrays. It also includes the four `_high` tasks, `test_mixed.jsonl`, `test_mixed_shift.jsonl`, and `test_blur.jsonl`. Each record stores its clean scene, variant, degradation parameters and independent component seeds. Weather images are generated from these records when read. Mixed weather uses rain→fog on half the test scenes and snow→low light on the other half, reusing the corresponding single-task test settings. Shifted mixtures compose the saved high-severity settings. No mixtures enter training.

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

Each of the 9,900 training pairs supplies four crops, giving 39,600 candidates per task. Both crop arrays are quantized to uint8 before storage; the degraded crop is decoded to float32 for embedding. Herding first picks the normalized embedding nearest the task mean, then greedily matches the mean with the average of selected embeddings. Float64 distances and lexicographic scene/variant/crop ordering resolve exact ties. The plan stores image IDs, crop seeds, task order and selection order. This fixed order defines smaller memory budgets and later truncation. The reference serves patch selection only.

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

Each run writes `config.json`, `stageN.pt`, `stageN_loss.jsonl`, and `stageN_summary.json`. Checkpoints include the model, task order, seed, selected step, augmentation/random states, uint8 paired replay patches, and any EWC state. Historical full training sets are not read by later stages. Replay first samples an old task uniformly, then a stored pair uniformly. Resume restores random streams at completed stage boundaries; matrix evaluation should receive all stage checkpoints in one run directory. `--stage-limit` applies to sequential methods.

## 4. Baseline, ablation and stress-test suites

`sweep` runs each experiment and seed sequentially. `--dry-run` prints the commands. Use a separate output root for a new experiment series.

```bash
talc sweep --suite main --replay-plan data/replay_plan.json --seeds 17 29 43 --output runs
talc sweep --suite ablations --replay-plan data/replay_plan.json --seeds 17 29 43 --output runs
talc sweep --suite capacity --replay-plan data/replay_plan.json --seeds 17 29 43 --output runs
talc sweep --suite memory --replay-plan data/replay_plan.json --seeds 17 29 43 --output runs
talc sweep --suite orders --order-replay-plans data/replay_plan.json data/replay_plan_fog.json data/replay_plan_snow.json --seeds 17 29 43 --output runs
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

Each alternate task order needs its own seed-17 first-task embedding reference:

```bash
talc train --variant talc --tasks fog,low_light,rain,snow --stage-limit 1 --pairs-per-task 0 --seed 17 --output runs/embedding_fog
talc select-memory --reference runs/embedding_fog/stage1.pt --tasks fog,low_light,rain,snow --max-pairs 64 --output data/replay_plan_fog.json
talc train --variant talc --tasks snow,rain,low_light,fog --stage-limit 1 --pairs-per-task 0 --seed 17 --output runs/embedding_snow
talc select-memory --reference runs/embedding_snow/stage1.pt --tasks snow,rain,low_light,fog --max-pairs 64 --output data/replay_plan_snow.json
```

Create these plans before running the `orders` suite. Each order uses a common reference and plan across its methods and seeds.

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

`per_image.csv` records restored and input PSNR/SSIM/LPIPS, their differences, scene/task identity and available routing probabilities. `summary.json` gives task means and their equal-weight mean. `evaluate-matrix` also writes the lower-triangular PSNR task matrix, `stageN_per_image.csv` for earlier stages, and final router accuracy/ECE. Checkpoint method, seed, configuration and stage identities are checked before forming the matrix. Uniform routing reports its chance reference separately from learned classification accuracy. `--no-lpips` omits LPIPS explicitly.

For a matrix R with stage t and test task j:

- **AP** is the mean final-row PSNR over tasks.
- **F** averages `max(R[j:T−1,j]) − R[T,j]` over earlier tasks. The final stage is excluded from the maximum.
- **BWT** averages `R[T,j] − R[j,j]` over earlier tasks.
- Eight-stage reports additionally give final PSNR on the original four tasks, their loss relative to stage four, and mean PSNR on the new-severity tasks.

PSNR is computed per image on RGB [0,1], with no border crop. SSIM uses an 11×11 Gaussian window, σ 1.5, population covariance and five-pixel reflection padding for a full-size map. LPIPS uses AlexNet v0.1 on RGB [−1,1], with `normalize=False` and `spatial=False`. Positive ΔPSNR/ΔSSIM and negative ΔLPIPS indicate improvement over the degraded input. RealRain L/H summaries use equal density weights. The density prefix in each image ID prevents collisions while the scene field retains shared L/H scene identity.

`--save-images` saves native float32 `ID.npy` predictions and PNG previews. Scoring and perception commands prefer the `.npy` prediction when both exist, preserving the restoration output before image quantization.

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

Both segmenters stay frozen. Default evaluation uses native resolution, ImageNet normalization once and right/bottom zero padding in normalized space to a multiple of 32. CUDA convolution uses FP16 by default; interpolation and reductions use FP32. `--precision fp32` selects FP32 evaluation. Logits are resized to the padded input dimensions, unpadded, and returned to the label dimensions. `--short-side` enables an explicitly resized evaluation, which should be reported separately. For each condition, intersections/unions are pooled over images before computing 19-class mIoU. The main average gives each condition weight 1/4. Pooled-archive class IoUs and their differences are reported separately in `segmentation.json`.

For any external restoration method, replace `--checkpoint` with `--predictions DIRECTORY` containing HWC float32 `ID.npy` or RGB `ID.png` files. Use `--seed` to identify separately trained external front ends. This also enables the paper's offline front-end comparisons with a common frozen evaluator.

After evaluating the three restoration seeds with the same frozen segmenter, aggregate run-wise condition means and class IoUs:

```bash
talc compare --kind segmentation --results runs/talc/seed17/segformer/segmentation.json runs/talc/seed29/segformer/segmentation.json runs/talc/seed43/segformer/segmentation.json --names TALC TALC TALC --output runs/comparisons/segformer
```

Replace `segformer` with `deeplab` for the second segmenter. Add results from other front ends and give each method its own name to build a comparison.

### Object detection

The detection entry point covers the appendix's Faster R-CNN ResNet-50-FPN protocol. Convert Cityscapes instance maps to COCO boxes; the eight classes are person, rider, car, truck, bus, train, motorcycle and bicycle. Group instances are COCO crowd regions.

```bash
talc prepare-detection --manifest data/weather/train_rain.jsonl --output data/detector_train.json
talc prepare-detection --manifest data/weather/val_rain.jsonl --output data/detector_val.json
talc train-detector --manifest data/weather/train_rain.jsonl --annotations data/detector_train.json --validation-manifest data/weather/val_rain.jsonl --validation-annotations data/detector_val.json --epochs 12 --output checkpoints/cityscapes_detector.pt
talc prepare-detection --manifests data/weather/test_rain.jsonl data/weather/test_fog.jsonl data/weather/test_snow.jsonl data/weather/test_low_light.jsonl --output data/detector_test.json
talc detection --manifests data/weather/test_rain.jsonl data/weather/test_fog.jsonl data/weather/test_snow.jsonl data/weather/test_low_light.jsonl --annotations data/detector_test.json --detector-weights checkpoints/cityscapes_detector.pt --checkpoint runs/talc/seed17/stage4.pt --output runs/talc/seed17/detection
```

Detector training reads the clean targets and collapses repeated weather variants to one record per clean scene. It fine-tunes COCO-initialized Faster R-CNN on the 2,475 clean training scenes, selects the checkpoint by AP on the 500 clean internal-validation scenes, and saves the nine-label state (background plus eight classes). Training and evaluation use the same FrozenBatchNorm backbone. Existing weights in this format can be supplied directly.

Group regions are split into 8-connected crowd components. Box area is the original `width×height`, including inclusive pixel extrema, rather than mask area. Evaluation discards boxes whose centers lie on the matched semantic ignore mask. The detector applies its own normalization once, short-edge resize 800/long-edge cap 1,333, score threshold 0.001, NMS 0.5, 100 detections/image and 1,000 pre-/post-NMS RPN proposals. COCO evaluation uses IoU 0.50:0.05:0.95, 101 recall points and maxDets 1/10/100. It reports AP50:95, AP50, APsmall, person AP and car AP, their input-relative differences in percentage points, and equal-condition means. These four-weather manifests contain 2,000 views of 500 base scenes. Use `--manifest` to evaluate one weather separately. A metric with no supported ground-truth instances is reported as null.

## 8. External restoration references

Run [TransWeather](https://github.com/jeya-maria-jose/TransWeather), [WGWS](https://github.com/zhuyr97/WGWS-Net), [LDR](https://github.com/noxsine/LDR), [Histoformer](https://github.com/sunshangquan/Histoformer), [SemiDDM-Weather](https://github.com/longfafffa/SemiDDM-Weather), or [DRR-Weather](https://github.com/GZHU-DVL/DRR-Weather) with their released code and weights. Preserve the manifest image ID as the prediction filename, RGB color order and original geometry.

Export synthetic or external inputs for those models:

```bash
talc export-data --manifest data/weather/test_mixed.jsonl --output data/export/mixed
talc export-data --manifest data/realrain_L.jsonl --output data/export/realrain_L
```

The exported `inputs/ID.png` and, for paired data, `targets/ID.png` share the scoring manifest's IDs. PNG export quantizes inputs to 8-bit RGB; use the same exported inputs when comparing front ends through file-based interfaces. Score the real prediction files after running each external model:

```bash
talc score-predictions --name Histoformer --manifests data/realrain_L.jsonl data/realrain_H.jsonl --predictions predictions/histoformer --output runs/references/histoformer
talc compare --kind restoration --results runs/talc/seed17/realrain/summary.json runs/references/histoformer/summary.json --names TALC Histoformer --reference TALC --output runs/comparisons/realrain
```

Use the manifest ID, including the density prefix, as each prediction filename. Predictions may be HWC float32 `.npy` arrays in [0,1] or RGB PNGs. The same command scores mixtures and any other aligned manifest. The comparison checks image/task identities and exports TALC-minus-reference gaps for both densities and their equal-weight mean. Shared L/H scene identity remains in the per-image file; training seeds are the units used for sample standard deviations. External references retain their own architectures and data access. `segmentation --predictions` and `detection --predictions` provide the corresponding downstream comparisons. External implementations and weights remain outside this package.

## 9. Inference, complexity and plots

Restore an image, an image tree, or a manifest:

```bash
talc infer --checkpoint runs/talc/seed17/stage4.pt --input data/example.png --output predictions/talc
talc infer --checkpoint runs/talc/seed17/stage4.pt --manifest data/weather/test_rain.jsonl --output predictions/talc_rain
```

Measure the selected checkpoint on the target device:

```bash
talc profile --checkpoint runs/talc/seed17/stage4.pt --manifest data/weather/test_rain.jsonl --height 512 --width 1024 --precision fp16 --warmup 100 --frames 500 --repeats 3 --telemetry --device-setup "AC power; performance profile; automatic boost" --ambient-temperature 25 --output runs/talc/seed17/profile
```

`profile.json` contains stored parameters, FP32 parameter storage in decimal MB, paired replay in MiB, active convolution/linear FLOPs, per-frame latency, mean/median/p95 latency, FPS, peak CUDA allocation and runtime/device metadata. Routing probe work and the full sparse restoration pass are included. Separate-expert profiling sums storage across the saved bank and activates the expert matching the timing scene's task. One MAC counts as two FLOPs; normalization, activation and other elementwise operations are excluded from this FLOP count.

The lexicographically first rain test frame is resized once to 1024×512 and preloaded. Each of three repeats uses 100 warm-up and 500 synchronized forwards; FPS uses the pooled 1,500 wall-clock latencies. cuDNN autotuning and TF32 are disabled, deterministic cuDNN is enabled, and one CPU thread is used. Decoding, transfer and downstream models are excluded. `--telemetry` saves clock/power/temperature samples at 100 ms intervals using `nvidia-smi` or `tegrastats`. Set Orin's power mode and clocks on the device before measuring and record them with `--device-setup`. `--ambient-temperature` records the measured room temperature.

After evaluating each run, aggregate real results:

```bash
talc evaluate-matrix --run runs/talc/seed29 --output runs/talc/seed29/evaluation
talc evaluate-matrix --run runs/talc/seed43 --output runs/talc/seed43/evaluation
talc report --evaluations runs/talc/seed17/evaluation runs/talc/seed29/evaluation runs/talc/seed43/evaluation --output runs/figures/main
talc qualitative --manifest data/weather/test_mixed.jsonl --predictions runs/talc/seed17/mixed/images --names TALC-AWR --limit 4 --output runs/figures/qualitative
```

Train seeds 29/43 before evaluating them. Add baseline evaluation directories to `--evaluations` for comparison. AP/F/BWT are calculated within each run before means and sample standard deviations are computed. A single run has null standard deviation. `comparison.csv` and `comparison.json` contain the aggregates; plots include stage curves, task matrices and final comparisons in PNG and editable SVG.

Use `evaluate-matrix --name memory0` or another experiment name when comparing runs of the same variant with different memory budgets or task orders. Reports group by that name and reject duplicate seeds within a group.

`compare` also accepts `--kind routing`, `detection` or `profile`, with the corresponding `summary.json`, `detection.json` or `profile.json` result files. Give one name per file; repeated names aggregate distinct seeds. Evaluation settings and image identities must match within a group and for reference comparisons. Results are averaged within each run before reporting the mean and sample standard deviation; unsupported metrics and single-run deviations are not filled with zero.

```bash
talc plot-comparison --comparison runs/comparisons/realrain/comparison.json --output runs/figures/realrain
talc plot-comparison --comparison runs/comparisons/segformer/comparison.json --output runs/figures/segformer_comparison
```

Use `--metrics` to select specific scalar metrics. For storage/forgetting and FPS/PSNR plots, pass a profile comparison together with `--restoration-comparison` pointing to a restoration comparison using the same experiment names.

## Experiment coverage

| Paper experiment | Commands | Outputs and metrics |
| --- | --- | --- |
| Main continual comparison | `sweep --suite main`, `evaluate-matrix`, `report` | Task matrices, PSNR/SSIM/LPIPS, AP/F/BWT, seed means and sample SD |
| Components and capacity controls | `sweep --suite ablations` / `capacity`, `evaluate-matrix`, `segmentation`, `profile` | Restoration, router accuracy, mIoU, parameters and memory |
| Memory and task order | `sweep --suite memory` / `orders`, `evaluate-matrix` | Final quality, forgetting, BWT and paired storage |
| Mixed weather / E3 diagnostics | `evaluate`, `diagnostics`, `plot-diagnostics` | Familiar/shifted mixture scores, four routing policies, both-component recall, ECE, confusion, blur gain and entropy AUROC |
| E1 RealRain and recent references | `evaluate`, `score-predictions`, `compare --kind restoration` | Per-density absolute scores, input-relative gains and equal-weight TALC-minus-reference gaps |
| E2 selective-update controls | `sweep --suite capacity`, `evaluate-matrix` | Matched backbone, old-expert unfreezing and distillation-removal comparisons |
| E4 eight-stage growth | `sweep --suite eight`, `evaluate-matrix`, `profile` | Original-four-task retention/loss, new-severity quality, eight-task matrix, stored growth and throughput |
| E5 second frozen segmenter | `segmentation --segmenter deeplab`, `compare --kind segmentation` | Condition mIoU, macro-condition mean, pooled class IoUs and gains |
| Device efficiency | `profile`, `compare --kind profile`, `plot-comparison` | Model storage, active convolution/linear FLOPs, replay MiB, latency, FPS and device telemetry |
| Appendix frozen detection | `prepare-detection`, `train-detector`, `detection`, `compare --kind detection` | AP50:95, AP50, APsmall, person/car AP, weather means and input-relative changes |

## Implementation conventions

The shared widths are 32/64/128/256, encoder depths 3/3/5/8, and shallow-to-deep decoder depths 3/2/5. Independent adapters have ranks 8/16/32/64 and 4/3/7/8 sites per scale. The router uses a shared-only two-level probe followed by a 96→1664→T MLP; the restoration pass recomputes the prefix with routed adapters. The four-task model has 6,487,012 parameters under this configuration.

After stage one, trainable shared parameters are channel normalization affine parameters, decoder pointwise projections and the RGB reconstruction head. Router parameters and the new expert also update. Stem, encoder convolution weights, scale changes, skip fusion, depthwise kernels, residual scales and old experts stay frozen unless an intervention explicitly unfreezes old experts. New adapter up-projections start at zero; adding a router logit can still change the whole model's output.

The restoration objective is elementwise Charbonnier with ε=1e−3 plus 0.2×(1−SSIM). Replay is a separate 128×128 forward with its own mean loss. Distillation is replay-output L1 plus 0.05 times the sum of four feature-scale L1 means, multiplied by 0.5. Router cross-entropy is one mean over current and replay labels, weighted by 0.1. Orthogonality uses normalized residuals at the last adapter site of each scale, removes the Gram diagonal, detaches old columns and averages over ordered expert pairs; its coefficient is 0.01. These terms are omitted at their empty-memory or single-expert boundaries.

Split seed 2025 is independent of training seeds. For lexicographically indexed clean scene `i`, split code `a=0/1/2`, weather–severity ID `t=1…8`, variant `v`, and component `c`, degradation seeds are `41001 + 10^9*a + 10^7*t + 100*i + 10*v + c`. Components use separate PCG64 streams. Candidate crop seeds are `4000073001 + 10^7*t + 100*i + 10*v + crop_id`. Current/replay augmentation streams use `90000+seed` and `91000+seed`, respectively. These identities are retained in manifests, replay plans or checkpoints.

Fog depth uses `(disparity_uint16−1)/256` and camera focal length/baseline. Missing pixels use the valid median of their 8-connected semantic region, then the frame median, then 80 m when the frame has no valid depth; final depth is clipped to 1–80 m. Rain bands use length multipliers 0.5/1/2, widths 1/2/3 px, counts 600/300/100 per 2048×1024 frame scaled by area, brightness U[0.6,1), and Gaussian σ 0.3/0.6/1.0. Snow assigns 90% of particles to Gaussian-blurred disks and 10% to 9-pixel vertical streaks. Low light uses `(RGB×exposure)^gamma`, Poisson peak 256 saved in each manifest record, and independent Gaussian read noise. Synthetic inputs are float RGB; replay uses half-up uint8 quantization. Padding is right/bottom reflection to a multiple of 8 and is removed before float32 clamping/scoring; very small images use edge replication when reflection is undefined. Perfect PSNR is capped at 120 dB for finite serialization. PNG export quantizes images; float32 `.npy` predictions preserve the native scoring interface.

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
