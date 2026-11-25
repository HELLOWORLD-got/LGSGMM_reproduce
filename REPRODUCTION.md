# MDMT reproduction

## Data and entry points

The supplied dataset can be consumed directly through `lgsgmm.data.MDMTDataset`.
The legacy four-directory TXT loader cannot read these JSONL manifests.
The adapter keeps image files in place, removes invisible identity-union placeholders,
converts normalized center boxes into pixel corner boxes, and retains shared IDs.
Occluded boxes remain candidates when their annotations mark them visible.
Categories use the class-agnostic labels in the supplied manifests.

Validated split counts are 12,026 training pairs across 25 sequences, 1,930 validation
pairs across 5 sequences, and 5,883 official test pairs across 14 sequences.
The official test manifest has 600,923 boxes. The supplied XML-derived `test.jsonl`
has 601,270 boxes, so `configs/mdmt.json` selects `test_official.jsonl` explicitly.
XML frame indices refer to positions in numerically sorted image lists. For example,
validation sequence 22 starts at JPEG 00000201 while its XML frame starts at zero.

```bash
python -m pip install -r requirements.txt
python tools/audit_mdmt.py
python -m unittest discover -s tests -v
python -m lgsgmm.run smoke
```

The current Anaconda Python environment already supplies the dependencies and a
cached torchvision ResNet-50 DEFAULT checkpoint. The system `python3` environment
has different packages. Run these commands with the intended environment's `python`.
For a fresh CUDA environment, install a matching torch/torchvision CUDA build before
installing the remaining requirements.

Regenerate manifests after changing raw data with the following command. It preserves
images and atomically replaces train/val/test_official manifests. Official test TXT
files must be present under the existing official_test directory.

```bash
python tools/prepare_mdmt.py
python tools/audit_mdmt.py
```

## Training and evaluation

```bash
python train_main_resnet.py --device cuda:0
python train_main_resnet.py --device cuda:0 --resume runs/mdmt_paper/last.pt
python -m lgsgmm.run evaluate --device cuda:0 --checkpoint runs/mdmt_paper/best.pt
```

All configuration paths are relative to the repository root. Set `CUDA_VISIBLE_DEVICES`
externally to choose the physical GPU. The implementation supports one CUDA device, CPU, and torchrun DDP. Batch size is
the global number of image pairs per optimizer update and must be divisible by the
process count. Pair losses accumulate locally, with gradients synchronized at the
last pair of each local batch. BatchNorm is updated per view on each rank, with
rank-zero buffers synchronized before validation. SyncBatchNorm is not enabled.

The default run keeps original 1920x1080 input, uses seven attention layers, six
neighbors, Adam at 1e-4, 200 epochs, and ReduceLROnPlateau with patience 5 and factor
0.5. The batch size 4 and scheduler come from the supplied training script. The
repository README's recommendation of batch size 8 applies to its updated model.
Change `--batch-size` or the JSON configuration to conduct that experiment.

`--max-side` is intended for resource-limited checks and changes the input protocol.
The smoke command uses at most two pairs per split at 160 pixels by default.
Smoke metrics describe execution checks and have no benchmark significance.

Training saves `last.pt`, `best.pt`, `history.jsonl` and the resolved configuration.
Checkpoints contain optimizer, scheduler, AMP scaler and Python/NumPy/torch RNG states.
Best selection uses validation one-stage sequence-macro MDA. Validation RNG draws are
restored before the next training epoch. Resume requires matching model, loss,
postprocessing, batch size, image size and sampling settings. Use `--epochs` to extend
a run. The `--resume` checkpoint must be a trusted checkpoint produced by this code.

Evaluation strictly reloads the checkpoint architecture, loss and postprocessing
configuration. It exports per-pair TP/FP/FN/GT and latency to CSV, plus per-sequence,
pair-macro, sequence-macro and micro metrics to JSON. The sequence-macro MDA follows
the averaging hierarchy of the MDMT official evaluator. A pair with no GT and no
predictions contributes zero. Network latency excludes image decoding and transfer,
uses CUDA synchronization, and includes feature extraction and enhancement.
The initial measured pair includes cold-start overhead.

The one-stage and two-stage results are evaluated on the same candidate GT boxes.
The detector/ByteTrack-based tracking experiment in paper Table II requires separate
YOLOv8 detections, temporal tracking outputs, and MOTA/IDF1 evaluation. This repository
currently provides the association experiments corresponding to Tables I/III and
module ablations. M3Track images and converted manifests must be supplied separately.

## Paper implementation choices

`lgsgmm/model.py` implements PGA equations (1)-(4), local normalized graph Laplacian
spectra (5)-(8), spectrum-guided cross-attention (9)-(10), cosine similarity and
bidirectional softmax (11). ResNet-50 feature-map cropping/average pooling reuses
`zz/all_nn/roi_pooling.py`, supplied with corner-derived top-left coordinates.
Both cross-attention directions consume the same pre-cross-attention features.
Small graphs use the available neighbors and zero-pad their spectra. Empty graphs
are supported without artificial targets.

The following choices fill details that the paper does not fully specify, or resolve
ambiguities between the paper and released code. They are recorded here for comparison
with author-provided settings, and exact table reproduction remains unverified.

- Descriptor dimension 256, four heads, shared weights across views, learned MLP
  positional encoding, and FFN hidden LayerNorm follow the available model's scale.
- Local spectra use all e+1 eigenvalues, zero self-edges and a mean-neighbor-distance
  kernel scale. PDM uses the mean off-diagonal inverse distance and unit self-weight.
- Backbone input uses ImageNet normalization with torchvision DEFAULT weights.
  The released legacy input loader only scales RGB to [0,1].
- BCE supervises the full valid association matrix, including missed true pairs.
  Squared-distance triplet loss uses shared-ID positives and random different-ID
  negatives from the other view. Loss weights are 0.6/0.3 and margin is 1.
- S-GMM implements the log-space version of equation (19), followed by weighted
  similarity-transform updates, sigma and outlier updates, and one-to-one posterior
  filtering. Shared coordinate centering/scaling makes sigma2=1 numerically usable.
  The transform family, initialization and coordinate convention are implementation
  choices. The paper's equation (16) and equation (19) differ in their Gaussian
  normalization factors. Equation (19) is the implemented E-step.
- An initial score below 0.5 contributes to the estimated outlier proportion. The
  paper's experimental prose calls scores below 0.5 inliers, which conflicts with
  its association-confidence definition. This ambiguity needs author clarification.
- Homography estimation uses RANSAC on S-GMM inliers, then supplements unmatched
  candidates by one-to-one projected-center assignment. The 20/80 pixel thresholds
  follow the supplied evaluation code. Degenerate homographies preserve filtered
  associations and return an explicit status.
- MDA uses TP/(GT+FP+FN), matching the MDMT official implementation. Paper equation
  (31) and the released trainer differ from this definition.

Use `--no-pga` and `--no-lga` for module ablations. To test plain Re-ID, use a copied
configuration with `model.layers=0`. Use `evaluate --one-stage` to omit geometric
refinement. Train each architecture separately and keep its output directory distinct.
The paper's MDMT reference MDA values are 39.29% one-stage and 45.77% two-stage.

The new paper model initializes its backbone through torchvision and does not consume
the repository's LightGlue checkpoint. The five original `.pth` files remain LFS
pointer files. To use the released updated model or author test weights, fetch the
real LFS objects and use a compatible legacy architecture. Author checkpoints are
not structurally compatible with the newly implemented paper model.

## Verification performed

All image paths, image dimensions, graph array lengths, visible boxes, identity
uniqueness and sequence split isolation were checked across the full supplied data.
Raw XML and official test TXT were independently converted in a temporary directory.
The resulting sequence/frame/ID/label/size/box/visibility fields matched all existing
train, val and test_official rows exactly.

Regression tests cover empty/unequal graphs and backpropagation, small spectral graphs,
MDA false-match penalties, mutual one-to-one matching and known geometric transforms.
CPU smoke training, two-stage evaluation, checkpoint save/reload and resume were run.
The resumed second epoch matched uninterrupted two-epoch training for every model
parameter and buffer. Full-resolution checks are recorded in `runs/`.

## Four A6000 GPUs

From the server project directory, activate the Python environment and run the small
check first. The script selects visible devices 0,1,2,3 and launches four NCCL DDP
processes. An existing CUDA_VISIBLE_DEVICES environment variable takes precedence.

```bash
bash scripts/train_4gpu.sh smoke
```

The small configuration is `configs/mdmt_4gpu_smoke.json`. It uses 16 training pairs,
8 validation pairs, one epoch, maximum image side 640, seven attention layers,
FP32 and global batch size 4 (one pair per rank). Its backbone initializes randomly
to avoid checkpoint downloads during the execution check. It runs both association
stages and checks saved weight reload. It writes into `runs/mdmt_4gpu_smoke/`.
`Smoke passed` confirms completion. Startup logs list all four rank/device mappings.

```bash
# Increase the small check to original input resolution.
bash scripts/train_4gpu.sh smoke --max-side 1920 --output-dir runs/mdmt_4gpu_smoke_fullres

# Train the complete dataset at original resolution using ImageNet initialization.
bash scripts/train_4gpu.sh train

# Continue the full run, preserving per-rank RNG and optimizer state.
bash scripts/train_4gpu.sh train --resume runs/mdmt_4gpu/last.pt

# Evaluate a full-training checkpoint, using all four GPUs.
bash scripts/train_4gpu.sh evaluate --checkpoint runs/mdmt_4gpu/best.pt
```

The full configuration is `configs/mdmt_4gpu.json`, with 200 epochs, global batch 4,
2 loading workers per process and original image resolution. To use two pairs per
rank, pass `--batch-size 8`. Single-process commands remain supported. Resume requires
the same process count as the saved run. A four-rank checkpoint can be evaluated in
one process because saved parameter names do not have a DDP `module.` prefix.

The training DistributedSampler reshuffles with the epoch and pads a split by at most
world_size-1 pairs to give all ranks equal training steps. Validation and test use an
unpadded sampler. Counts and rows are gathered before computing dataset metrics, so
uneven splits and ranks with no evaluation examples are handled correctly. Rank zero
alone writes checkpoints, configurations, logs and CSV results. Each checkpoint stores
all ranks' RNG states and the process count.

Local checks use four CPU/Gloo processes. The four-A6000 CUDA/NCCL run is to be checked
on the server. Single-GPU and four-GPU training can differ because per-rank random
sampling and BatchNorm updates differ.
