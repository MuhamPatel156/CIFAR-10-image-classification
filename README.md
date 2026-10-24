# CIFAR-10 Image Classification with PyTorch

An end-to-end image-classification project comparing a fully connected neural
network with a custom convolutional neural network (CNN), followed by data
augmentation, held-out evaluation, GPU profiling, and a mixed-precision training
microbenchmark.

**Selected model:** custom CNN with random-crop and horizontal-flip augmentation.  
**Test accuracy:** 67.94% on 10,000 test images.  
**Macro F1:** 0.6783.  
**Hardware:** NVIDIA Tesla T4 on Google Colab.

## Notebook

Open [CIFAR10_PyTorch_Complete_Project.ipynb](CIFAR10_PyTorch_Complete_Project.ipynb)
to inspect the implementation, training logs, charts, and saved evaluation results.
The notebook contains the complete implementation and does not require a
separate source-code archive.

## Project workflow

1. Load and normalize CIFAR-10 images using torchvision.
2. Create disjoint training and validation subsets with a fixed random seed.
3. Train an MLP baseline and a custom CNN from random initialization.
4. Compare the same CNN with and without training-data augmentation.
5. Select the model and checkpoint using validation accuracy.
6. Evaluate the selected model on the held-out test set.
7. Analyse per-class metrics, a confusion matrix, and incorrect predictions.
8. Benchmark FP32 versus automatic mixed-precision training steps.
9. Profile GPU operations with `torch.profiler`.

## Dataset and experimental setup

CIFAR-10 contains 32 x 32 RGB images across ten classes: airplane, automobile,
bird, cat, deer, dog, frog, horse, ship, and truck.

| Setting | Value |
|---|---|
| Training images | 45,000 |
| Validation images | 5,000 |
| Test images | 10,000 |
| Random seed | 42 |
| Training batch size | 128 |
| DataLoader workers | 2 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 0 |
| Loss | Cross-entropy |
| Main classification experiments | FP32 |
| Augmentation | Random crop with padding 4; random horizontal flip |

Augmentation is applied only to training images. Validation and test images
use normalization without random augmentation. The test set is evaluated after
validation-based model selection.

## Model architectures

### MLP baseline

```text
3 x 32 x 32 image
  -> Flatten (3072 features)
  -> Linear(3072, 256) -> ReLU
  -> Linear(256, 128)  -> ReLU
  -> Linear(128, 10)
```

**Trainable parameters:** 820,874.

### Custom CNN

```text
3 x 32 x 32 image
  -> Conv(3, 32, 3x3)   -> ReLU
  -> Conv(32, 32, 3x3)  -> ReLU -> MaxPool
  -> Conv(32, 64, 3x3)  -> ReLU -> MaxPool
  -> Conv(64, 128, 3x3) -> ReLU -> MaxPool
  -> Adaptive average pooling
  -> Flatten -> Linear(128, 10)
```

All convolutions use padding 1. **Trainable parameters:** 103,786.
Both models output raw logits for `CrossEntropyLoss`.

## Classification results

These values are taken from the executed notebook outputs.

| Experiment | Epochs | Best validation accuracy | Training time* |
|---|---:|---:|---:|
| MLP baseline | 3 | 50.82% | 29.00 s |
| CNN without augmentation | 8 | 68.22% | 82.00 s |
| CNN with augmentation | 8 | **68.76%** | 115.63 s |

\*Sum of recorded training-epoch times. Includes the training-loop data loading
and device transfers; excludes validation, checkpoint writing, and dataset download.

Augmentation increased the CNN's best validation accuracy by **0.54 percentage
points** in this run. The MLP and CNN were trained for different epoch counts,
so their comparison is not an equal-training-budget experiment.

### Final held-out evaluation

The augmented CNN was selected using validation accuracy. Its selected checkpoint
was from epoch 8.

| Metric | Test result |
|---|---:|
| Accuracy | **67.94%** |
| Macro F1 | **0.6783** |
| Cross-entropy loss | 0.9115 |
| Correct predictions | 6,794 / 10,000 |

Per-class F1 scores range from **0.5040 for cat** to **0.8257 for automobile**.
The confusion matrix records 184 dog images predicted as cat and 156 cat images
predicted as dog, illustrating an important failure case.

## Mixed-precision training microbenchmark

A separate experiment compares FP32 with automatic mixed precision using FP16
autocasting and gradient scaling. Both modes use the same initial CNN weights,
batch size 128, a fresh optimizer per repeat, ten warm-up steps, and thirty timed
training steps per repeat. Each mode is measured three times.

| Mode | Median training throughput | Peak PyTorch-allocated GPU memory |
|---|---:|---:|
| FP32 | 8,281.3 images/s | 0.3402 GiB |
| AMP FP16 | 26,178.1 images/s | 0.0624 GiB |

In this microbenchmark, AMP achieved **3.16x throughput** with approximately
**81.65% less peak allocated memory**.

These measurements use a fixed synthetic batch already on the GPU. They exclude
data loading, augmentation, and host-to-device transfers. They do not establish
a 3.16x end-to-end training speedup or equivalent final model accuracy under AMP.
The classification results above come from FP32 training.

## GPU profiling

The executed notebook includes a `torch.profiler` run and exports a Chrome trace.
In the displayed profile, `aten::convolution_backward` accounts for approximately
55.11% of self CUDA time. Profiling is used for operation-level inspection;
the separate unprofiled microbenchmark supplies throughput measurements.

## Run in Google Colab

1. Open the notebook in Google Colab.
2. Select a GPU runtime under **Runtime > Change runtime type**.
3. Run the cells in order. CIFAR-10 downloads automatically.
4. Review the training curves, final test report, and performance results.
5. Run `download_results()` in a new cell to export metrics and charts.

The completed run used:

```text
GPU:          Tesla T4
PyTorch:      2.11.0+cu130
torchvision:  0.26.0+cu130
CUDA build:   13.0
```

The implementation uses PyTorch, torchvision, NumPy, Matplotlib, tqdm, and Pillow.
Use a compatible PyTorch/torchvision pair; Colab's existing installation was used
for the recorded run. Exact timings may differ on another runtime.

The notebook saves results under `/content/cifar10_project_results` in Colab.
Download them before deleting the runtime, or configure `OUT` to use persistent
storage before training. Checkpoint resume is implemented at epoch boundaries;
retain the same settings when resuming an existing experiment.

## Generated artifacts

| File | Contents |
|---|---|
| `environment.json` | Runtime versions, GPU, and configuration |
| `split_indices.json` | Training and validation indices |
| `experiment_summary.json` / `.csv` | Validation results and training times |
| `learning_curves.png` | Training and validation loss/accuracy curves |
| `final_test_report.json` | Test metrics, per-class scores, and confusion matrix |
| `confusion_matrix.png` | Visual test-set confusion matrix |
| `incorrect_predictions.png` | First twelve incorrect test predictions |
| `precision_microbenchmark.json` | Repeated FP32/AMP performance measurements |
| `training_trace.json` | Profiler trace |
| `inference_model.pt` | Selected model weights and preprocessing metadata |

The results-download helper exports JSON, CSV, and PNG files, excluding the dataset
and model checkpoints. Retain model weights separately if you need inference later.
Once `inference_model.pt` is available, the notebook's `predict_image(path)` helper
returns the top three CIFAR-10 predictions for an image.

## Scope and limitations

- Results are from one random seed; they are not a multi-seed statistical study.
- Accuracy reflects short training budgets, not a state-of-the-art CIFAR-10 claim.
- Transfer learning and tiny-subset debugging are implemented as optional code
  but were disabled in this recorded run.
- Single-image inference is implemented; no example prediction call is recorded
  in the saved outputs.
- Incorrect-image plots show the first errors, not a random sample of all errors.
- Softmax scores are not calibrated probabilities of correctness.
- GPU microbenchmark results apply to the stated workload and environment.
