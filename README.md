# CIFAR-10 Image Classification with a CNN

A convolutional neural network built in **PyTorch** to classify images from the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset into 10 categories. The project explores how **learning rate, batch size, dropout, network depth, L2 regularization, and early-stopping patience** affect model performance, through a series of controlled hyperparameter experiments.


---

## Results at a Glance

| Best single run | Validation accuracy |
|---|---|
| Patience = 15 (LR 0.001, batch 128, dropout 0.3, 3 conv blocks) | **82.25%** |

**Recommended configuration:** `LR = 0.001`, `batch size = 128`, `dropout = 0.3`, `3 conv blocks`, `patience = 15`

---

## Dataset

- **CIFAR-10**: 50,000 training images (32×32 RGB), downloaded automatically via `torchvision`
- **Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Split:** 40,000 training / 10,000 validation
- Perfectly balanced (5,000 images per class), so **accuracy** is used as the evaluation metric

## Project Structure

The notebook is organized into seven sections:

1. **Loading Data and Libraries**
2. **Exploratory Data Analysis** — class distribution, sample images per class, per-channel pixel statistics and distributions
3. **Data Augmentation Pipeline**
4. **CNN Model**
5. **Model Training**
6. **Early Stopping**
7. **Hyperparameter Experiments** — with a written analysis of each

## Methodology

### Exploratory Data Analysis
Per-channel means and standard deviations were computed from the training data and used for input normalization:

| Channel | Mean | Std |
|---|---|---|
| Red | 0.4914 | 0.2470 |
| Green | 0.4822 | 0.2435 |
| Blue | 0.4465 | 0.2616 |

### Data Augmentation (training set only)
- **Random Horizontal Flip** (p = 0.5): teaches the model that left/right orientation doesn't change the class
- **Random Crop** (32×32 with 4-px padding): teaches the model that object position doesn't matter
- **Color Jitter** (brightness, contrast, saturation ±0.2): makes the model robust to lighting conditions

The validation set uses normalization only, with no augmentation.

### Model Architecture
A configurable CNN with 2 or 3 convolutional blocks:

```
Input (3×32×32)
 → [Conv 3×3, 32]  → BatchNorm → ReLU → MaxPool   → 32×16×16
 → [Conv 3×3, 64]  → BatchNorm → ReLU → MaxPool   → 64×8×8
 → [Conv 3×3, 128] → BatchNorm → ReLU → MaxPool   → 128×4×4   (3-block only)
 → Flatten → FC(256) → ReLU → Dropout → FC(10)
```

### Training
- **Loss:** Cross-entropy
- **Optimizer:** Adam (optional L2 via `weight_decay`)
- **Epochs:** up to 30
- **Early stopping:** monitors validation loss and restores the best weights

## Hyperparameter Experiments

Each experiment changes one hyperparameter from the baseline (`LR=0.001, batch=128, dropout=0.3, 3 blocks, L2=0, patience=7`).

| Experiment | Setting | Best Val Acc |
|---|---|---|
| **Learning rate** | 0.01 | 0.7638 |
| | **0.001** | **0.8192** |
| | 0.0001 | 0.7741 |
| **Batch size** | 32 | 0.8203 |
| | **128** | **0.8216** |
| | 512 | 0.7940 |
| **Dropout** | 0.0 | 0.8149 |
| | **0.3** | **0.8184** |
| | 0.6 | 0.7884 |
| **Network depth** | 2 blocks | 0.7648 |
| | **3 blocks** | **0.8165** |
| | 3 blocks + L2 (1e-4) | 0.8165 |
| **Early stopping patience** | 3 | 0.8082 |
| | 7 | 0.8104 |
| | **15** | **0.8225** |

### Key Takeaways
- **Learning rate:** 0.01 overshoots and fails to converge well; 0.0001 learns too slowly within 30 epochs. 0.001 is the best balance.
- **Batch size:** Very large batches (512) hurt generalization. 32 and 128 perform similarly, with 128 slightly ahead.
- **Dropout:** Moderate dropout (0.3) helps a little; heavy dropout (0.6) removes too much information.
- **Depth:** Depth matters most. The 3-block model beats the 2-block model by about 5 points. L2 regularization added nothing, which suggests the model wasn't significantly overfitting.
- **Patience:** Validation loss can plateau temporarily before improving again, so a longer patience (15) let the model find a better solution.

