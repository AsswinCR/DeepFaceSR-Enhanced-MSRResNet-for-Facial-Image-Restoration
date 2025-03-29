# DeepFaceSR: Advanced Face Super-Resolution

## Project Overview

This repository contains the implementation of advanced deep learning models for Blind Face Super Resolution, developed for the course AI6126 Advanced Computer Vision at Nanyang Technological University.

## Introduction

Blind Face Super Resolution aims to enhance low-quality (LQ) face images to high-quality (HQ) versions without prior knowledge of the specific degradation process. This task is particularly challenging as real-world low-quality images can suffer from various types of degradation including blur, noise, compression artifacts, and low resolution.

<div align="center">
  <img width="626" alt="Super-Resolution Example" src="https://github.com/user-attachments/assets/9ca43485-8e89-4f10-a33f-e8a7b1183a62" />
  <p><em>Example of low-quality face images (left) and their corresponding super-resolved outputs (right)</em></p>
</div>

## Dataset

The project utilizes the FFHQ (Flickr-Faces-HQ) dataset:
- 5,000 high-quality (HQ) face images for training
- 400 HQ-LQ image pairs for validation
- LQ images pre-processed to 128×128 pixels through a second-order degradation pipeline
- Target HQ resolution: 512×512 pixels

The degradation pipeline includes:
- Gaussian blur
- Downsampling
- Noise addition
- Compression artifacts

## Methodology

### Model Architectures

<div align="center">
  <img width="969" alt="Model Architecture" src="https://github.com/user-attachments/assets/60cdbcf2-675c-4b1f-9d0e-ad5c7533ad39" />
  <p><em>Left: Comparison of SRResNet and MSRResNet blocks. Right: Overall EDSR Architecture</em></p>
</div>

Two main model architectures were explored:

1. **Modified SRResNet (MSRResNet)**
   - A variant of SRResNet with Batch Normalization (BN) layers removed
   - Designed to preserve high-frequency details essential for image clarity
   - Experimented with different numbers of residual blocks (16, 20, 24)
   - 3 input and 3 output channels for RGB image processing
   - 64 feature maps per convolutional layer to capture diverse features
   - 4× upscaling factor (128×128 → 512×512 pixels)

2. **Enhanced Deep Super-Resolution Network (EDSR)**
   - BN layers removed to avoid artifacts and preserve fine details
   - 3 input/output channels for RGB images
   - 64 feature maps across convolutional layers
   - 24 residual blocks to enhance detail capture
   - 4× upscaling factor

### Data Augmentation

To improve model robustness, various image augmentations were applied:
- 50% chance of horizontal flip
- Rotation by 90, 180, or 270 degrees
- 50% probability of brightness adjustment (0.7 to 1.3 factor)
- 50% chance of contrast adjustment (0.7 to 1.3 factor)

### Optimization Strategy

- **Optimizer**: Adam with learning rate 2×10⁻⁴, beta coefficients (0.9, 0.99)
- **Learning Rate Scheduler**: Cosine Annealing with two periods of 150,000 iterations
- **Loss Function**: L1 loss
- **Evaluation Metric**: Peak Signal-to-Noise Ratio (PSNR)
- Model checkpoints saved based on highest PSNR value

## Experiments & Results

Multiple model configurations were trained and evaluated:

| Architecture | Augmentation | Parameters | Loss | Iterations Trained | Best Val PSNR |
|--------------|--------------|------------|------|-------------------|--------------|
| MSRResNet-B16 | No | 1,517,571 | L1 | 125,000 | 26.51530 |
| MSRResNet-B16 | Yes | 1,517,571 | L1 | 100,000 | 26.44951 |
| EDSR B-16 | No | 1,517,571 | L1 | 80,000 | 26.420882 |
| MSRResNet-B24 | Yes | 2,108,419 | L1 | 55,000 | 26.508414 |
| MSRResNet-B20 | Yes | 1,812,995 | L1 | 75,000 | 26.561710 |

**Best Model Performance**: MSRResNet-B20 with augmentation achieved 26.63871 PSNR on the test set.

<div align="center">
  <img width="783" alt="Training Curves" src="https://github.com/user-attachments/assets/32a53bcc-099d-41ba-b386-2ba825096a10" />
  <p><em>Training curves showing PSNR (left) and L1 Loss (right) for different model configurations</em></p>
</div>

### Key Findings:

1. **Optimal Network Depth**: MSRResNet-B20 (20 residual blocks) provided the best balance between model complexity and performance.
2. **Importance of Augmentation**: Models with augmentation strategies generally achieved higher PSNR values.
3. **L1 Loss Effectiveness**: L1 loss proved effective for guiding the super-resolution process.
4. **Architecture Considerations**: The specific configuration of the network design significantly impacts super-resolution performance.

<div align="center">
  <img width="1018" alt="Visual Results" src="https://github.com/user-attachments/assets/b9624d89-0cc2-4ed4-a3aa-9ff3d8bbdd05" />
  <p><em>Visual comparison of low-quality inputs (left) and super-resolved outputs (right) from our best model</em></p>
</div>

## Repository Structure

```
├── assets/               # Images for README documentation
├── code_files/
│   ├── evaluate.py       # Evaluation script
│   ├── inferrold.py      # Inference script
│   ├── *.yml             # Configuration file
│   └── ffsub_dataset.py  # Dataset handling
├── Obtained Test Images/ # Test images from best model
├── lj07_CodaLab_score.png # CodaLab score screenshot
├── Report.pdf            # Detailed project report
└── README.md             # This file
```



## Computing Environment

The models were trained on the SCSE GPU Cluster with the following specifications:

| Cluster | GPU | CPU | Memory | MaxWall |
|---------|-----|-----|--------|---------|
| q_amsai | 7 | 1 | 12G | 8Hrs |
| q_dmsai | 10 | 1 | 30G | 8Hrs |

## Usage Instructions

### For Training:

1. Ensure all file paths are correctly set according to your machine configuration
2. Verify your cluster setup is correct
3. Run the training script:
   ```
   sbatch run.sh
   ```

### For Testing:

1. Use the trained model weights
2. Ensure file paths are correctly set
3. Run the testing script:
   ```
   sbatch test.sh
   ```

## References

1. Wang, Xintao, et al. "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data." Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops. 2021.
2. Wang, Xintao, et al. "GFP-GAN: Towards Real-World Blind Face Restoration with Generative Facial Prior." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.
3. [Real-ESRGAN GitHub Repository](https://github.com/xinntao/Real-ESRGAN/tree/master)
4. [Single Image Super Resolution Challenge](https://bozliu.medium.com/single-image-super-resolution-challenge-6f4835e5a156)
5. [BasicSR GitHub Repository](https://github.com/XPixelGroup/BasicSR)
