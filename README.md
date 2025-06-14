# DeepSub: Deep Learning for Thermal Background Subtraction in Heavy-Ion Collisions

## Overview

This is the official repository for DeepSub, a novel deep learning approach for background subtraction in heavy-ion collisions. DeepSub employs a Swin Transformer-based architecture for image restoration to process jet images and separate the hard jet signal from the thermal background. The model is trained to preserve jet substructure observables while effectively removing the underlying event background.

## Abstract

Jet reconstruction in an ultra-relativistic heavy-ion collision suffers from a notoriously large thermal background. Traditional background subtraction methods struggle to remove this soft background while preserving the jet's hard substructure. In this study, we present DeepSub, the first machine learning-based approach for full-event background subtraction. DeepSub utilizes Swin Transformer layers to denoise jet images and effectively disentangle hard jets from the heavy-ion background. DeepSub significantly outperforms existing subtraction techniques on key observables, achieving sub-percent to percent-level closure on distributions of jet $p_\mathrm{T}$, mass, girth, and energy correlation functions. As such, DeepSub paves the way for precision measurements in heavy-ion collisions.

## Key Highlights

- First machine learning-based approach for full-event background subtraction.
- Utilizes Swin Transformer architecture for effective image processing.
- Achieves sub-percent to percent level closure on key jet observables:
  - Jet $p_\mathrm{T}$.
  - Jet mass.
  - Jet girth.
  - Energy correlation functions.
- As such, preserves jet substructure while removing thermal background.

## Technical Implementation

### Architecture

The model is based on the Swin Transformer architecture with the following key components:
- Input: Noisy jet images (64x64 pixels).
- Swin Transformer layers with:
  - 6 stages of depth.
  - 180 embedding dimensions.
  - 6 attention heads per stage.
  - Window size of 8.
- Output: Denoised jet images.

### Pre-trained Models

The training process maintains:
- Regular checkpoints: `models/model_{epoch}`
- Best model: `models/best_model` (saved when validation loss improves)

## Usage

1. Prepare your datasets:
   - Training data: `datasets/train.pt`
   - Validation data: `datasets/val.pt`

2. Run the training:
   ```bash
   python train.py
   ```

## Dependencies

- pytorch torchvision
- timm
- tqdm
- numpy

## Acknowledgements

This project is based on SwinTR ([paper](https://arxiv.org/abs/2103.14030), [code](https://github.com/microsoft/Swin-Transformer)) and SwinIR ([paper](https://arxiv.org/abs/2108.10257), [code](https://github.com/JingyunLiang/SwinIR)). Thanks for their outstanding work and code ❤️.

## Citation

If you use this code in your research, please cite our paper:
```
[Citation information to be added]
```

## License

[License information to be added]

## Contact

[Contact information to be added]
