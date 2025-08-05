UNet Image Segmentation – Project Report

📌 Hyperparameters

Tried

* Image Size: 128×128, 256×256
Rationale: Smaller sizes trained faster but lost fine details; 256×256 chosen for better boundary preservation.

* Batch Size: 4, 8
Rationale: Larger batches gave smoother convergence but were limited by GPU memory.

* Learning Rate: 1e-4
Rationale: Higher learning rates caused oscillations; 1e-4 provided stable training without overshooting.

* Weight Decay: 1e-5
Rationale: Helped generalization without slowing convergence.

* Loss Functions: CrossEntropy, Dice, BCE + Dice (α=0.5)

Rationale: Dice loss improved small-object segmentation; BCE+Dice gave the best overall IoU.

Final Settings

image_size: (256, 256)

batch_size: 8

num_epochs: 50

learning_rate: 1e-4

weight_decay: 1e-5

loss_function: BCE + Dice (alpha=0.5)

optimizer: Adam

scheduler: ReduceLROnPlateau

🏗 Architecture

Base Model: UNet with encoder–decoder skip connections.

* UNet Autoencoder–Decoder with encoder–decoder skip connections.

* Encoder: 4 downsampling blocks (Conv → BN → ReLU → Conv → BN → ReLU → MaxPool).

* Decoder: 4 upsampling blocks (TransposeConv / Bilinear → Concatenate skip → Conv layers).

Conditioning Method Experiments

I experimented with multiple conditioning strategies before finalizing the color-index approach:

1. Text-to-Vector (Hugging Face Tokenizer)

* Used a pretrained tokenizer + embedding model to map text labels to vectors.

* Issue: Instead of segmenting, the model started generating new images.

2. One-Hot Encoding of Text Labels

* Preserved discrete class information effectively.

* Issue: Increased input dimensionality significantly, causing slow training.

3. Final Approach – Numeric Color Index + Reverse Color Mapping

* Assigned each color a numeric index (e.g., 0 = red, 1 = blue, etc.).

* Added the index as a normalized conditioning channel:
* Used reverse color mapping in post-processing to restore original colors.


✅ Benefits:

* Lightweight, minimal computation overhead.

* Stable segmentation without unintended image generation.

* Easy to interpret and debug.

Conditioning: Bilinear upsampling enabled (bilinear=True) for smoother feature maps.

Augmentation Challenges

* Dataset had very simple shapes (mostly circles and squares).

* Augmentations (flip, rotation, brightness/contrast) applied, but image diversity did not increase significantly.

* Only 2–3 images changed meaningfully due to shape invariance to transformations.

* Brightness/contrast changes sometimes caused blurriness without adding useful variation.


📈  Training Dynamics

* Loss: BCE+Dice decreased steadily, plateaued around epoch 35.

* Metrics: IoU reached ~0.86 on validation.

* Qualitative progression:

* Early: Large blobs, poor boundaries.

* Mid: Boundaries sharpened, but some colors misclassified.

* Late: Clean separation of shapes, colors respected.

 Key Learnings
 
* Learned how to segment images containing multiple color text using a conditioning approach.
* Successfully combined two loss functions (BCE + Dice) to improve IoU and small-object detection.
* Uploaded trained model to the Hugging Face Model Hub for sharing and reproducibility.
* Skip connections are critical for retaining fine spatial details.
* Loss choice has a greater impact on small-object segmentation than minor architecture tweaks.

Skip connections are critical for retaining fine spatial details.

Loss choice has a greater impact on small-object segmentation than minor architecture tweaks.
