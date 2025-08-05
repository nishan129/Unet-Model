UNet Image Segmentation – Project Report
📌 Hyperparameters
Tried

Image Size: 128×128, 256×256

Rationale: Smaller sizes train faster but lose detail; 256×256 chosen for better boundary preservation.

Batch Size: 4, 8,

Rationale: Larger batches improved stability but were limited by GPU memory.

Learning Rate:  1e-4

Rationale: High LR sped up convergence but caused oscillations; 1e-4 gave smooth, stable training.

Weight Decay:  1e-5 

Rationale: Small regularization improved generalization without slowing training.

Loss Functions: CrossEntropy, Dice, BCE + Dice (α=0.5)

Rationale: Dice improved small-object segmentation; BCE+Dice gave best overall IoU.

Augmentation: Random flips, rotations, brightness/contrast

Rationale: Addressed overfitting and improved robustness to spatial variation.

Final Settings
yaml
Copy
Edit
image_size: (256, 256)
batch_size: 8
num_epochs: 50
learning_rate: 1e-4
weight_decay: 1e-5
loss_function: BCE + Dice (alpha=0.5)
augmentations: Horizontal/Vertical Flip, Rotation (±15°), Random Brightness/Contrast
optimizer: Adam
scheduler: ReduceLROnPlateau
🏗 Architecture
Base Model: UNet with encoder–decoder skip connections.

Encoder: 4 downsampling blocks (Conv → BN → ReLU → Conv → BN → ReLU → MaxPool).

Decoder: 4 upsampling blocks (TransposeConv → Concatenate skip → Conv layers).

Conditioning: Bilinear upsampling enabled (bilinear=True) for smoother feature maps.

Ablations:

Removed skip connections → significant drop in fine-edge accuracy.

Tried deeper network → better small-object segmentation but slower inference.

Compared bilinear vs transposed convolution upsampling → bilinear was smoother, reduced checkerboard artifacts.

📈 Training Dynamics
Loss Curves:

BCE+Dice loss decreased smoothly, plateaued around epoch 35.

Early high variance in validation loss reduced after adding augmentations.

Metric Trends (IoU / Dice Score):

IoU improved steadily; plateau ~0.86 validation IoU.

Qualitative Output Trends:

Early epochs: Coarse blob segmentation.

Mid-training: Shape boundaries start forming, but some small objects missed.

Late training: Clean edges, better object separation, minimal over-segmentation.

Failure Modes & Fixes

Issue	Cause	Fix
Missing small objects	Class imbalance	Weighted BCE + Dice loss
Jagged edges	Transposed convolution artifacts	Switched to bilinear upsampling
Overfitting	Small dataset	Heavy augmentation + weight decay

💡 Key Learnings
Loss choice matters more than architecture tweaks for small-object detection.

Skip connections are critical for retaining fine spatial details.

Augmentation + small weight decay significantly improve generalization on small datasets.

Monitoring qualitative samples is as important as metric curves for diagnosing failure modes.

Bilinear upsampling often yields cleaner edges than naive transpose convolutions.

