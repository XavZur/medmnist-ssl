## Week 4 – Self-Supervised Learning (SSL)

### Method Choice
For this stage, I implemented a SimCLR-lite self-supervised learning approach. I chose SimCLR-lite because it is conceptually simple, does not require a memory queue or momentum encoder, and is well-suited for smaller datasets like MedMNIST. Using a single encoder with a contrastive loss allowed me to focus on learning meaningful image representations before introducing labels, which aligns well with the goals of this week.

### Augmentations
Each image was augmented twice using a combination of random resized crops, horizontal flips, and small rotations. These augmentations are realistic for chest X-ray images, as they preserve the overall anatomical structure while encouraging the model to become invariant to small positional and orientation changes. Since PneumoniaMNIST images are grayscale, they were replicated across three channels to match the ResNet-18 input format.

### Hyperparameters
- Backbone: ResNet-18  
- Projection dimension: 128  
- Batch size: 64  
- Epochs: 30  
- Optimizer: AdamW  
- Learning rate: 3e-4  
- Temperature: 0.2
  
### Training Behavior
The contrastive loss decreased steadily over the course of training, starting around 1.03 and gradually declining to approximately 0.83 by the final epoch. The loss dropped relatively quickly in the early epochs and then decreased more slowly, suggesting that the encoder learned useful representations early on and continued refining them over time. Training was stable, and no major instabilities were observed, so no learning rate adjustments were necessary.

### Notes for Week 5–6
The trained encoder weights are saved as `ssl_encoder.pt` and will be reused for downstream supervised tasks in the next stages. Based on the smooth loss curve and stable convergence, the encoder does not appear to be severely undertrained or overtrained, although further evaluation using linear probing or fine-tuning will provide a clearer assessment of representation quality.
