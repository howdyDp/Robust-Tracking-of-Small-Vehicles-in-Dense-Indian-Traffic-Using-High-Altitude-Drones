# Object Detection and Tracking with DETR, DINO, and Particle Filters

This repository contains the codebase for an advanced object detection and tracking pipeline, developed as a MTP. It leverages state-of-the-art transformer-based object detectors like DETR and DINO, and integrates them with robust tracking algorithms including ByteTrack and custom GPU-optimized Particle Filters.

## Dataset

The custom dataset used for training, fine-tuning, and evaluating the models in this project can be accessed here:
[Project Dataset (Google Drive)](https://drive.google.com/drive/folders/1qADpdZjQMXsAaU6J_jIvtJrCxzrT9D0U?usp=sharing)

## Project Structure & File Descriptions

The project is structured into four main Jupyter Notebooks, each responsible for a distinct part of the pipeline:

### 1. `ByteTrack_+_DETR.ipynb`
This notebook focuses on the integration of DETR for object detection with ByteTrack for multi-object tracking.
- **Dataset Conversion:** Includes scripts to convert annotations from CVAT JSON format into the standard COCO format.
- **DETR + ByteTrack Pipeline:** Uses a fine-tuned DETR model to generate bounding box detections on video frames, which are then passed to the BYTETracker for consistent ID assignment across frames.
- **Grid-Based Processing:** Implements a technique to process high-resolution videos/images by splitting frames into a grid (e.g., 3x3) to maintain detection accuracy on smaller objects.
- **Inference & Visualization:** Saves tracking results as JSON and generates debug videos with bounding boxes and track IDs.

### 2. `DINO_Finetuning.ipynb`
This notebook contains the code for fine-tuning the DINO (DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection) model.
- **Environment Setup:** Clones the official `IDEA-Research/DINO` repository and builds custom MultiScaleDeformableAttention CUDA operations.
- **CUDA Patching:** Patches C++ code to ensure compatibility with newer PyTorch versions (`value.type()` to `value.scalar_type()`).
- **Training Configuration:** Sets up a training loop using a ResNet-50 backbone with 5-scale features, customized for the specific dataset classes.

### 3. `Fine_Tuning_DETR.ipynb`
This notebook focuses on fine-tuning the base DETR model (`facebook/detr-resnet-50`) using PyTorch Lightning.
- **Dataset Loading:** Uses `torchvision.datasets.CocoDetection` to load training, validation, and test splits.
- **PyTorch Lightning Module:** Defines a custom `Detr` LightningModule for streamlined training, validation, and optimization using AdamW.
- **Checkpoints & Logging:** Integrates TensorBoard logging and automatic checkpoint saving based on validation performance.
- **Evaluation:** Evaluates the fine-tuned model using `pycocotools.coco.COCO` and `coco_eval.CocoEvaluator`, applying Non-Maximum Suppression (NMS) via the `supervision` library to filter overlapping detections.

### 4. `Particle_filter.ipynb`
This notebook is dedicated to exploring and implementing highly advanced, custom tracking algorithms based on Particle Filters, utilizing DETR and DINO for frame-by-frame detections.
It implements several iterations of the Particle Filter algorithm:
- **Vanilla Particle Filter:** A standard implementation using Euclidean distance and color histograms for particle weighting and state updates.
- **GPU-Optimized Repulsive Force Particle Filter (APF):** Introduces artificial potential fields where particles are repelled by other tracked objects to prevent track merging/ID switching. Entirely vectorized using PyTorch for fast GPU execution.
- **Boundary Heuristics:** Enhances the APF by incorporating boundary constraints and velocity-based predictions to handle objects entering and leaving the frame.
- **Weighted Anisotropic Distance:** Uses custom longitudinal and lateral distance weightings depending on the object's direction of motion to improve likelihood estimation.
- **Steering Physics:** Blends Interaction Over Union (IoU) matching with Euclidean distance and steering physics to create a highly robust tracker capable of handling severe occlusions and complex trajectories.

## Dependencies

Major libraries required to run this project:
- `torch`, `torchvision`, `pytorch-lightning`
- `transformers` (Hugging Face)
- `supervision`
- `opencv-python` (cv2)
- `numpy`, `scipy`
- `pycocotools`
- `loguru`, `tqdm`
- Custom built CUDA ops for DINO and ByteTrack.

## Usage

1. Open the notebooks in an environment like Google Colab or a local Jupyter server with GPU support (CUDA highly recommended).
2. Ensure the dataset is downloaded or mounted from Google Drive as expected by the paths in the notebooks.
3. Run the cells sequentially to install dependencies, compile custom operations, and execute the training/tracking pipelines.