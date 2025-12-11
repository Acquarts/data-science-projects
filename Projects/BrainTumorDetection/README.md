# Brain Tumor Detection with YOLOv8

## Project Description

This project implements a brain tumor detection system using YOLOv8 (You Only Look Once version 8) to automatically identify the presence of tumors in brain MRI images.

## Dataset

- **Name**: Brain Tumor Dataset
- **Classes**: 
  - `0: negative` - No tumor
  - `1: positive` - With tumor
- **Split**:
  - **Training**: 893 images
  - **Validation**: 223 images
- **Format**: Object detection with bounding boxes

## Model Architecture

- **Base model**: YOLOv8n (nano) - Lightweight and fast version
- **Parameters**: 3,011,238 trainable parameters
- **GFLOPs**: 8.2
- **Layers**: 129 layers in total

## Training Configuration

```python
model = YOLO('yolov8n.pt')
model.train(
    data='datasets/brain-tumor.yaml',
    epochs=50,
    imgsz=640,
    batch=8,
    device=0  # GPU
)
```

### Hyperparameters

- **Epochs**: 50
- **Image size**: 640x640 pixels
- **Batch size**: 8
- **Optimizer**: AdamW (lr=0.001667, momentum=0.9)
- **Data augmentation**: Enabled (mosaic, mixup, albumentations)

## Training Results

### Final Metrics
- **mAP@50**: 49.9% - Mean average precision at IoU 0.5
- **mAP@50-95**: 36.7% - Mean average precision across IoU range 0.5-0.95
- **Precision**: 42.0% - Proportion of correct detections
- **Recall**: 82.0% - Proportion of detected tumors

### Metrics by Class
| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|-----------|--------|--------|-----------|
| Negative | 54.5% | 70.8% | 59.1% | 43.7% |
| Positive | 29.6% | 93.1% | 40.8% | 29.7% |

### Performance
- **Inference speed**: ~1ms per image
- **Training time**: 0.130 hours (7.8 minutes)

## Results Analysis

### Strengths
- **High Recall (82%)**: The model detects most present tumors
- **Speed**: Very fast inference, suitable for real-time applications
- **Efficiency**: Lightweight model with few parameters

### Areas for Improvement
- **Moderate Precision (42%)**: Generates some false positives
- **Class imbalance**: The "positive" class has lower precision
- **Small dataset**: Only 893 training images

## Installation and Usage

### Requirements
```bash
pip install ultralytics torch torchvision matplotlib
```

### Training
```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n.pt')

# Train
results = model.train(
    data='path/to/brain-tumor.yaml',
    epochs=50,
    imgsz=640,
    batch=8
)
```

### Inference
```python
# Load trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Predict on new image
results = model.predict('path/to/image.jpg', conf=0.25)

# Visualize results
import matplotlib.pyplot as plt
plt.imshow(results[0].plot())
plt.show()
```

## Project Structure

```
brain-tumor-detection/
├── datasets/
│   └── brain-tumor/
│       ├── train/
│       ├── valid/
│       └── brain-tumor.yaml
├── runs/
│   └── detect/
│       └── train/
│           ├── weights/
│           │   ├── best.pt
│           │   └── last.pt
│           └── results.png
└── README.md
```

## Potential Applications

- **Medical screening**: Early detection of brain tumors
- **Diagnostic assistance**: Support for radiologists in MRI interpretation
- **Research**: Automated analysis of large medical datasets
- **Telemedicine**: Remote diagnosis in resource-limited areas

## Ethical and Medical Considerations

⚠️ **Important**: This model is for educational and research purposes only. It should not be used as a medical diagnostic tool without proper professional supervision.

## Future Work

1. **Dataset expansion**: Include more images and case diversity
2. **Architecture improvement**: Test YOLOv8s or YOLOv8m for higher precision
3. **Segmentation**: Implement semantic tumor segmentation
4. **Clinical validation**: Evaluation with medical specialists
5. **Optimization**: Quantization and pruning for mobile deployment

## References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com)
- [Brain Tumor Dataset](https://github.com/ultralytics/assets/releases/download/v0.0.0/brain-tumor.zip)

## License

This project uses the AGPL-3.0 license from Ultralytics YOLO.
