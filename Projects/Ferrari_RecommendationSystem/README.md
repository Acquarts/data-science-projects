# Ferrari Image Recommendation System 🏎️

An intelligent image recommendation system for Ferrari vehicles using Deep Learning and cosine similarity.

## 📋 Description

This project implements a content-based recommendation system that uses the ResNet50 convolutional neural network pre-trained on ImageNet to extract features from Ferrari images and recommend similar vehicles based on cosine similarity.

## ✨ Features

- **🤖 Deep Learning**: Uses ResNet50 pre-trained on ImageNet
- **📊 Cosine Similarity**: Calculates similarities between feature vectors
- **🖼️ Visualization**: Displays recommendation results visually
- **⚡ Efficient Processing**: Optimized image dataset handling
- **🎯 High Accuracy**: Recommendations based on deep visual features

## 🛠️ Technologies

- **Python 3.x**
- **TensorFlow/Keras** - For ResNet50 model
- **NumPy** - Numerical operations
- **Pandas** - Data manipulation
- **Scikit-learn** - Cosine similarity
- **PIL (Pillow)** - Image processing
- **Matplotlib** - Visualization
- **tqdm** - Progress bars

## 📦 Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/ferrari-recommendation-system.git
cd ferrari-recommendation-system
```

2. **Install dependencies**
```bash
pip install tensorflow numpy pandas scikit-learn Pillow matplotlib tqdm
```

## 📁 Project Structure

```
ferrari-recommendation-system/
│
├── ferrari_recommendation.py          # Main code
├── ferrari_metadata.csv              # Dataset metadata
├── ferrari_dataset/                  # Image dataset
│   └── ferrari_images/
│       ├── 512/                     # Ferrari 512 models
│       ├── roma/                    # Ferrari Roma
│       ├── formula_1/               # F1 vehicles
│       └── ...
├── test_images/                      # Test images
│   ├── test_ferrari1.jpg
│   └── test_ferrari2.jpg
└── README.md
```

## 🚀 Usage

### 1. Initial Setup

```python
import os
import numpy as np
import pandas as pd
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
```

### 2. Dataset Processing

```python
# Configure paths
catalog_dir = "path/to/ferrari-dataset"
csv_path = "path/to/ferrari_metadata.csv"

# Process catalog images
df = pd.read_csv(csv_path)
features = []
image_names = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    img_name = row['image_path']
    img_path = os.path.join(catalog_dir, img_name)
    img_array = load_and_preprocess_image(img_path)
    if img_array is not None:
        feat = model.predict(img_array)[0]
        features.append(feat)
        image_names.append(img_name)
```

### 3. Generate Recommendations

```python
# Recommendation function
def recommend_similar_images(query_image_path, top_k=5):
    img_array = load_and_preprocess_image(query_image_path)
    query_vector = model.predict(img_array)[0].reshape(1, -1)
    similarity_scores = cosine_similarity(query_vector, np.array(features))[0]
    top_indices = np.argsort(similarity_scores)[::-1][:top_k]
    
    for i, idx in enumerate(top_indices):
        print(f"{i+1}. {image_names[idx]} — similarity: {similarity_scores[idx]:.4f}")

# Usage example
recommend_similar_images("path/to/query_image.jpg", top_k=5)
```

### 4. Results Visualization

```python
# Show recommendations visually
show_similar_images("path/to/query_image.jpg", top_k=5)
```

## 📊 Example Results

### Query: 1970 Ferrari 512
```
🔎 Query image: 1970_Ferrari_512_M_2.jpg

1. ferrari_dataset/ferrari_images/512/1970_Ferrari_512_M_2.jpg — similarity: 1.0000
2. ferrari_dataset/ferrari_images/512/1970_Ferrari_512_M_1.jpg — similarity: 0.8254
3. ferrari_dataset/ferrari_images/512/1970_Ferrari_512_M_3.jpg — similarity: 0.7996
4. ferrari_dataset/ferrari_images/512/1970_Ferrari_512_S_4.jpg — similarity: 0.7349
5. ferrari_dataset/ferrari_images/formula_1/2024_Ferrari_SF-24_2.jpg — similarity: 0.7319
```

### Query: 2024 Ferrari Roma Spider
```
🔎 Query image: 2024_Ferrari_Roma_Spider_1.jpg

1. ferrari_dataset/ferrari_images/roma/2024_Ferrari_Roma_Spider_1.jpg — similarity: 1.0000
2. ferrari_dataset/ferrari_images/roma/2024_Ferrari_Roma_Spider_3.jpg — similarity: 0.7566
3. ferrari_dataset/ferrari_images/roma/2024_Ferrari_Roma_Spider_2.jpg — similarity: 0.7505
4. ferrari_dataset/ferrari_images/roma/2024_Ferrari_Roma_Spider_6.jpg — similarity: 0.7405
5. ferrari_dataset/ferrari_images/roma/2024_Ferrari_Roma_Spider_4.jpg — similarity: 0.7358
```

## 🧠 Algorithm

1. **Feature Extraction**: Uses ResNet50 to extract 2048-dimensional feature vectors
2. **Cosine Similarity**: Calculates similarity between query vector and all dataset vectors
3. **Ranking**: Sorts results by descending similarity
4. **Top-K**: Returns the K most similar images

## ⚙️ Main Functions

- `load_and_preprocess_image()`: Loads and preprocesses images for the model
- `recommend_similar_images()`: Generates text recommendations
- `show_similar_images()`: Visualizes recommendations with matplotlib
- `recommend_from_path()`: Wrapper for recommendations from external files

## 📈 Performance

- **Processing**: ~100ms per image on GPU
- **Accuracy**: High visual similarity between recommendations
- **Scalability**: Efficient for datasets with thousands of images
- **Memory**: Compact feature vectors (2048 dim)

## 🔧 Advanced Configuration

### Adjust Model Parameters
```python
# Change input size
target_size = (224, 224)  # Standard ResNet50 size

# Adjust number of recommendations
top_k = 10  # Increase for more results
```

### GPU Optimization
```python
# Configure GPU (if available)
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
```

## 🤝 Contributions

Contributions are welcome. To contribute:

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is under the MIT License. See `LICENSE` for more details.

## 🙏 Acknowledgments

- **ImageNet** - Pre-training dataset
- **TensorFlow/Keras** - Deep Learning framework
- **Ferrari** - For creating such iconic vehicles
- **ResNet** - Neural network architecture

## 📧 Contact

Adrián Zambrana - [LinkedIn](https://www.linkedin.com/in/adrianzambranaacquaroni/) - info.aza.future@gmail.com

Project Link: [https://github.com/your-username/ferrari-recommendation-system](https://github.com/your-username/ferrari-recommendation-system)

---

**Made with ❤️ for Ferrari enthusiasts and Deep Learning practitioners**
