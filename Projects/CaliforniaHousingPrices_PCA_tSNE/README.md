# 🏠 Dimensionality Reduction and Clustering in California Housing Dataset

## 🎯 Objective
Explore the data structure of the **California Housing Dataset** through **dimensionality reduction** and **unsupervised clustering**, evaluating whether it's possible to identify patterns similar to the categorical variable `ocean_proximity` without using it directly in training.

---

## 📂 Dataset
- **Source**: California Housing Dataset.
- **Rows**: 20,640
- **Columns**: 9 (numerical + `ocean_proximity` categorical)
- **Target categorical variable (evaluation only)**: `ocean_proximity`.

---

## 🛠 Methodology

### 1. Preprocessing
- Encoding of categorical variable using `OneHotEncoder`.
- Scaling of numerical variables using `StandardScaler`.
- Unified pipeline with `ColumnTransformer`.

---

### 2. PCA (Principal Component Analysis)
- Reduction to **2 components**.
- **Cumulative explained variance**: ~62%.
- Visualization colored by `ocean_proximity` → high overlap.
- Clustering with **K-Means (K=5)** → **ARI = 0.125** → low similarity with real categories.

---

### 3. t-SNE (Non-Linear Dimensionality Reduction)
- Reduction to 2 dimensions preserving non-linear relationships.
- Initial parameterization (`perplexity=30`, `learning_rate=200`) showed greater visual separation.
- Hyperparameter optimization:
  - Best combination: **perplexity=30, learning_rate=500**.
  - **ARI with K-Means on optimized t-SNE = 0.419** → significant improvement over PCA.

---

### 4. Clustering (K-Means)
- **Number of clusters (K)**: equal to the number of real categories (5).
- Evaluation with **Adjusted Rand Index (ARI)** using `ocean_proximity` as reference.
- Contingency table → several clusters clearly represent specific categories (e.g., `INLAND`, `NEAR BAY`).

---

## 📊 Results

| Technique                    | ARI   | Observations |
|------------------------------|-------|---------------|
| PCA + K-Means                | 0.125 | Much overlap, insufficient linear structure. |
| Initial t-SNE + K-Means      | 0.341 | Notable improvement, more visually defined groups. |
| Optimized t-SNE + K-Means    | 0.419 | Better separation, several clusters aligned with real categories. |

---

## 📈 Comparative Visualization
- **Left**: Optimized t-SNE colored by real categories.  
- **Right**: Optimized t-SNE colored by K-Means clusters.  
*(Insert here the images generated in the notebook)*

---

## 🧠 Conclusions
- PCA is useful for quick visualization and reduction, but limited to linear patterns.
- t-SNE captures non-linear relationships, achieving better separation in this dataset.
- Even without using `ocean_proximity` for training, clustering detected part of its structure.
- ARI = 0.419 → moderate correlation between clusters and real categories.

---

## 🚀 Next Steps
- Test **UMAP** as a faster and more scalable alternative to t-SNE.
- Enrich features with geographic transformations to further improve clustering.
- Use more flexible clustering methods (Gaussian Mixtures, optimized DBSCAN).

---

## ⚙️ Execution Requirements
Install necessary dependencies:
```bash
pip install -r requirements.txt
```

---

## 📦 Main Dependencies
```
numpy
pandas
scikit-learn
matplotlib
seaborn
```

---

## 🔧 Usage

### Run the complete notebook:
```bash
jupyter notebook dimensionality_reduction_clustering.ipynb
```

### Run the main script:
```python
python main.py
```

---

## 📁 Project Structure
```
california-housing-clustering/
├── data/
│   └── housing.csv
├── notebooks/
│   └── dimensionality_reduction_clustering.ipynb
├── src/
│   ├── preprocessing.py
│   ├── dimensionality_reduction.py
│   └── clustering.py
├── results/
│   ├── pca_visualization.png
│   ├── tsne_visualization.png
│   └── comparison_plot.png
├── requirements.txt
└── README.md
```

---

## 📊 Key Metrics

### PCA
- **Explained Variance (2 components)**: 62%
- **ARI with K-Means**: 0.125

### t-SNE (Optimized)
- **Perplexity**: 30
- **Learning Rate**: 500
- **ARI with K-Means**: 0.419

---

## 🎓 Educational Value

This project demonstrates:
- Effective use of dimensionality reduction techniques
- Comparison between linear (PCA) and non-linear (t-SNE) methods
- Unsupervised clustering evaluation with ARI
- Hyperparameter optimization for t-SNE
- Data visualization best practices

---

## 👨‍💻 Author

Developed as part of Machine Learning coursework exploring unsupervised learning techniques.

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🔗 References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [t-SNE: A Tutorial](https://distill.pub/2016/misread-tsne/)
- [California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
