# Data Science Projects Portfolio 📊

<div align="center">

**Collection of data science projects applied to real-world problems**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![Deep Learning](https://img.shields.io/badge/DL-PyTorch%20%7C%20YOLOv8-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Apps-Streamlit-FF4B4B.svg)](https://streamlit.io/)

[About](#-about-the-portfolio) •
[Projects](#-projects) •
[Technologies](#-tech-stack) •
[Interactive Apps](#-interactive-applications)

</div>

---

## 📖 About the Portfolio

This repository brings together a collection of data science projects that address real-world problems through data-driven methodologies. Each project combines:

- 🔍 **Exploratory Data Analysis (EDA)**
- 📊 **Statistical Insights**
- 🤖 **Machine Learning Techniques**
- 💡 **Practical Solutions and Predictive Models**

Each project is fully documented and includes datasets, analysis notebooks, and detailed documentation.

---

## 🚀 Projects

### 1. ✈️ Airlines Flights Analysis and Prediction

**Objective**: Predict flight prices for Indian airlines

**Technologies**: Pandas, RandomForest, XGBoost, SHAP

**Dataset**: Flight bookings with features including origin/destination, schedules, duration, and prices

**Key Results**:
- XGBoost model with **R² = 0.9883**
- RMSE ≈ 2,454 | MAE ≈ 1,258
- Key variables: ticket class, days remaining, duration, and stops

**Use Cases**:
- Dynamic pricing optimization
- Booking strategies for consumers
- Competitive airline analysis

📁 [View Project →](https://github.com/Acquarts/data-science-projects/tree/main/Projects/AirlinesFlights_AnalysisAndPrediction)

---

### 2. 🧠 Brain Tumor Detection

**Objective**: Automatic detection of brain tumors in MRI images

**Technologies**: YOLOv8n, PyTorch, Ultralytics, Matplotlib

**Dataset**: 893 training images + 223 validation images with bounding boxes

**Key Results**:
- **Recall: 82%** - High tumor detection rate
- Inference: **~1ms per image**
- Model optimized for medical screening

**Potential Applications**:
- Automated medical screening
- Diagnostic support for radiologists
- Large-scale medical dataset analysis
- Telemedicine in resource-limited areas

📁 [View Project →](https://github.com/Acquarts/data-science-projects/tree/main/Projects/BrainTumorDetection)

---

### 3. 🏠 California Housing Prices - PCA & t-SNE

**Objective**: Analyze California housing prices using dimensionality reduction techniques

**Technologies**: PCA (Principal Component Analysis), t-SNE, scikit-learn

**Approach**:
- Dimensionality reduction for visualization
- Pattern identification in real estate data
- Comparison of dimensionality reduction techniques

**Applications**:
- Real estate valuation
- Market segmentation
- Identification of price-influencing factors

📁 [View Project →](https://github.com/Acquarts/data-science-projects/tree/main/Projects/CaliforniaHousingPrices_PCA_tSNE)

---

### 4. 🏎️ Ferrari Recommendation System

**Objective**: Specialized recommendation system for Ferrari models

**Technologies**: Recommendation systems, collaborative filtering

**Features**:
- Personalized recommendations
- User preference analysis
- Suggestion optimization

📁 [View Project →](https://github.com/Acquarts/data-science-projects/tree/main/Projects/Ferrari_RecommendationSystem)

---

### 5. 🌾 Smart Farming Sensor

**Objective**: Analysis of smart agricultural sensor data

**Technologies**: IoT, time series analysis, machine learning

**Applications**:
- Precision agriculture
- Resource optimization
- Prediction of optimal growing conditions
- Environmental monitoring

📁 [View Project →](https://github.com/Acquarts/data-science-projects/tree/main/Projects/SmartFarmingSensor)

---

### 6. 🎓 Students Academic Outcome APP

**Objective**: Web application to predict student academic risk

**Technologies**: Streamlit, scikit-learn, XGBoost, pickle

**Features**:
- ✅ Bilingual interactive web interface
- ✅ Instant predictions
- ✅ Analysis of academic, economic, and demographic variables
- ✅ Risk factor visualization

**Applications**:
- Early warning systems
- Preventive interventions
- Educational management
- Student support

📁 [View Project →](https://github.com/Acquarts/data-science-projects/tree/main/Projects/StudentsAcademicOutcomeAPP)

🌐 **[Interactive Demo Available]**

---

### 7. 📚 Students Dropout & Success

**Objective**: Analysis of student dropout and success

**Technologies**: Machine learning, predictive analytics

**Approach**:
- Identification of dropout patterns
- Factors influencing academic success
- Predictive retention models

**Applications**:
- Student retention policies
- Academic support programs
- Educational planning

📁 [View Project →](https://github.com/Acquarts/data-science-projects/tree/main/Projects/StudentsDropout&Success)

---

### 8. 🏥 Thyroid Cancer Recurrence

**Objective**: Prediction of thyroid cancer recurrence

**Technologies**: Machine learning, medical analysis

**Features**:
- Recurrence predictive models
- Risk factor analysis
- Clinical decision support

**Applications**:
- Predictive medicine
- Treatment planning
- Patient monitoring
- Oncological research

📁 [View Project →](https://github.com/Acquarts/data-science-projects/tree/main/Projects/ThyroidCancerRecurrence)

---

### 9. 🏥 Thyroid Cancer Recurrence APP

**Objective**: Interactive application for predicting thyroid cancer recurrence

**Technologies**: Streamlit, machine learning, data visualization

**Features**:
- Intuitive medical interface
- Real-time predictions
- Clinical factor analysis

🌐 **[Deployed Application Available]**

📁 [View Project →](https://github.com/Acquarts/data-science-projects/tree/main/Projects/ThyroidCancerRecurrenceAPP)

---

## 🛠️ Tech Stack

### Languages
- **Python 3.8+** - Primary language

### Machine Learning & Data Science
- **scikit-learn** - Classic ML algorithms
- **XGBoost** - Optimized gradient boosting
- **RandomForest** - Ensemble learning
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Visualization

### Deep Learning
- **PyTorch** - Deep learning framework
- **YOLOv8** - Object detection
- **Ultralytics** - YOLO tools

### Model Explainability
- **SHAP** - SHapley Additive exPlanations

### Application Development
- **Streamlit** - Interactive web apps
- **Jupyter Notebooks** - Exploratory analysis

### Advanced Techniques
- **PCA** - Dimensionality reduction
- **t-SNE** - Complex data visualization
- **Recommendation Systems**
- **Time Series**
- **Computer Vision**

---

## 🌐 Interactive Applications

This portfolio includes deployed web applications that allow interaction with trained models:

| Application | Description | Technology | Status |
|-------------|-------------|------------|--------|
| **Students Academic Outcome** | Academic risk prediction | Streamlit + XGBoost | ✅ Live |
| **Thyroid Cancer Recurrence** | Oncological recurrence prediction | Streamlit + ML | ✅ Live |

---

## 📊 Application Areas

Projects cover multiple domains:

- 🏥 **Healthcare**: Tumor detection, oncological prediction
- 🎓 **Education**: Student dropout, performance prediction
- ✈️ **Transportation**: Flight price prediction
- 🏠 **Real Estate**: Property valuation
- 🌾 **Agriculture**: Smart sensors, precision farming
- 🏎️ **Retail**: Recommendation systems

---

## 🎯 Methodology

Each project follows a consistent structure:

```
📁 ProjectName/
├── 📂 data/              # Datasets (CSV, images, etc.)
├── 📂 notebooks/         # Jupyter notebooks with analysis
├── 📂 models/            # Trained models (pickle, pt, etc.)
├── 📄 app.py             # Streamlit application (if applicable)
├── 📄 requirements.txt   # Project dependencies
└── 📄 README.md          # Detailed documentation
```

### Typical Pipeline

1. **Data Exploration (EDA)**
   - Descriptive statistical analysis
   - Distribution visualization
   - Pattern and outlier identification

2. **Preprocessing**
   - Data cleaning
   - Feature engineering
   - Normalization/standardization
   - Train/test split

3. **Modeling**
   - Algorithm selection
   - Model training
   - Cross-validation
   - Hyperparameter optimization

4. **Evaluation**
   - Performance metrics (RMSE, MAE, R², Accuracy, Recall, etc.)
   - Explainability analysis (SHAP)
   - Model comparison

5. **Deployment** (when applicable)
   - Web application development
   - Model serialization
   - Cloud deployment

---

## 📈 Success Metrics

### Regression Projects
- **R² (Coefficient of determination)**: up to 0.9883
- **RMSE/MAE**: Optimized per context

### Classification Projects
- **Recall**: up to 82% in medical detection
- **Precision & F1-Score**: Balanced per use case

### Deployed Applications
- **Latency**: <1s for predictions
- **Availability**: 99%+ uptime

---

## 🚀 How to Use This Repository

### Clone the Repository

```bash
git clone https://github.com/Acquarts/data-science-projects.git
cd data-science-projects/Projects
```

### Explore a Specific Project

```bash
cd ProjectName
pip install -r requirements.txt
jupyter notebook  # To explore notebooks
```

### Run an Application

```bash
cd StudentsAcademicOutcomeAPP
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔮 Future Projects

- 🔄 Advanced recommendation systems
- 🗣️ Natural Language Processing (NLP)
- 📸 Applied computer vision
- ⏰ Complex time series analysis
- 🎯 AutoML and pipeline optimization

---

## 📚 Learning Resources

Each project includes educational documentation covering:

- Theoretical problem foundations
- Design decisions and justification
- Results interpretation
- Lessons learned
- Bibliographic references

---

## 🤝 Contributions

Contributions are welcome. If you have ideas for improvements or new projects:

1. Fork the repository
2. Create a branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

**GitHub**: [@Acquarts](https://github.com/Acquarts)
**Repository**: [data-science-projects](https://github.com/Acquarts/data-science-projects)

---

## ⭐ Acknowledgments

Special thanks to the open source community and the following tools:

- [scikit-learn](https://scikit-learn.org/)
- [PyTorch](https://pytorch.org/)
- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

---

<div align="center">

**Constantly evolving portfolio | Data science applied to real-world problems**

⭐ If you find this portfolio useful, consider giving it a star

[⬆ Back to top](#data-science-projects-portfolio-)

</div>
