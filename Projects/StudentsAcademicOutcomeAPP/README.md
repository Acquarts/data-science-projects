# Student Academic Outcome Prediction App

A functional prototype of an academic risk prediction tool built with Streamlit that enables users to simulate student profiles and receive instant predictions about their academic outcomes.

## 📋 Overview

This application predicts student academic outcomes using machine learning, classifying students into three categories:
- 🔴 **Dropout** - Students at risk of dropping out
- 🟡 **Enrolled** - Students currently active in their studies
- 🟢 **Graduate** - Students predicted to successfully complete their program

The tool analyzes student data across academic, economic, and demographic dimensions, functioning as both a demonstration and testing platform for educational institutions.

## ✨ Features

- **Interactive Student Profile Simulation**: Create custom student profiles with various academic and demographic parameters
- **Real-Time Predictions**: Instant classification using a pre-trained XGBoost model
- **Bilingual Interface**: Full support for both English and Spanish languages
- **User-Friendly Design**: Intuitive web interface with dynamic form fields
- **Two-Tab Layout**:
  - Information tab with project context
  - Prediction tab with the forecasting tool

## 🛠️ Technologies Used

- **Framework**: [Streamlit](https://streamlit.io/) - Interactive web interface
- **Machine Learning**:
  - [scikit-learn](https://scikit-learn.org/) - Data preprocessing and model utilities
  - [XGBoost](https://xgboost.readthedocs.io/) - Gradient boosting classifier
- **Data Processing**:
  - [Pandas](https://pandas.pydata.org/) - Data manipulation
  - [NumPy](https://numpy.org/) - Numerical computing
- **Model Persistence**: [Joblib](https://joblib.readthedocs.io/) - Model serialization
- **Language**: Python 3.x

## 📊 Model Input Features

The prediction model analyzes seven key variables:

1. **Curricular units 2nd sem (approved)**: Number of subjects passed in the 2nd semester
2. **Curricular units 2nd sem (grade)**: Grade average for 2nd semester coursework
3. **Curricular units 2nd sem (evaluations)**: Total evaluations completed in 2nd semester
4. **Admission grade**: Student's admission score
5. **Tuition fees up to date**: Payment status (binary: Yes/No)
6. **Age at enrollment**: Student's age when enrolling
7. **Previous qualification (grade)**: Grade from previous educational qualification

## 📁 Project Structure

```
StudentsAcademicOutcomeAPP/
├── Model_XGBoost_Classifier.pkl    # Pre-trained XGBoost model
├── XGB_Clf_Trained.ipynb           # Model training notebook
├── main.py                          # Streamlit application entry point
├── df_final.csv                     # Training dataset
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

## 🚀 Installation & Setup

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/Acquarts/data-science-projects.git
   cd data-science-projects/Projects/StudentsAcademicOutcomeAPP
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify required files**

   Ensure these files are in the project directory:
   - `Model_XGBoost_Classifier.pkl`
   - `df_final.csv`
   - `main.py`

4. **Run the application**
   ```bash
   streamlit run main.py
   ```

5. **Access the app**

   Open your browser and navigate to:
   ```
   http://localhost:8501
   ```

## 💻 Usage

1. **Select Language**: Choose between English or Spanish at the top of the page
2. **Navigate to Prediction Tab**: Click on the "Prediction" tab
3. **Enter Student Data**:
   - Fill in all required fields using the interactive form
   - Binary features use dropdown menus
   - Continuous variables use sliders or number inputs
4. **Get Prediction**: The model will instantly classify the student outcome
5. **Interpret Results**: Review the prediction with its corresponding emoji indicator

## 📦 Dependencies

```
streamlit
pandas
numpy
scikit-learn
xgboost
joblib
```

## 🔬 Model Details

- **Algorithm**: XGBoost Classifier
- **Training Data**: Historical student academic records (`df_final.csv`)
- **Output Classes**: 3 categories (Dropout, Enrolled, Graduate)
- **Model File**: `Model_XGBoost_Classifier.pkl`

## 📝 Development

The model training process is documented in `XGB_Clf_Trained.ipynb`, which includes:
- Data exploration and preprocessing
- Feature engineering
- Model training and hyperparameter tuning
- Performance evaluation
- Model serialization

## 🎯 Use Cases

This application can be utilized in various educational contexts:

- **Early Warning Systems**: Identify at-risk students before they drop out
- **Preventive Interventions**: Allocate resources to students who need them most
- **Educational Management**: Data-driven decision making for academic administrators
- **Student Support Services**: Personalized guidance and support programs
- **Policy Development**: Inform retention and success strategies

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

This project is part of the [Data Science Projects](https://github.com/Acquarts/data-science-projects) repository.

## 👤 Author

**Acquarts**
- GitHub: [@Acquarts](https://github.com/Acquarts)

## 🙏 Acknowledgments

- Built as part of a data science portfolio project
- Designed to help educational institutions identify at-risk students early
- Aims to support student retention and success initiatives

---

**Note**: This is a prototype application for demonstration purposes. For production use in educational settings, additional validation, testing, and ethical considerations should be implemented.
