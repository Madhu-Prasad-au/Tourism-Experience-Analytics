# Tourism Experience Analytics: Prediction & Recommendation System

An end-to-end **Machine Learning and Data Analytics project** that analyzes tourism transaction data and provides **attraction rating prediction, visitor behavior classification, and personalized attraction recommendations** through an interactive Streamlit application.

---

## 📌 Project Overview

Tourism platforms generate large amounts of user–attraction interaction data. This project uses historical tourism data to build machine learning models that can:

* Predict the **rating** a visitor may give to an attraction.
* Predict the visitor's **visit mode**.
* Recommend relevant attractions based on historical user preferences.
* Provide interactive analytics and predictions through a **Streamlit web application**.

The project covers the complete machine learning workflow:

**Data Collection → Data Cleaning → Data Integration → EDA → Feature Engineering → Model Training → Evaluation → Recommendation → Deployment**

---

## 🎯 Objectives

The main objectives of the project are:

1. Analyze tourism transaction and visitor behavior data.
2. Integrate multiple tourism-related datasets into a consolidated dataset.
3. Perform exploratory data analysis to identify tourism trends.
4. Build a regression model for attraction rating prediction.
5. Build a classification model for visit-mode prediction.
6. Develop a hybrid recommendation system for attractions.
7. Deploy the trained models through an interactive Streamlit application.

---

## 🗂️ Dataset

The project uses multiple related tourism datasets, including:

* **Transaction** – User-attraction interactions, ratings, visit year/month and visit mode.
* **User** – User-related information.
* **Item / Attraction** – Attraction details.
* **City** – City information.
* **Country** – Country information.
* **Region** – Regional information.
* **Continent** – Continent information.
* **Visit Mode** – Visit-mode categories.
* **Attraction Type** – Attraction category/type information.

The transaction data forms the primary interaction table and is joined with the supporting datasets to create a consolidated **master dataset**.

---

## 🔄 Data Processing Pipeline

### 1. Data Cleaning

The preprocessing stage includes:

* Loading the individual datasets.
* Inspecting data types and missing values.
* Removing duplicate records.
* Handling missing values.
* Validating rating values.
* Standardizing categorical information.
* Preparing datasets for merging.

### 2. Data Integration

The related tables are merged using common identifiers such as:

* `UserId`
* `AttractionId`
* `CityId`
* `CountryId`
* `RegionId`
* `ContinentId`
* `VisitMode`

This produces a master dataset containing both **visitor behavior** and **attraction information**.

### 3. Feature Engineering

Temporal and behavioral features are generated to improve model performance.

Examples include:

* Visit Year
* Visit Month
* Season
* Quarter
* User average rating
* User visit count
* User unique attraction count
* Attraction average rating
* Attraction visit count
* Visit-mode information

These features allow the models to capture both **individual user behavior** and **attraction popularity**.

---

# 📊 Exploratory Data Analysis

The notebook performs exploratory analysis of tourism behavior, including:

* Rating distribution
* Visit-mode distribution
* Popular attractions
* Popular attraction types
* Visits by year
* Visits by month
* Rating behavior across visit modes
* User activity
* Attraction popularity
* Numerical feature correlations

These visualizations help identify patterns and trends before model training.

---

# 🤖 Machine Learning Models

The project contains three major ML components.

---

## 1. ⭐ Rating Prediction — Regression

### Objective

Predict the rating that a visitor may give to an attraction.

### Target

```text
Rating
```

The rating is represented on a **1–5 scale**.

### Input Features

The regression model uses information such as:

* Visit year
* Visit month
* Visit mode
* Attraction type
* Season
* User behavioral statistics
* Attraction rating statistics
* Attraction popularity

### Preprocessing

Categorical features are encoded and numerical features are scaled before model training.

### Models Compared

Several regression algorithms are evaluated, including:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor

### Evaluation Metrics

The models are evaluated using:

* **RMSE** – Root Mean Squared Error
* **MAE** – Mean Absolute Error
* **R² Score** – Coefficient of Determination

The best model is selected based primarily on validation performance.

---

# 2. 👥 Visit Mode Prediction — Classification

### Objective

Predict how a visitor is likely to travel/visit an attraction.

### Target

```text
VisitMode
```

The dataset contains multiple visitor behavior classes such as:

* Business
* Couples
* Family
* Friends
* Solo

### Input Features

The classifier uses:

* Visit year
* Visit month
* Season
* Attraction type
* User behavioral features
* Attraction statistics
* Rating-related information

### Models Compared

The notebook compares classification algorithms such as:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

### Evaluation Metrics

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

### Result

The classification task achieves approximately **50% accuracy**.

This indicates that visitor visit mode is relatively difficult to predict from the available attributes. The overlap between behavioral classes can be observed through the confusion matrix.

This provides an opportunity for future improvement using richer user-history and behavioral features.

---

# 3. 🎯 Attraction Recommendation System

The recommendation component generates personalized attraction recommendations based on historical user–attraction interactions.

### User–Item Matrix

A user–item interaction matrix is constructed:

```text
Rows    → Users
Columns → Attractions
Values  → Ratings / Interactions
```

This matrix represents the historical relationship between users and attractions.

### Collaborative Filtering

Collaborative filtering learns latent representations of:

* Users
* Attractions

SVD-based factorization is used to capture hidden preference patterns from the user–item interaction matrix.

The learned representations are then used to calculate predicted attraction scores.

### Content-Based Component

Attraction characteristics can also be used to identify attractions with similar characteristics.

The content component helps provide recommendations based on attraction attributes rather than only user interactions.

### Popularity Component

Attraction popularity is calculated using historical interaction/rating information.

Popularity acts as an additional recommendation signal and provides a fallback for users with limited interaction history.

### Hybrid Recommendation

The final recommendation system combines:

```text
Collaborative Filtering
        +
Content-Based Information
        +
Popularity
        ↓
Hybrid Recommendation Score
```

Previously visited attractions are removed before presenting the final recommendations.

### Cold-Start Handling

For users without sufficient historical interaction data, the system falls back to popularity-based recommendations.

This prevents the recommendation module from returning empty results for new or unknown users.

---

# 🖥️ Streamlit Application

The trained models and preprocessing artifacts are integrated into a Streamlit application.

The application contains four major modules:

### 1. 📈 Analytics

Displays:

* Number of transactions
* Number of users
* Number of attractions
* Average rating
* Rating distribution
* Visit-mode distribution
* Top attractions

---

### 2. ⭐ Predict Rating

Users can select:

* Visit year
* Visit month
* Visit mode
* Attraction
* Season

The application processes the input using the saved preprocessing components and generates a predicted attraction rating.

The result is displayed on a **1–5 rating scale**.

---

### 3. 👥 Predict Visit Mode

Users provide information such as:

* Visit year
* Visit month
* Season
* Attraction
* Expected rating

The classifier predicts the most likely visit mode.

The application also displays the **class probabilities** to show the model's confidence across different visit modes.

---

### 4. 🧭 Recommendations

The user selects a `UserId` and the number of recommendations required.

The application:

1. Retrieves the user's learned representation.
2. Calculates attraction scores.
3. Removes attractions already visited by the user.
4. Combines recommendation signals.
5. Returns the highest-ranked attractions.

For cold-start users, popularity-based recommendations are used.

---

# 🏗️ System Architecture

```text
                Tourism Datasets
                       │
                       ▼
              Data Cleaning
                       │
                       ▼
              Data Integration
                       │
                       ▼
              Master Dataset
                       │
             ┌─────────┼─────────┐
             ▼         ▼         ▼
           EDA    Feature      User-Item
                  Engineering    Matrix
             │         │            │
             ▼         ▼            ▼
        ┌────────┐ ┌──────────┐ ┌──────────────┐
        │ Rating │ │ Visit    │ │Recommendation│
        │Regression│ │Mode      │ │   System     │
        │        │ │Classifier│ │              │
        └────┬───┘ └────┬─────┘ └──────┬───────┘
             │          │               │
             └──────────┼───────────────┘
                        ▼
                 Saved ML Artifacts
                        │
                        ▼
                Streamlit Application
                        │
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
     Analytics     Predictions      Recommendations
```

---

# 🛠️ Technology Stack

### Programming

* Python

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn

### Recommendation

* Collaborative Filtering
* SVD / Matrix Factorization
* Content-Based Filtering
* Popularity-Based Recommendation

### Deployment / Interface

* Streamlit

### Model Persistence

* Joblib / Pickle

### Development

* Jupyter Notebook
* GitHub

---

# 📁 Project Structure

```text
Tourism-Experience-Analytics/
│
├── NB4.ipynb
│
├── app.py
│
├── data/
│   ├── Transaction.csv
│   ├── User.csv
│   ├── Item.csv
│   ├── City.csv
│   ├── Country.csv
│   ├── Region.csv
│   ├── Continent.csv
│   ├── VisitMode.csv
│   └── Type.csv
│
├── models/
│   ├── regression_model.pkl
│   ├── classification_model.pkl
│   ├── user_item_matrix.pkl
│   ├── user_factors.pkl
│   ├── item_factors.pkl
│   └── popularity_scores.pkl
│
├── requirements.txt
│
└── README.md
```

> File names may vary depending on the final project artifact names.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Tourism-Experience-Analytics
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Notebook

Open the notebook using Jupyter:

```bash
jupyter notebook
```

Then open:

```text
NB4.ipynb
```

Run the notebook sequentially to:

1. Load the datasets.
2. Clean and integrate the data.
3. Perform EDA.
4. Engineer features.
5. Train regression models.
6. Train classification models.
7. Build the recommendation system.
8. Evaluate the models.
9. Save the required model artifacts.

---

# 🚀 Running the Streamlit Application

Run:

```bash
streamlit run app.py
```

The application will open in the browser.

The main workflow is:

```text
Analytics
    ↓
Predict Rating
    ↓
Predict Visit Mode
    ↓
Recommendations
```

---

# 📈 Model Evaluation

### Regression

| Metric | Purpose                                                             |
| ------ | ------------------------------------------------------------------- |
| RMSE   | Measures average prediction error with larger errors penalized more |
| MAE    | Measures average absolute prediction error                          |
| R²     | Measures explained variance                                         |

### Classification

| Metric    | Purpose                                                      |
| --------- | ------------------------------------------------------------ |
| Accuracy  | Overall percentage of correct predictions                    |
| Precision | Correct positive predictions relative to predicted positives |
| Recall    | Correct positive predictions relative to actual positives    |
| F1-score  | Harmonic mean of precision and recall                        |

### Recommendation

The recommendation system is evaluated using held-out user–attraction interactions where applicable.

The evaluation focuses on how effectively the system predicts unseen user preferences rather than simply reproducing interactions used during training.

---

# ⚠️ Limitations

The current system has several limitations:

* Visit-mode classification achieves approximately 50% accuracy.
* Visitor behavior can overlap significantly between classes.
* Recommendation quality depends on the amount of historical user–attraction interaction data.
* New users have limited historical information.
* Content-based recommendations depend on the available attraction attributes.
* Larger and richer behavioral datasets could improve personalization.

---

# 🔮 Future Enhancements

Possible improvements include:

### Classification

* Add richer user-history features.
* Use sequential visit behavior.
* Add demographic/contextual features where available.
* Perform hyperparameter optimization.
* Handle class imbalance if present.
* Experiment with XGBoost/LightGBM or other advanced classifiers.

### Recommendation

* Implement stronger hybrid weighting strategies.
* Add item-to-item similarity.
* Use implicit feedback such as clicks, views and visits.
* Implement neural collaborative filtering.
* Improve cold-start personalization.
* Evaluate using Precision@K, Recall@K, MAP@K and NDCG@K.

### Application

* Add interactive recommendation explanations.
* Add filtering by city, country and attraction type.
* Add user preference controls.
* Improve dashboard interactivity.
* Deploy the application to a cloud platform.

---

# 💡 Key Learning Outcomes

This project demonstrates practical experience with:

* Multi-table data integration
* Data cleaning and preprocessing
* Exploratory data analysis
* Feature engineering
* Regression
* Classification
* Recommendation systems
* Collaborative filtering
* Matrix factorization
* Cold-start handling
* Model evaluation
* Model persistence
* Streamlit application development
* End-to-end ML deployment

---

# 👨‍💻 Project Workflow

```text
1. Load Tourism Datasets
          ↓
2. Clean & Validate Data
          ↓
3. Merge Related Tables
          ↓
4. Perform EDA
          ↓
5. Feature Engineering
          ↓
6. Train Regression Model
          ↓
7. Train Classification Model
          ↓
8. Build Recommendation Engine
          ↓
9. Evaluate Models
          ↓
10. Save Model Artifacts
          ↓
11. Integrate with Streamlit
          ↓
12. Deploy Interactive Application
```

---

# 📌 Conclusion

The **Tourism Experience Analytics** project combines predictive analytics and recommendation techniques to transform historical tourism interaction data into an interactive decision-support system.

The system provides three core capabilities:

**Rating Prediction → Visit Mode Prediction → Personalized Recommendations**

The Streamlit application makes these capabilities accessible through an interactive interface, demonstrating a complete machine-learning pipeline from **data preprocessing and model development to application deployment**.
