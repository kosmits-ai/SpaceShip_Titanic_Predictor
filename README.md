# 🪐 SpaceShip Titanic - Classification problem  

This is my **third machine learning project**, based on the popular [Kaggle Spaceship Titanic competition](https://www.kaggle.com/competitions/spaceship-titanic).  

The goal of the challenge is to **predict whether a passenger was transported to another dimension** (`Transported = True/False`) using information about their home planet, cabin, VIP status, destination, and more.

---

##  Workflow  
1. **Exploratory Data Analysis (EDA)**  
   - Inspected data types, missing values, and feature distributions.
   - Visualized relationships between categorical features and the target variable.
   - Explored correlations and feature importance to guide feature engineering.  

2. **Feature Engineering**  
   - **Missing values:** Imputed numerical and categorical columns with appropriate strategies.
   - **Binary features:** Converted `CryoSleep` and `VIP` columns into `0`/`1`.
   - **New features:** Engineered additional features such as `FamilySize` based on passenger groups.
   - **Categorical encoding:** Used `OneHotEncoder` within a `ColumnTransformer` to encode all categorical variables.
   - Added `*_was_missing` indicator columns to capture missingness as a potential signal. 

3. **Modeling**  
   - Built a reproducible ML workflow using a `Pipeline`:
   - **Step 1:** Preprocessing (`ColumnTransformer`)
   - **Step 2:** Model (`RandomForestClassifier`)
   - Performed **hyperparameter tuning** using `RandomizedSearchCV`.
   - Ensured reproducibility with `random_state=42` and `n_jobs=-1` for parallel training.  

4. **Evaluation**  
   - Compared **training vs validation accuracy** to check for overfitting.
   - Used cross-validation scores (`RandomizedSearchCV.best_score_`) to estimate generalization performance.
   - Evaluated the model with key metrics:
     - Accuracy
     - Confusion Matrix
     - Classification Report (Precision, Recall, F1-score)  

---

##  Files in Repository  
- `spaceshiptitanic_predictor.ipynb` → Full notebook workflow  
- `train.csv`, `test.csv` → Kaggle datasets  
