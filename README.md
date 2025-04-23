# Titanic Survival Prediction

## Overview
This project implements a machine learning pipeline to predict passenger survival on the Titanic using a dataset containing passenger information. Five classification models are trained and evaluated to determine survival outcomes based on features such as passenger class, sex, age, and fare. The models are saved for future use, and their performance is visualized using ROC curves.

## Dataset
The dataset used is `tested.csv`, which includes the following key columns:
- **PassengerId**: Unique identifier for each passenger
- **Survived**: Survival status (0 = No, 1 = Yes)
- **Pclass**: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)
- **Sex**: Gender of the passenger
- **Age**: Age of the passenger
- **Fare**: Ticket fare
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Ticket**: Ticket number
- **Cabin**: Cabin number
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

The dataset is preprocessed by selecting relevant features (`Pclass`, `Sex`, `Age`, `Fare`, `Survived`), handling missing values, normalizing numerical features, and encoding categorical variables.

## Models
Five classification models are trained and evaluated:
1. **Decision Tree**
2. **Logistic Regression**
3. **Random Forest**
4. **Support Vector Machine (SVM)**
5. **Gaussian Naive Bayes**

Each model is saved as a `.pkl` file for easy reuse:
- `DecisionTree_model.pkl`
- `LogisticRegression_model.pkl`
- `RandomForest_model.pkl`
- `SVM_model.pkl`
- `GaussianNB_model.pkl`

## Requirements
To run the project, install the required Python packages:
```bash
pip install numpy pandas scikit-learn matplotlib
```

## Project Structure
- `TITANIC_SURVIVAL_PREDICTION.ipynb`: Jupyter notebook containing the full pipeline (data preprocessing, model training, evaluation, and visualization).
- `tested.csv`: Input dataset (not included in the repo; source from [Kaggle](https://www.kaggle.com/datasets/brendan45774/test-file)).
- `*_model.pkl`: Saved model files.
- `roc_curves.png`: Output visualization of ROC curves for all models.
- `README.md`: This file.

## Usage
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/titanic-survival-prediction.git
   cd titanic-survival-prediction
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook**:
   - Open `TITANIC_SURVIVAL_PREDICTION.ipynb` in Jupyter Notebook or JupyterLab.
   - Ensure `tested.csv` is in the same directory.
   - Execute the cells to preprocess data, train models, and generate ROC curves.

4. **Load a Saved Model**:
   To use a saved model for predictions, load it using Python:
   ```python
   import pickle
   import pandas as pd

   # Load a model
   with open('RandomForest_model.pkl', 'rb') as file:
       model = pickle.load(file)

   # Example input (adjust based on your preprocessed data)
   sample_data = pd.DataFrame({
       'Pclass': [2],  # Normalized value
       'Sex': ['male'],
       'Age': [34],  # Normalized value
       'Fare': [15.0]  # Normalized value
   })

   # Preprocess sample_data (encode 'Sex', etc., as done in the notebook)
   # Make prediction
   prediction = model.predict(sample_data)
   print(f"Survival Prediction: {'Survived' if prediction[0] == 1 else 'Did not survive'}")
   ```

## Results
The performance of each model is evaluated using ROC curves and AUC scores, visualized in `roc_curves.png`. The AUC scores indicate the models' ability to distinguish between survivors and non-survivors.

## Notes
- The dataset contains missing values in `Age` and `Fare`, which are handled by dropping rows with missing values in the current implementation. Alternative imputation strategies could be explored.
- The `Sex` column requires encoding (e.g., using `pd.get_dummies`) before model training, which is not explicitly shown in the provided notebook snippet but is assumed to be part of the preprocessing.
- The ROC curve plotting assumes the models are trained and split into `X_test` and `y_test` datasets, which should be defined in the notebook.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Dataset sourced from [Kaggle](https://www.kaggle.com/datasets/brendan45774/test-file).
- Built using Python, scikit-learn, pandas, numpy, and matplotlib.
