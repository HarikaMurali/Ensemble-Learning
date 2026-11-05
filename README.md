# EnsembleLab — Interactive Ensemble Learning Studio

A Streamlit project comparing four classification algorithms on the Breast Cancer Wisconsin Diagnostic dataset:
- Decision Tree (baseline)
- Bagging
- AdaBoost
- Random Forest

## Features
- Dark dashboard theme
- Dataset exploration and interactive 3D PCA projection
- Algorithm explorer with tree-member visualization
- Model Arena: metrics, radar chart, confusion matrix, ROC curves and feature importance
- Prediction Studio: test sample or manual feature entry, four model predictions and majority vote
- Experiment Lab: adjustable test split, estimator count, tree depth and AdaBoost learning rate

## Requirements
- Python 3.9 or newer recommended
- Internet connection for initial package installation (and Google Fonts in the theme; app still works without the font)

## Windows setup (PowerShell)
```powershell
cd path\to\EnsembleLab
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, use Command Prompt:
```cmd
.venv\Scripts\activate.bat
```

## Run
Run `streamlit run app.py` from the project root folder. Streamlit opens the app at `http://localhost:8501`.

## Dataset notes
The dataset is loaded from Scikit-learn; no manual CSV download is required.
Scikit-learn target encoding:
- `0 = malignant`
- `1 = benign`

Precision, recall and F1 in this starter app use benign (class 1) as the positive class. The 3D PCA chart is a visualization only; model training uses all 30 original features.

## Important
This is an educational machine-learning project, not a medical diagnostic system. Do not use its predictions for healthcare decisions.
