# Loan Approval ML

A machine learning project that predicts whether a loan application will be approved,
based on applicant details such as income, credit history, loan amount, and employment.
It includes a training pipeline built with scikit-learn and a Streamlit web app where you
can enter an applicant's information and get a prediction.

> **Status:** early development. The project structure and environment are set up; the
> data pipeline, model, and app are being built out next (see [Roadmap](#roadmap)).

## Project structure

```
loan-approval-ml/
├── app/
│   └── streamlit_app.py   # Streamlit web app for interactive predictions
├── data/                  # Raw and processed datasets (not committed)
├── models/                # Saved trained models
├── notebooks/             # Jupyter notebooks for exploration and analysis
├── src/
│   ├── preprocess.py      # Data cleaning, encoding, and feature engineering
│   ├── train.py           # Model training
│   ├── evaluate.py        # Model evaluation and metrics
│   └── predict.py         # Load a trained model and make predictions
├── requirements.txt
└── README.md
```

## Tech stack

- **Python 3.11+**
- **pandas** and **NumPy** for data handling
- **scikit-learn** for modeling
- **Matplotlib** and **Altair** for visualization
- **Streamlit** for the web app
- **Jupyter** for exploration

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/saalderete/Loan-Approval-ML.git
cd Loan-Approval-ML
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
```

Or with conda:

```bash
conda create -n loan-approval-ml python=3.11
conda activate loan-approval-ml
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Check the setup

```bash
python src/train.py
```

You should see `Project setup is working.`

## Usage

Once the pipeline is in place, the workflow will be:

```bash
python src/preprocess.py         # Clean and prepare the data in data/
python src/train.py              # Train the model and save it to models/
python src/evaluate.py           # Print evaluation metrics
python src/predict.py            # Make predictions with the saved model
streamlit run app/streamlit_app.py
```

The Streamlit app opens in your browser at `http://localhost:8501`.

## Data

Put the dataset in the `data/` folder. Large data files should not be committed to Git
(GitHub rejects files over 100 MB); add them to `.gitignore` or use
[Git LFS](https://git-lfs.com/) instead.

## Roadmap

- [x] Project structure and environment
- [ ] Add the loan dataset and explore it in a notebook
- [ ] Preprocessing: handle missing values, encode categorical features, scale numeric features
- [ ] Train and compare models (e.g. logistic regression, random forest)
- [ ] Evaluate with accuracy, precision, recall, F1, and ROC-AUC
- [ ] Save the best model to `models/`
- [ ] Build the Streamlit app for interactive predictions
- [ ] Deploy the app

## Author

**Samuel Alderete**
GitHub: [@saalderete](https://github.com/saalderete)
