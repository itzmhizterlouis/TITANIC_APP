# Start App - Titanic Survival Analytics

Predict passenger survival probabilities using historical data and machine learning.

## Prerequisites

- Python 3.8+
- Required libraries: FastAPI, Uvicorn, Pandas, Scikit-Learn

## Installation

1. Navigate to the project directory:
   ```bash
   cd "TITANIC APP"
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the model file (`titanicmodel.pkl`) is present. If it's missing, you may need to run:
   ```bash
   python titanicmodel.py
   ```

## Running the Application

Execute the following command to start the server:

```bash
uvicorn app:app --reload
```

The system will bind to `0.0.0.0` and default to port `8000`. You can override the port by setting the `PORT` environment variable.

## System Overview

- **Dataset**: Historical Titanic passenger manifests.
- **Model**: Random Forest Classifier.
- **Features**: Social class, sex, age, fare, and family relationships.
- **Analytics**: Real-time synthesis of survival probability with confidence intervals.
