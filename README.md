# TARUMT Student Assistance Chatbot

## Project Overview

This project is a TARUMT Student Assistance Chatbot developed using Natural Language Processing (NLP) and machine learning.

The chatbot is designed to help students obtain common TARUMT-related information and guidance on what action to take next.

The chatbot supports enquiries related to:

- Admissions
- Timetables
- Examinations
- Fees
- Scholarships
- Programmes
- Campus facilities

Two machine learning classifiers are implemented and evaluated:

- Logistic Regression
- Linear Support Vector Classification (LinearSVC)

The student-facing chatbot uses LinearSVC as the deployed classification model, while both models can be compared through the Technical Evaluation page.

## Requirements

- Python 3.11
- Streamlit
- pandas
- numpy
- scikit-learn
- joblib
- nltk

## Installation

Open a terminal in the project folder and move into the `AI_Assignment` folder:

```bash
cd AI_Assignment
```

Install all required dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

Run the Streamlit application:

```bash
python -m streamlit run StreamlitApp.py
```

The application will normally open automatically in the browser.

If it does not open automatically, open:

```text
http://localhost:8501
```

## Using the Student Chatbot

1. Open the **Student Chatbot** page.

2. Enter a TARUMT-related question in the chat input.

3. The chatbot processes the question using NLP and predicts the user's intent using the trained LinearSVC model.

4. The chatbot provides a predefined response based on the predicted intent.

5. For supported enquiries, the chatbot also provides **What you can do next** guidance to help the user take the appropriate next action.

The **Quick Help** buttons can also be used to quickly ask common questions related to:

- Timetable
- Fees
- Examination
- Admission

Other supported topics such as scholarships, programmes, and campus facilities can be entered directly into the chat input.

## Technical Evaluation

The **Technical Evaluation** page compares Logistic Regression and LinearSVC using a separate unseen test dataset.

The page displays:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices
- Detailed prediction results

The evaluation results can also be viewed in the terminal by running:

```bash
python EvaluateUnseen.py
```

## Retraining the Models

Retraining is not required to run the chatbot because the trained models and TF-IDF vectorizer are already included.

Individual model training can be performed using:

```bash
python TrainLogistic.py
```

```bash
python TrainLinearSVC.py
```

To train both final models and save the trained models and TF-IDF vectorizer, run:

```bash
python TrainModels.py
```

The generated files will be saved in the `model` folder.

## Main Files

- `StreamlitApp.py` - Streamlit user interface for the Student Chatbot and Technical Evaluation
- `Chatbot.py` - Intent prediction and response selection
- `NLP.py` - Text preprocessing
- `Responses.py` - Predefined chatbot responses
- `TrainLogistic.py` - Logistic Regression training and internal evaluation
- `TrainLinearSVC.py` - LinearSVC training and internal evaluation
- `TrainModels.py` - Final training and saving of both models and the TF-IDF vectorizer
- `EvaluateUnseen.py` - Evaluation of both models using the unseen test dataset
- `datasets/` - Training and unseen test datasets
- `data_builder/` - Dataset construction files
- `model/` - Saved TF-IDF vectorizer and trained models

## Notes

The chatbot is designed to answer common TARUMT student enquiries within the supported intent categories.

Questions outside the supported scope may be classified as `unknown`.

The Student Chatbot uses LinearSVC as the deployed model because it achieved better performance on the separate unseen test dataset.

An internet connection may be required during the first run to download the required NLTK resources.