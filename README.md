# TARUMT Student Assistance Chatbot

## Project Overview

This project is a TARUMT Student Assistance Chatbot developed using Natural Language Processing (NLP) and machine learning.

The chatbot is designed to help students obtain common TARUMT-related information and provide guidance on what action to take next.

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

The student-facing chatbot uses LinearSVC as the deployed classification model. Both models are evaluated and compared through the Technical Evaluation page.

## Live Demo

The deployed application is available on Streamlit Community Cloud:

[Open TARUMT Student Assistance Chatbot](https://tarumt-student-assistance-chatbotgit-s2ycueajfitftumbam2z2v.streamlit.app/)

## Key Features

The chatbot includes the following features:

- Student-facing conversational interface
- NLP text preprocessing
- TF-IDF feature extraction
- LinearSVC intent classification
- Predefined responses based on predicted intent
- Quick Help buttons for common student enquiries
- "What you can do next" guidance for supported enquiries
- Fallback suggestions for unsupported or unknown questions
- Helpful / Not Helpful user feedback
- Feedback summary by predicted intent
- Error analysis for misclassified questions
- Confusion matrix visualization
- Detailed prediction results
- Technical comparison between Logistic Regression and LinearSVC

## Requirements

- Python 3.11
- Streamlit
- pandas
- NumPy
- scikit-learn
- joblib
- NLTK
- matplotlib

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

Open the **Student Chatbot** page and enter a TARUMT-related question in the chat input.

The chatbot processes the question using NLP, converts the processed text into TF-IDF features, and predicts the user's intent using the trained LinearSVC model.

The chatbot then provides a predefined response according to the predicted intent.

For supported enquiries, the chatbot also provides **What you can do next** guidance to help the student take the appropriate next action.

The **Quick Help** section provides shortcuts for common questions related to:

- Timetable
- Fees
- Examination
- Admission

Other supported topics such as scholarships, programmes, and campus facilities can be entered directly into the chat input.

If a question is outside the supported scope, the chatbot may classify it as `unknown`. In this case, fallback buttons are provided to help the user continue with common supported topics.

Users can also rate supported chatbot responses as **Helpful** or **Not Helpful**.

The **Clear Chat** button clears the current conversation while keeping the feedback statistics for the current session.

## Technical Evaluation

The **Technical Evaluation** page is mainly used to evaluate and demonstrate the machine learning models rather than as part of the normal student interaction.

It compares Logistic Regression and LinearSVC using a separate unseen test dataset.

The page includes:

- Accuracy
- Precision
- Recall
- F1-score
- Accuracy comparison between both models
- User feedback summary
- Feedback breakdown by predicted intent
- Error counts
- Misclassified cases
- Confusion matrices
- Detailed prediction results

The confusion matrices show the actual intent in the rows and the predicted intent in the columns. Higher values along the main diagonal indicate correct classifications.

The current unseen test evaluation shows that LinearSVC performs better than Logistic Regression, so LinearSVC is used as the deployed model for the Student Chatbot.

The evaluation results can also be viewed in the terminal by running:

```bash
python EvaluateUnseen.py
```

## User Feedback

Users can rate chatbot responses using:

- 👍 Helpful
- 👎 Not Helpful

The Technical Evaluation page displays:

- Number of responses rated
- Number of helpful responses
- Satisfaction rate
- Number of not helpful responses
- Feedback results grouped by predicted intent

The **Reset Feedback** button clears the feedback statistics and allows displayed chatbot responses to be rated again.

Feedback is stored only in the current Streamlit session and is not permanently stored after the session ends.

## Retraining the Models

Retraining is not required to run the chatbot because the trained models and TF-IDF vectorizer are already included.

To train and internally evaluate Logistic Regression:

```bash
python TrainLogistic.py
```

To train and internally evaluate LinearSVC:

```bash
python TrainLinearSVC.py
```

To train both final models and save the trained models and TF-IDF vectorizer:

```bash
python TrainModels.py
```

The generated model files will be saved in the `model` folder.

After changing the training dataset or NLP preprocessing, the models should be retrained before running the final evaluation.

## Main Files

- `StreamlitApp.py` - Streamlit interface for the Student Chatbot and Technical Evaluation
- `Chatbot.py` - Intent prediction and chatbot response selection
- `NLP.py` - Text preprocessing functions
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

The Technical Evaluation page is included to demonstrate model performance, analyse classification errors, and provide evidence for the model selection.

An internet connection may be required during the first run to download the required NLTK resources.