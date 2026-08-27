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

The dataset also includes conversational intents such as greetings and goodbyes, together with an `unknown` intent for questions outside the supported scope.

Two machine learning classifiers are implemented and evaluated:

- Logistic Regression
- Linear Support Vector Classification (LinearSVC)

Both models use TF-IDF features and are trained under the same conditions. Hyperparameter tuning is performed using GridSearchCV with stratified cross-validation.

The student-facing chatbot automatically loads the selected classification model from `selected_model.txt`. In the current version, Logistic Regression is selected as the student-facing model based on the internal model selection procedure.

## Live Demo

The deployed application is available on Streamlit Community Cloud:

[Open TARUMT Student Assistance Chatbot](https://tarumt-student-assistance-chatbotgit-s2ycueajfitftumbam2z2v.streamlit.app/)

## Key Features

The chatbot includes the following features:

- Student-facing conversational interface
- NLP text preprocessing
- TF-IDF feature extraction
- Logistic Regression and LinearSVC intent classification
- GridSearchCV hyperparameter tuning
- Stratified cross-validation
- Automatic model selection
- Predefined responses based on predicted intent
- Quick Help buttons for common student enquiries
- "What you can do next" guidance for supported enquiries
- Fallback suggestions for unsupported or unknown questions
- Helpful / Not Helpful user feedback
- Feedback summary by predicted intent
- Error analysis for misclassified questions
- Heuristic error evaluation with possible explanations and recommended improvements
- Confusion matrix visualization
- Detailed prediction results
- Technical comparison between Logistic Regression and LinearSVC

## Dataset

The current training dataset contains 677 questions across 10 intent categories:

- Admission
- Timetable
- Examination
- Fees
- Scholarship
- Programme
- Campus facility
- Greeting
- Goodbye
- Unknown

A separate unseen test dataset containing 208 questions is used for final model evaluation.

The unseen test dataset is kept separate from the training, hyperparameter tuning, and internal model selection process.

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

The chatbot first applies NLP preprocessing to the user's question. The processed text is then passed through the saved machine learning pipeline, which performs TF-IDF feature extraction and predicts the user's intent using the selected model.

The selected student-facing model is loaded from:

```text
model/selected_model.txt
```

In the current version, the selected model is:

```text
Logistic Regression
```

After the intent is predicted, the chatbot provides a predefined response associated with the predicted intent.

For supported enquiries, the chatbot also provides **What you can do next** guidance to help the student take the appropriate next action.

The **Quick Help** section provides shortcuts for common questions related to:

- Timetable
- Fees
- Examination
- Admission

Other supported topics such as scholarships, programmes, and campus facilities can be entered directly into the chat input.

If a question is outside the supported scope, the chatbot may classify it as `unknown`. In this case, fallback buttons are provided to help the user continue with common supported topics.

Users can also rate supported chatbot responses as:

- 👍 Helpful
- 👎 Not Helpful

The **Clear Chat** button clears the current conversation while keeping the feedback statistics for the current session.

## Model Training and Selection

The training process is handled by:

```bash
python TrainModels.py
```

The training process includes the following stages:

1. Load the training dataset.
2. Apply NLP preprocessing to the student questions.
3. Separate the processed questions and intent labels.
4. Split the dataset into an 80:20 stratified train-test split.
5. Use the 80% training portion for hyperparameter tuning.
6. Perform 5-fold stratified cross-validation using GridSearchCV.
7. Tune the TF-IDF and classifier hyperparameters.
8. Evaluate the best Logistic Regression and LinearSVC configurations using the untouched 20% internal test set.
9. Compare the models using Macro F1-score and accuracy.
10. Select the student-facing model.
11. Retrain both final pipelines using the complete training dataset.
12. Save the trained pipelines and model selection results.

The current dataset contains 677 records. The 80:20 internal split produces 541 training records and 136 internal testing records.

### Hyperparameter Tuning

GridSearchCV is used to test different combinations of:

- TF-IDF `ngram_range`
- TF-IDF `min_df`
- TF-IDF `sublinear_tf`
- Classifier regularization parameter `C`

The grid search uses 5-fold stratified cross-validation and `f1_macro` as the scoring metric so that each intent category contributes equally to model evaluation.

The best cross-validation Macro F1-scores obtained were:

- Logistic Regression: 91.99%
- LinearSVC: 91.41%

### Internal Evaluation

After hyperparameter tuning, the best configuration of each model is evaluated using the untouched 20% internal test set.

The internal evaluation results are:

| Model | Accuracy | Macro F1-score |
|---|---:|---:|
| Logistic Regression | 95.59% | 94.14% |
| LinearSVC | 91.91% | 90.78% |

Logistic Regression achieved better performance during the internal evaluation and was therefore selected as the student-facing model.

### Model Selection

The student-facing model is selected mainly using the Macro F1-score obtained from the untouched internal 20% test set.

Accuracy is used as a tie-breaker if both models obtain the same Macro F1-score.

If both Macro F1-score and accuracy are equal, LinearSVC is selected.

The selected model name is saved in:

```text
model/selected_model.txt
```

The current selected model is:

```text
Logistic Regression
```

## Saved Model Files

The final Logistic Regression and LinearSVC models are saved as complete scikit-learn pipelines.

Each pipeline contains:

```text
NLP-preprocessed text
        ↓
TF-IDF Vectorizer
        ↓
Classification Model
```

The generated model files include:

```text
model/logistic_pipeline.joblib
model/linearsvc_pipeline.joblib
model/selected_model.txt
```

Additional training and tuning results are also saved:

```text
model/logistic_cv_results.csv
model/linearsvc_cv_results.csv
model/logistic_internal_predictions.csv
model/linearsvc_internal_predictions.csv
model/tuning_summary.csv
```

Retraining is not required to run the chatbot because the trained pipelines are already included.

However, the models should be retrained whenever the training dataset or NLP preprocessing is changed.

## Technical Evaluation

The **Technical Evaluation** page is mainly used to evaluate and demonstrate the machine learning models rather than as part of normal student interaction.

It compares the final Logistic Regression and LinearSVC pipelines using the separate unseen test dataset containing 208 questions.

The unseen dataset is not used during model training, hyperparameter tuning, or internal model selection.

The page includes:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Comparison between Logistic Regression and LinearSVC
- User feedback summary
- Feedback breakdown by predicted intent
- Error counts
- Misclassified cases
- Error Evaluation
- Confusion matrices
- Detailed prediction results

The final unseen test results are:

| Model | Accuracy | Macro F1-score |
|---|---:|---:|
| Logistic Regression | 96.63% | 96.63% |
| LinearSVC | 97.60% | 97.84% |

LinearSVC achieved slightly higher performance on the separate unseen test dataset. However, the unseen test dataset is used only for final evaluation and is not used to change the student-facing model selected during the internal evaluation process.

Therefore, Logistic Regression remains the selected student-facing model.

The confusion matrices show the actual intent in the rows and the predicted intent in the columns. Higher values along the main diagonal indicate correct classifications.

The evaluation results can also be viewed in the terminal by running:

```bash
python EvaluateUnseen.py
```

## Error Analysis and Error Evaluation

The Technical Evaluation page includes two forms of error inspection.

### Misclassified Cases

The system displays questions that were classified incorrectly together with:

- Question
- Expected Intent
- Predicted Intent

### Error Evaluation

A heuristic diagnostic analysis is also provided to help identify possible patterns in misclassified questions.

Possible error categories include:

- Insufficient or sparse wording
- Ambiguous or multi-intent questions
- Possible preprocessing information loss
- Likely training-data coverage issues
- Unseen vocabulary or training coverage

For each identified error pattern, the system provides:

- Error category
- Possible explanation
- Recommended improvement

These categories are diagnostic heuristics and should not be treated as confirmed root causes of classification errors.

## User Feedback

Users can rate supported chatbot responses using:

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

## NLP Preprocessing

The chatbot applies several NLP preprocessing techniques before classification.

The preprocessing process includes:

- Converting text to lowercase
- Expanding common contractions
- Removing URLs
- Removing unnecessary special characters
- Tokenisation
- Stop-word removal
- Preserving important negation words such as `not`, `no`, and `cannot`
- Lemmatization
- Rejoining processed tokens into cleaned text

The required NLTK resources are downloaded automatically when needed.

## Main Files

- `StreamlitApp.py` - Streamlit interface for the Student Chatbot and Technical Evaluation
- `Chatbot.py` - Loads the selected model, predicts intents, and retrieves chatbot responses
- `NLP.py` - NLP preprocessing functions
- `Responses.py` - Predefined chatbot responses
- `TrainModels.py` - Hyperparameter tuning, internal evaluation, model selection, final retraining, and saving of both pipelines
- `EvaluateUnseen.py` - Final evaluation of both trained pipelines using the separate unseen test dataset
- `datasets/tarumt_dataset.csv` - Main training dataset
- `datasets/unseen_test.csv` - Separate unseen test dataset
- `data_builder/` - Dataset construction and organization files
- `model/` - Saved pipelines, selected model, cross-validation results, internal predictions, and tuning summary

## Notes

The chatbot is designed to answer common TARUMT student enquiries within the supported intent categories.

Questions outside the supported scope may be classified as `unknown`.

The student-facing model is selected during the internal training and evaluation process rather than using the separate unseen test dataset.

The current selected student-facing model is Logistic Regression.

Although LinearSVC achieved slightly higher performance on the separate unseen test dataset, the unseen dataset is used only for final evaluation and does not affect the previously completed internal model selection process.

The Technical Evaluation page is included to demonstrate model performance, analyse classification errors, compare both classifiers, and provide supporting evidence for the evaluation of the chatbot.

An internet connection may be required during the first run to download the required NLTK resources.