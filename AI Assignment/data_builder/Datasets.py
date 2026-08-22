from pathlib import Path
import pandas as pd

from Greeting import greeting_data, greeting_unseen_data
from Admission import admission_data, admission_unseen_data
from Timetable import timetable_data, timetable_unseen_data
from Examination import examination_data, examination_unseen_data
from Fees import fees_data, fees_unseen_data
from Scholarship import scholarship_data, scholarship_unseen_data
from Programme import programme_data, programme_unseen_data
from CampusFacility import campus_facility_data, campus_facility_unseen_data
from Goodbye import goodbye_data, goodbye_unseen_data
from Unknown import unknown_data, unknown_unseen_data


BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR.parent / "datasets"

training_data = (
    []
    + greeting_data
    + admission_data
    + timetable_data
    + examination_data
    + fees_data
    + scholarship_data
    + programme_data
    + campus_facility_data
    + goodbye_data
    + unknown_data
)

unseen_data = (
    []
    + greeting_unseen_data
    + admission_unseen_data
    + timetable_unseen_data
    + examination_unseen_data
    + fees_unseen_data
    + scholarship_unseen_data
    + programme_unseen_data
    + campus_facility_unseen_data
    + goodbye_unseen_data
    + unknown_unseen_data
)

training_df = pd.DataFrame(training_data)
unseen_df = pd.DataFrame(unseen_data).rename(
    columns={"intent": "expected_intent"}
)

DATASETS_DIR.mkdir(parents=True, exist_ok=True)

training_output = DATASETS_DIR / "tarumt_dataset.csv"
unseen_output = DATASETS_DIR / "unseen_test.csv"

training_df.to_csv(training_output, index=False)
unseen_df.to_csv(unseen_output, index=False)

print("Training dataset created:", training_output)
print(training_df["intent"].value_counts())

print("\nUnseen dataset created:", unseen_output)
print(unseen_df["expected_intent"].value_counts())
