import pandas as pd

from Registration import registration_data

data = (
    registration_data
)

df = pd.DataFrame(data)
df.to_csv("tarumt_dataset.csv", index=False)
print(df)