# prepare.py as submitted in HW1

import sklearn

## Normalizers

def StandardNormalize(df, columns):
  scaler = sklearn.preprocessing.StandardScaler()
  scaler.fit(df[columns])
  df[columns] = scaler.transform(df[columns])

def MinMaxNormalize(df, columns):
  scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
  scaler.fit(df[columns])
  df[columns] = scaler.transform(df[columns])


### Prepare Data
def prepare_data(training_data, new_data):
  prepared_data = new_data.copy()

  #Replace NaN values for household_income:
  median = training_data["household_income"].median()
  prepared_data["household_income"] = prepared_data["household_income"].fillna(median)

  #Prepare Blood Types
  prepared_data["SpecialProperty"] = prepared_data["blood_type"].isin(["O+", "B+"])
  prepared_data = prepared_data.drop("blood_type", axis=1)

  # New: Drop non continuous columns
  columns_to_drop = ['patient_id', 'current_location', 'pcr_date' ,"sex", "SpecialProperty", "happiness_score", "conversations_per_day", "sport_activity"]
  prepared_data = prepared_data.drop(columns_to_drop, axis=1)

  #Normalization
  StandardNormalize(prepared_data, ["PCR_01", "PCR_02", "PCR_05", "PCR_06", "PCR_07", "PCR_08", "sugar_levels", "household_income", "age", "weight", "num_of_siblings"])
  MinMaxNormalize(prepared_data, ["PCR_10", "PCR_03", "PCR_04", "PCR_09"])
  return prepared_data