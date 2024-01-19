
### please change the Train and Test path accordingly.

### For better understanding please visit the original repo: https://github.com/sozo-lab/abc2023-wearingoffchallenge

### Loading Necessary Libary

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score

#### define Data path and participant

TRAIN_DATA_PATH = 'TrainData'
TEST_DATA_PATH = 'TestData'

USER = 'participant10'


# In[ ]:


### Feature Selection

features = ['heart_rate', 'steps', 'stress_score',
            'awake', 'deep', 'light', 'rem',
            'nonrem_total', 'total', 'nonrem_percentage', 'sleep_efficiency']


features += ['timestamp_hour', 'timestamp_dayofweek',
             'timestamp_hour_sin', 'timestamp_hour_cos']

TARGET_COLUMN = 'wearing_off'
features.append(TARGET_COLUMN)

columns = ['timestamp'] + features + ['participant']


normalize_features = features


SHIFT = 1
N_IN = 2   # t-2, t-1, t
N_OUT = 2  # t+1
RECORD_SIZE_PER_DAY = 96  # 60 minutes / 15 minutes * 24 hours = 96
FIGSIZE = (20, 7)
FIGSIZE_CM = (13, 7)


# In[ ]:


test_horizons = {
  "participant1": ["2021-12-02 0:00", "2021-12-03 23:45"],
  "participant2": ["2021-11-28 0:00", "2021-11-29 23:45"],
  "participant3": ["2021-11-25 0:00", "2021-11-26 23:45"],
  "participant4": ["2021-12-06 0:00", "2021-12-07 7:15"],
  "participant5": ["2021-11-28 0:00", "2021-11-29 23:45"],
  "participant6": ["2021-12-06 0:00", "2021-12-07 23:45"],
  "participant7": ["2021-12-12 0:00", "2021-12-13 9:45"],
  "participant8": ["2021-12-23 0:00", "2021-12-24 23:45"],
  "participant9": ["2021-12-23 0:00", "2021-12-24 23:45"],
  "participant10": ["2021-12-23 0:00", "2021-12-24 23:45"],
}

test_horizons_df = pd.DataFrame(
  [[participant, test_start_date, test_end_date]
   for participant, (test_start_date, test_end_date) in test_horizons.items()],
  columns=['participant', 'test_start_date', 'test_end_date']
)



dataset = pd.read_excel(f'{TRAIN_DATA_PATH}/combined_data.xlsx',
                        index_col="timestamp",
                        usecols=columns,
                        engine='openpyxl').query(f'participant == "{USER}"')

# Fill missing data with 0

dataset.fillna(0, inplace=True)

dataset = dataset.query(f'participant == "{USER}"').drop(
    columns=['participant'])


dataset_test = pd.read_excel(f'{TEST_DATA_PATH}/combined_data.xlsx',
                             index_col="timestamp",
                             usecols=columns,
                             engine='openpyxl').query(f'participant == "{USER}"')

dataset_test.fillna(0, inplace=True)

dataset_test = dataset_test.query(f'participant == "{USER}"').drop(
    columns=['participant'])



train_df = dataset.copy()
test_df = dataset_test.copy()



def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
  var_names = data.columns
  n_vars = len(var_names)
  df = pd.DataFrame(data)
  cols, names = list(), list()  # new column values, new columne names

  for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
    names += list(
        map(lambda var_name: f'{var_name}(t-{i})', var_names)
    )

  for i in range(0, n_out):
    cols.append(df.shift(-i))
    if i == 0:
      names += list(map(lambda var_name: f'{var_name}(t)', var_names))
    else:
      names += list(map(lambda var_name: f'{var_name}(t+{i})', var_names))


  agg = pd.concat(cols, axis=1)
  agg.columns = names

  # drop rows with NaN values

  if dropnan:
    agg.dropna(inplace=True)

  return agg



# Split into X and y

def split_x_y(df, target_columns, SHIFT=SHIFT, drop_wearing_off=True):
  # Drop extra columns i.e., (t+1), (t+2), (t+3), (t+4)

  regex = r".*\(t\+[1-{SHIFT}]\)$"  # includes data(t)
  # regex = r"\(t(\+([1-{SHIFT}]))?\)$" # removes data(t)

  # Drop extra columns except target_columns

  df.drop(
    [x for x in df.columns if re.search(regex, x) and x not in target_columns],
    axis=1, inplace=True
  )

  # Split into X and y

  y = df[target_columns].copy()
  X = df.drop(target_columns + [f'{TARGET_COLUMN}(t)'], axis=1)

  if drop_wearing_off:
    # Delete past wearing_off data, because it will not be provided in test data.
    #   Predicted past weaering_off can be used as feature, but make it yourself.
    wearing_off_features = X.filter(like='wearing_off').columns
    X = X.drop(columns=wearing_off_features)

  return (X, y)


# Load submission file to get test data times

submission_df = pd.read_csv(f'{TEST_DATA_PATH}/submission.csv',
                            index_col=0
                            ).query(f'participant == "{USER}"')
submission_df['Timestamp'] = pd.to_datetime(submission_df['Timestamp'])
submission_df['reframed_timestamp'] = pd.to_datetime(
  submission_df['reframed_timestamp']
)
submission_df.rename(columns={'Timestamp': 'timestamp'}, inplace=True)


def keep_forecast_times(full_test_data, submission_df):
  return full_test_data.reset_index().merge(
    submission_df[['reframed_timestamp']],
    left_on=['timestamp'],
    right_on=['reframed_timestamp'],
    how='right'
  ).drop(columns='reframed_timestamp').set_index('timestamp')



# TRAIN SET

reframed_train_df = series_to_supervised(train_df,
                                         n_in=N_IN,
                                         n_out=N_OUT,
                                         dropnan=True)

train_X, train_y = split_x_y(
    reframed_train_df, [f'{TARGET_COLUMN}(t+{N_OUT-1})'])

#display(train_y.head(5))
#display(train_X.head(5))



# TEST SET

reframed_test_df = series_to_supervised(test_df,
                                        n_in=N_IN,
                                        n_out=N_OUT,
                                        dropnan=True)


# Keep only the test data times based on submission file
tmp_df = keep_forecast_times(reframed_test_df,
                             submission_df=submission_df)

test_X, test_y = split_x_y(tmp_df, [f'{TARGET_COLUMN}(t+{N_OUT-1})'])

#display(test_y.head(10))
#display(test_X.head(10))


# MinMaxScaler was used but feel free to change
scaler = MinMaxScaler(feature_range=(0, 1))

# TRAIN SET
train_X_scaled = scaler.fit_transform(train_X)
train_X_scaled = pd.DataFrame(train_X_scaled,
                              columns=train_X.columns,
                              index=train_X.index)

# TEST SET
test_X_scaled = scaler.fit_transform(test_X)
test_X_scaled = pd.DataFrame(test_X_scaled,
                             columns=test_X.columns,
                             index=test_X.index)


# Normalizer was used but feel free to change

normalizer = Normalizer()

# TRAIN SET

train_X_scaled_normalized = normalizer.fit_transform(train_X_scaled)
train_X_scaled_normalized = pd.DataFrame(train_X_scaled_normalized,
                                         columns=train_X.columns,
                                         index=train_X.index)

# TEST SET
test_X_scaled_normalized = normalizer.fit_transform(test_X_scaled)
test_X_scaled_normalized = pd.DataFrame(test_X_scaled_normalized,
                                        columns=test_X.columns,
                                        index=test_X.index)
                                        

# Keep original processed data for later use
original_train_X_scaled_normalized = train_X_scaled_normalized.copy()
original_train_X = train_X.copy()
original_train_y = train_y.copy()
original_test_X = test_X.copy()



# Split train set to train and validation set for participant's internal validation

train_X, val_X, train_y, val_y = train_test_split(original_train_X_scaled_normalized,
                                                  original_train_y,
                                                  test_size=0.33,
                                                  random_state=4)
test_X = test_X_scaled_normalized.copy()


# create model instance
# seed=56456462
# ##### XGB classifer

# xgb_model = XGBClassifier(random_state=seed, n_estimators=1000)
# xgb_model.fit(train_X, train_y)

# ## random forest

# rf_model = RandomForestClassifier(n_estimators=1000, random_state=seed)
# rf_model.fit(train_X, train_y)

# ######## gradient boosting ########
# gb_model = GradientBoostingClassifier(random_state=seed)
# gb_model.fit(train_X, train_y)

# ######## svc ############ 
# svc_model = SVC(probability=True, random_state=seed)
# svc_model.fit(train_X, train_y)

######## ensemble ################
# ######## ensemble 3 models
# rf_clf = RandomForestClassifier(random_state=42)
# #gb_clf = GradientBoostingClassifier(random_state=42)
# svm_clf = SVC(probability=True, random_state=42)
# xgb_clf = XGBClassifier(random_state=42)

# Create the ensemble model using VotingClassifier
# MLen_model = VotingClassifier(
#     estimators=[('rf', rf_model), ('svm', svm_clf), ('xgb', xgb_model)],
#     voting='soft'  # Use 'soft' voting to get class probabilities for better performance
# )
# en_model.fit(train_X, train_y)


# ########### cnn ############## model
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from tensorflow.keras.layers import LSTM

# # Assuming you have your dataset loaded into a pandas DataFrame called 'data'
# # Make sure your DataFrame has features as columns and the target (Parkinson's disease) as a separate column

# # Assuming your features are stored in X and the target in y
# #X = data.drop('Parkinsons', axis=1)
# #y = data['Parkinsons']

# # Split the data into training and testing sets
# #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train = train_X
# y_train = train_y
# X_test = val_X
# y_test = val_y
# # Reshape the data for CNN input (assuming 1D data, modify the input shape if needed)
# input_shape = (X_train.shape[1], 1)
# X_train = X_train.values.reshape(-1, X_train.shape[1], 1)
# X_test = X_test.values.reshape(-1, X_test.shape[1], 1)

# # Normalize the data to values between 0 and 1
# X_train = X_train / np.max(X_train)
# X_test = X_test / np.max(X_test)

# # Create the CNN model
# cnn_model = tf.keras.Sequential([
#     tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
#     tf.keras.layers.MaxPooling1D(pool_size=2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# # Compile the model
# cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Train the model
# cnn_model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

# ########
# # Create the LSTM model
# # lstm_model = tf.keras.Sequential([
# #     LSTM(64, input_shape=input_shape, activation='relu', return_sequences=True),
# #     tf.keras.layers.Dropout(0.5),
# #     LSTM(32, activation='relu'),
# #     tf.keras.layers.Dropout(0.5),
# #     tf.keras.layers.Dense(1, activation='sigmoid')
# # ])

# # # Compile the model
# # lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # # Train the model using validation data for early stopping
# # #model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_val, y_val))
# # lstm_model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))


# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier
# from xgboost import XGBClassifier
# from sklearn.linear_model import LogisticRegression

# # Assuming you have your dataset loaded into a pandas DataFrame called 'data'
# # Make sure your DataFrame has features as columns and the target (Parkinson's disease wearing-off) as a separate column

# X_val = X_test
# y_val = y_test

# # Create the individual models
# model1 = RandomForestClassifier(random_state=42)
# model2 = XGBClassifier(random_state=42)

# # Train the individual models
# model1.fit(X_train.reshape(X_train.shape[0], -1), y_train)
# model2.fit(X_train.reshape(X_train.shape[0], -1), y_train)

# # Train the CNN model
# model = tf.keras.Sequential([
#     tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
#     tf.keras.layers.MaxPooling1D(pool_size=2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_val, y_val))

# # Get predictions from the individual models
# ml_model1_predictions = model1.predict(X_val.reshape(X_val.shape[0], -1))
# ml_model2_predictions = model2.predict(X_val.reshape(X_val.shape[0], -1))
# cnn_model_predictions = model.predict(X_val)

# # Create the meta-features for the meta-model (stacking)
# meta_features = np.column_stack((ml_model1_predictions, ml_model2_predictions, cnn_model_predictions))

# # Train the meta-model (stacking) on the meta-features
# meta_model = LogisticRegression()
# meta_model.fit(meta_features, y_val)

# # Get predictions from the individual models for the test set
# ml_model1_test_predictions = model1.predict(X_test.reshape(X_test.shape[0], -1))
# ml_model2_test_predictions = model2.predict(X_test.reshape(X_test.shape[0], -1))
# cnn_model_test_predictions = model.predict(X_test)

# # Create the meta-features for the test set
# test_meta_features = np.column_stack((ml_model1_test_predictions, ml_model2_test_predictions, cnn_model_test_predictions))

# # Make predictions using the meta-model on the test set
# meta_model_predictions = meta_model.predict(test_meta_features)

# # Evaluate the meta-model (stacking) performance
# print("Meta-Model (Stacking) Results:")
# print(classification_report(y_test, meta_model_predictions))



# # Evaluate the model on the test set
# from sklearn.metrics import accuracy_score, classification_report
# y_pred_probs = cnn_model.predict(X_test)
# y_pred = (y_pred_probs >= 0.5).astype(int)

# # Calculate the accuracy of the CNN model
# accuracy = accuracy_score(y_test, y_pred)
# print("CNN Model Accuracy:", accuracy)

# val_classification_report = classification_report(y_test, y_pred)
# print("Validation Classification Report:")
# print(val_classification_report)

# # y_pred_probs = None
# # y_pred = None
# # val_classification_report = None


# y_pred_probs = lstm_model.predict(X_test)
# y_pred = (y_pred_probs >= 0.5).astype(int)

# # Calculate the accuracy of the CNN model
# accuracy = accuracy_score(y_test, y_pred)
# print("CNN Model Accuracy:", accuracy)

# val_classification_report = classification_report(y_test, y_pred)
# print("Validation Classification Report:")
# print(val_classification_report)

# y_pred_probs = None
# y_pred = None
# val_classification_report = None


# rf_clf = RandomForestClassifier(random_state=seed)
# gb_clf = GradientBoostingClassifier(random_state=seed)
# svm_clf = SVC(probability=True, random_state=seed)
# xgb_clf = XGBClassifier(random_state=seed)

# en_model = VotingClassifier(
#     estimators=[('rf', rf_clf), ('svm', svm_clf),('gb', gb_clf), ('xgb', xgb_clf)],
#     voting='soft'  # Use 'soft' voting to get class probabilities for better performance
# )
# en_model.fit(train_X, train_y)

# # Ensemble using VotingClassifier (voting='hard' for majority voting)
# ensemble_model = VotingClassifier(estimators=[('ml_model1', model1), ('ml_model2', model2), ('cnn_model', cnn_model)], voting='hard')
# ensemble_model.fit(X_train, y_train)

# # Get ensemble predictions
# ensemble_predictions = ensemble_model.predict(X_val)

# # Combine predictions using majority voting
# final_predictions = np.round((ml_model1_predictions + ml_model2_predictions + cnn_model_predictions) / 3).astype(int)

# # Compare ensemble_predictions and final_predictions
# # You can use accuracy_score, classification_report, etc. to evaluate the performance



seed=56456462
rf_clf = RandomForestClassifier(random_state=seed)
#gb_clf = GradientBoostingClassifier(random_state=seed)
svm_clf = SVC(probability=True, random_state=seed)
xgb_clf = XGBClassifier(random_state=seed)

en_model = VotingClassifier(
    estimators=[('rf', rf_clf), ('svm', svm_clf), ('xgb', xgb_clf)],
    voting='hard'  # Use 'soft' voting to get class probabilities for better performance
)
en_model.fit(train_X, train_y)

# y_pred_probs = en_model.predict(val_X)
# y_pred = (y_pred_probs >= 0.5).astype(int)

# # Calculate the accuracy of the CNN model
# #accuracy = accuracy_score(y_test, y_pred)
# #print("CNN Model Accuracy:", accuracy)

# val_classification_report = classification_report(val_y, y_pred)
# print("Validation Classification Report:")
# print(val_classification_report)


# Make forecasts for validation set

#xgb_model = xgb_model
#xgb_model = rf_model
#xgb_model = gb_model
#xgb_model = svc_model
#xgb_model = en_model

forecasts = en_model.predict(
  val_X
)

# Get the probability for 1s class

forecasts_proba = (forecasts >= 0.5).astype(int)
# forecasts_proba = xgb_model.predict_proba(
#   val_X
# )[:, 1]

# Transform as dataframe with timestamp

forecasts_output = pd.DataFrame(
  {
    'participant': [USER] * len(forecasts),
    'forecasted_wearing_off': forecasts,
    'forecasted_wearing_off_probability': forecasts_proba,
    'ground_truth': val_y.values.flatten(),
  },
  index=val_X.index
)

# Sort by timestamp

forecasts_output.sort_index(inplace=True)


# Make forecasts for test set and submission

forecasts_test = en_model.predict(
  test_X
)

# Get the probability for 1s class
# forecasts_proba_test = xgb_model.predict_proba(
#   test_X
# )[:, 1]

forecasts_proba_test = (forecasts_test >= 0.5).astype(int)

# Transform as dataframe with timestamp
forecasts_output_test = pd.DataFrame(
  {
    'participant': [USER] * len(forecasts_test),
    'forecasted_wearing_off': forecasts_test,
    'forecasted_wearing_off_probability': forecasts_proba_test,
    'ground_truth': test_y.values.flatten(),
  },
  index=test_X.index
)

# Sort by timestamp
# forecasts_output_test.sort_index(inplace=True)

# forecasts_test = None
# forecasts_proba_test = None



# For Validation Set

# Plot ground truth, & predicted probability on the same plot to show the difference

plt.figure(figsize=FIGSIZE)
plt.plot(forecasts_output.ground_truth,
         label='actual', color='red', marker='o',)
plt.plot(forecasts_output.forecasted_wearing_off_probability,
         label='predicted', color='blue', marker='o')
# plt.plot(forecasts_output.forecasted_wearing_off,
#          label='predicted', color='blue', marker='o')
plt.legend()

# Dashed horizontal line at 0.5

plt.axhline(0.5, linestyle='--', color='gray')

# Set y-axis label

plt.ylabel('Wearing-off Forecast Probability')

# Set x-axis label

plt.xlabel('Time')

# Set title

plt.title(
    f"Forecasted vs Actual Wearing-off for {USER.upper()}, Validation Set")

#plt.savefig("ForecastedPlot.png")

plt.show()



# For Test Set

# Plot ground truth, & predicted probability on the same plot to show the difference

plt.figure(figsize=FIGSIZE)
plt.plot(forecasts_output_test.ground_truth,
         label='actual', color='red', marker='o',)
plt.plot(forecasts_output_test.forecasted_wearing_off_probability,
         label='predicted', color='blue', marker='o')
# plt.plot(forecasts_output_test.forecasted_wearing_off,
#          label='predicted', color='blue', marker='o')
plt.legend()

# Dashed horizontal line at 0.5

plt.axhline(0.5, linestyle='--', color='gray')

# Dashed vertical lines on each hour

for i in forecasts_output_test.index:
  if pd.Timestamp(i).minute == 0:
    plt.axvline(i, linestyle='--', color='gray')

# Set y-axis label

plt.ylabel('Wearing-off Forecast Probability')

# Set x-axis label

plt.xlabel('Time')

# Set title

plt.title(f"Forecasted vs Actual Wearing-off for {USER.upper()}, Test Set")

plt.show()


# Plot confusion matrix for each participant


# Set labels for confusion matrix

labels = ['No Wearing-off', 'Wearing-off']

# Calculate confusion matrix

conf_matrix = confusion_matrix(forecasts_output.ground_truth,
                               forecasts_output.forecasted_wearing_off)
# Plot confusion matrix

plt.figure(figsize=FIGSIZE_CM)
sns.heatmap(conf_matrix,
            xticklabels=labels, yticklabels=labels,
            annot=True, fmt=".2f", cmap='Blues_r')

# Set y-axis label

plt.ylabel('True class')

# Set x-axis label

plt.xlabel('Predicted class')

# Set title

plt.title(f"confusion matrix for {USER.upper()}, Validation Set")

plt.show()



# Calculate fpr, tpr, thresholds

fpr, tpr, thresholds = metrics.roc_curve(forecasts_output.sort_index().ground_truth,
                                         forecasts_output.sort_index().forecasted_wearing_off_probability)

######################
# Evaluate predictions with f1 score, recall, precision, accuracy, auc-roc, auc-prc

model_metric_scores = pd.DataFrame(
  [
    metrics.f1_score(
      forecasts_output.ground_truth,
      forecasts_output.forecasted_wearing_off),
    metrics.recall_score(
      forecasts_output.ground_truth,
      forecasts_output.forecasted_wearing_off),
    metrics.precision_score(
      forecasts_output.ground_truth,
      forecasts_output.forecasted_wearing_off),
    metrics.accuracy_score(
      forecasts_output.ground_truth,
      forecasts_output.forecasted_wearing_off),
    metrics.auc(fpr, tpr),
    metrics.average_precision_score(
      forecasts_output.sort_index().ground_truth,
      forecasts_output.sort_index().forecasted_wearing_off_probability)
  ],
  index=['f1 score', 'recall', 'precision', 'accuracy', 'auc-roc', 'auc-prc'],
  columns=['metrics']
).T.round(3).assign(participant=USER)
model_metric_scores.set_index(['participant'], inplace=True)

######################
# Generate classification report

model_classification_report = pd.DataFrame(
  classification_report(
    forecasts_output.ground_truth,
    forecasts_output.forecasted_wearing_off,
    output_dict=True
  )
).T.round(3).assign(participant=USER)
# Set index's name to 'classification report'

model_classification_report.index.name = 'classification report'

# Remove row that has 'accuracy' as index

model_classification_report = model_classification_report.drop(
  ['accuracy'], axis=0)

model_classification_report = model_classification_report.reset_index()
model_classification_report.set_index(
    ['participant', 'classification report'], inplace=True)

model_metric_scores.reset_index(inplace=True)
model_classification_report.reset_index(inplace=True)

#display(model_metric_scores)
#display(model_classification_report)


# Load submission file as template

template_df = pd.read_csv(f'{TEST_DATA_PATH}/submission.csv', index_col=0)
template_df['Timestamp'] = pd.to_datetime(template_df['Timestamp'])
template_df['reframed_timestamp'] = pd.to_datetime(
    template_df['reframed_timestamp'])
template_df.head(5)



# Merge template with forecasts_output_test

output_df = template_df.merge(
  forecasts_output_test.reset_index(),
  left_on=['reframed_timestamp', 'participant'],
  right_on=['timestamp', 'participant']
)[
  list(template_df.columns)[:-1] + ['forecasted_wearing_off']
].rename(
  columns={'forecasted_wearing_off': 'final_wearing_off'}
)
output_df.head(5)


for_saving_file = 'submissions/submission_BAUCVL.csv'

if os.path.exists(for_saving_file):
  # Append csv
  output_df.to_csv(for_saving_file, index=True, mode='a', header=False)
else:
  # Create csv
  output_df.to_csv(for_saving_file, index=True)



# import matplotlib.pyplot as plt

# # Average values for class 0 and class 1
# average_precision_class0 = 0.933
# average_recall_class0 = 0.997
# average_f1_score_class0 = 0.964
# average_accuracy_class0 = 0.932

# average_precision_class1 = 0.440
# average_recall_class1 = 0.247
# average_f1_score_class1 = 0.314

# # List of metrics
# metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy']

# # Average values for class 0
# average_values_class0 = [average_precision_class0, average_recall_class0, average_f1_score_class0, average_accuracy_class0]

# # Create the bar chart for class 0
# plt.bar(metrics, average_values_class0)
# plt.xlabel('Metrics')
# plt.ylabel('Average Value')
# plt.title('Average Evaluation Metrics for Class 0')
# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np

# # Average values for class 0 and class 1
# average_precision_class0 = 0.933
# average_recall_class0 = 0.997
# average_f1_score_class0 = 0.964
# average_accuracy = 0.932

# average_precision_class1 = 0.440
# average_recall_class1 = 0.247
# average_f1_score_class1 = 0.314

# # List of metrics
# metrics = ['Precision', 'Recall', 'F1-Score']
# NWmetrics = ['Precision', 'Recall', 'F1-Score','Accuracy']

# # Average values for class 0 and class 1
# average_values_class0 = [average_precision_class0, average_recall_class0, average_f1_score_class0]
# average_values_class1 = [average_precision_class1, average_recall_class1, average_f1_score_class1]

# # Positions for the bars
# positions = np.arange(len(metrics))

# # Width for each bar
# bar_width = 0.35

# # Create the grouped bar chart
# plt.bar(positions - bar_width/2, average_values_class0, bar_width, label='Class 0')
# plt.bar(positions + bar_width/2, average_values_class1, bar_width, label='Class 1')

# # Add a separate bar for accuracy
# plt.bar(len(metrics) + bar_width, average_accuracy, bar_width, label='Accuracy', color='red')

# # Customize the plot
# plt.xticks(positions, metrics)
# plt.xlabel('Metrics')
# plt.ylabel('Average Value')
# plt.title('Average Evaluation Metrics for Class 0 and Class 1')
# plt.legend()
# plt.savefig("AverageValue2.png")

# # Display the plot
# plt.show()


# import matplotlib.pyplot as plt
# import numpy as np

# # Average values for class 0 and class 1
# average_precision_class0 = 0.933
# average_recall_class0 = 0.997
# average_f1_score_class0 = 0.964
# average_accuracy = 0.932

# average_precision_class1 = 0.440
# average_recall_class1 = 0.247
# average_f1_score_class1 = 0.314

# # List of metrics
# metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy']

# # Average values for class 0 and class 1
# average_values_class0 = [average_precision_class0, average_recall_class0, average_f1_score_class0, average_accuracy]
# average_values_class1 = [average_precision_class1, average_recall_class1, average_f1_score_class1]

# # Positions for the bars
# positions = np.arange(len(metrics) * 2)

# # Width for each bar
# bar_width = 0.35

# # Create the grouped bar chart
# plt.bar(positions[::2], average_values_class0, bar_width, label='Class 0')
# plt.bar(positions[1::2], average_values_class1, bar_width, label='Class 1')

# # Add a separate bar for accuracy
# plt.bar(len(metrics) * 2, average_accuracy, bar_width, label='Accuracy', color='orange')

# # Add text "Accuracy" below the accuracy bar
# plt.annotate('Accuracy', xy=(len(metrics) * 2, average_accuracy), xytext=(len(metrics) * 2, average_accuracy - 0.1),
#              ha='center', va='center', color='black')

# # Customize the plot
# plt.xticks(positions[1::2], metrics)
# plt.xlabel('Metrics')
# plt.ylabel('Average Value')
# plt.title('Average Evaluation Metrics for Class 0 and Class 1')
# plt.legend()

# # Display the plot
# plt.show()






