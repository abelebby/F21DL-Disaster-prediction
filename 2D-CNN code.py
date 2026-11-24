# Flood Prediction in Nepal: 2D Convolutional Neural Network (2D-CNN)
#
# Environment : Python 3 (Spyder / VSCode)
# Libraries   : pandas, numpy, matplotlib, scikit-learn, tensorflow
# Setup       : Place "nepal_flood_clean.csv" in the same working directory.
#
# Input       : Cleaned historical meteorological dataset (4-day x 4-variable grids).
# Output      : Classification metrics on unseen future data and Train/Val loss curve.
# Metrics     : Precision, Recall, F1-Score, PR-AUC, and Day-1 Flood Onset Recall.
#-------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Preprocessing, evaluation metrics, and Keras layers for a 2D Convolutional Neural Network
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, precision_recall_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten

# Fix random seeds for TensorFlow and numpy to ensure reproducible results
tf.random.set_seed(42)
np.random.seed(42)


# Data Loading and Temporal Lag Creation

# Load the cleaned dataset, parse dates, and sort chronologically within each station
dataFrame = pd.read_csv("nepal_flood_clean.csv")
dataFrame['date'] = pd.to_datetime(dataFrame['date'])
dataFrame = dataFrame.sort_values(['location', 'date']).reset_index(drop=True)

# Group by station to isolate time-dependent operations and prevent cross-station data leakage
byStation = dataFrame.groupby('location')

def sum7Days(column):
    """ Compute the cumulative rainfall over a 7-day rolling window to match the MLP filtering criteria. """
    return column.rolling(7).sum()

dataFrame['rain7d'] = byStation['precipitation_mm'].transform(sum7Days)

# Build a 4-day historical window (days j-1, j-2, j-3) for the 4 core physical variables
for num in [1, 2, 3]:
    dataFrame[f'rain_{num}'] = byStation['precipitation_mm'].shift(num)
    dataFrame[f'hours_{num}'] = byStation['precipitation_hours'].shift(num)
    dataFrame[f'soil_{num}'] = byStation['soil_moisture_0_100cm_m3m3'].shift(num)
    dataFrame[f'discharge_{num}'] = byStation['river_discharge_m3s'].shift(num)

# Shift tomorrow's river discharge up as the target to predict
dataFrame['dischargeTomorrow'] = byStation['river_discharge_m3s'].shift(-1)

# Remove rows with missing values to keep the exact same sample set as the MLP, then sort globally by date
dataFrame = dataFrame.dropna(subset=['rain7d', 'discharge_1', 'dischargeTomorrow'])
dataFrame = dataFrame.sort_values('date').reset_index(drop=True)


# Chronological Split and Alert Thresholds

# Split the dataset chronologically (64% train, 16% validation, 20% test) instead of shuffling randomly
# This prevents future data leakage and ensures the model is evaluated strictly on unseen future dates
splitDate = dataFrame['date'].quantile(0.80)
fullTrain = dataFrame[dataFrame['date'] < splitDate].copy()
test = dataFrame[dataFrame['date'] >= splitDate].copy()

valSplitIndex = int(len(fullTrain) * 0.80)
train = fullTrain.iloc[:valSplitIndex].copy()
val = fullTrain.iloc[valSplitIndex:].copy()

# Define a flood threshold per station (top 5% of river discharge) using only training data to avoid leakage
stationThresholds = train.groupby('location')['dischargeTomorrow'].quantile(0.95)

def addAlertColumns(alert):
    """ Compute next-day flood labels and normalize current and past river discharges for each station."""
    alert = alert.copy()
    alert['threshold'] = alert['location'].map(stationThresholds)
    alert['alertTomorrow'] = (alert['dischargeTomorrow'] > alert['threshold']).astype(int)
    alert['ratioToday'] = alert['river_discharge_m3s'] / alert['threshold']
    alert['ratio_1'] = alert['discharge_1'] / alert['threshold']
    alert['ratio_2'] = alert['discharge_2'] / alert['threshold']
    alert['ratio_3'] = alert['discharge_3'] / alert['threshold']
    return alert

# Apply threshold mapping and target creation across train, validation, and test splits
train = addAlertColumns(train)
val = addAlertColumns(val)
test = addAlertColumns(test)

# Create a persistence baseline which predicts a flood tomorrow if the river is already flooding today
persistence = (test['ratioToday'] > 1).astype(int)


# 2D Grid Structuring and Scaling

# Arrange the 16 features in chronological order (rows: j-3 to j, columns: 4 physical variables)
# Ordering the columns this way allows 2D filters to scan across both time steps and variable interactions simultaneously
features = [
    'rain_3',           'hours_3',             'soil_3',                     'ratio_3',
    'rain_2',           'hours_2',             'soil_2',                     'ratio_2',
    'rain_1',           'hours_1',             'soil_1',                     'ratio_1',
    'precipitation_mm', 'precipitation_hours', 'soil_moisture_0_100cm_m3m3', 'ratioToday'
]

# Standardize features so NN weights converge smoothly
# Fit the scaler strictly on training data to prevent validation or test data from leaking
scaler = StandardScaler()
xTrainFlat = scaler.fit_transform(train[features])
xValFlat = scaler.transform(val[features])
xTestFlat = scaler.transform(test[features])

# Reshape each sample into a 4x4x1 grid (height = 4 days, width = 4 variables, channel = 1) for 2D convolution
xTrain = xTrainFlat.reshape(-1, 4, 4, 1)
xVal = xValFlat.reshape(-1, 4, 4, 1)
xTest = xTestFlat.reshape(-1, 4, 4, 1)

# Extract binary flood targets as numpy arrays for TensorFlow model training and evaluation
yTrain = train['alertTomorrow'].values
yVal = val['alertTomorrow'].values
yTest = test['alertTomorrow'].values


# 2D-CNN Architecture and Fine-Tuned Training

# Compact 2D-CNN designed to extract local patterns across the 4-day weather grid without overfitting
# Uses 2x2 filters with tanh activations and pooling before the sigmoid classification
modelCNN = Sequential([
    Input(shape=(4, 4, 1)),
    Conv2D(6, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='tanh'),
    MaxPooling2D(pool_size=(2, 1), strides=(2, 1)),
    Conv2D(6, kernel_size=(2, 2), strides=(1, 1), padding='valid', activation='tanh'),
    Flatten(),
    Dense(6, activation='tanh'),
    Dense(1, activation='sigmoid')
])

# Halve the learning rate if validation loss stalls for 3 epochs to help the model fine-tune
reduceLr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=0
)

# Stop training early if validation loss stops improving for 6 epochs and restore the best weights to avoid overfitting
earlyStop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0.0005, patience=6, restore_best_weights=True
)

# Use Adam optimizer with a conservative learning rate (0.0005) for steady convergence, and binary cross-entropy for binary flood classification
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
modelCNN.compile(optimizer=optimizer, loss='binary_crossentropy')

# Balance validation weights so the proportion of flood events matches the training set
# Since flood frequency varies across years, this keeps the validation error fair and comparable
trainRate = yTrain.mean()
valRate = yVal.mean()
valWeights = np.where(yVal == 1, (trainRate / valRate), ((1 - trainRate) / (1 - valRate)))

# Train the model for up to 100 passes (64 samples at a time) using the adjusted validation set
# Callbacks automatically lower the learning rate and stop training early once performance stops improving
historyCNN = modelCNN.fit(
    xTrain, yTrain,
    validation_data=(xVal, yVal, valWeights),
    epochs=100, batch_size=64,
    callbacks=[earlyStop, reduceLr],
    verbose=1
)


# Threshold Optimization and Evaluation

# Generate next-day flood probabilities for the validation and test sets
# Flatten the predictions into 1D arrays to simplify threshold tuning and evaluation
pVal = modelCNN.predict(xVal, verbose=0).ravel()
pTest = modelCNN.predict(xTest, verbose=0).ravel()

# Test different probability thresholds on the validation set to find the best balance
# between catching floods and avoiding false alarms
# Pick the threshold that maximizes the F1 score instead of using a default 0.5 cutoff
prec, rec, thr = precision_recall_curve(yVal, pVal)
f1Val = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
bestIndex = np.argmax(f1Val)
decisionThreshold = thr[bestIndex]

# Apply the tuned threshold to turn test probabilities into final binary flood alerts (1 or 0)
classPredictions = (pTest >= decisionThreshold).astype(int)

# Measure overall ranking performance with PR-AUC (same as before so suited for rare flood events) and compare against a random guess baseline
modelPrauc = average_precision_score(yTest, pTest)
randomBaseline = yTest.mean()

print("\n    TEST SET PERFORMANCE    ")
print(f"Operational Threshold : {decisionThreshold:.3f}")
print(f"Recall (Floods Caught): {recall_score(yTest, classPredictions, zero_division=0):.1%}")
print(f"Precision             : {precision_score(yTest, classPredictions, zero_division=0):.1%}")
print(f"F1-Score              : {f1_score(yTest, classPredictions, zero_division=0):.3f}")
print("\n")
print(f"Model PR-AUC          : {modelPrauc:.3f}")
print(f"Random Baseline       : {randomBaseline:.3f}")
print(f"Around {modelPrauc / randomBaseline:.1f}x better than random")

# Evaluate model's ability to anticipate new floods (Day 1 onsets),
# isolating days where a flood occurs tomorrow but the river is normal today.
onsetMask = (yTest == 1) & (persistence == 0)

yTestOnset = yTest[onsetMask]
predsOnset = classPredictions[onsetMask]

print("\n    FLOOD ONSET EVALUATION    ")
if len(yTestOnset) > 0:
    onsetRecall = recall_score(yTestOnset, predsOnset, zero_division=0)
    print(f"Total onsets in test set: {len(yTestOnset)}")
    print(f"Successfully anticipated: {predsOnset.sum()}")
    print(f"Anticipation Recall     : {onsetRecall:.1%}")
else:
    print("No onsets found in test set.")


# Plot training and validation loss across epochs to check how well the model learns and ensure it is not overfitting
plt.figure(figsize=(9, 5))
plt.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.3)
plt.plot(historyCNN.history['loss'], label='Training Loss', color='#008080', linewidth=2)
plt.plot(historyCNN.history['val_loss'], label='Validation Loss', color='#00CCCC', linewidth=2)

# Add chart labels and a fixed y-axis scale for easier comparison
plt.title('2D-CNN Training and Validation Loss', fontsize=14, fontweight='bold', color='#333333')
plt.xlabel('Epochs', fontsize=11, fontweight='bold', color='#555555')
plt.ylabel('Binary Crossentropy Loss', fontsize=11, fontweight='bold', color='#555555')
plt.ylim(0, 0.6)
plt.legend(frameon=True, shadow=True, facecolor='white')
plt.gca().set_facecolor('#F8F9FA')
plt.show()