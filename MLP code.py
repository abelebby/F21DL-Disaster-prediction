# Flood Prediction in Nepal: Multi-Layer Perceptron (MLP)
#
# Environment : Python 3 (Spyder / VSCode)
# Libraries   : pandas, numpy, matplotlib, scikit-learn, tensorflow
# Setup       : Place "nepal_flood_clean.csv" in the same working directory.
#
# Input       : Cleaned historical meteorological dataset (daily measurements).
# Output      : Classification metrics on unseen future data and Train/Val loss curve.
# Metrics     : Precision, Recall, F1-Score, PR-AUC, and Day-1 Flood Onset Recall.
#-------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Preprocessing, evaluation metrics, and Keras model layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, precision_recall_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.constraints import MaxNorm

# Fix random seeds for TensorFlow and numpy to ensure reproducible results
tf.random.set_seed(42)
np.random.seed(42)


# Data Loading and Feature Engineering

# Load the cleaned dataset, parse dates, and sort chronologically within each station
dataFrame = pd.read_csv("nepal_flood_clean.csv")
dataFrame['date'] = pd.to_datetime(dataFrame['date'])
dataFrame = dataFrame.sort_values(['location', 'date']).reset_index(drop=True)

def sum3Days(column):
    """ Compute the cumulative rainfall over a 3-day rolling window to capture short-term soil saturation."""
    return column.rolling(3).sum()

def sum7Days(column):
    """ Compute the cumulative rainfall over a 7-day rolling window to capture medium-term watershed saturation."""
    return column.rolling(7).sum()

# Group by station to isolate time-dependent operations and prevent cross-station data leakage
byStation = dataFrame.groupby('location')

# Get rainfall from 1 and 2 days ago to account for the time water takes to reach the river
dataFrame['rain1'] = byStation['precipitation_mm'].shift(1)
dataFrame['rain2'] = byStation['precipitation_mm'].shift(2)

# Sum rainfall over the last 3 and 7 days to measure how saturated the ground already is
dataFrame['rain3d'] = byStation['precipitation_mm'].transform(sum3Days)
dataFrame['rain7d'] = byStation['precipitation_mm'].transform(sum7Days)

# Get yesterday's discharge for current river level, and shift tomorrow's discharge up as the target to predict
dataFrame['dischargeYesterday'] = byStation['river_discharge_m3s'].shift(1)
dataFrame['dischargeTomorrow'] = byStation['river_discharge_m3s'].shift(-1)

# Encode the month cyclically using sine and cosine to help the model capture seasonal monsoon patterns
# This prevents an artificial jump between December (12) and January (1), keeping winter months adjacent
month = dataFrame['date'].dt.month
dataFrame['monthSin'] = np.sin(2 * np.pi * month / 12)
dataFrame['monthCos'] = np.cos(2 * np.pi * month / 12)

# Remove rows with missing values, then sort globally by date
dataFrame = dataFrame.dropna(subset=['rain7d', 'dischargeYesterday', 'dischargeTomorrow'])
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
    """ Compute next-day flood labels and normalize river discharges for each station."""
    alert = alert.copy()
    alert['threshold'] = alert['location'].map(stationThresholds)
    alert['alertTomorrow'] = (alert['dischargeTomorrow'] > alert['threshold']).astype(int)
    alert['ratioToday'] = alert['river_discharge_m3s'] / alert['threshold']
    alert['ratioYesterday'] = alert['dischargeYesterday'] / alert['threshold']
    return alert

# Apply threshold mapping and target creation across train, validation, and test splits
train = addAlertColumns(train)
val = addAlertColumns(val)
test = addAlertColumns(test)

# Create a persistence baseline which basically predicts a flood tomorrow if the river is already flooding today
persistence = (test['ratioToday'] > 1).astype(int)


# Preprocessing and Scaling

# Select the 12 input features: recent rainfall, soil moisture, atmospheric conditions, seasonality, and normalized river flow
features = [
    'precipitation_mm', 'rain1', 'rain2', 'rain3d', 'rain7d',
    'soil_moisture_0_100cm_m3m3', 'precipitation_hours',
    'relative_humidity_mean_pct', 'monthSin', 'monthCos',
    'ratioToday', 'ratioYesterday'
]


# Standardize features so NN weights converge smoothly
# Fit the scaler strictly on training data to prevent validation or test data from leaking
scaler = StandardScaler()
xTrain = scaler.fit_transform(train[features])
xVal = scaler.transform(val[features])
xTest = scaler.transform(test[features])

# Extract binary flood targets as numpy arrays for TensorFlow model training and evaluation
yTrain = train['alertTomorrow'].values
yVal = val['alertTomorrow'].values
yTest = test['alertTomorrow'].values


# MLP Architecture and Fine-Tuned Training

# Compact MLP with two small hidden layers to capture non-linear patterns without overfitting on weather noise
# Uses tanh activations and MaxNorm weight constraints for stability, ending with a sigmoid flood probability
modelMLP = Sequential([
    Input(shape=(len(features),)),
    Dense(32, activation='tanh', kernel_constraint=MaxNorm(1.0)),
    Dense(16, activation='tanh', kernel_constraint=MaxNorm(1.0)),
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

# Use Adam with a learning rate (0.0005) for steady convergence, and binary cross-entropy as the loss for binary flood classification
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
modelMLP.compile(optimizer=optimizer, loss='binary_crossentropy')

# Balance validation weights so the proportion of flood events matches the training set
# Since flood frequency varies across years, this keeps the validation error fair and comparable
trainRate = yTrain.mean()
valRate = yVal.mean()
valWeights = np.where(yVal == 1, (trainRate / valRate), ((1 - trainRate) / (1 - valRate)))

# Train the model for up to 100 passes (64 samples at a time) using the adjusted validation set
# Callbacks automatically lower the learning rate and stop training early once performance stops improving
historyMLP = modelMLP.fit(
    xTrain, yTrain,
    validation_data=(xVal, yVal, valWeights),
    epochs=100, batch_size=64,
    callbacks=[earlyStop, reduceLr],
    verbose=1
)


# Threshold Optimization and Evaluation

# Generate next-day flood probabilities for the validation and test sets
# Flatten the predictions into 1D arrays to simplify threshold tuning and evaluation
pVal = modelMLP.predict(xVal, verbose=0).ravel()
pTest = modelMLP.predict(xTest, verbose=0).ravel()

# Test different probability thresholds on the validation set to find the best balance
# between catching floods and avoiding false alarms
# Pick the threshold that maximizes the F1 score instead of using a default 0.5 cutoff
prec, rec, thr = precision_recall_curve(yVal, pVal)
f1Val = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
bestIndex = np.argmax(f1Val)
decisionThreshold = thr[bestIndex]

# Apply the tuned threshold to turn test probabilities into final binary flood alerts (1 or 0)
classPredictions = (pTest >= decisionThreshold).astype(int)

# Measure overall ranking performance with PR-AUC (well-suited for rare flood events) and compare against a random guess baseline
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
plt.plot(historyMLP.history['loss'], label='Training Loss', color='#008080', linewidth=2)
plt.plot(historyMLP.history['val_loss'], label='Validation Loss', color='#00CCCC', linewidth=2)

# Add chart labels and a fixed y-axis scale for easier comparison
plt.title('MLP Training and Validation Loss', fontsize=14, fontweight='bold', color='#333333')
plt.xlabel('Epochs', fontsize=11, fontweight='bold', color='#555555')
plt.ylabel('Binary Crossentropy Loss', fontsize=11, fontweight='bold', color='#555555')
plt.ylim(0, 0.6)
plt.legend(frameon=True, shadow=True, facecolor='white')
plt.gca().set_facecolor('#F8F9FA')
plt.show()