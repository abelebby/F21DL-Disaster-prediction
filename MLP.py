# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 11:00:45 2026

@author: User
"""

# Flood Prediction in Nepal: Multi-Layer Perceptron (MLP)
# --------------------------------------------------------------------------------------------

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, precision_recall_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.constraints import MaxNorm

# Fixed random seeds so that weight initialization and results are reproducible
tf.random.set_seed(42)
np.random.seed(42)


# Data Loading and Feature Engineering

dataFrame = pd.read_csv(Path(__file__).resolve().parent / "data" / "processed" / "nepal_flood_clean.csv")
dataFrame['date'] = pd.to_datetime(dataFrame['date'])
dataFrame = dataFrame.sort_values(['location', 'date']).reset_index(drop=True)   # sort by station first then chronologically


def sum3Days(column):
    """ Compute the cumulative sum over a 3-day rolling window (current day + 2 previous days)
    to capture short-term rain accumulation. """
    return column.rolling(3).sum()


def sum7Days(column):
    """ Compute the cumulative sum over a 7-day rolling window to capture ground saturation
    over a full week of rain. """
    return column.rolling(7).sum()


# Group by station before shifting so values from one river do not carry over to the next station
byStation = dataFrame.groupby('location')
dataFrame['rain1'] = byStation['precipitation_mm'].shift(1)                      # rain at day j-1
dataFrame['rain2'] = byStation['precipitation_mm'].shift(2)                      # rain at day j-2
dataFrame['rain3d'] = byStation['precipitation_mm'].transform(sum3Days)          # 3-day cumulative rain
dataFrame['rain7d'] = byStation['precipitation_mm'].transform(sum7Days)          # 7-day cumulative rain
dataFrame['dischargeYesterday'] = byStation['river_discharge_m3s'].shift(1)      # river level at day j-1
dataFrame['dischargeTomorrow'] = byStation['river_discharge_m3s'].shift(-1)      # target to predict (day j+1)

# Cyclical encoding of the month so the model understands that December (12) and January (1) are close
month = dataFrame['date'].dt.month
dataFrame['monthSin'] = np.sin(2 * np.pi * month / 12)
dataFrame['monthCos'] = np.cos(2 * np.pi * month / 12)

# Remove rows with missing values created by the 7-day rolling window and the -1/+1 shifts
dataFrame = dataFrame.dropna(subset=['rain7d', 'dischargeYesterday', 'dischargeTomorrow'])
dataFrame = dataFrame.sort_values('date').reset_index(drop=True)                 # global chronological order for splitting


# Chronological Split and Alert Thresholds

# Time-based split (no random shuffle) to prevent using future weather to predict the past
splitDate = dataFrame['date'].quantile(0.80)
fullTrain = dataFrame[dataFrame['date'] < splitDate].copy()                      # first 80% of the timeline (past)
test = dataFrame[dataFrame['date'] >= splitDate].copy()                          # last 20% of the timeline (future)

valSplitIndex = int(len(fullTrain) * 0.80)
train = fullTrain.iloc[:valSplitIndex].copy()                                    # 64% of total data for training
val = fullTrain.iloc[valSplitIndex:].copy()                                      # 16% of total data for validation

# A flood alert is defined as the top 5% highest river discharges for each specific station.
# Computed strictly on the train set to avoid any data leakage from validation or test sets.
stationThresholds = train.groupby('location')['dischargeTomorrow'].quantile(0.95)


def addAlertColumns(alert):
    """ Map each station's flood threshold to the dataframe, create the binary target label
    (alertTomorrow = 1 if tomorrow's discharge exceeds the 95th percentile), and normalize
    today's and yesterday's river discharges by the local threshold. """
    alert = alert.copy()
    alert['threshold'] = alert['location'].map(stationThresholds)
    alert['alertTomorrow'] = (alert['dischargeTomorrow'] > alert['threshold']).astype(int)
    alert['ratioToday'] = alert['river_discharge_m3s'] / alert['threshold']      # > 1 means the river is already flooding today
    alert['ratioYesterday'] = alert['dischargeYesterday'] / alert['threshold']
    return alert


train = addAlertColumns(train)
val = addAlertColumns(val)
test = addAlertColumns(test)

# Compute persistence before log1p transformation (if ratioToday > 1, log1p would change the cutoff to log(2))
persistence = (test['ratioToday'] > 1).astype(int)


# Preprocessing and Scaling

# Initial full list of candidate meteorological/hydrological variables
features = [
    'precipitation_mm', 'rain1', 'rain2', 'rain3d', 'rain7d',
    'soil_moisture_0_100cm_m3m3', 'precipitation_hours',
    'relative_humidity_mean_pct', 'monthSin', 'monthCos',
    'ratioToday', 'ratioYesterday'
]

# Reduced subset keeping only the main variables
features = [
    'precipitation_mm', 'rain1', 'rain2', 'rain3d', 'rain7d',
    'precipitation_hours', 'ratioToday', 'ratioYesterday'
]

# Apply log(1 + x) transformation to compress extreme spikes and help the MLP converge
for col in features:
    if col in train.columns:
        train[col] = np.log1p(np.maximum(train[col], 0))                         # np.maximum(..., 0) prevents log of negative values
        val[col] = np.log1p(np.maximum(val[col], 0))
        test[col] = np.log1p(np.maximum(test[col], 0))

# Standardize features (mean = 0, variance = 1) fitted only on the training set
scaler = StandardScaler()
xTrain = scaler.fit_transform(train[features])
xVal = scaler.transform(val[features])
xTest = scaler.transform(test[features])

yTrain = train['alertTomorrow'].values
yVal = val['alertTomorrow'].values
yTest = test['alertTomorrow'].values


# MLP Architecture and Training

# Two hidden layers (32 and 16 neurons) with tanh activation and MaxNorm(1.0) weight constraint
# to keep weights bounded and prevent overfitting without having an impact on the crossentropy loss
modelMLP = Sequential([
    Input(shape=(len(features),)),
    Dense(32, activation='tanh', kernel_constraint=MaxNorm(1.0)),
    Dense(16, activation='tanh', kernel_constraint=MaxNorm(1.0)),
    Dense(1, activation='sigmoid')                                               # outputs a flood probability between 0 and 1
])

# The learning rate is divided by 2 if validation loss stops improving for 3 epochs
reduceLr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-5,
    verbose=0
)

modelMLP.compile(optimizer='adam', loss='binary_crossentropy')

# Stop training when val_loss fails to drop by at least 0.001 over 5 consecutive epochs
earlyStop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=5,
    restore_best_weights=True                                                    # roll back to the best epoch's weights
)

historyMLP = modelMLP.fit(
    xTrain,
    yTrain,
    validation_data=(xVal, yVal),
    epochs=100,
    batch_size=64,
    callbacks=[earlyStop, reduceLr],
    verbose=1
)


# Threshold Optimization and Evaluation

pVal = modelMLP.predict(xVal, verbose=0).ravel()                                 # predicted probabilities on validation set
pTest = modelMLP.predict(xTest, verbose=0).ravel()                               # predicted probabilities on test set

# Instead of using a naive 0.5 threshold (which fails on rare events), scan all possible
# thresholds on the validation set and pick the one that maximizes the F1 score
prec, rec, thr = precision_recall_curve(yVal, pVal)
f1Val = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)                 # + 1e-9 avoids division by zero
bestIndex = np.argmax(f1Val)
decisionThreshold = thr[bestIndex]

# Apply the validation-tuned threshold to classify the unseen future test set
classPredictions = (pTest >= decisionThreshold).astype(int)

print("\n RESULTS ON FUTURE DATA (TEST SET) ")
print("Selected decision threshold :", round(decisionThreshold, 3))
print("Precision :", precision_score(yTest, classPredictions, zero_division=0))
print("Recall :", recall_score(yTest, classPredictions, zero_division=0))
print("F1 Score :", f1_score(yTest, classPredictions, zero_division=0))
print("PR-AUC :", average_precision_score(yTest, pTest), "(random baseline :", yTest.mean(), ")")

print("\nPersistence baseline -> Precision :", precision_score(yTest, persistence, zero_division=0),
      "| Recall :", recall_score(yTest, persistence, zero_division=0),
      "| F1 :", f1_score(yTest, persistence, zero_division=0))

# Plot learning curves to visually check that validation loss does not diverge from training loss
plt.figure(figsize=(8, 5))
plt.plot(historyMLP.history['loss'], label='Training (Loss)')
plt.plot(historyMLP.history['val_loss'], label='Validation (Val Loss)')
plt.title('Train vs Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(0, 0.5)
plt.legend()
plt.show()