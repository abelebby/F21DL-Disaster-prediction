# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 17:12:05 2026

@author: User
"""

# Flood Prediction in Nepal: 1D Convolutional Neural Network (CNN)
# --------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, precision_recall_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten

# Same random seed initialization as in the MLP
tf.random.set_seed(42)
np.random.seed(42)


# Data Loading and Temporal Lag Creation

# Same data loading and station-chronological sorting as in the MLP
dataFrame = pd.read_csv("nepal_flood_clean.csv")
dataFrame['date'] = pd.to_datetime(dataFrame['date'])
dataFrame = dataFrame.sort_values(['location', 'date']).reset_index(drop=True)

byStation = dataFrame.groupby('location')


# Build a 4-day historical window (days j-1, j-2, j-3) for the 4 core physical variables
for num in [1, 2, 3]:
    dataFrame[f'rain_{num}'] = byStation['precipitation_mm'].shift(num)           # daily rainfall at j-num
    dataFrame[f'hours_{num}'] = byStation['precipitation_hours'].shift(num)       # rain duration at j-num
    dataFrame[f'soil_{num}'] = byStation['soil_moisture_0_100cm_m3m3'].shift(num) # soil saturation at j-num
    dataFrame[f'discharge_{num}'] = byStation['river_discharge_m3s'].shift(num)   # river discharge at j-num

dataFrame['dischargeTomorrow'] = byStation['river_discharge_m3s'].shift(-1)       # target to predict (day j+1)

dataFrame = dataFrame.dropna().sort_values('date').reset_index(drop=True)


# Chronological Split and Alert Thresholds

# Same chronological split (64% train, 16% val, 20% test) and 95th percentile station threshold as in the MLP
splitDate = dataFrame['date'].quantile(0.80)
fullTrain = dataFrame[dataFrame['date'] < splitDate].copy()
test = dataFrame[dataFrame['date'] >= splitDate].copy()

valSplitIndex = int(len(fullTrain) * 0.80)
train = fullTrain.iloc[:valSplitIndex].copy()
val = fullTrain.iloc[valSplitIndex:].copy()

stationThresholds = train.groupby('location')['dischargeTomorrow'].quantile(0.95)

def addAlertColumns(part):
    """ Same target labeling and discharge normalization as in the MLP, extended here
    down to day J-3 (ratio_1, ratio_2, ratio_3) to complete the 4-day window. """
    part = part.copy()
    part['threshold'] = part['location'].map(stationThresholds)
    part['alertTomorrow'] = (part['dischargeTomorrow'] > part['threshold']).astype(int)
    part['ratioToday'] = part['river_discharge_m3s'] / part['threshold']
    part['ratio_1'] = part['discharge_1'] / part['threshold']
    part['ratio_2'] = part['discharge_2'] / part['threshold']
    part['ratio_3'] = part['discharge_3'] / part['threshold']
    return part

train = addAlertColumns(train)
val = addAlertColumns(val)
test = addAlertColumns(test)


# 2D Grid Structuring and Scaling

# Arrange the 16 features so that reshaping into (4, 4) creates a coherent spatio-temporal matrix:
# Rows = chronological progression (j-3, j-2, j-1, Today j)
# Columns = physical variables (Rain amount, Rain hours, Soil moisture, Discharge ratio)
features = [
    'rain_3',           'hours_3',             'soil_3',                     'ratio_3',
    'rain_2',           'hours_2',             'soil_2',                     'ratio_2',
    'rain_1',           'hours_1',             'soil_1',                     'ratio_1',
    'precipitation_mm', 'precipitation_hours', 'soil_moisture_0_100cm_m3m3', 'ratioToday'
]

# Same standardization as in the MLP, fitted strictly on the flat training set
scaler = StandardScaler()
xTrainFlat = scaler.fit_transform(train[features])
xValFlat = scaler.transform(val[features])
xTestFlat = scaler.transform(test[features])

# Reshape each sample from a 16-value vector into a 4x4x1 tensor (height=4 days, width=4 variables, channel=1)
xTrain = xTrainFlat.reshape(-1, 4, 4, 1)
xVal = xValFlat.reshape(-1, 4, 4, 1)
xTest = xTestFlat.reshape(-1, 4, 4, 1)

yTrain = train['alertTomorrow'].values
yVal = val['alertTomorrow'].values
yTest = test['alertTomorrow'].values


# 2D-CNN Architecture and Training

# Compact 2D-CNN scanning 2x2 local receptive fields (2 consecutive days x 2 adjacent physical variables).
# MaxPooling2D uses a (2, 1) window to halve the time dimension (4 days -> 2 steps) without losing variables.
modelCNN = Sequential([
    Input(shape=(4, 4, 1)),
    Conv2D(6, kernel_size=(2, 2), strides=(1, 1), padding='same', activation='tanh'),
    MaxPooling2D(pool_size=(2, 1), strides=(2, 1)),
    Conv2D(6, kernel_size=(2, 2), strides=(1, 1), padding='valid', activation='tanh'),
    Flatten(),                                                                   # flatten feature maps into a 1D column
    Dense(6, activation='tanh'),
    Dense(1, activation='sigmoid')
])

# Lower learning rate (0.0005 instead of default 0.001) for smoother convergence on small 4x4 grids
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
modelCNN.compile(optimizer=optimizer, loss='binary_crossentropy')

# Same EarlyStopping logic as in the MLP (with patience set to 8 epochs)
earlyStop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True
)

historyCNN = modelCNN.fit(
    xTrain,
    yTrain,
    validation_data=(xVal, yVal),
    epochs=100,
    batch_size=64,
    callbacks=[earlyStop],
    verbose=1
)


# Threshold Optimization and Evaluation

# Same validation threshold tuning, test evaluation, persistence baseline, and loss plotting as in the MLP
pVal = modelCNN.predict(xVal, verbose=0).ravel()
pTest = modelCNN.predict(xTest, verbose=0).ravel()

prec, rec, thr = precision_recall_curve(yVal, pVal)
f1Val = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
bestIndex = np.argmax(f1Val)
decisionThreshold = thr[bestIndex]

classPredictions = (pTest >= decisionThreshold).astype(int)

print("\n RESULTS ON FUTURE DATA (TEST SET) ")
print("Selected decision threshold :", round(decisionThreshold, 3))
print("Precision :", precision_score(yTest, classPredictions, zero_division=0))
print("Recall :", recall_score(yTest, classPredictions, zero_division=0))
print("F1 Score :", f1_score(yTest, classPredictions, zero_division=0))
print("PR-AUC :", average_precision_score(yTest, pTest), "(random baseline :", yTest.mean(), ")")

persistence = (test['ratioToday'] > 1).astype(int)
print("\nPersistence baseline -> Precision :", precision_score(yTest, persistence, zero_division=0),
      "| Recall :", recall_score(yTest, persistence, zero_division=0),
      "| F1 :", f1_score(yTest, persistence, zero_division=0))

plt.figure(figsize=(8, 5))
plt.plot(historyCNN.history['loss'], label='Training (Loss)')
plt.plot(historyCNN.history['val_loss'], label='Validation (Val Loss)')
plt.title('Overfitting Diagnostic (Train vs Validation)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(0, 0.5)
plt.legend()
plt.show()