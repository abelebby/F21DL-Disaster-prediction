# Next-Day High-Flow Prediction in Nepal: 2D Convolutional Neural Network (2D-CNN)
#
# Environment : Python 3 (Spyder / VSCode)
# Libraries   : pandas, numpy, matplotlib, scikit-learn, tensorflow
# Setup       : Install requirements-neural.txt; reads data/processed/nepal_flood_clean.csv.
#
# Input       : Cleaned historical meteorological dataset (4-day x 4-variable grids).
# Output      : Classification metrics on unseen future data and Train/Val loss curve.
# Metrics     : Precision, Recall, F1-Score, PR-AUC, and Day-1 High-Flow Onset Recall.
#-------------------------------------------------------------------------------------

import argparse
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / '.mpl_cache'))
import matplotlib
matplotlib.use('Agg')
from src.high_flow_experiment import ROOT, prepare_data, arrays, save_results
import joblib
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
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()
np.random.seed(42)


# Shared data preparation: same rows, labels and purged temporal boundaries.
parser = argparse.ArgumentParser()
parser.add_argument('--results-dir', type=Path, default=ROOT / 'results/high_flow/cnn')
parser.add_argument('--epochs', type=int, default=100)
args = parser.parse_args()
if args.epochs < 1:
    parser.error('--epochs must be positive')
splits, stationThresholds = prepare_data()
xs, ys, scaler, features = arrays(splits, 'cnn')
train, val, test = (splits[k] for k in ['train', 'val', 'test'])
xTrain, xVal, xTest = (xs[k] for k in ['train', 'val', 'test'])
yTrain, yVal, yTest = (ys[k] for k in ['train', 'val', 'test'])
xTrain, xVal, xTest = (x.reshape(-1, 4, 4, 1) for x in [xTrain, xVal, xTest])
persistence = (test.ratioToday > 1).astype(int)

# 2D-CNN Architecture and Fine-Tuned Training

# Compact 2D-CNN designed to extract local patterns across the 4-day weather grid with regularisation
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

# Use Adam optimizer with a conservative learning rate (0.0005) for steady convergence, and binary cross-entropy for binary high-flow classification
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
modelCNN.compile(optimizer=optimizer, loss='binary_crossentropy')

# Balance validation weights so the proportion of high-flow events matches the training set
# This changes the validation-loss weighting; it does not remove distribution shift.
trainRate = yTrain.mean()
valRate = yVal.mean()
valWeights = np.where(yVal == 1, (trainRate / valRate), ((1 - trainRate) / (1 - valRate)))

# Train the model for up to 100 passes (64 samples at a time) using the adjusted validation set
# Callbacks automatically lower the learning rate and stop training early once performance stops improving
historyCNN = modelCNN.fit(
    xTrain, yTrain,
    validation_data=(xVal, yVal, valWeights),
    epochs=args.epochs, batch_size=64,
    callbacks=[earlyStop, reduceLr],
    verbose=1
)


# Threshold Optimization and Evaluation

# Generate next-day high-flow probabilities for the validation and test sets
# Flatten the predictions into 1D arrays to simplify threshold tuning and evaluation
pVal = modelCNN.predict(xVal, verbose=0).ravel()
pTest = modelCNN.predict(xTest, verbose=0).ravel()

# Test different probability thresholds on the validation set to find the best balance
# between catching high-flows and avoiding false alarms
# Pick the threshold that maximizes the F1 score instead of using a default 0.5 cutoff
prec, rec, thr = precision_recall_curve(yVal, pVal)
f1Val = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
bestIndex = np.argmax(f1Val)
decisionThreshold = thr[bestIndex]

# Apply the tuned threshold to turn test probabilities into final binary high-flow alerts (1 or 0)
classPredictions = (pTest >= decisionThreshold).astype(int)

# Measure overall ranking performance with average precision (same as before so suited for rare high-flow events) and compare against a random guess baseline
modelPrauc = average_precision_score(yTest, pTest)
randomBaseline = yTest.mean()

print("\n    TEST SET PERFORMANCE    ")
print(f"Validation-selected cutoff : {decisionThreshold:.3f}")
print(f"Recall (High-Flow Days Caught): {recall_score(yTest, classPredictions, zero_division=0):.1%}")
print(f"Precision             : {precision_score(yTest, classPredictions, zero_division=0):.1%}")
print(f"F1-Score              : {f1_score(yTest, classPredictions, zero_division=0):.3f}")
print("\n")
print(f"Average precision (AP)          : {modelPrauc:.3f}")
print(f"Random Baseline       : {randomBaseline:.3f}")
print(f"Around {modelPrauc / randomBaseline:.1f}x better than random")

# Evaluate model's ability to anticipate new high-flows (Day 1 onsets),
# isolating days where a high-flow occurs tomorrow but the river is normal today.
onsetMask = (yTest == 1) & (persistence == 0)

yTestOnset = yTest[onsetMask]
predsOnset = classPredictions[onsetMask]

print("\n    HIGH-FLOW ONSET EVALUATION    ")
if len(yTestOnset) > 0:
    onsetRecall = recall_score(yTestOnset, predsOnset, zero_division=0)
    print(f"Total onsets in test set: {len(yTestOnset)}")
    print(f"Successfully anticipated: {predsOnset.sum()}")
    print(f"Anticipation Recall     : {onsetRecall:.1%}")
else:
    print("No onsets found in test set.")


# Plot training and validation loss across epochs to check how well the model learns and inspect possible overfitting
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
args.results_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(args.results_dir / 'training_loss.png', dpi=160, bbox_inches='tight')
plt.close()
pd.DataFrame(historyCNN.history).to_csv(args.results_dir / 'training_history.csv', index_label='epoch')
modelCNN.save(args.results_dir / 'model.keras')
joblib.dump(dict(scaler=scaler, features=features), args.results_dir / 'preprocessing.joblib')
metrics = save_results(args.results_dir, splits, stationThresholds,
    dict(val=pVal, test=pTest), float(decisionThreshold), 'cnn', features,
    dict(tensorflow=tf.__version__, max_epochs=args.epochs, epochs_run=len(historyCNN.history['loss']),
         validation_loss='weighted to training prevalence; F1 cutoff uses unweighted validation labels'))
print(metrics.to_string(index=False))
