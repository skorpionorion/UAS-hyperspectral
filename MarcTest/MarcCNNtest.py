import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load Data
Bands = sio.loadmat('../ML_Data/Bands.mat')
Signals = sio.loadmat('../ML_Data/Signals.mat')
Moisture_Percentage = sio.loadmat('../ML_Data/Moisture_Percentage.mat')
# Extracting Out Data
Bands = Bands[list(Bands.keys())[-1]].T
X = Signals[list(Signals.keys())[-1]].T
y = Moisture_Percentage[list(Moisture_Percentage.keys())[-1]].T

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Split into training and testing sets (70/30)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(y_train.shape)
print(X_train.max())
print(X_train.min())
print(y_train.max())
print(y_train.min())


## Build the model
# CNN Model
X_train_cnn = X_train[..., np.newaxis]
X_test_cnn = X_test[..., np.newaxis]

cnn_model = Sequential([
    Conv1D(64, 5, activation='relu', input_shape=(X_train_cnn.shape[1], 1)), # Conv1D layer
    Conv1D(32, 3, activation='relu'),
    Conv1D(16, 3, activation='relu'),
    Flatten(),
    Dense(64, activation='relu'), # Dense layer
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='linear')
])

cnn_model.summary()

## Train the model
batch_size = 8
epochs = 300

cnn_model.compile(optimizer=Adam(1e-4), loss='mse', metrics=['mae'])
hist = cnn_model.fit(X_train_cnn, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)

# Evaluate CNN Model
y_pred_cnn = cnn_model.predict(X_test_cnn).flatten()
mae_cnn = mean_absolute_error(y_test, y_pred_cnn)
rmse_cnn = np.sqrt(mean_squared_error(y_test, y_pred_cnn))

## RMSE and R2 score
score = cnn_model.evaluate(X_test_cnn, y_test, verbose=0)
print("Test MSE:", score[0])
print("Test Mean Absolute Error:", score[1])


r2_test = r2_score(y_test, y_pred_cnn)
print("Test R2 Score:", r2_test)

# Results
print("\n1D CNN Model Performance")
print(f"RMSE = {rmse_cnn:.6f}")

# Plot training & validation MAE values
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(hist.history["mae"])
plt.plot(hist.history["val_mae"])
plt.title("Model Mean Absolute Error")
plt.ylabel("Mean Absolute Error")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper left")
plt.grid(True)

# Plot training & validation MSE values
plt.subplot(1, 2, 2)
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.title("Model Loss (MSE)")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper left")
plt.grid(True)

plt.tight_layout()
plt.show()

#Plotting Results
plt.figure(figsize=(10, 6))
plt.plot(y_test, 'o-', label='Actual Moisture', markersize=6)
plt.plot(y_pred_cnn, 's--', label='Predicted Moisture', markersize=6)
plt.xlabel('Sample Index')
plt.ylabel('Moisture Percentage')
plt.title('Predicted vs Actual Moisture on Test Set')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()