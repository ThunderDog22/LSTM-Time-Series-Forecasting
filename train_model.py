import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

# Load dataset
zip_path = tf.keras.utils.get_file(
    origin="https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
    fname="jena_climate_2009_2016.csv.zip",
    extract=True
)

csv_path, _ = os.path.splitext(zip_path)
df = pd.read_csv(r"C:\Users\aljla\.keras\datasets\jena_climate_2009_2016_extracted\jena_climate_2009_2016.csv")

# Downsample to hourly data
df = df[5::6]
df.index = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
temp = df["T (degC)"]

# Function to create input sequences
def df_to_X_y(df, window_size=5):
    df_as_np = df.to_numpy()
    X, y = [], []
    for i in range(len(df_as_np) - window_size):
        X.append([[a] for a in df_as_np[i:i+window_size]])
        y.append(df_as_np[i+window_size])
    return np.array(X), np.array(y)

# Prepare data
WIN_SIZE = 5
X, y = df_to_X_y(temp, WIN_SIZE)
X_train, y_train = X[:60000], y[:60000]
X_val, y_val = X[60000:65000], y[60000:65000]
X_test, y_test = X[65000:], y[65000:]

# Define model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(WIN_SIZE, 1)),
    LSTM(64, return_sequences=False),
    Dense(32, activation="relu"),
    Dense(1)
])

# Compile model
model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001))

# Model checkpoint to save best model
cp = ModelCheckpoint("model1/best_model.keras", save_best_only=True)

# Train model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[cp])

# Save final model
model.save("model1/final_model.keras")

# Load best model and evaluate
best_model = load_model("model1/best_model.keras")
test_predictions = best_model.predict(X_test).flatten()

# Save results
test_results = pd.DataFrame({"Test Predictions": test_predictions, "Actuals": y_test})
test_results.to_csv("test_results.csv", index=False)

print("Model training complete! Predictions saved to test_results.csv.")

# Plot of temperature data or model predictions
plt.figure(figsize=(10, 6))

# Plotting example data (e.g., your temperature data or model predictions)
plt.plot(test_results["Test Predictions"][:100])
plt.plot(test_results["Actuals"][:100])
plt.title('Predicted vs Actual Temperature')
plt.xlabel('Time (Hrs)')
plt.ylabel('Temperature (°C)')
plt.legend()

# Save the plot to a file
plt.savefig("temperature_plot.png", dpi=300)  # Save as a PNG file with 300 dpi for high quality
plt.close()
 