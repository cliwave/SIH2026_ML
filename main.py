import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# --------------------------------
# 1. Read the water quality data
# --------------------------------

data = pd.read_csv("data/water_quality.csv")

# --------------------------------
# 2. Select the four ML features
# --------------------------------

features = ["pH", "turbidity", "tds", "temperature"]

X = data[features]

# --------------------------------
# 3. Create the ML model
# --------------------------------

model = IsolationForest(
    n_estimators=100,
    contamination=0.15,
    random_state=42
)

# Train the model and predict
data["prediction"] = model.fit_predict(X)

import joblib

joblib.dump(model, "models/water_quality_model.joblib")

print("\nModel saved successfully!")

# --------------------------------
# 4. Convert prediction to status
# --------------------------------

data["status"] = data["prediction"].apply(
    lambda x: "NORMAL" if x == 1 else "UNUSUAL"
)

# --------------------------------
# 5. Display ML results
# --------------------------------

print("\nWater Quality Pattern Detection Results:\n")

print(
    data[
        ["timestamp", "pH", "turbidity", "tds",
         "temperature", "status"]
    ].to_string(index=False)
)

# --------------------------------
# 6. Show unusual readings
# --------------------------------

print("\n--------------------------------")
print("UNUSUAL PATTERNS DETECTED")
print("--------------------------------")

unusual = data[data["status"] == "UNUSUAL"]

print(
    unusual[
        ["timestamp", "pH", "turbidity",
         "tds", "temperature"]
    ].to_string(index=False)
)

# -------------------------------
# GRAPH 1: pH
# -------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    data["timestamp"],
    data["pH"],
    marker="o",
    label="pH"
)

# Highlight unusual points
unusual = data[data["status"] == "UNUSUAL"]

plt.scatter(
    unusual["timestamp"],
    unusual["pH"],
    color="red",
    s=100,
    label="Unusual Pattern",
    zorder=5
)

plt.xlabel("Time")
plt.ylabel("pH")
plt.title("pH Pattern Over Time")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# -------------------------------
# GRAPH 2: TURBIDITY
# -------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    data["timestamp"],
    data["turbidity"],
    marker="o",
    label="Turbidity"
)

plt.scatter(
    unusual["timestamp"],
    unusual["turbidity"],
    color="red",
    s=100,
    label="Unusual Pattern",
    zorder=5
)

plt.xlabel("Time")
plt.ylabel("Turbidity (NTU)")
plt.title("Turbidity Pattern Over Time")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# -------------------------------
# GRAPH 3: TDS
# -------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    data["timestamp"],
    data["tds"],
    marker="o",
    label="TDS"
)

plt.scatter(
    unusual["timestamp"],
    unusual["tds"],
    color="red",
    s=100,
    label="Unusual Pattern",
    zorder=5
)

plt.xlabel("Time")
plt.ylabel("TDS (ppm)")
plt.title("TDS Pattern Over Time")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# -------------------------------
# GRAPH 4: TEMPERATURE
# -------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    data["timestamp"],
    data["temperature"],
    marker="o",
    label="Temperature"
)

plt.scatter(
    unusual["timestamp"],
    unusual["temperature"],
    color="red",
    s=100,
    label="Unusual Pattern",
    zorder=5
)

plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.title("Temperature Pattern Over Time")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()