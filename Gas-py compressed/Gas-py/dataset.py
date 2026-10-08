import numpy as np
import pandas as pd

# ============================================================
# SYNTHETIC GAS / SMELL CLASSIFICATION DATASET
# ============================================================
# Sensors:
#   gas1 = MQ-2
#   gas2 = MQ-8
#   gas3 = MQ-135
#
# Classes:
#   Normal
#   Smoke
#   Butane
#   LPG
#   Alcohol
#   Ammonia
#   Methane
# ============================================================

np.random.seed(42)


# ============================================================
# SETTINGS
# ============================================================

SAMPLES_PER_CLASS = 7143

classes = [
    "Normal",
    "Smoke",
    "Butane",
    "LPG",
    "Alcohol",
    "Ammonia",
    "Methane"
]

FEATURES = [
    "gas1",
    "gas2",
    "gas3"
]


# ============================================================
# SENSOR INFORMATION
# ============================================================
# gas1 = MQ-2
# gas2 = MQ-8
# gas3 = MQ-135
#
# The ranges below follow the thresholds provided.
# ============================================================

SENSOR_LIMITS = {
    "gas1": (0, 10000),    # MQ-2
    "gas2": (0, 10000),    # MQ-8
    "gas3": (0, 10000)     # MQ-135
}


# ============================================================
# CLASS GENERATION PROFILES
# ============================================================
#
# Each class has:
#   mean  -> typical sensor response
#   std   -> natural variation
#
# The important sensor is additionally constrained later
# to remain inside the required class range.
# ============================================================

profiles = {

    # --------------------------------------------------------
    # NORMAL
    # --------------------------------------------------------
    "Normal": {
        "mean": [90, 2500, 1400],
        "std":  [20, 700, 350]
    },

    # --------------------------------------------------------
    # SMOKE
    # MQ-2 > 2500
    # --------------------------------------------------------
    "Smoke": {
        "mean": [5000, 3000, 1500],
        "std":  [600, 700, 350]
    },

    # --------------------------------------------------------
    # BUTANE
    # MQ-8 = 5001 - 6000
    # --------------------------------------------------------
    "Butane": {
        "mean": [900, 5500, 1500],
        "std":  [250, 250, 300]
    },

    # --------------------------------------------------------
    # LPG
    # MQ-8 = 6001 - 7000
    # --------------------------------------------------------
    "LPG": {
        "mean": [1000, 6500, 1600],
        "std":  [250, 250, 300]
    },

    # --------------------------------------------------------
    # ALCOHOL
    # MQ-135 = 2900 - 3299
    # --------------------------------------------------------
    "Alcohol": {
        "mean": [1000, 3500, 3100],
        "std":  [250, 500, 100]
    },

    # --------------------------------------------------------
    # AMMONIA
    # MQ-135 >= 3300
    # --------------------------------------------------------
    "Ammonia": {
        "mean": [900, 3500, 6500],
        "std":  [250, 500, 700]
    },

    # --------------------------------------------------------
    # METHANE
    # MQ-8 > 7000
    # --------------------------------------------------------
    "Methane": {
        "mean": [1000, 8200, 1600],
        "std":  [250, 400, 300]
    }
}


# ============================================================
# FUNCTION TO GENERATE ONE CLASS
# ============================================================

def generate_class_data(class_name, n_samples):

    # Generate realistic base values.
    mean = np.array(profiles[class_name]["mean"], dtype=float)
    std = np.array(profiles[class_name]["std"], dtype=float)

    X = np.random.normal(
        loc=mean,
        scale=std,
        size=(n_samples, 3)
    )

    # --------------------------------------------------------
    # CLASS-SPECIFIC RANGE CONTROL
    # --------------------------------------------------------
    # The defining sensor is generated AFTER all random variation.
    # This prevents noise/drift from moving a sample into another
    # class and teaches the Random Forest the intended boundaries.
    # --------------------------------------------------------

    if class_name == "Normal":

        # MQ-2: Normal range
        X[:, 0] = np.random.uniform(20, 145, n_samples)

        # MQ-8: normal/other range
        X[:, 1] = np.random.uniform(500, 4800, n_samples)

        # MQ-135: 0-2800 = Normal
        X[:, 2] = np.random.uniform(300, 2790, n_samples)

    elif class_name == "Smoke":

        # MQ-2 > 2500 = Smoke
        X[:, 0] = np.random.uniform(2501, 8000, n_samples)

        X[:, 1] = np.random.uniform(1000, 5000, n_samples)
        X[:, 2] = np.random.uniform(500, 2500, n_samples)

    elif class_name == "Butane":

        # MQ-8 = 5001-6000
        X[:, 1] = np.random.uniform(5001, 6000, n_samples)

        X[:, 0] = np.random.uniform(250, 1500, n_samples)
        X[:, 2] = np.random.uniform(500, 2200, n_samples)

    elif class_name == "LPG":

        # MQ-8 = 6001-7000
        X[:, 1] = np.random.uniform(6001, 7000, n_samples)

        X[:, 0] = np.random.uniform(300, 1800, n_samples)
        X[:, 2] = np.random.uniform(600, 2300, n_samples)

    elif class_name == "Alcohol":

        # MQ-135 = 2900-3299
        # Keep the distribution centered around 3100 so values
        # near 3200 are strongly represented as Alcohol.
        X[:, 2] = np.random.uniform(2900, 3299, n_samples)

        X[:, 0] = np.random.uniform(250, 1800, n_samples)
        X[:, 1] = np.random.uniform(1000, 5000, n_samples)

    elif class_name == "Ammonia":

        # MQ-135 >= 3300
        X[:, 2] = np.random.uniform(3300, 10000, n_samples)

        X[:, 0] = np.random.uniform(250, 1700, n_samples)
        X[:, 1] = np.random.uniform(1000, 5000, n_samples)

    elif class_name == "Methane":

        # MQ-8 > 7000
        X[:, 1] = np.random.uniform(7001, 10000, n_samples)

        X[:, 0] = np.random.uniform(300, 1800, n_samples)
        X[:, 2] = np.random.uniform(500, 2300, n_samples)

    # Keep every value inside the sensor limits.
    X[:, 0] = np.clip(X[:, 0], 0, 10000)
    X[:, 1] = np.clip(X[:, 1], 0, 10000)
    X[:, 2] = np.clip(X[:, 2], 0, 10000)

    # Integer sensor readings.
    X = np.round(X).astype(int)

    df = pd.DataFrame(X, columns=FEATURES)
    df["label"] = class_name

    return df


# ============================================================
# GENERATE ALL CLASSES
# ============================================================

all_data = []

for gas_class in classes:

    print(
        f"Generating {SAMPLES_PER_CLASS} samples "
        f"for {gas_class}..."
    )

    class_data = generate_class_data(
        gas_class,
        SAMPLES_PER_CLASS
    )

    all_data.append(class_data)


# ============================================================
# COMBINE ALL CLASSES
# ============================================================

dataset = pd.concat(
    all_data,
    ignore_index=True
)


# ============================================================
# SHUFFLE DATASET
# ============================================================

dataset = dataset.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# SAVE DATASET
# ============================================================

OUTPUT_FILE = "gas_sensor.csv"

dataset.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DATASET INFORMATION
# ============================================================

print("\n")
print("=" * 65)
print("DATASET CREATED SUCCESSFULLY")
print("=" * 65)

print("\nTotal rows:")
print(len(dataset))

print("\nColumns:")
print(dataset.columns.tolist())

print("\nSensors:")
print("gas1 = MQ-2")
print("gas2 = MQ-8")
print("gas3 = MQ-135")

print("\nClasses:")
print(classes)

print("\nSamples per class:")
print(
    dataset["label"].value_counts()
    .sort_index()
)

print("\nMinimum sensor values:")
print(
    dataset[FEATURES].min()
)

print("\nMaximum sensor values:")
print(
    dataset[FEATURES].max()
)

print("\nAverage sensor values by class:")
print(
    dataset.groupby("label")[FEATURES]
    .mean()
    .round(2)
)

print("\nFirst 20 rows:")
print(
    dataset.head(20)
)

print("\nDataset saved as:")
print(OUTPUT_FILE)

print("\n")
print("=" * 65)
print("THRESHOLD SUMMARY")
print("=" * 65)

print("""
MQ-2 (gas1):
    0 - 150       -> Normal
    > 2500        -> Smoke

MQ-8 (gas2):
    0 - 5000      -> Normal / other
    5001 - 6000   -> Butane
    6001 - 7000   -> LPG
    > 7000        -> Methane

MQ-135 (gas3):
    0 - 2899      -> Normal
    2900 - 3299   -> Alcohol
    >= 3300       -> Ammonia
""")