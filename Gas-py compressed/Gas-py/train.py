import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# GAS / SMELL CLASSIFICATION
# SYNTHETIC DATA + RANDOM FOREST
# MQ-2 + MQ-8 + MQ-135
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

SAMPLES_PER_CLASS = 7143

DATASET_PATH = "gas_sensor.csv"

MODEL_PATH = "gas_sensor_random_forest.pkl"

LABEL_ENCODER_PATH = "gas_label_encoder.pkl"

REPORT_PATH = "gas_classification_report.txt"

CONFUSION_MATRIX_PATH = "gas_confusion_matrix.png"

FEATURE_IMPORTANCE_PATH = "gas_feature_importance.png"

TREE_ACCURACY_PATH = "gas_tree_accuracy.png"

MODEL_INFO_PATH = "gas_model_info.json"


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "gas1",
    "gas2",
    "gas3"
]


# ============================================================
# SENSOR MAPPING
# ============================================================

SENSOR_NAMES = {
    "gas1": "MQ-2",
    "gas2": "MQ-8",
    "gas3": "MQ-135"
}


# ============================================================
# CLASSES
# ============================================================

CLASSES = [
    "Normal",
    "Smoke",
    "Butane",
    "LPG",
    "Alcohol",
    "Ammonia",
    "Methane"
]


# ============================================================
# RANDOM SEED
# ============================================================

np.random.seed(RANDOM_STATE)


# ============================================================
# LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING GAS SENSOR DATASET")
print("=" * 70)

if not os.path.exists(DATASET_PATH):

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET_PATH}\n\n"
        "Make sure gas_sensor.csv is in the same folder "
        "as this training script."
    )


df = pd.read_csv(DATASET_PATH)


print("\nDataset loaded successfully.")

print("\nDataset shape:")
print(df.shape)

print("\nDataset columns:")
print(df.columns.tolist())


# ============================================================
# VERIFY REQUIRED COLUMNS
# ============================================================

required_columns = [
    "gas1",
    "gas2",
    "gas3",
    "label"
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    raise ValueError(
        "\nMissing required columns: "
        + str(missing_columns)
        + "\n\nExpected columns:\n"
        + str(required_columns)
    )


# ============================================================
# REMOVE MISSING VALUES
# ============================================================

print("\nChecking missing values...")

missing_count = df[required_columns].isnull().sum()

print(missing_count)


if missing_count.sum() > 0:

    print("\nMissing values found.")

    df = df.dropna(
        subset=required_columns
    ).reset_index(drop=True)

    print(
        f"Rows remaining after cleaning: {len(df)}"
    )

else:

    print("No missing values found.")


# ============================================================
# VERIFY LABELS
# ============================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print(
    df["label"].value_counts()
)


unexpected_classes = set(
    df["label"].unique()
) - set(CLASSES)


if unexpected_classes:

    print(
        "\nWARNING: Unexpected classes detected:"
    )

    print(
        unexpected_classes
    )


# ============================================================
# SENSOR INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("SENSOR CONFIGURATION")
print("=" * 70)

print("\ngas1 = MQ-2")
print("gas2 = MQ-8")
print("gas3 = MQ-135")


# ============================================================
# SENSOR VALUE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SENSOR VALUE RANGES")
print("=" * 70)

print("\nMinimum values:")

print(
    df[FEATURES].min()
)


print("\nMaximum values:")

print(
    df[FEATURES].max()
)


print("\nMean values:")

print(
    df[FEATURES].mean().round(2)
)


# ============================================================
# CLASS-WISE SENSOR MEANS
# ============================================================

print("\n" + "=" * 70)
print("CLASS-WISE SENSOR MEAN VALUES")
print("=" * 70)

class_means = (
    df.groupby("label")[FEATURES]
    .mean()
    .round(2)
)

print(
    class_means
)


# ============================================================
# VERIFY MQ-135 DATA RANGES
# ============================================================

print("\n" + "=" * 70)
print("MQ-135 CLASS RANGE VERIFICATION")
print("=" * 70)


for class_name in CLASSES:

    class_data = df[
        df["label"] == class_name
    ]

    if len(class_data) > 0:

        print(
            f"{class_name:<10} -> "
            f"MQ-135 Min: {class_data['gas3'].min():>5} | "
            f"Max: {class_data['gas3'].max():>5}"
        )


# ============================================================
# FEATURES AND TARGET
# ============================================================

print("\n" + "=" * 70)
print("PREPARING TRAINING DATA")
print("=" * 70)

X = df[FEATURES].values

y_text = df["label"].values


print(
    f"\nNumber of input features: {X.shape[1]}"
)

print(
    f"Input features: {FEATURES}"
)


# ============================================================
# LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    y_text
)


print("\n" + "=" * 70)
print("CLASS ENCODING")
print("=" * 70)


for i, class_name in enumerate(
    label_encoder.classes_
):

    print(
        f"{i} -> {class_name}"
    )


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("SPLITTING DATASET")
print("=" * 70)


X_train, X_temp, y_train, y_temp = train_test_split(

    X,
    y,

    test_size=0.30,

    random_state=RANDOM_STATE,

    stratify=y
)


X_val, X_test, y_val, y_test = train_test_split(

    X_temp,
    y_temp,

    test_size=0.50,

    random_state=RANDOM_STATE,

    stratify=y_temp
)


print(
    f"\nTraining samples   : {len(X_train)}"
)

print(
    f"Validation samples : {len(X_val)}"
)

print(
    f"Testing samples    : {len(X_test)}"
)


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)


model = RandomForestClassifier(

    n_estimators=300,

    criterion="gini",

    max_depth=None,

    min_samples_split=2,

    min_samples_leaf=2,

    max_features="sqrt",

    bootstrap=True,

    class_weight="balanced",

    random_state=RANDOM_STATE,

    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


print(
    "\nRandom Forest training completed."
)


# ============================================================
# TRAINING PREDICTIONS
# ============================================================

train_predictions = model.predict(
    X_train
)


train_accuracy = accuracy_score(
    y_train,
    train_predictions
)


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

val_predictions = model.predict(
    X_val
)


val_accuracy = accuracy_score(
    y_val,
    val_predictions
)


# ============================================================
# TEST PREDICTIONS
# ============================================================

test_predictions = model.predict(
    X_test
)


test_accuracy = accuracy_score(
    y_test,
    test_predictions
)


# ============================================================
# PRINT ACCURACIES
# ============================================================

print("\n" + "=" * 70)
print("MODEL ACCURACY")
print("=" * 70)


print(
    f"\nTraining Accuracy   : "
    f"{train_accuracy * 100:.2f}%"
)


print(
    f"Validation Accuracy : "
    f"{val_accuracy * 100:.2f}%"
)


print(
    f"Test Accuracy       : "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    y_test,

    test_predictions,

    target_names=label_encoder.classes_,

    digits=4
)


print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    report
)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "MQ-2 + MQ-8 + MQ-135 GAS CLASSIFICATION REPORT\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        "Sensors:\n"
    )

    f.write(
        "gas1 = MQ-2\n"
    )

    f.write(
        "gas2 = MQ-8\n"
    )

    f.write(
        "gas3 = MQ-135\n\n"
    )

    f.write(
        f"Training Accuracy   : "
        f"{train_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Validation Accuracy : "
        f"{val_accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Test Accuracy       : "
        f"{test_accuracy * 100:.2f}%\n\n"
    )

    f.write(
        report
    )


print(
    f"\nClassification report saved: "
    f"{REPORT_PATH}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    test_predictions
)


fig, ax = plt.subplots(
    figsize=(10, 8)
)


disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=label_encoder.classes_
)


disp.plot(

    ax=ax,

    values_format="d",

    xticks_rotation=45
)


plt.title(
    "MQ-2 + MQ-8 + MQ-135 Gas Classification"
)


plt.xlabel(
    "Predicted Class"
)


plt.ylabel(
    "Actual Class"
)


plt.tight_layout()


plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300
)


plt.close()


print(
    f"Confusion matrix saved: "
    f"{CONFUSION_MATRIX_PATH}"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = model.feature_importances_


importance_df = pd.DataFrame({

    "Feature": FEATURES,

    "Sensor": [
        SENSOR_NAMES[feature]
        for feature in FEATURES
    ],

    "Importance": importance

})


importance_df = importance_df.sort_values(

    by="Importance",

    ascending=False

)


print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)


print(
    importance_df.to_string(
        index=False
    )
)


plt.figure(
    figsize=(10, 6)
)


plt.bar(

    importance_df["Sensor"],

    importance_df["Importance"]

)


plt.xlabel(
    "Gas Sensor"
)


plt.ylabel(
    "Importance"
)


plt.title(
    "Random Forest Feature Importance"
)


plt.tight_layout()


plt.savefig(
    FEATURE_IMPORTANCE_PATH,
    dpi=300
)


plt.close()


print(
    f"\nFeature importance saved: "
    f"{FEATURE_IMPORTANCE_PATH}"
)


# ============================================================
# VALIDATION ACCURACY VS NUMBER OF TREES
# ============================================================

print("\n" + "=" * 70)
print("TREE COUNT VALIDATION TEST")
print("=" * 70)


tree_values = [
    25,
    50,
    100,
    150,
    200,
    300,
    400,
    500
]


tree_accuracies = []


for n_trees in tree_values:

    temp_model = RandomForestClassifier(

        n_estimators=n_trees,

        criterion="gini",

        max_depth=None,

        min_samples_split=2,

        min_samples_leaf=2,

        max_features="sqrt",

        bootstrap=True,

        class_weight="balanced",

        random_state=RANDOM_STATE,

        n_jobs=-1
    )


    temp_model.fit(

        X_train,

        y_train
    )


    temp_prediction = temp_model.predict(

        X_val
    )


    temp_accuracy = accuracy_score(

        y_val,

        temp_prediction
    )


    tree_accuracies.append(

        temp_accuracy
    )


    print(

        f"{n_trees:>4} trees -> "

        f"{temp_accuracy * 100:.2f}% "
        f"validation accuracy"
    )


# ============================================================
# TREE ACCURACY GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(

    tree_values,

    np.array(tree_accuracies) * 100,

    marker="o"
)


plt.xlabel(
    "Number of Trees"
)


plt.ylabel(
    "Validation Accuracy (%)"
)


plt.title(
    "Random Forest Validation Accuracy vs Number of Trees"
)


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(
    TREE_ACCURACY_PATH,
    dpi=300
)


plt.close()


print(
    f"\nTree accuracy graph saved: "
    f"{TREE_ACCURACY_PATH}"
)


# ============================================================
# SAVE TRAINED MODEL
# ============================================================

joblib.dump(

    model,

    MODEL_PATH
)


print(
    f"\nTrained model saved: "
    f"{MODEL_PATH}"
)


# ============================================================
# SAVE LABEL ENCODER
# ============================================================

joblib.dump(

    label_encoder,

    LABEL_ENCODER_PATH
)


print(
    f"Label encoder saved: "
    f"{LABEL_ENCODER_PATH}"
)


# ============================================================
# SAVE MODEL INFORMATION
# ============================================================

model_info = {

    "algorithm":
        "Random Forest",

    "n_estimators":
        300,

    "random_state":
        RANDOM_STATE,

    "features":
        FEATURES,

    "sensor_mapping":
        SENSOR_NAMES,

    "classes":
        label_encoder.classes_.tolist(),

    "train_samples":
        int(len(X_train)),

    "validation_samples":
        int(len(X_val)),

    "test_samples":
        int(len(X_test)),

    "training_accuracy":
        float(train_accuracy),

    "validation_accuracy":
        float(val_accuracy),

    "test_accuracy":
        float(test_accuracy),

    "thresholds": {

        "MQ-2": {
            "Normal": "0-150",
            "Smoke": ">2500"
        },

        "MQ-8": {
            "Normal/Other": "0-5000",
            "Butane": "5001-6000",
            "LPG": "6001-7000",
            "Methane": ">7000"
        },

        "MQ-135": {
            "Normal": "0-2899",
            "Alcohol": "2900-3299",
            "Ammonia": ">=3300"
        }
    }
}


with open(
    MODEL_INFO_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(

        model_info,

        f,

        indent=4
    )


print(
    f"Model information saved: "
    f"{MODEL_INFO_PATH}"
)


# ============================================================
# REAL-TIME PREDICTION FUNCTION
# ============================================================

def predict_gas(
    gas1,
    gas2,
    gas3
):

    sensor_values = np.array([

        [
            gas1,
            gas2,
            gas3
        ]

    ])


    prediction = model.predict(

        sensor_values

    )[0]


    probabilities = model.predict_proba(

        sensor_values

    )[0]


    predicted_class = (

        label_encoder.inverse_transform(

            [prediction]

        )[0]

    )


    confidence = (

        np.max(probabilities) * 100

    )


    class_probabilities = {}


    for class_name, probability in zip(

        label_encoder.classes_,

        probabilities

    ):

        class_probabilities[class_name] = (

            probability * 100

        )


    return (

        predicted_class,

        confidence,

        class_probabilities

    )


# ============================================================
# REAL-TIME SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("REAL-TIME SAMPLE PREDICTIONS")
print("=" * 70)


samples = [

    # --------------------------------------------------------
    # NORMAL
    # --------------------------------------------------------

    {
        "gas1": 100,
        "gas2": 3000,
        "gas3": 1500
    },


    # --------------------------------------------------------
    # SMOKE
    # --------------------------------------------------------

    {
        "gas1": 4500,
        "gas2": 3000,
        "gas3": 1500
    },


    # --------------------------------------------------------
    # BUTANE
    # --------------------------------------------------------

    {
        "gas1": 800,
        "gas2": 5500,
        "gas3": 1600
    },


    # --------------------------------------------------------
    # LPG
    # --------------------------------------------------------

    {
        "gas1": 900,
        "gas2": 6500,
        "gas3": 1800
    },


    # --------------------------------------------------------
    # ALCOHOL
    # MQ-135 = 2900-3299
    # --------------------------------------------------------

    {
        "gas1": 700,
        "gas2": 3500,
        "gas3": 3135
    },


    # --------------------------------------------------------
    # ALCOHOL - UPPER RANGE
    # --------------------------------------------------------

    {
        "gas1": 700,
        "gas2": 3500,
        "gas3": 3200
    },


    # --------------------------------------------------------
    # AMMONIA
    # MQ-135 >= 3300
    # --------------------------------------------------------

    {
        "gas1": 650,
        "gas2": 3200,
        "gas3": 5000
    },


    # --------------------------------------------------------
    # METHANE
    # --------------------------------------------------------

    {
        "gas1": 900,
        "gas2": 8200,
        "gas3": 1700
    }

]


for sample in samples:

    predicted_class, confidence, probabilities = predict_gas(

        sample["gas1"],

        sample["gas2"],

        sample["gas3"]

    )


    print("\nInput sensor readings:")

    print(
        f"MQ-2   (gas1) : {sample['gas1']}"
    )

    print(
        f"MQ-8   (gas2) : {sample['gas2']}"
    )

    print(
        f"MQ-135 (gas3) : {sample['gas3']}"
    )


    print(
        f"\nPrediction : {predicted_class}"
    )


    print(
        f"Confidence : {confidence:.2f}%"
    )


    print("\nClass probabilities:")


    for class_name, probability in probabilities.items():

        print(

            f"  {class_name:<10} : "
            f"{probability:.2f}%"

        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)


print(
    f"\nTraining Accuracy   : "
    f"{train_accuracy * 100:.2f}%"
)


print(
    f"Validation Accuracy : "
    f"{val_accuracy * 100:.2f}%"
)


print(
    f"Test Accuracy       : "
    f"{test_accuracy * 100:.2f}%"
)


print("\nInput sensors:")

print(
    "gas1 -> MQ-2"
)

print(
    "gas2 -> MQ-8"
)

print(
    "gas3 -> MQ-135"
)


print("\nClasses:")


for class_name in CLASSES:

    print(
        f"  - {class_name}"
    )


print("\nFinal MQ-135 thresholds:")

print(
    "  0 - 2899   -> Normal"
)

print(
    "  2900-3299  -> Alcohol"
)

print(
    "  >= 3300    -> Ammonia"
)


print("\nSaved files:")

print(
    f"1. {DATASET_PATH}"
)

print(
    f"2. {MODEL_PATH}"
)

print(
    f"3. {LABEL_ENCODER_PATH}"
)

print(
    f"4. {REPORT_PATH}"
)

print(
    f"5. {CONFUSION_MATRIX_PATH}"
)

print(
    f"6. {FEATURE_IMPORTANCE_PATH}"
)

print(
    f"7. {TREE_ACCURACY_PATH}"
)

print(
    f"8. {MODEL_INFO_PATH}"
)


print("\n" + "=" * 70)
print("ALL PROCESSING COMPLETED")
print("=" * 70)