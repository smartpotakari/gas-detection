from flask import Flask, jsonify, render_template
import requests
import sqlite3
import os
import joblib
import numpy as np
from datetime import datetime


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

SENSOR_API_URL = "https://aislyntech.com/Api/44-get.php"

MODEL_PATH = "gas_sensor_random_forest.pkl"

LABEL_ENCODER_PATH = "gas_label_encoder.pkl"

DATABASE = "gas_records.db"


# ============================================================
# PREDICTION MODE
# ============================================================

# 0 = Random Forest Model + Sensor Threshold Correction
# 1 = Fully Condition Based

PREDICTION_MODE = 0


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
# MODEL VARIABLES
# ============================================================

model = None
label_encoder = None


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    global model
    global label_encoder

    # Already loaded
    if model is not None:
        return

    # Check model
    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"{MODEL_PATH} not found. "
            "Train the Random Forest model first."
        )

    # Load Random Forest model
    try:

        model = joblib.load(
            MODEL_PATH
        )

    except Exception as e:

        model = None

        raise RuntimeError(
            f"Could not load {MODEL_PATH}: {e}"
        )

    # Load label encoder
    if os.path.exists(
        LABEL_ENCODER_PATH
    ):

        try:

            label_encoder = joblib.load(
                LABEL_ENCODER_PATH
            )

        except Exception as e:

            label_encoder = None

            raise RuntimeError(
                f"Could not load "
                f"{LABEL_ENCODER_PATH}: {e}"
            )

    else:

        print(
            "WARNING: Label encoder file not found."
        )

    print(
        "Random Forest model loaded for prediction."
    )


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db():

    conn = sqlite3.connect(
        DATABASE
    )

    conn.row_factory = sqlite3.Row

    return conn


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def init_database():

    conn = get_db()

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS gas_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gas1 REAL NOT NULL,
            gas2 REAL NOT NULL,
            gas3 REAL NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            prediction_method TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    conn.commit()

    conn.close()


init_database()


# ============================================================
# CONDITION-BASED PREDICTION
# ============================================================

def predict_by_condition(
    gas1,
    gas2,
    gas3
):

    # --------------------------------------------------------
    # MQ-2
    # >2500 = Smoke
    # --------------------------------------------------------

    if gas1 > 2500:

        return "Smoke", 100.0


    # --------------------------------------------------------
    # MQ-8
    # >7000 = Methane
    # --------------------------------------------------------

    elif gas2 > 7000:

        return "Methane", 100.0


    # --------------------------------------------------------
    # MQ-8
    # 6001-7000 = LPG
    # --------------------------------------------------------

    elif gas2 > 6000:

        return "LPG", 100.0


    # --------------------------------------------------------
    # MQ-8
    # 1701-6000 = Butane
    # --------------------------------------------------------

    elif gas2 > 1700:

        return "Butane", 100.0


    # --------------------------------------------------------
    # MQ-135
    # >=3300 = Ammonia
    # --------------------------------------------------------

    elif gas3 >= 3300:

        return "Ammonia", 100.0


    # --------------------------------------------------------
    # MQ-135
    # 2900-3299 = Alcohol
    # --------------------------------------------------------

    elif gas3 >= 2900:

        return "Alcohol", 100.0


    # --------------------------------------------------------
    # Normal
    # MQ-2 <=150 OR MQ-135 <=2899
    # --------------------------------------------------------

    elif gas1 <= 150 or gas3 <= 2899:

        return "Normal", 100.0


    # --------------------------------------------------------
    # Default
    # --------------------------------------------------------

    return "Normal", 100.0


# ============================================================
# MQ-8 THRESHOLD CORRECTION
# ============================================================

def apply_mq8_rule(
    gas2,
    rf_prediction
):

    """
    MQ-8 classification:

    >7000      -> Methane
    6001-7000  -> LPG
    1701-6000  -> Butane

    Values <=1700 keep the Random Forest result.
    """

    # --------------------------------------------------------
    # MQ-8 >7000
    # --------------------------------------------------------

    if gas2 > 7000:

        return "Methane"


    # --------------------------------------------------------
    # MQ-8 6001-7000
    # --------------------------------------------------------

    elif gas2 > 6000:

        return "LPG"


    # --------------------------------------------------------
    # MQ-8 1701-6000
    # --------------------------------------------------------

    elif gas2 > 1700:

        return "Butane"


    # --------------------------------------------------------
    # <=1700
    # Keep Random Forest result
    # --------------------------------------------------------

    return rf_prediction


# ============================================================
# MQ-135 THRESHOLD CORRECTION
# ============================================================

def apply_mq135_rule(
    gas3,
    rf_prediction
):

    """
    MQ-135 classification:

    0-2899     -> Normal
    2900-3299  -> Alcohol
    >=3300     -> Ammonia
    """

    # --------------------------------------------------------
    # MQ-135 >=3300
    # --------------------------------------------------------

    if gas3 >= 3300:

        return "Ammonia"


    # --------------------------------------------------------
    # MQ-135 2900-3299
    # --------------------------------------------------------

    elif gas3 >= 2900:

        return "Alcohol"


    # --------------------------------------------------------
    # MQ-135 0-2899
    # --------------------------------------------------------

    elif gas3 <= 2899:

        if rf_prediction in [
            "Normal",
            "Alcohol",
            "Ammonia"
        ]:

            return "Normal"


    # --------------------------------------------------------
    # Keep Random Forest result
    # --------------------------------------------------------

    return rf_prediction


# ============================================================
# MAIN GAS PREDICTION
# ============================================================

def predict_gas(
    gas1,
    gas2,
    gas3
):

    # ========================================================
    # CONDITION-BASED MODE
    # ========================================================

    if PREDICTION_MODE == 1:

        prediction, confidence = predict_by_condition(

            gas1,
            gas2,
            gas3

        )

        probabilities = {

            cls:
            100.0 if cls == prediction else 0.0

            for cls in CLASSES

        }

        return (

            prediction,
            confidence,
            probabilities,
            "Condition Based"

        )


    # ========================================================
    # RANDOM FOREST MODE
    # ========================================================

    load_model()


    # --------------------------------------------------------
    # Prepare sensor values
    # --------------------------------------------------------

    values = np.array(
        [
            [
                gas1,
                gas2,
                gas3
            ]
        ],
        dtype=float
    )


    # --------------------------------------------------------
    # Random Forest prediction
    # --------------------------------------------------------

    result = model.predict(
        values
    )[0]


    # --------------------------------------------------------
    # Convert prediction to class
    # --------------------------------------------------------

    if label_encoder is not None:

        prediction = label_encoder.inverse_transform(
            [result]
        )[0]

    else:

        prediction = str(result)


    # --------------------------------------------------------
    # Random Forest probabilities
    # --------------------------------------------------------

    probabilities = {}


    if hasattr(
        model,
        "predict_proba"
    ):

        probs = model.predict_proba(
            values
        )[0]


        # Use model.classes_ to correctly
        # map probabilities.

        for model_class, probability in zip(
            model.classes_,
            probs
        ):

            if label_encoder is not None:

                cls = label_encoder.inverse_transform(
                    [model_class]
                )[0]

            else:

                cls = str(model_class)


            probabilities[cls] = round(
                float(probability) * 100,
                2
            )


        if probabilities:

            rf_confidence = max(
                probabilities.values()
            )

        else:

            rf_confidence = 100.0

    else:

        rf_confidence = 100.0


    # ========================================================
    # APPLY MQ-8 RULE FIRST
    # ========================================================

    prediction_after_mq8 = apply_mq8_rule(

        gas2,

        prediction

    )


    # ========================================================
    # APPLY MQ-135 RULE
    # ========================================================

    corrected_prediction = apply_mq135_rule(

        gas3,

        prediction_after_mq8

    )


    # ========================================================
    # DETERMINE PREDICTION METHOD
    # ========================================================

    if corrected_prediction != prediction:

        prediction_method = (
            "Random Forest + Sensor Threshold"
        )

    else:

        prediction_method = (
            "Random Forest Model"
        )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    if corrected_prediction != prediction:

        confidence = 100.0

    else:

        confidence = rf_confidence


    # ========================================================
    # UPDATE PROBABILITIES IF THRESHOLD CORRECTED
    # ========================================================

    if corrected_prediction != prediction:

        probabilities = {

            cls:
            100.0 if cls == corrected_prediction else 0.0

            for cls in CLASSES

        }


    elif corrected_prediction not in probabilities:

        probabilities = {

            cls:
            100.0 if cls == corrected_prediction else 0.0

            for cls in CLASSES

        }


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return (

        corrected_prediction,

        confidence,

        probabilities,

        prediction_method

    )


# ============================================================
# SAVE RECORD
# ============================================================

def save_record(

    gas1,
    gas2,
    gas3,
    prediction,
    confidence,
    prediction_method

):

    created_at = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


    conn = get_db()


    cursor = conn.execute(
        """
        INSERT INTO gas_records
        (
            gas1,
            gas2,
            gas3,
            prediction,
            confidence,
            prediction_method,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            gas1,
            gas2,
            gas3,
            prediction,
            confidence,
            prediction_method,
            created_at
        )
    )


    record_id = cursor.lastrowid


    conn.commit()

    conn.close()


    return record_id, created_at


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    conn = get_db()


    total_records = conn.execute(
        """
        SELECT COUNT(*) AS total
        FROM gas_records
        """
    ).fetchone()["total"]


    latest_records = conn.execute(
        """
        SELECT *
        FROM gas_records
        ORDER BY id DESC
        LIMIT 5
        """
    ).fetchall()


    conn.close()


    return render_template(

        "index.html",

        total_records=total_records,

        latest_records=[
            dict(row)
            for row in latest_records
        ]

    )


# ============================================================
# DATABASE PAGE
# ============================================================

@app.route("/database")
def database():

    conn = get_db()


    records = conn.execute(
        """
        SELECT *
        FROM gas_records
        ORDER BY id DESC
        """
    ).fetchall()


    total_records = len(records)


    conn.close()


    return render_template(

        "database.html",

        records=[
            dict(row)
            for row in records
        ],

        total_records=total_records

    )


# ============================================================
# READ LIVE SENSOR
# ============================================================

@app.route("/api/read-sensor")
def read_sensor():

    try:

        # ----------------------------------------------------
        # Request sensor API
        # ----------------------------------------------------

        response = requests.get(

            SENSOR_API_URL,

            timeout=5

        )


        response.raise_for_status()


        sensor_response = response.json()


        # ----------------------------------------------------
        # Get sensor data
        # ----------------------------------------------------

        data = sensor_response.get(

            "data",

            sensor_response

        )


        # ----------------------------------------------------
        # MQ-2
        # ----------------------------------------------------

        gas1 = float(

            data.get(
                "gas1",
                0
            )

        )


        # ----------------------------------------------------
        # MQ-8
        # ----------------------------------------------------

        gas2 = float(

            data.get(
                "gas2",
                0
            )

        )


        # ----------------------------------------------------
        # MQ-135
        # ----------------------------------------------------

        gas3 = float(

            data.get(
                "gas3",
                0
            )

        )


        # ----------------------------------------------------
        # Validate values
        # ----------------------------------------------------

        if (
            gas1 < 0
            or gas2 < 0
            or gas3 < 0
        ):

            raise ValueError(
                "Sensor values cannot be negative."
            )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        (
            prediction,
            confidence,
            probabilities,
            method

        ) = predict_gas(

            gas1,
            gas2,
            gas3

        )


        # ----------------------------------------------------
        # Save database record
        # ----------------------------------------------------

        record_id, created_at = save_record(

            gas1,
            gas2,
            gas3,

            prediction,

            confidence,

            method

        )


        # ----------------------------------------------------
        # Get total records
        # ----------------------------------------------------

        conn = get_db()


        total_records = conn.execute(
            """
            SELECT COUNT(*) AS total
            FROM gas_records
            """
        ).fetchone()["total"]


        # ----------------------------------------------------
        # Get latest records
        # ----------------------------------------------------

        latest_records = conn.execute(
            """
            SELECT *
            FROM gas_records
            ORDER BY id DESC
            LIMIT 5
            """
        ).fetchall()


        conn.close()


        # ----------------------------------------------------
        # Return JSON
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "id": record_id,

            "gas1": gas1,

            "gas2": gas2,

            "gas3": gas3,

            "mq2": gas1,

            "mq8": gas2,

            "mq135": gas3,

            "prediction": prediction,

            "confidence": round(
                confidence,
                2
            ),

            "probabilities": probabilities,

            "prediction_method": method,

            "created_at": created_at,

            "total_records": total_records,

            "latest_records": [

                dict(row)

                for row in latest_records

            ]

        })


    except Exception as e:

        import traceback


        print(
            "\n========== API ERROR =========="
        )


        print(
            type(e).__name__
        )


        print(
            str(e)
        )


        traceback.print_exc()


        print(
            "================================\n"
        )


        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# GET LAST 20 RECORDS
# ============================================================

@app.route("/api/records")
def get_records():

    conn = get_db()


    rows = conn.execute(
        """
        SELECT *
        FROM gas_records
        ORDER BY id DESC
        LIMIT 20
        """
    ).fetchall()


    conn.close()


    return jsonify({

        "success": True,

        "records": [

            dict(row)

            for row in rows

        ]

    })


# ============================================================
# GET ALL RECORDS
# ============================================================

@app.route("/api/all-records")
def get_all_records():

    conn = get_db()


    rows = conn.execute(
        """
        SELECT *
        FROM gas_records
        ORDER BY id DESC
        """
    ).fetchall()


    conn.close()


    return jsonify({

        "success": True,

        "records": [

            dict(row)

            for row in rows

        ]

    })


# ============================================================
# DATABASE INFORMATION
# ============================================================

@app.route("/api/database-info")
def database_info():

    conn = get_db()


    total = conn.execute(
        """
        SELECT COUNT(*) AS total
        FROM gas_records
        """
    ).fetchone()["total"]


    conn.close()


    return jsonify({

        "success": True,

        "total_records": total

    })


# ============================================================
# DELETE DATABASE RECORDS
# ============================================================

@app.route(
    "/api/delete-database",
    methods=["DELETE", "POST"]
)
def delete_database():

    conn = get_db()


    conn.execute(
        "DELETE FROM gas_records"
    )


    conn.commit()

    conn.close()


    return jsonify({

        "success": True,

        "message": "All database records deleted."

    })


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/api/health")
def health():

    method = (

        "Random Forest Model + Sensor Threshold"

        if PREDICTION_MODE == 0

        else

        "Condition Based"

    )


    return jsonify({

        "success": True,

        "status": "running",

        "prediction_mode": PREDICTION_MODE,

        "prediction_method": method,

        "model_loaded": model is not None

    })


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    print(
        "\n==================================="
    )

    print(
        "       GASSENSE AI SYSTEM"
    )

    print(
        "==================================="
    )


    print(

        "Prediction Mode:",

        "Random Forest Model + Sensor Threshold"

        if PREDICTION_MODE == 0

        else

        "Condition Based"

    )


    print(
        "MQ-2   -> gas1"
    )

    print(
        "MQ-8   -> gas2"
    )

    print(
        "MQ-135 -> gas3"
    )


    print(
        "\nMQ-8 Classification:"
    )

    print(
        "0-1700    -> Random Forest"
    )

    print(
        "1701-6000 -> Butane"
    )

    print(
        "6001-7000 -> LPG"
    )

    print(
        ">7000     -> Methane"
    )


    print(
        "\nMQ-135 Classification:"
    )

    print(
        "0-2899    -> Normal"
    )

    print(
        "2900-3299 -> Alcohol"
    )

    print(
        ">=3300    -> Ammonia"
    )


    print(
        "===================================\n"
    )


    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )