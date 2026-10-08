# SmartSense AI – Real-Time Gas Detection & Smell Classification

An "electronic nose" web application that reads three MQ gas sensors (**MQ-2**, **MQ-8**, **MQ-135**), classifies the detected gas pattern with a **Random Forest** model backed by **sensor-threshold correction rules**, and shows the results on a live Flask dashboard. Every reading and prediction is stored in a SQLite database.

---

## Table of Contents

- [Features](#features)
- [Detected Classes](#detected-classes)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Sensor Thresholds](#sensor-thresholds)
- [Model Details & Results](#model-details--results)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Database Schema](#database-schema)
- [Configuration](#configuration)
- [Limitations & Notes](#limitations--notes)
- [Future Improvements](#future-improvements)

---

## Features

- **3-sensor gas array**: MQ-2 (`gas1`), MQ-8 (`gas2`), MQ-135 (`gas3`)
- **7-class classification**: Normal, Smoke, Butane, LPG, Alcohol, Ammonia, Methane
- **Two prediction modes**
  - `0` – Random Forest + sensor-threshold correction (default)
  - `1` – Fully condition (rule) based
- **Live dashboard** with sensor cards, prediction, confidence ring and a live Chart.js graph (updates every 3 seconds, last 30 points)
- **Alert popup** whenever a new prediction is detected
- **SQLite storage** of all readings, predictions, confidence values, method and timestamp
- **Database page** to browse the complete history, plus a "Clear Database" action
- **JSON REST API** for sensor reading, records, health check and database management
- **Synthetic dataset generator** and **training pipeline** with reports and plots included

---

## Detected Classes

| Class   | Primary Indicator                 |
|---------|-----------------------------------|
| Normal  | MQ-2 low and MQ-135 ≤ 2899        |
| Smoke   | MQ-2 > 2500                       |
| Butane  | MQ-8 in the Butane range          |
| LPG     | MQ-8 = 6001 – 7000                |
| Methane | MQ-8 > 7000                       |
| Alcohol | MQ-135 = 2900 – 3299              |
| Ammonia | MQ-135 ≥ 3300                     |

---

## Tech Stack

| Layer      | Technology                                   |
|------------|----------------------------------------------|
| Backend    | Python, Flask                                |
| ML         | scikit-learn (RandomForestClassifier), joblib |
| Data       | NumPy, pandas                                |
| Plots      | Matplotlib (training), Chart.js (dashboard)  |
| Database   | SQLite                                       |
| Frontend   | HTML, CSS, JavaScript (Jinja2 templates)     |

---

## Project Structure

```
.
├── app.py                          # Flask application (API, prediction logic, DB)
├── dataset.py                      # Synthetic dataset generator -> gas_sensor.csv
├── train.py                        # Trains Random Forest, saves model + reports + plots
├── gas_sensor.csv                  # Generated dataset (50,001 rows)
├── gas_sensor_random_forest.pkl    # Trained Random Forest model
├── gas_label_encoder.pkl           # Fitted LabelEncoder
├── gas_model_info.json             # Model metadata, accuracies and thresholds
├── gas_classification_report.txt   # Precision / recall / F1 report
├── gas_confusion_matrix.png        # Confusion matrix (test set)
├── gas_feature_importance.png      # Feature importance plot
├── gas_tree_accuracy.png           # Validation accuracy vs number of trees
├── gas_records.db                  # SQLite database (created automatically)
└── templates/
    ├── index.html                  # Main dashboard
    └── database.html               # Full database records page
```

> **Important:** Flask loads HTML files from a `templates/` folder. Place `index.html` and `database.html` inside `templates/`.

---

## How It Works

```
 MQ-2 / MQ-8 / MQ-135 sensors
            │
            ▼
   Sensor API (JSON: gas1, gas2, gas3)
            │   polled every 3 s by the dashboard
            ▼
   Flask  /api/read-sensor
            │
            ├── Mode 0: Random Forest prediction
            │            └─► MQ-8 threshold rule ─► MQ-135 threshold rule
            │
            └── Mode 1: Pure condition-based rules
            │
            ▼
   Prediction + confidence + probabilities
            │
            ├── Saved to SQLite (gas_records)
            └── Returned as JSON ─► Dashboard (cards, graph, alert, table)
```

### Prediction pipeline (Mode 0)

1. The three sensor values are fed to the trained **Random Forest**.
2. The **MQ-8 rule** overrides the result for strong MQ-8 readings (Methane / LPG / Butane).
3. The **MQ-135 rule** then corrects Alcohol / Ammonia / Normal decisions.
4. If a rule changed the Random Forest result, the method is stored as `Random Forest + Sensor Threshold` with **100%** confidence; otherwise it is `Random Forest Model` with the model's probability as confidence.

---

## Sensor Thresholds

**Dataset / training thresholds** (`dataset.py`, `gas_model_info.json`)

| Sensor | Range        | Class         |
|--------|--------------|---------------|
| MQ-2   | 0 – 150      | Normal        |
| MQ-2   | > 2500       | Smoke         |
| MQ-8   | 0 – 5000     | Normal / other|
| MQ-8   | 5001 – 6000  | Butane        |
| MQ-8   | 6001 – 7000  | LPG           |
| MQ-8   | > 7000       | Methane       |
| MQ-135 | 0 – 2899     | Normal        |
| MQ-135 | 2900 – 3299  | Alcohol       |
| MQ-135 | ≥ 3300       | Ammonia       |

**Runtime correction rules** (`app.py`)

| Sensor | Range        | Result                         |
|--------|--------------|--------------------------------|
| MQ-8   | > 7000       | Methane                        |
| MQ-8   | 6001 – 7000  | LPG                            |
| MQ-8   | 1701 – 6000  | Butane                         |
| MQ-8   | ≤ 1700       | Keep Random Forest result      |
| MQ-135 | ≥ 3300       | Ammonia                        |
| MQ-135 | 2900 – 3299  | Alcohol                        |
| MQ-135 | ≤ 2899       | Normal (if RF said Normal / Alcohol / Ammonia) |

---

## Model Details & Results

| Parameter           | Value                      |
|---------------------|----------------------------|
| Algorithm           | Random Forest              |
| Trees               | 300                        |
| Criterion           | Gini                       |
| `min_samples_leaf`  | 2                          |
| `max_features`      | sqrt                       |
| `class_weight`      | balanced                   |
| Features            | `gas1`, `gas2`, `gas3`     |
| Dataset size        | 50,001 samples (7,143 per class) |
| Split               | 70% train / 15% validation / 15% test (stratified) |
| Train / Val / Test  | 35,000 / 7,500 / 7,501     |

**Accuracy**

| Set        | Accuracy |
|------------|----------|
| Training   | 100.00%  |
| Validation | 100.00%  |
| Test       | 100.00%  |

### Visual results

**Confusion matrix**

![Confusion Matrix](gas_confusion_matrix.png)

**Feature importance** (MQ-8 is the most influential sensor)

![Feature Importance](gas_feature_importance.png)

**Validation accuracy vs number of trees**

![Tree Accuracy](gas_tree_accuracy.png)

---

## Installation

### Prerequisites

- Python 3.9+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install flask requests joblib numpy pandas scikit-learn matplotlib

# 4. Make sure the HTML files are inside a templates/ folder
mkdir -p templates
mv index.html database.html templates/
```

A `requirements.txt` you can use:

```
flask
requests
joblib
numpy
pandas
scikit-learn
matplotlib
```

> The pickled model must be loaded with the same scikit-learn version used for training. If you see a version warning or load error, simply re-run `train.py` (see below).

---

## Usage

### 1. (Optional) Regenerate the dataset

```bash
python dataset.py
```

Creates `gas_sensor.csv` with 7,143 samples for each of the 7 classes.

### 2. (Optional) Retrain the model

```bash
python train.py
```

Produces the model, label encoder, classification report, confusion matrix, feature-importance plot, tree-accuracy plot and `gas_model_info.json`.

### 3. Run the web application

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

### 4. Using the dashboard

1. Click **START MONITORING** – the app polls the sensor every 3 seconds.
2. Watch the live sensor values, predicted gas, confidence and graph.
3. Acknowledge the popup when a new prediction appears.
4. Click **DATABASE** to see the complete stored history.
5. Click **CLEAR DATABASE** to delete all records (asks for confirmation).

---

## API Endpoints

| Method          | Endpoint              | Description                                             |
|-----------------|-----------------------|---------------------------------------------------------|
| GET             | `/`                   | Main dashboard                                          |
| GET             | `/database`           | Full database records page                              |
| GET             | `/api/read-sensor`    | Fetch live sensor values, predict, save, return JSON    |
| GET             | `/api/records`        | Last 20 records                                         |
| GET             | `/api/all-records`    | All records                                             |
| GET             | `/api/database-info`  | Total record count                                      |
| DELETE / POST   | `/api/delete-database`| Delete all records                                      |
| GET             | `/api/health`         | Status, prediction mode and model-loaded flag           |

### Example response – `/api/read-sensor`

```json
{
  "success": true,
  "id": 42,
  "gas1": 850.0,
  "gas2": 6500.0,
  "gas3": 1700.0,
  "mq2": 850.0,
  "mq8": 6500.0,
  "mq135": 1700.0,
  "prediction": "LPG",
  "confidence": 100.0,
  "probabilities": { "LPG": 100.0, "Butane": 0.0, "...": 0.0 },
  "prediction_method": "Random Forest Model",
  "created_at": "2026-10-08 14:30:00",
  "total_records": 42,
  "latest_records": []
}
```

---

## Database Schema

SQLite file: `gas_records.db`, table `gas_records` (created automatically on first run).

| Column              | Type    | Description                                   |
|---------------------|---------|-----------------------------------------------|
| `id`                | INTEGER | Primary key (auto-increment)                  |
| `gas1`              | REAL    | MQ-2 reading                                  |
| `gas2`              | REAL    | MQ-8 reading                                  |
| `gas3`              | REAL    | MQ-135 reading                                |
| `prediction`        | TEXT    | Predicted class                               |
| `confidence`        | REAL    | Confidence in %                               |
| `prediction_method` | TEXT    | `Random Forest Model`, `Random Forest + Sensor Threshold` or `Condition Based` |
| `created_at`        | TEXT    | Timestamp (`YYYY-MM-DD HH:MM:SS`)             |

---

## Configuration

Edit the constants at the top of `app.py`:

| Variable             | Default                                   | Purpose                                  |
|----------------------|-------------------------------------------|------------------------------------------|
| `SENSOR_API_URL`     | `https://aislyntech.com/Api/44-get.php`   | Endpoint that returns `gas1`, `gas2`, `gas3` as JSON |
| `MODEL_PATH`         | `gas_sensor_random_forest.pkl`            | Trained model file                       |
| `LABEL_ENCODER_PATH` | `gas_label_encoder.pkl`                   | Label encoder file                       |
| `DATABASE`           | `gas_records.db`                          | SQLite database file                     |
| `PREDICTION_MODE`    | `0`                                       | `0` = RF + threshold, `1` = condition based |

The sensor API is expected to return either a flat JSON object or one wrapped in a `data` key:

```json
{ "gas1": 120, "gas2": 3000, "gas3": 1500 }
```

---

## Limitations & Notes

- **The 100% accuracy comes from synthetic data.** The dataset is generated from the same threshold rules the classes are defined by, so the model effectively learns those rules. Real MQ sensors drift with temperature, humidity, warm-up time and cross-sensitivity, so real-world accuracy will be lower. Collecting and training on real labelled sensor data is recommended before relying on this for safety decisions.
- The training thresholds for Butane (MQ-8 5001–6000) and the runtime rule (MQ-8 1701–6000) differ; the runtime rule is intentionally broader.
- This project is a demonstration / educational prototype and **must not be used as a certified gas-leak or safety alarm**.
- The Flask app runs with `debug=True` and `host="0.0.0.0"`, and the delete endpoint has no authentication. Disable debug mode and add authentication before deploying publicly.
- The sensor API URL is hard-coded; change it for your own hardware / backend.

---

## Future Improvements

- Train on real, labelled MQ-sensor data
- Add temperature / humidity compensation
- Sensor calibration and warm-up handling
- Authentication for destructive endpoints
- Export records to CSV
- Email / SMS / buzzer alerts for dangerous gases
- Docker support and a production WSGI server (Gunicorn)

---

## License

Add your preferred license here (e.g. MIT).

## Author

Your name – [GitHub profile](https://github.com/<your-username>)
