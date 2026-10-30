# MPLAD AI — AI-Powered Detection of Anomalies, Fraud and Inefficiencies in MPLADS

An AI/ML-powered monitoring platform for the **Members of Parliament Local Area
Development Scheme (MPLADS)**. It ingests project implementation data and uses
machine learning to surface **anomalies, fraud-risk indicators, cost
inefficiencies, delays, duplicate projects, and geographical clustering** for
human review — without ever automatically accusing anyone of wrongdoing.

> Every risk score is shown with the underlying evidence and labelled with
> neutral language ("High Risk", "Potential Anomaly", "Requires Review").
> **Risk scores identify unusual patterns for further review and do not
> establish fraud or wrongdoing.**

---

## 1. Features

- Isolation Forest + Local Outlier Factor unsupervised anomaly detection
- Rule-based, weighted fraud-risk indicator scoring (0–100, explainable)
- Inefficiency scoring (budget utilization, duration, cost overrun, coverage)
- Leakage-safe Gradient Boosting delay-risk prediction with a calibrated
  high-precision operating threshold (90.6% precision on a stratified holdout
  of the supplied sample data)
- TF-IDF + cosine-similarity duplicate/near-duplicate project detection
- Transparent, weighted **Overall Risk Score** (Anomaly 30% / Financial 25% /
  Delay 20% / Inefficiency 15% / Duplicate 10%) with a per-project factor
  breakdown ("Explainable AI")
- Role-based access (Admin / Analyst / Auditor / Viewer) with JWT-secured REST API
- Bootstrap 5 dashboard: 8 summary cards, 10 Chart.js charts, project table
  with search/filter/sort/pagination, project detail with Leaflet map,
  anomaly monitoring, fraud-risk dashboard, state/category analytics with a
  clustered map, and CSV/PDF report generation
- CSV upload → validate → clean → feature-engineer → store in MongoDB → run
  full ML pipeline → dashboard, all in one workflow

---

## 2. Technology Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11+, Django 4.2, Django REST Framework |
| Auth | Custom JWT (simplejwt tokens issued/verified against MongoEngine users) |
| Database | MongoDB via MongoEngine / PyMongo |
| ML | Pandas, NumPy, Scikit-learn, SciPy, Joblib |
| Frontend | HTML5, CSS3, Bootstrap 5, Chart.js, Leaflet.js, vanilla JS / fetch |

No MySQL/PostgreSQL is used for application data. Django's internal
session/admin framework requires *some* SQL database, so a local SQLite file
(`django_internal.sqlite3`) is used **only** for that — every MPLADS record,
transaction, anomaly and prediction lives in MongoDB.

---

## 3. Architecture

```
MPLAD_AI/
├── manage.py
├── requirements.txt
├── .env / .env.example
├── config/                # settings, urls, wsgi/asgi
├── accounts/              # MongoEngine User model, JWT auth, login pages
├── projects/              # Project/Transaction/Anomaly/MLPrediction models,
│                             CSV import + ML orchestration (services.py),
│                             REST views, page views
├── analytics/              # Dashboard aggregations, Chart.js datasets,
│                             CSV/PDF report generation
├── ml_engine/
│   ├── feature_engineering.py   # cleaning, derived features
│   ├── anomaly_detection.py     # Isolation Forest + LOF
│   ├── fraud_detection.py       # fraud-risk indicators
│   ├── inefficiency_detection.py# efficiency score, delay model, duplicates
│   ├── train_models.py          # standalone training script
│   ├── predict.py               # orchestrates all models → overall risk
│   └── models/                  # saved .joblib models + metadata.json
├── data/
│   ├── generate_sample_data.py  # synthetic dataset generator
│   └── sample_mplad_data.csv    # 1200+ synthetic records
├── templates/              # Bootstrap pages
├── static/css, static/js
└── utils/                  # Mongo connection + shared helpers
```

---

## 4. Installation

### 4.1 Prerequisites
- Python 3.11+
- A running MongoDB instance (local install, Docker, or MongoDB Atlas)

### 4.2 Set up the virtual environment

```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```
macOS/Linux:
```bash
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 4.3 MongoDB setup

Any of the following works:

```bash
# Local install (Ubuntu/Debian, MongoDB 7.x from mongodb.org's own repo)
sudo apt-get install -y mongodb-org
sudo systemctl start mongod

# OR Docker
docker run -d -p 27017:27017 --name mplad-mongo mongo:7

# OR MongoDB Atlas (cloud) — just point MONGO_URI at your Atlas connection string
```

### 4.4 Environment variables

Copy `.env.example` to `.env` and adjust as needed:

```env
MONGO_URI=mongodb://localhost:27017/mplad_ai
MONGO_DB_NAME=mplad_ai
SECRET_KEY=change-this-in-production
DEBUG=True
ALLOWED_HOSTS=127.0.0.1,localhost
```

### 4.5 Django internal database (sessions/admin)

```bash
python manage.py migrate
```

### 4.6 Create demo users (all 4 roles)

```bash
python manage.py seed_users
```

This creates:

| Username | Password | Role |
|---|---|---|
| `admin` | `Admin@123` | Admin |
| `analyst` | `Analyst@123` | Analyst |
| `auditor` | `Auditor@123` | Auditor |
| `viewer` | `Viewer@123` | Viewer |

### 4.7 Generate the sample dataset (if not already present)

```bash
python data/generate_sample_data.py
```

Generates 1200+ synthetic MPLADS records (`data/sample_mplad_data.csv`) with
realistic values **and intentionally injected anomalies** (cost overruns,
stalled-but-fully-spent projects, extreme delays, duplicate locations,
abnormal cost-per-beneficiary, expenditure spikes, and near-duplicate
projects) so the ML models have genuine patterns to learn from.

### 4.8 Train the ML models

```bash
python ml_engine/train_models.py
```

Trains and saves (via Joblib) the Isolation Forest anomaly detector and the
Gradient Boosting delay-risk model into `ml_engine/models/`, plus a
`model_metadata.json` version record. The delay model excludes `delay_days`
from its input features to prevent target leakage, uses a stratified 80/20
holdout, tunes a probability threshold for high precision, and records
accuracy, precision, recall, F1, ROC-AUC, average precision, and the confusion
matrix. On the supplied synthetic sample data, the holdout metrics are:

| Metric | Value |
|---|---:|
| Accuracy | 81.1% |
| Precision | 90.6% |
| Recall | 40.3% |
| F1 | 55.8% |
| ROC-AUC | 78.4% |

These figures are not a guarantee of real-world performance: the included CSV
is synthetic and must be replaced/revalidated with official SIH/MPLADS data
before deployment. (LOF is transductive and is scored fresh at prediction time
— it has no model file to save.)

### 4.9 Import the sample data + run full ML analysis

```bash
python manage.py import_sample_data
```

This validates and cleans the CSV, engineers features, stores everything in
MongoDB, then runs the complete ML pipeline (anomaly detection, fraud-risk,
inefficiency, delay-risk, duplicate detection, overall risk scoring) and
persists `anomalies` + `ml_predictions`.

### 4.10 Run the server

```bash
python manage.py runserver
```

Open **http://127.0.0.1:8000/** and log in with any demo user above.

---

## 5. Dataset Format

`data/sample_mplad_data.csv` columns:

```
project_id, mp_name, constituency, district, state, project_category,
sanctioned_amount, released_amount, expenditure_amount, estimated_cost,
actual_cost, start_date, expected_completion_date, actual_completion_date,
completion_percentage, contractor, implementing_agency, beneficiaries,
latitude, longitude, project_status
```

Categories: Roads, Bridges, Schools, Hospitals, Drinking Water, Sanitation,
Community Halls, Electricity, Public Infrastructure, Sports Facilities.

You can upload your own CSV (same columns) from the **Upload Data** page —
it will be validated, cleaned, engineered, stored, and analyzed automatically.

---

## 6. ML Workflow

```
CSV Upload
  → Validate columns (Stage 28)
  → Clean (missing values, duplicates, invalid values)
  → Engineer features (expenditure_ratio, cost_overrun, delay_days,
    utilization_ratio, completion_efficiency, cost_per_beneficiary)
  → Store in MongoDB (projects collection)
  → Isolation Forest + Local Outlier Factor  → anomaly_score
  → Fraud-risk indicator scoring             → fraud_risk_score
  → Efficiency scoring                       → inefficiency_score
  → Gradient Boosting delay classifier       → delay_risk_score
  → TF-IDF + cosine similarity               → duplicate_score
  → Weighted Overall Risk Score (30/25/20/15/10)
  → Persist to anomalies + ml_predictions collections
  → Dashboard / Anomaly page / Fraud-risk page / Project detail page
```

Every score comes with plain-language "Detected Patterns" reasons and a
per-factor contribution breakdown, so the AI's output is explainable rather
than a single opaque number.

---

## 7. API Documentation

Base URL: `http://127.0.0.1:8000/api/`

### Auth
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/auth/register/` | Create a user `{username,email,password,role,full_name}` |
| POST | `/api/auth/login/` | `{username,password}` → `{user, access, refresh}` |
| GET | `/api/auth/me/` | Current user (requires `Authorization: Bearer <access>`) |

### Projects
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/projects/` | List (filters: `state`, `category`, `status`, `risk`, `search`, `page`, `page_size`) |
| POST | `/api/projects/` | Create (Admin) |
| GET | `/api/projects/<project_id>/` | Detail + latest ML prediction + anomalies |
| PUT | `/api/projects/<project_id>/` | Update |
| DELETE | `/api/projects/<project_id>/` | Delete (Admin only) |
| POST | `/api/projects/upload-csv/` | Multipart CSV upload → import + auto-analyze |

### Anomalies
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/anomalies/` | List (filters: `severity`, `type`, `review_status`) |
| GET | `/api/anomalies/<anomaly_id>/` | Detail |
| PATCH | `/api/anomalies/<anomaly_id>/` | Update `review_status` |

### ML
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/ml/predict/<project_id>/` | Latest (or freshly computed) prediction for one project |
| POST | `/api/ml/analyze/` | `{"project_id": "..."}` to analyze one project, or `{}` to re-analyze the whole dataset |

### Analytics & Reports
| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/analytics/dashboard/` | Summary cards + all chart datasets |
| GET | `/api/analytics/states/` | State-level rollup |
| GET | `/api/analytics/categories/` | Category-level rollup |
| GET | `/api/reports/?type=<project\|anomaly\|financial\|risk\|state\|constituency>&file_format=<csv\|pdf>` | Download a report |

> Note: the report-format query parameter is named **`file_format`**, not
> `format` — `format` is reserved by Django REST Framework for content
> negotiation and would otherwise collide with it.

### Example: `POST /api/ml/analyze/`

Request:
```json
{ "project_id": "MPLAD00001" }
```

Response:
```json
{
  "project_id": "MPLAD00001",
  "anomaly_probability": 12.4,
  "fraud_risk_score": 8.1,
  "inefficiency_score": 21.7,
  "delay_risk_score": 4.2,
  "overall_risk_score": 11.9,
  "risk_level": "Low",
  "reasons": ["No significant risk indicators detected"]
}
```

---

## 8. Authentication & Roles

| Role | Permissions |
|---|---|
| Admin | Add/edit/delete projects, run ML analysis, view all reports, manage users |
| Analyst | View projects, run analysis, view analytics, generate reports |
| Auditor | View risk reports, review anomalies, change review status |
| Viewer | View dashboard, project information, public analytics |

The REST API is JWT-secured (`Authorization: Bearer <access-token>`); the
server-rendered dashboard uses a lightweight Django-session login separately
from the JWT API.

---

## 9. AI/ML Concepts Used

- **Unsupervised anomaly detection**: Isolation Forest (tree-based isolation
  of outliers), Local Outlier Factor (density-based local outlier scoring)
- **Feature engineering**: ratio and rate features (utilization, cost
  overrun, cost per beneficiary, completion efficiency) that are far more
  informative to models than raw amounts
- **Supervised classification**: Gradient Boosting for delay-risk prediction
- **NLP / text similarity**: TF-IDF vectorization + cosine similarity for
  duplicate-project detection
- **Explainable AI**: every prediction surfaces the specific features and
  weighted contribution factors that drove the score, not just a number
- **Ensemble score blending**: Isolation Forest and LOF scores are blended,
  with a bonus when both methods agree — reducing false positives from any
  single method's blind spots
- **Weighted multi-criteria scoring**: the Overall Risk Score explicitly
  documents its weights (30/25/20/15/10) rather than hiding them in a
  black-box model

---

## 10. Viva Questions & Answers

**Q: Why Isolation Forest AND Local Outlier Factor, not just one?**
A: Isolation Forest is efficient and works well for global outliers across
many features; LOF captures *local* density-based outliers that IF can miss
(a project that looks normal by dataset-wide standards but is unusual next
to its true peers). Blending both, with a bonus when they agree, gives more
robust anomaly detection than either alone.

**Q: How is "fraud" avoided as a legal claim?**
A: The system only ever outputs a 0–100 "fraud_risk_score" and labels of
Low/Medium/High Risk, always alongside the specific indicators that fired
and an explicit disclaimer. No component claims to prove fraud — that
determination is left to human auditors.

**Q: Why MongoDB instead of a relational database?**
A: MPLADS project records vary in structure across states/schemes, and the
system stores flexible nested data (contributing_factors, detected_features)
per anomaly/prediction. MongoDB's schema flexibility suits this better than
a rigid relational schema, and it scales horizontally for large datasets.

**Q: How is the Overall Risk Score computed?**
A: A weighted sum: 30% anomaly score (Isolation Forest + LOF blend), 25%
financial risk (fraud-risk score blended with cost overrun), 20% delay-risk
(Gradient Boosting classifier probability), 15% inefficiency score, and 10%
duplicate/suspicious-pattern score — normalized to 0–100 and bucketed into
Low/Moderate/High/Critical.

**Q: How does the system explain its predictions?**
A: `ml_engine/predict.py` computes a per-project `contributing_factors` dict
(each risk component's weighted contribution) and a `reasons` list in plain
English (e.g. "Cost overrun exceeds the acceptable threshold"), both stored
in MongoDB and rendered directly in the Project Detail and Fraud Risk pages.

**Q: How would this scale to real MPLADS data (thousands of MPs, lakhs of
projects)?**
A: MongoDB indexes on `state`, `district`, `project_category`, and
geolocation support efficient filtering; the ML pipeline is vectorized with
pandas/NumPy/scikit-learn and can be batched or scheduled (e.g., via Celery)
for periodic re-scoring rather than synchronous analysis on every request.

**Q: What's the false-positive risk, and how is it mitigated?**
A: Every "anomaly" is a statistical outlier, not a proven violation — hence
the neutral language and mandatory reviewer workflow (Pending → Under Review
→ Verified/Dismissed). The IsolationForest/LOF agreement bonus and multi-
indicator fraud scoring (10 separate indicators, weighted) reduce single-
signal false positives.

---

## 11. Future Enhancements

- Real-time streaming ingestion from state MPLADS portals
- Model retraining pipeline with feedback loop from Auditor "Verified/Dismissed" outcomes
- SHAP/LIME-based per-feature attribution for even finer-grained explainability
- Multi-language (regional language) dashboard support
- Email/SMS alerting for newly detected Critical-risk projects
- Integration with GIS shapefiles for richer geographical anomaly detection
- Audit-trail logging of every review-status change with the reviewing user

---

## 12. Testing Notes

This project was validated end-to-end during development using
[`mongomock`](https://github.com/mongomock/mongomock) (an in-memory MongoDB
stand-in, not part of `requirements.txt`) since a live MongoDB server wasn't
available in the build environment. All pages, the full REST API, CSV
upload, ML training/prediction, and CSV/PDF report generation were verified
this way. Install it separately (`pip install mongomock`) only if you want
to repeat that kind of test without a real MongoDB server; production use
should always point `MONGO_URI` at a real MongoDB instance.
