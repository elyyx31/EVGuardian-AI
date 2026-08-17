# EVGuardian-AI: AI-Based Battery Health Monitoring & Remaining Useful Life (RUL) Estimation

EVGuardian-AI is an end-to-end machine learning and web application framework designed to monitor Lithium-ion battery State of Health ($\text{SOH}$), estimate Remaining Useful Life ($\text{RUL}$) to End-of-Life ($\text{EOL}$), classify operational health status, and provide actionable maintenance recommendations.

---

## Table of Contents
- [1. Problem Statement](#1-problem-statement)
- [2. Project Objectives](#2-project-objectives)
- [3. System Architecture](#3-system-architecture)
- [4. Dataset & Features](#4-dataset--features)
- [5. SOH Modeling & Regression Methodology](#5-soh-modeling--regression-methodology)
- [6. Experimental Evaluation & Validation](#6-experimental-evaluation--validation)
- [7. Derived RUL Prognostics Methodology](#7-derived-rul-prognostics-methodology)
- [8. Battery Health Classification](#8-battery-health-classification)
- [9. Advisory Recommendation Engine](#9-advisory-recommendation-engine)
- [10. Web Dashboard & REST API](#10-web-dashboard--rest-api)
- [11. Installation & Setup](#11-installation--setup)
- [12. How to Run](#12-how-to-run)
- [13. Repository Structure](#13-repository-structure)
- [14. Scientific Limitations & Future Work](#14-scientific-limitations--future-work)

---

## 1. Problem Statement

Lithium-ion battery degradation is a non-linear electrochemical process influenced by cycling aging, temperature fluctuations, charge/discharge rates ($\text{C-rate}$), and internal impedance growth. In Electric Vehicle (EV) battery management systems (BMS), accurate State of Health ($\text{SOH}$) estimation and Remaining Useful Life ($\text{RUL}$) prognostics are critical for:
- Preventing sudden in-service power failures or thermal runaway events.
- Optimizing maintenance schedules and second-life battery repurposing.
- Providing transparent battery health diagnostics to vehicle owners and fleet managers.

---

## 2. Project Objectives

1. Develop a supervised regression pipeline to estimate continuous battery $\text{SOH}$ from measurable cycle and operational parameters.
2. Formulate a model-derived prognostic method to estimate $\text{RUL}$ cycles until the standard industrial End-of-Life threshold ($\text{SOH} = 80\%$).
3. Categorize battery condition into clear operational tiers (`Healthy`, `Moderate`, `Poor`).
4. Provide a thermal-safety-first maintenance advisory engine.
5. Deploy the diagnostic pipeline via an interactive Flask web dashboard and REST API.

---

## 3. System Architecture

```
+-------------------------------------------------------------------------------+
|                                Web Dashboard UI                               |
|                  (HTML5 / CSS3 / JavaScript / Chart.js)                       |
+---------------------------------------+---------------------------------------+
                                        |  HTTP POST / JSON
                                        v
+-------------------------------------------------------------------------------+
|                            Flask Application (app.py)                         |
|   - Input Validation (range & type verification)                              |
|   - In-memory dataset caching                                                 |
|   - REST API endpoint (/api/predict)                                          |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                         Inference Engine (ml/predict.py)                      |
+-------------------+-------------------+-------------------+-------------------+
|                   |                   |                   |                   |
| 1. SOH Regression | 2. Derived RUL    | 3. Health State   | 4. Advisory Logic |
|    Gradient       |    Prognostics    |    Categorization |    Thermal First  |
|    Boosting       |    (EOL @ 80%)    |    (Thresholds)   |    Rule Engine    |
|   (soh_model.pkl) | (rul_calculator)  |  (battery_health) |  (recommendation) |
+---------+---------+---------+---------+---------+---------+---------+---------+
          |                   |                   |                   |
          +-------------------+---------+---------+-------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                             Diagnostic Output                                 |
|  - Predicted SOH (%)                - Estimated RUL (cycles)                  |
|  - Health Category (Badge)          - Maintenance Advisory                    |
|  - Dynamic Degradation Chart with Current Evaluated Operating Point           |
+-------------------------------------------------------------------------------+
```

---

## 4. Dataset & Features

The active dataset (`dataset/new_battery_dataset.csv`) contains **2,000 cycle records** across multiple battery batches.

### Feature Definition

| Feature Name | Description | Range in Dataset | Typical Units |
| :--- | :--- | :--- | :--- |
| `Cycle` | Battery operational cycle count | $1$ to $2,000$ | Cycles |
| `Voltage` | Terminal cell voltage | $3.00$ to $4.20$ | $\text{V}$ |
| `Current` | Operating current | $0.50$ to $2.00$ | $\text{A}$ |
| `Temperature` | Cell operating temperature | $9.90$ to $40.97$ | $^\circ\text{C}$ |
| `ChargeTime` | Cycle charging duration | $30.00$ to $119.88$ | Step units |
| `DischargeTime` | Cycle discharging duration | $30.02$ to $119.97$ | Step units |
| `InternalResistance` | Equivalent series internal resistance | $0.0465$ to $0.2519$ | $\Omega$ |
| `Capacity` | Measurable discharge capacity | $1.460$ to $2.521$ | $\text{Ah}$ |
| `AmbientHumidity` | Ambient relative humidity | $30.03$ to $69.97$ | $\%$ |
| `C_Rate` | Current charge/discharge rate | $0.501$ to $1.999$ | $\text{C}$ |
| **`SOH` (Target)** | **Battery State of Health** | **$27.79$ to $100.00$** | **$\%$** |

---

## 5. SOH Modeling & Regression Methodology

The primary regression model is a **Gradient Boosting Regressor** (`sklearn.ensemble.GradientBoostingRegressor`) trained to predict continuous SOH:

```python
GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
```

The model is serialized and deployed as `models/soh_model.pkl`.

---

## 6. Experimental Evaluation & Validation

To maintain rigorous scientific honesty, the model was evaluated under two distinct data partitioning protocols:

### Protocol A: Random Train-Test Split (Interpolation)
- **Configuration:** 80% Train ($1,600$ samples), 20% Test ($400$ samples), shuffled randomly (`random_state=42`).
- **Results:**
  - **Mean Absolute Error ($\text{MAE}$):** $0.8007\%$
  - **Root Mean Squared Error ($\text{RMSE}$):** $1.0255\%$
  - **Coefficient of Determination ($R^2$):** $\mathbf{0.9971}$

*Interpretation:* Under random splitting, test cycles are interleaved throughout the historical cycle range ($1$ to $2,000$). The model exhibits high interpolative precision across the known domain.

### Protocol B: Chronological / Cycle-Based Split (Extrapolation Validation)
- **Configuration:** Train on Cycles $1$ to $1,600$ ($1,600$ samples), Test on unseen future Cycles $1,601$ to $2,000$ ($400$ samples).
- **Results:**
  - **Mean Absolute Error ($\text{MAE}$):** $6.7418\%$
  - **Root Mean Squared Error ($\text{RMSE}$):** $7.6645\%$
  - **Coefficient of Determination ($R^2$):** $\mathbf{-3.4854}$

*Scientific Finding & Limitation:* Tree-based ensemble models (like Gradient Boosting) partition feature space using orthogonal decision boundaries and predict constant leaf values outside the convex hull of training features. When evaluating future cycles ($>1600$), the regressor cannot extrapolate degradation beyond the lowest leaf value encountered in training, demonstrating that tree models require periodic recalibration or hybrid physics-informed formulations for long-term extrapolation.

---

## 7. Derived RUL Prognostics Methodology

> [!NOTE]
> $\text{RUL}$ in EVGuardian-AI is **not an independently trained ML model**. It is a **model-derived prognostic estimation** based on the calibrated global SOH degradation model and the standard End-of-Life ($\text{EOL}$) boundary.

In battery engineering literature (e.g., ISO/IEC standards, NASA PCoE prognostics), End-of-Life ($\text{EOL}$) is reached when capacity degrades to **$80\%$ of nominal capacity ($\text{SOH} = 80\%$)**.

### Formulation & Calibration
1. **Calibrated SOH Degradation Trendline:**
   $$\text{SOH}(\text{Cycle}) = 93.5244 - 0.032399 \times \text{Cycle}$$
2. **Empirical EOL Cycle Boundary (at $\text{SOH} = 80.0\%$):**
   $$\text{EOL\_cycle} = \frac{93.5244 - 80.0}{0.032399} = \mathbf{417.4326 \text{ cycles}} \approx \mathbf{417.43 \text{ cycles}}$$
3. **RUL Calculation Rule:**
   $$\text{RUL} = \begin{cases} 0 & \text{if } \text{SOH} \le 80.0\% \\ \max(0, \text{round}(\text{EOL\_cycle} - \text{current\_cycle})) & \text{if } \text{SOH} > 80.0\% \end{cases}$$

### Benchmark Verification
- **$\text{Cycle} = 10$ ($\text{SOH} \ge 90\%$):** $\text{RUL} = \max(0, \text{round}(417.43 - 10)) = \mathbf{407 \text{ cycles}}$
- **$\text{Cycle} = 200$ ($80\% \le \text{SOH} < 90\%$):** $\text{RUL} = \max(0, \text{round}(417.43 - 200)) = \mathbf{217 \text{ cycles}}$
- **$\text{Cycle} = 250$ ($80\% \le \text{SOH} < 90\%$):** $\text{RUL} = \max(0, \text{round}(417.43 - 250)) = \mathbf{167 \text{ cycles}}$
- **$\text{Cycle} = 600$ ($\text{SOH} < 80\%$):** $\text{RUL} = \mathbf{0 \text{ cycles}}$


---

## 8. Battery Health Classification

Battery operational condition is mapped deterministically from predicted $\text{SOH}$:

$$\text{Health Status} = \begin{cases} \textbf{Healthy} & \text{if } \text{SOH} \ge 90\% \\ \textbf{Moderate} & \text{if } 80\% \le \text{SOH} < 90\% \\ \textbf{Poor (EOL)} & \text{if } \text{SOH} < 80\% \end{cases}$$

---

## 9. Advisory Recommendation Engine

The recommendation engine executes a strict priority hierarchy to ensure user safety and prevent conflicting diagnostics:

1. **Thermal Safety Warning (Priority 1):** Triggered if cell temperature exceeds $45^\circ\text{C}$ regardless of $\text{SOH}$.
2. **Healthy Advisory ($\text{SOH} \ge 90\%$):** Normal operation confirmed with estimated cycles remaining to $\text{EOL}$.
3. **Moderate Advisory ($80\% \le \text{SOH} < 90\%$):** Preventive maintenance recommended; cell degradation flagged for periodic inspection.
4. **End-of-Life Advisory ($\text{SOH} < 80\%$):** Battery replacement recommended to maintain vehicle range and reliability.

---

## 10. Web Dashboard & REST API

### Web Dashboard
- **Glassmorphism UI:** Modern dark theme built with responsive vanilla CSS.
- **Dynamic SOH Circular Gauge:** Visualized via CSS conic gradients.
- **Interactive Degradation Chart:** Chart.js dual-dataset chart plotting both the historical degradation profile and the user's evaluated `(Cycle, SOH)` state point.
- **Pre-loaded AI Demo:** One-click demo populated with in-distribution parameters from Cycle 200.

### REST JSON API
- **Endpoint:** `POST /api/predict`
- **Request Format:**
  ```json
  {
    "Cycle": 200,
    "Voltage": 3.70,
    "Current": 1.00,
    "Temperature": 30.0,
    "ChargeTime": 75.0,
    "DischargeTime": 75.0,
    "InternalResistance": 0.052,
    "Capacity": 2.48,
    "AmbientHumidity": 50.0,
    "C_Rate": 1.00
  }
  ```
- **Response Format:**
  ```json
  {
    "success": true,
    "inputs": { ... },
    "prediction": {
      "SOH": 88.35,
      "RUL": 143,
      "Health": "Moderate",
      "Recommendation": "Battery health is moderate (88.4%). Schedule preventive maintenance and monitor cell degradation (~143 cycles remaining to EOL)."
    }
  }
  ```

---

## 11. Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.11 / 3.12 / 3.13)
- `pip` package manager

### Installation
```bash
# Clone the repository
git clone https://github.com/elyyx31/EVGuardian-AI.git
cd EVGuardian-AI

# Create and activate a virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 12. How to Run

### Run the Flask Application
```bash
python app.py
```
Open your browser and navigate to: `http://127.0.0.1:5000`

### Run Model Evaluation & Scripts
```bash
# Run SOH regression training & evaluation
python ml/soh_regression.py

# Run temporal / cycle validation experiment
python ml/soh_validation.py

# Run standalone inference test
python ml/predict.py
```

---

## 13. Repository Structure

```
EVGuardian-AI/
├── app.py                      # Flask backend & REST API server
├── requirements.txt            # Pinned project dependencies
├── README.md                   # Project documentation & research report
├── dataset/
│   └── new_battery_dataset.csv # Primary 2,000-cycle battery dataset
├── ml/
│   ├── battery_health.py       # SOH health category mapper
│   ├── hyperparameter_tuning.py# Gradient Boosting classification tuning
│   ├── predict.py              # Central inference module
│   ├── recommendation.py       # Advisory recommendation engine
│   ├── rul_calculator.py       # Model-derived EOL RUL calculator
│   ├── soh_regression.py       # SOH Gradient Boosting training pipeline
│   ├── soh_validation.py       # Temporal / cycle-based validation script
│   └── xgboost_comparison.py   # XGBoost benchmark script
├── models/
│   └── soh_model.pkl           # Pre-trained Gradient Boosting Regressor
├── static/
│   └── style.css               # Dashboard styling
└── templates/
    └── dashboard.html          # Interactive Jinja2 web interface
```

---

## 14. Scientific Limitations & Future Work

### Current Limitations
1. **Extrapolation Constraints:** As demonstrated in the temporal validation experiment ($R^2 = -3.4854$), tree-based regressor models cannot extrapolate degradation trends beyond the cycle bounds of their training set.
2. **Derived RUL Simplification:** RUL assumes monotonic linear degradation from the current operating state. Real electrochemical cells may experience accelerated degradation (knees) near EOL.
3. **Synthetic Dataset Features:** Voltage, Current, Temperature, and Humidity in the current dataset have lower statistical variance relative to SOH than observed in raw hardware cyclers.

### Planned Future Work
- Integrate physics-informed battery degradation models (e.g., Equivalent Circuit Models coupled with Kalman Filters).
- Benchmark sequence architectures (LSTM, Transformer, Neural ODEs) for multi-step future degradation forecasting.
- Implement streaming MQTT / CAN-bus ingestion for real-time BMS telematics.
