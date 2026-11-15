# Punjab Mandi Flask Dashboard

## Files
- `app.py` - Flask app, loads your existing cleaning, feature engineering and saved pipeline.
- `templates/index.html` - dashboard UI and forms.
- `static/style.css` - dashboard styling.

## Put these files in your existing project root
Your structure should look like:

punjab_mandi_price_forecasting/
│
├── data/
│   ├── proceed/
│   │   └── clean_data.csv
│   └── raw/
│       └── bathinda_crops_mandi_data.csv
│
├── models/
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── sgd_model.pkl
│
├── Notebooks/
├── src/
│   ├── data_cleaning.py
│   ├── data_preprocessing.py
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── forecast.py
│   └── train.py
│
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── app.py
├── README.txt
├── requirements.txt
└── .gitignore

If your Python modules are directly in the project root instead of `src/`, move them into `src/` or adjust the import path in `app.py`.

## Install and run
```bash
pip install flask pandas numpy scikit-learn joblib
python app.py
```
Open `http://127.0.0.1:5000` in your browser.

## Important
1. `models/model.pkl` must be the **fitted sklearn Pipeline**, not an unfitted `ColumnTransformer`.
2. The pipeline must accept raw feature columns and perform preprocessing internally.
3. Feature names must match your training code exactly. This app uses `lag_30` (underscore), matching the feature name in your latest forecasting code.
4. The forecast form uses only rows on or before `last_data_date`; it does not leak later records into the forecast history.
5. The single-prediction form forecasts a future date after the latest available date for that commodity-market pair.
6. The displayed evaluation metrics are the values you supplied. They are not recalculated by the dashboard.

Author:- **Arman Ali**
Aspiring Data Scientists
Mob:- 7050762210
Gmail:- armanarbaj179@gmail.com
