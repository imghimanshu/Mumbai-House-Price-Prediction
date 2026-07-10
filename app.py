"""
Mumbai House Price Estimator — FREE NUMERIC INPUT VERSION
-----------------------------------------------------------
- Categorical fields (Locality, City Region, Property Type, Furnishing)
  stay as dropdowns — these MUST match the model's trained categories.
- Numeric fields (Area, BHK, Bathrooms, Balconies, Property Age) are now
  plain number input boxes — the user can type ANY numeric value, not
  just the preset list.
- Server-side validation makes sure whatever the user types is actually
  a valid number and within a sane range before it reaches the model.

Folder needed:
    mumbai_final/
    ├── app.py                          <- this file
    ├── mumbai_house_price_model.pkl    <- your trained model
    └── model_columns.json              <- your saved column list

Run with:
    pip install flask scikit-learn pandas numpy joblib
    python app.py
Then open your browser and type: localhost:5000
"""

from flask import Flask, request
import joblib
import json
import numpy as np
import pandas as pd

app = Flask(__name__)

model = joblib.load("mumbai_house_price_model.pkl")
with open("model_columns.json") as f:
    MODEL_COLUMNS = json.load(f)


def options_for(prefix):
    return sorted([c[len(prefix):] for c in MODEL_COLUMNS if c.startswith(prefix)])


LOCALITY_OPTIONS = options_for("locality_")
CITY_OPTIONS = options_for("city_")
PROPERTY_TYPE_OPTIONS = options_for("property_type_")
FURNISHED_OPTIONS = options_for("furnished_")

DEFAULT_LATITUDE = 19.0760
DEFAULT_LONGITUDE = 72.8777

# Sane server-side bounds for the free-typed numeric fields.
# (min, max, field label — used for validation error messages)
NUMERIC_BOUNDS = {
    "area": (100, 20000, "Area (sqft)"),
    "bedroom_num": (1, 20, "BHK (Bedrooms)"),
    "bathroom_num": (1, 20, "Bathrooms"),
    "balcony_num": (0, 10, "Balconies"),
    "age": (0, 100, "Property Age (yrs)"),
}


def validate_numeric_fields(form):
    """Parse and validate the free-typed numeric inputs.
    Returns (values_dict, errors_list)."""
    values = {}
    errors = []

    for field, (lo, hi, label) in NUMERIC_BOUNDS.items():
        raw = (form.get(field) or "").strip()
        if raw == "":
            errors.append(f"{label} is required.")
            continue
        try:
            num = float(raw)
        except ValueError:
            errors.append(f"{label} must be a number (you entered '{raw}').")
            continue
        if num < lo or num > hi:
            errors.append(f"{label} should be between {lo} and {hi} (you entered {raw}).")
            continue
        values[field] = num

    return values, errors


def build_feature_vector(form, numeric_values):
    area = numeric_values["area"]
    bedroom_num = numeric_values["bedroom_num"]
    bathroom_num = numeric_values["bathroom_num"]
    balcony_num = numeric_values["balcony_num"]
    age = numeric_values["age"]
    locality = form["locality"]
    city = form["city"]
    property_type = form["property_type"]
    furnished = form["furnished"]

    area_per_bedroom = area / bedroom_num if bedroom_num > 0 else area

    row = {col: 0 for col in MODEL_COLUMNS}

    field_values = {
        "area": area,
        "bedroom_num": bedroom_num,
        "bathroom_num": bathroom_num,
        "balcony_num": balcony_num,
        "age": age,
        "latitude": DEFAULT_LATITUDE,
        "longitude": DEFAULT_LONGITUDE,
        "area_per_bedroom": area_per_bedroom,
    }
    for key, val in field_values.items():
        if key in row:
            row[key] = val

    for prefix, chosen in [
        ("locality_", locality),
        ("city_", city),
        ("property_type_", property_type),
        ("furnished_", furnished),
    ]:
        col_name = f"{prefix}{chosen}"
        if col_name in row:
            row[col_name] = 1

    return pd.DataFrame([row], columns=MODEL_COLUMNS)


def option_tags(values, selected=None):
    tags = ""
    for v in values:
        sel = " selected" if str(v) == str(selected) else ""
        tags += f'<option value="{v}"{sel}>{v}</option>\n'
    return tags


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mumbai House Price Estimator</title>
<style>
  :root {{
    --ink: #0F1E2E;
    --panel: #142B40;
    --gold: #C8A24A;
    --gold-bright: #E4C878;
    --paper: #F3EFE6;
    --muted: #8FA3B5;
    --error: #C2544A;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: Georgia, 'Times New Roman', serif;
    background: linear-gradient(180deg, var(--ink) 0%, #0A1520 100%);
    color: var(--paper);
    min-height: 100vh;
    display: flex;
    justify-content: center;
    padding: 48px 20px;
  }}
  .wrap {{ max-width: 560px; width: 100%; }}
  .eyebrow {{
    font-family: Arial, sans-serif;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    font-size: 11px;
    color: var(--gold);
    margin-bottom: 10px;
  }}
  h1 {{
    font-size: 2.2rem;
    margin: 0 0 24px 0;
    font-weight: 400;
    border-bottom: 1px solid rgba(200,162,74,0.3);
    padding-bottom: 18px;
  }}
  form {{
    background: var(--panel);
    border: 1px solid rgba(200,162,74,0.25);
    padding: 30px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px;
  }}
  .field {{ display: flex; flex-direction: column; }}
  label {{
    font-family: Arial, sans-serif;
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 6px;
  }}
  select, input[type="number"] {{
    font-family: Arial, sans-serif;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(200,162,74,0.3);
    color: var(--paper);
    padding: 10px 12px;
    font-size: 0.9rem;
  }}
  select {{ cursor: pointer; }}
  select option {{ color: #000; }}
  input[type="number"]::placeholder {{ color: var(--muted); }}
  button {{
    grid-column: 1 / -1;
    font-family: Arial, sans-serif;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-size: 0.85rem;
    background: var(--gold);
    color: var(--ink);
    border: none;
    padding: 15px;
    margin-top: 6px;
    cursor: pointer;
  }}
  button:hover {{ background: var(--gold-bright); }}
  .result {{
    margin-top: 24px;
    padding: 24px 28px;
    background: rgba(200,162,74,0.08);
    border: 1px solid var(--gold);
    text-align: center;
  }}
  .result .label {{
    font-family: Arial, sans-serif;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-size: 11px;
    color: var(--gold);
  }}
  .result .value {{ font-size: 2.2rem; margin-top: 6px; }}
  .error-box {{
    margin-top: 24px;
    padding: 16px 20px;
    background: rgba(194,84,74,0.1);
    border: 1px solid var(--error);
    font-family: Arial, sans-serif;
    font-size: 0.85rem;
    color: #E8A39C;
  }}
  .error-box ul {{ margin: 6px 0 0 0; padding-left: 18px; }}
  @media (max-width: 600px) {{
    form {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<div class="wrap">
  <div class="eyebrow">Mumbai Real Estate</div>
  <h1>House Price Estimator</h1>

  <form method="POST">
    <div class="field">
      <label>Place / Locality</label>
      <select name="locality" required>
        {locality_options}
      </select>
    </div>

    <div class="field">
      <label>City Region</label>
      <select name="city" required>
        {city_options}
      </select>
    </div>

    <div class="field">
      <label>Property Type</label>
      <select name="property_type" required>
        {property_type_options}
      </select>
    </div>

    <div class="field">
      <label>Furnishing</label>
      <select name="furnished" required>
        {furnished_options}
      </select>
    </div>

    <div class="field">
      <label>Area (sqft)</label>
      <input type="number" name="area" min="100" max="20000" step="any"
             placeholder="e.g. 850" value="{area_val}" required>
    </div>

    <div class="field">
      <label>BHK (Bedrooms)</label>
      <input type="number" name="bedroom_num" min="1" max="20" step="1"
             placeholder="e.g. 2" value="{bedroom_val}" required>
    </div>

    <div class="field">
      <label>Bathrooms</label>
      <input type="number" name="bathroom_num" min="1" max="20" step="1"
             placeholder="e.g. 2" value="{bathroom_val}" required>
    </div>

    <div class="field">
      <label>Balconies</label>
      <input type="number" name="balcony_num" min="0" max="10" step="1"
             placeholder="e.g. 1" value="{balcony_val}" required>
    </div>

    <div class="field">
      <label>Property Age (yrs)</label>
      <input type="number" name="age" min="0" max="100" step="any"
             placeholder="e.g. 5" value="{age_val}" required>
    </div>

    <button type="submit">Calculate Estimate Price</button>
  </form>

  {result_html}
  {error_html}
</div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    result_html = ""
    error_html = ""
    field_vals = {"area": "", "bedroom_num": "", "bathroom_num": "", "balcony_num": "", "age": ""}

    if request.method == "POST":
        for key in field_vals:
            field_vals[key] = request.form.get(key, "")

        numeric_values, errors = validate_numeric_fields(request.form)

        for cat_field, label in [("locality", "Place / Locality"), ("city", "City Region"),
                                  ("property_type", "Property Type"), ("furnished", "Furnishing")]:
            if not request.form.get(cat_field):
                errors.append(f"{label} is required.")

        if not errors:
            try:
                X = build_feature_vector(request.form, numeric_values)
                log_price_pred = model.predict(X)[0]
                price_pred = np.expm1(log_price_pred)
                result_html = f"""
                <div class="result">
                  <div class="label">Estimated Price</div>
                  <div class="value">\u20B9{price_pred:,.0f}</div>
                </div>
                """
            except Exception as e:
                errors.append(f"Prediction failed: {e}")

        if errors:
            items = "".join(f"<li>{e}</li>" for e in errors)
            error_html = f'<div class="error-box"><strong>Please fix the following:</strong><ul>{items}</ul></div>'

    page = PAGE_TEMPLATE.format(
        locality_options=option_tags(LOCALITY_OPTIONS, request.form.get("locality")),
        city_options=option_tags(CITY_OPTIONS, request.form.get("city")),
        property_type_options=option_tags(PROPERTY_TYPE_OPTIONS, request.form.get("property_type")),
        furnished_options=option_tags(FURNISHED_OPTIONS, request.form.get("furnished")),
        area_val=field_vals["area"],
        bedroom_val=field_vals["bedroom_num"],
        bathroom_val=field_vals["bathroom_num"],
        balcony_val=field_vals["balcony_num"],
        age_val=field_vals["age"],
        result_html=result_html,
        error_html=error_html,
    )
    return page


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
