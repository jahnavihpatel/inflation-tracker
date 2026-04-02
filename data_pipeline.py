"""
Inflation Tracker — Data Pipeline
Fetches CPI item prices from BLS public API (no key needed for basic access)
and saves to data/prices.csv
"""

import requests
import pandas as pd
import time
import os

os.makedirs("data", exist_ok=True)

# 70 BLS series IDs mapped to readable item names + category
# Series format: APU0000XXXXXX = Average Price, US city average
BLS_SERIES = {
    # Food & Beverages
    "APU0000708111": ("Eggs (dozen)", "Food & Beverages", "per dozen"),
    "APU0000709112": ("Milk (whole, gallon)", "Food & Beverages", "per gallon"),
    "APU0000703112": ("Ground beef (lb)", "Food & Beverages", "per lb"),
    "APU0000FC1101": ("Chicken breast (lb)", "Food & Beverages", "per lb"),
    "APU0000702111": ("Bread, white (loaf)", "Food & Beverages", "per loaf"),
    "APU0000711211": ("Apples (lb)", "Food & Beverages", "per lb"),
    "APU0000711311": ("Bananas (lb)", "Food & Beverages", "per lb"),
    "APU0000712311": ("Tomatoes (lb)", "Food & Beverages", "per lb"),
    "APU0000715211": ("Sugar (5 lb bag)", "Food & Beverages", "per 5 lb bag"),
    "APU0000717311": ("Coffee, ground (13 oz)", "Food & Beverages", "per 13 oz"),
    "APU0000710411": ("Butter (lb)", "Food & Beverages", "per lb"),
    "APU0000710212": ("Cheddar cheese (lb)", "Food & Beverages", "per lb"),
    "APU0000706111": ("Pork chops (lb)", "Food & Beverages", "per lb"),
    "APU0000704111": ("Bacon (lb)", "Food & Beverages", "per lb"),
    "APU0000701111": ("White rice (lb)", "Food & Beverages", "per lb"),
    "APU0000714221": ("Orange juice, frozen (16 oz)", "Food & Beverages", "per 16 oz"),
    "APU0000718311": ("Cola, 2-liter", "Food & Beverages", "per 2 liter"),
    "APU0000720111": ("Beer, at home (6 pack)", "Food & Beverages", "per 6 pack"),
    "APU0000FJ1101": ("Lettuce (head)", "Food & Beverages", "per head"),
    "APU0000712112": ("Potatoes (lb)", "Food & Beverages", "per lb"),

    # Energy
    "APU000074714" : ("Gasoline, regular (gallon)", "Energy", "per gallon"),
    "APU000072610" : ("Electricity (kWh)", "Energy", "per kWh"),
    "APU000072620" : ("Natural gas (therm)", "Energy", "per therm"),
    "APU000074715" : ("Gasoline, midgrade (gallon)", "Energy", "per gallon"),
    "APU000074716" : ("Gasoline, premium (gallon)", "Energy", "per gallon"),
    "APU000072631" : ("Heating oil (gallon)", "Energy", "per gallon"),
    "APU000074724" : ("Propane (gallon)", "Energy", "per gallon"),

    # Housing
    "CUSR0000SAH"  : ("Rent of primary residence", "Housing", "monthly index"),
    "CUSR0000SEHC" : ("Water & sewer utilities", "Housing", "monthly index"),
    "CUSR0000SEHD" : ("Garbage collection", "Housing", "monthly index"),
    "CUSR0000SEHA" : ("Owners' equiv rent", "Housing", "monthly index"),
    "CUSR0000SEHF" : ("Household furnishings", "Housing", "monthly index"),

    # Transportation
    "CUSR0000SETA01": ("New cars", "Transportation", "monthly index"),
    "CUSR0000SETA02": ("Used cars & trucks", "Transportation", "monthly index"),
    "CUSR0000SETD"  : ("Airline fares", "Transportation", "monthly index"),
    "CUSR0000SETB01": ("Auto insurance", "Transportation", "monthly index"),
    "CUSR0000SETC"  : ("Vehicle maintenance & repair", "Transportation", "monthly index"),

    # Healthcare
    "CUSR0000SAM"  : ("Medical care (overall)", "Healthcare", "monthly index"),
    "CUSR0000SAM1" : ("Prescription drugs", "Healthcare", "monthly index"),
    "CUSR0000SEMD" : ("Hospital services", "Healthcare", "monthly index"),
    "CUSR0000SEMC" : ("Dental services", "Healthcare", "monthly index"),
    "CUSR0000SEMB" : ("Physician services", "Healthcare", "monthly index"),

    # Education & Communication
    "CUSR0000SAE"  : ("Education & communication", "Education", "monthly index"),
    "CUSR0000SEEA" : ("College tuition", "Education", "monthly index"),
    "CUSR0000SEED" : ("Childcare & nursery school", "Education", "monthly index"),
    "CUSR0000SETE" : ("Internet services", "Education", "monthly index"),
    "CUSR0000SESD" : ("Telephone services", "Education", "monthly index"),

    # Apparel
    "CUSR0000SAA"  : ("Apparel (overall)", "Apparel", "monthly index"),
    "CUSR0000SAA1" : ("Men's apparel", "Apparel", "monthly index"),
    "CUSR0000SAA2" : ("Women's apparel", "Apparel", "monthly index"),
    "CUSR0000SEAE" : ("Footwear", "Apparel", "monthly index"),

    # Recreation
    "CUSR0000SAR"  : ("Recreation (overall)", "Recreation", "monthly index"),
    "CUSR0000SERA" : ("Televisions", "Recreation", "monthly index"),
    "CUSR0000SERC" : ("Pets & pet products", "Recreation", "monthly index"),
    "CUSR0000SERD" : ("Sporting goods", "Recreation", "monthly index"),

    # Food Away From Home
    "CUSR0000SEFV" : ("Full-service restaurants", "Food Away From Home", "monthly index"),
    "CUSR0000SEFX" : ("Fast food restaurants", "Food Away From Home", "monthly index"),
    "CUSR0000SEFS" : ("Food at employee sites", "Food Away From Home", "monthly index"),

    # Other
    "CUSR0000SABO" : ("Tobacco products", "Other", "monthly index"),
    "CUSR0000SAOL" : ("Personal care products", "Other", "monthly index"),
    "CUSR0000SAOS" : ("Personal care services", "Other", "monthly index"),
    "CUSR0000SEFW" : ("Alcoholic beverages away from home", "Other", "monthly index"),
}


def fetch_bls_data(series_ids: list, start_year: str = "2015", end_year: str = "2025") -> pd.DataFrame:
    """Fetch data from BLS public API v2 (no API key required for ≤25 series at a time)."""
    all_rows = []
    # BLS limits to 25 series per call without a key
    chunks = [series_ids[i:i+25] for i in range(0, len(series_ids), 25)]

    for chunk in chunks:
        payload = {
            "seriesid": chunk,
            "startyear": start_year,
            "endyear": end_year,
        }
        try:
            resp = requests.post(
                "https://api.bls.gov/publicAPI/v2/timeseries/data/",
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()

            if result.get("status") != "REQUEST_SUCCEEDED":
                print(f"  BLS warning: {result.get('message', 'unknown error')}")
                continue

            for series in result.get("Results", {}).get("series", []):
                sid = series["seriesID"]
                meta = BLS_SERIES.get(sid, (sid, "Unknown", ""))
                name, category, unit = meta
                for dp in series.get("data", []):
                    try:
                        all_rows.append({
                            "series_id": sid,
                            "item": name,
                            "category": category,
                            "unit": unit,
                            "year": int(dp["year"]),
                            "month": int(dp["period"].replace("M", "")),
                            "value": float(dp["value"]),
                        })
                    except (ValueError, KeyError):
                        continue
        except requests.RequestException as e:
            print(f"  Request failed for chunk: {e}")

        time.sleep(1)  # Be polite to the API

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    df = df.sort_values(["item", "date"]).reset_index(drop=True)
    return df


def run():
    print("Fetching CPI data from BLS...")
    series_ids = list(BLS_SERIES.keys())
    df = fetch_bls_data(series_ids)

    if df.empty:
        print("No data returned — check your internet connection or BLS API status.")
        return

    df.to_csv("data/prices.csv", index=False)
    print(f"Saved {len(df)} rows across {df['item'].nunique()} items → data/prices.csv")
    print(df.groupby("category")["item"].nunique().to_string())


if __name__ == "__main__":
    run()
