"""Generates synthetic datasets A-H for testing AutoDQ's detection accuracy,
plus a polished demo dataset (customer churn) with documented injected issues.

Run: python generate_datasets.py
Outputs CSVs into this directory.
"""
import os

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
OUT = os.path.dirname(os.path.abspath(__file__))


def base_dataset(n=1000):
    x1 = rng.normal(50, 10, n)
    x2 = rng.normal(20, 5, n)
    x3 = rng.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2])
    noise = rng.normal(0, 1, n)
    score = 0.3 * x1 + 0.5 * x2 + noise
    target = (score > np.median(score)).astype(int)
    return pd.DataFrame({"feature_1": x1, "feature_2": x2, "category": x3, "target": target})


def dataset_a_clean():
    """A: High-quality, balanced dataset -- no injected issues."""
    return base_dataset(1000)


def dataset_b_missing():
    """B: Missing values injected into two columns."""
    df = base_dataset(1000)
    idx1 = rng.choice(df.index, size=int(0.15 * len(df)), replace=False)
    idx2 = rng.choice(df.index, size=int(0.30 * len(df)), replace=False)
    df.loc[idx1, "feature_1"] = np.nan
    df.loc[idx2, "category"] = np.nan
    return df


def dataset_c_imbalance():
    """C: Heavy class imbalance (~1:9)."""
    df = base_dataset(1000)
    majority = df[df["target"] == 0]
    minority = df[df["target"] == 1].sample(frac=0.11, random_state=1)
    return pd.concat([majority, minority], ignore_index=True)


def dataset_d_duplicates():
    """D: Duplicate rows injected."""
    df = base_dataset(900)
    dupes = df.sample(frac=0.12, random_state=2)
    return pd.concat([df, dupes], ignore_index=True)


def dataset_e_outliers():
    """E: Statistical outliers injected into feature_1."""
    df = base_dataset(1000)
    idx = rng.choice(df.index, size=int(0.05 * len(df)), replace=False)
    df.loc[idx, "feature_1"] = df.loc[idx, "feature_1"] * rng.choice([8, -8], size=len(idx))
    return df


def dataset_f_redundant():
    """F: A near-duplicate feature (correlation ~0.98) with feature_1."""
    df = base_dataset(1000)
    df["feature_1_copy"] = df["feature_1"] * 1.02 + rng.normal(0, 0.3, len(df))
    return df


def dataset_g_leakage():
    """G: A feature derived directly from the target (leakage)."""
    df = base_dataset(1000)
    df["target_echo"] = df["target"].map({0: 0.01, 1: 0.99}) + rng.normal(0, 0.02, len(df))
    return df


def dataset_h_combined():
    """H: Combination of multiple problems -- missing values, imbalance,
    duplicates, outliers, redundancy, and leakage all in one dataset."""
    df = base_dataset(1000)
    majority = df[df["target"] == 0]
    minority = df[df["target"] == 1].sample(frac=0.15, random_state=3)
    df = pd.concat([majority, minority], ignore_index=True)

    idx_missing = rng.choice(df.index, size=int(0.10 * len(df)), replace=False)
    df.loc[idx_missing, "feature_1"] = np.nan

    idx_out = rng.choice(df.index, size=int(0.03 * len(df)), replace=False)
    df.loc[idx_out, "feature_2"] = df.loc[idx_out, "feature_2"] * 6

    df["feature_1_copy"] = df["feature_1"] * 1.01 + rng.normal(0, 0.2, len(df))
    df["target_echo"] = df["target"].map({0: 0.02, 1: 0.98}) + rng.normal(0, 0.03, len(df))

    dupes = df.sample(frac=0.08, random_state=4)
    df = pd.concat([df, dupes], ignore_index=True)
    return df


def demo_churn_dataset(n=2000):
    """Polished demo dataset: customer churn, with a controlled, documented
    set of injected quality problems for the live demonstration."""
    customer_id = np.arange(100000, 100000 + n)
    tenure_months = rng.integers(1, 72, n)
    monthly_charges = np.round(rng.normal(65, 20, n).clip(10, 150), 2)
    total_charges = np.round(monthly_charges * tenure_months + rng.normal(0, 50, n), 2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    support_calls = rng.poisson(2, n)
    age = rng.integers(18, 80, n)

    # Churn probability driven by tenure, contract type, and support calls
    contract_risk = pd.Series(contract).map({"Month-to-month": 0.45, "One year": 0.15, "Two year": 0.05}).to_numpy()
    churn_prob = (contract_risk + (support_calls > 4) * 0.2 + (tenure_months < 6) * 0.2).clip(0, 0.9)
    churn = (rng.random(n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customer_id": customer_id,
        "age": age,
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract,
        "support_calls": support_calls,
        "churn": churn,
    })

    # --- Intentionally injected quality problems (documented) ---
    # 1. Missing values in monthly_charges (~8%) and contract_type (~5%)
    idx = rng.choice(df.index, size=int(0.08 * n), replace=False)
    df.loc[idx, "monthly_charges"] = np.nan
    idx2 = rng.choice(df.index, size=int(0.05 * n), replace=False)
    df.loc[idx2, "contract_type"] = np.nan

    # 2. ~6% duplicate rows
    dupes = df.sample(frac=0.06, random_state=5)
    df = pd.concat([df, dupes], ignore_index=True)

    # 3. A handful of extreme age outliers (data entry errors)
    idx3 = rng.choice(df.index, size=8, replace=False)
    df.loc[idx3, "age"] = rng.choice([150, -5, 999], size=8)

    # 4. A highly correlated redundant feature
    df["estimated_annual_value"] = df["monthly_charges"] * 12 + rng.normal(0, 5, len(df))

    # 5. A leakage-like feature: a "retention_flag" that's essentially the inverse of churn
    df["retention_flag"] = 1 - df["churn"]

    df = df.sample(frac=1.0, random_state=6).reset_index(drop=True)
    return df


if __name__ == "__main__":
    datasets = {
        "A_clean.csv": dataset_a_clean(),
        "B_missing_values.csv": dataset_b_missing(),
        "C_class_imbalance.csv": dataset_c_imbalance(),
        "D_duplicates.csv": dataset_d_duplicates(),
        "E_outliers.csv": dataset_e_outliers(),
        "F_redundant_features.csv": dataset_f_redundant(),
        "G_leakage.csv": dataset_g_leakage(),
        "H_combined_problems.csv": dataset_h_combined(),
        "demo_customer_churn.csv": demo_churn_dataset(),
    }
    for name, df in datasets.items():
        df.to_csv(os.path.join(OUT, name), index=False)
        print(f"Wrote {name}: {df.shape[0]} rows x {df.shape[1]} cols")
