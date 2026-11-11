"""Preprocesamiento de los datos REALES (E-commerce_Customer_Segmentation_2026.csv, 50,000 clientes x 53 variables).

Objetivo: abandono = activity_status == "Inactive" (90 o más días sin comprar).

PREVENCIÓN DE FUGA DE INFORMACIÓN: en este archivo activity_status es una regla fija de
days_since_last_purchase (Active 0-29, At Risk 30-59, Dormant 60-89, Inactive >= 90) y purchase_frequency,
recency_score, rfm_*, churn_risk_*, customer_health_score y las categorías derivadas se calculan con ella.
Por eso NO entran como variables del modelo (ver src/models/control_fuga.py, que lo demuestra).
Tampoco entran variables demográficas (edad, género, país...), por no discriminar (Informe de diseño, 3.1).

Uso:  python src/data/preprocess_real.py --raw data/real/ecommerce_real.csv \
          --out data/processed/dataset_real_limpio.csv --models models/real/
"""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
CATEGORICAS = {"shopping_channel": "canal", "device_used": "dispositivo", "loyalty_tier": "lealtad"}


def construir(df: pd.DataFrame) -> pd.DataFrame:
    n = df["total_purchases"].clip(lower=1)
    out = pd.DataFrame({
        "id_cliente": df["customer_id"].map(lambda v: hashlib.sha256(("sprvc-2026" + v).encode()).hexdigest()[:12]),
        "antiguedad_meses": df["tenure_months"],
        "num_compras": df["total_purchases"],
        "ticket_promedio_usd": df["avg_order_value_usd"],
        "gasto_mensual_usd": df["total_spent_usd"] / df["tenure_months"].clip(lower=1),
        "quejas": df["complaint_count"],
        "devoluciones": df["return_count"],
        "quejas_por_compra": df["complaint_count"] / n,
        "devoluciones_por_compra": df["return_count"] / n,
        "satisfaccion": df["satisfaction_score"],
        "tasa_apertura_correos": df["email_open_rate"],
        "tasa_clics": df["click_through_rate"],
        "tasa_conversion": df["conversion_rate"],
    })
    for col, pref in CATEGORICAS.items():            # los NaN de loyalty_tier se tratan como categoría "None"
        out = pd.concat([out, pd.get_dummies(df[col].fillna("None"), prefix=pref, dtype=int)], axis=1)
    out["dias_sin_comprar"] = df["days_since_last_purchase"]   # SOLO para el control de fuga; no es variable del modelo
    out["abandono"] = (df["activity_status"] == "Inactive").astype(int)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/real/ecommerce_real.csv")
    ap.add_argument("--out", default="data/processed/dataset_real_limpio.csv")
    ap.add_argument("--models", default="models/real/")
    a = ap.parse_args()
    raw = pd.read_csv(a.raw)
    n0 = len(raw)
    raw = raw.drop_duplicates(subset="customer_id")
    df = construir(raw)
    features = [c for c in df.columns if c not in ("id_cliente", "dias_sin_comprar", "abandono")]
    tr, te = train_test_split(df.index, test_size=0.2, stratify=df["abandono"], random_state=SEED)
    df["split"] = "train"
    df.loc[te, "split"] = "test"
    scaler = StandardScaler().fit(df.loc[df.split == "train", features])   # solo con train
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.models).mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    joblib.dump(scaler, Path(a.models) / "scaler.joblib")
    meta = {"features": features, "medianas": df.loc[df.split == "train", features].median().to_dict(), "semilla": SEED,
            "reporte_limpieza": {"filas_iniciales": n0, "filas_finales": len(df), "faltantes_despues": int(df[features].isna().sum().sum())}}
    (Path(a.models) / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[INFO] Filas: {n0:,} -> {len(df):,} | variables del modelo: {len(features)} | abandono {df.abandono.mean():.1%}")
    print(f"[INFO] Train: {(df.split == 'train').sum():,} | Test: {(df.split == 'test').sum():,}")
    print(f"[SUCCESS] Dataset limpio: {a.out} | scaler y meta en {a.models}")
