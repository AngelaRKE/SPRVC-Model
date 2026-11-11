"""Control de fuga de información sobre los datos reales.

Entrena una regresión logística con y sin las variables que revelan la etiqueta y reporta el AUC en test.
Uso:  python src/models/control_fuga.py --data data/processed/dataset_real_limpio.csv --models models/real/
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/processed/dataset_real_limpio.csv")
ap.add_argument("--models", default="models/real/")
a = ap.parse_args()
df = pd.read_csv(a.data)
feats = json.loads((Path(a.models) / "meta.json").read_text())["features"]
tr, te = df[df.split == "train"], df[df.split == "test"]


def auc(cols):
    s = StandardScaler().fit(tr[cols])
    m = LogisticRegression(max_iter=3000).fit(s.transform(tr[cols]), tr.abandono)
    return roc_auc_score(te.abandono, m.predict_proba(s.transform(te[cols]))[:, 1])


print(f"[CONTROL] AUC-ROC en test SIN variables de recencia ({len(feats)} variables de comportamiento): {auc(feats):.4f}")
print(f"[CONTROL] AUC-ROC en test CON dias_sin_comprar (fuga: la etiqueta es una regla de esa variable): {auc(feats + ['dias_sin_comprar']):.4f}")
for c in ["antiguedad_meses", "num_compras", "gasto_mensual_usd", "quejas", "devoluciones", "satisfaccion", "tasa_apertura_correos"]:
    print(f"[CONTROL] AUC univariado de {c}: {roc_auc_score(te.abandono, te[c]):.3f}")
