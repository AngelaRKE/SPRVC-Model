"""Entrenamiento y experimentación del módulo de abandono (churn) del SPRVC.

Compara dos familias de modelos sobre el MISMO conjunto de prueba (partición 80/20 estratificada
creada por src/data/preprocess_real.py, columna `split`):

  * Regresión logística (referencia interpretable)      -> baseline y versión ajustada
  * Bosque aleatorio (Random Forest, modelo candidato)  -> configuración por defecto y versión ajustada

Reglas anti-fuga: el ajuste de hiperparámetros y la elección del umbral usan SOLO validación cruzada
sobre train; el conjunto de prueba se usa una única vez para reportar. El ganador se elige por AUC-ROC
de validación cruzada (no por el de prueba).

Uso:  python src/models/train.py --data data/processed/dataset_real_limpio.csv --output models/real/ --reports reports/real/
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve, precision_recall_curve,
                             classification_report, accuracy_score)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_predict

OBJETIVO = "abandono"   # 1 = Inactive (90 o más días sin comprar), construido en src/data/preprocess_real.py

SEED = 42
META_RECALL = 0.60      # meta O3 del informe de diseño: recall de la clase abandono >= 60%
META_AUC = 0.75         # meta O3: AUC-ROC >= 0.75
LOG: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


def umbral_operativo(y, p, meta_recall=META_RECALL) -> float:
    """Umbral MÁS ALTO (más precisión) que aún cumple recall >= meta, calculado sobre predicciones OOF."""
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= meta_recall)[0]
    return float(thr[ok[-1]]) if len(ok) else 0.5


def evaluar(modelo, X_tr, y_tr, X_te, y_te, umbral: float) -> dict:
    t0 = time.perf_counter()
    p_te = modelo.predict_proba(X_te)[:, 1]
    t_inf = time.perf_counter() - t0
    p_tr = modelo.predict_proba(X_tr)[:, 1]
    pred = (p_te >= umbral).astype(int)
    pred_tr = (p_tr >= umbral).astype(int)
    return {
        "auc_test": roc_auc_score(y_te, p_te), "auc_train": roc_auc_score(y_tr, p_tr),
        "pr_auc_test": average_precision_score(y_te, p_te),
        "recall_test": recall_score(y_te, pred), "precision_test": precision_score(y_te, pred, zero_division=0),
        "f1_test": f1_score(y_te, pred), "accuracy_test": accuracy_score(y_te, pred),
        "recall_train": recall_score(y_tr, pred_tr), "f1_train": f1_score(y_tr, pred_tr),
        "umbral": umbral, "t_inferencia_s": t_inf, "p_test": p_te,
    }


def diagnostico(auc_tr: float, auc_te: float) -> str:
    brecha = auc_tr - auc_te
    if auc_te < 0.65 and brecha < 0.08:
        return "Subajuste (underfitting)"
    if brecha > 0.15:
        return "Sobreajuste severo"
    if brecha > 0.06:
        return "Sobreajuste moderado"
    return "Óptimo (generalización sólida)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/dataset_real_limpio.csv")
    ap.add_argument("--output", default="models/real/")
    ap.add_argument("--reports", default="reports/real/")
    ap.add_argument("--rf-iter", type=int, default=30)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    salida, rep = Path(a.output), Path(a.reports)
    (rep / "figures").mkdir(parents=True, exist_ok=True)
    salida.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- datos
    df = pd.read_csv(a.data)
    log(f"[INFO] Cargando dataset: {a.data} ({len(df):,} registros)...")
    feats = json.loads((salida / "meta.json").read_text())["features"]   # variables definidas por preprocess_real.py
    tr, te = df[df.split == "train"], df[df.split == "test"]
    scaler = joblib.load(salida / "scaler.joblib")                       # ajustado solo con train en preprocess_real.py
    Xtr, ytr = scaler.transform(tr[feats].values), tr[OBJETIVO].values
    Xte, yte = scaler.transform(te[feats].values), te[OBJETIVO].values
    log(f"[INFO] Conjunto Train: {len(tr):,} muestras | Conjunto Test: {len(te):,} muestras")
    log(f"[INFO] Tasa de abandono: train {ytr.mean():.1%} | test {yte.mean():.1%} | variables: {len(feats)}")
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    # ---------------------------------------------------------------- experimentos
    exps = []   # (id, nombre, modelo_familia, hiperparametros_texto, estimador, cv_auc)

    def registrar(eid, nombre, familia, estimador, tiempo, cv_auc, hp):
        exps.append(dict(id=eid, nombre=nombre, familia=familia, modelo=estimador, t_train=tiempo, cv_auc=cv_auc, hp=hp))
        log(f"[INFO] {eid} finalizado: entrenamiento final en {tiempo:.2f} s | AUC-ROC validación cruzada (5 folds) = {cv_auc:.4f}")

    def ajustar(estim):
        t0 = time.perf_counter()
        estim.fit(Xtr, ytr)
        return time.perf_counter() - t0

    def cv_auc_de(estim):
        """AUC-ROC de validación cruzada (5 folds) calculado solo sobre train."""
        oof_ = cross_val_predict(estim, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
        return float(roc_auc_score(ytr, oof_))

    # Exp 1: baseline regresión logística (configuración por defecto)
    m = LogisticRegression(max_iter=1000)
    log("[INFO] Exp 1 - Entrenando Regresión Logística baseline (C=1.0, sin pesos de clase)...")
    t = ajustar(m)
    registrar("Exp 1", "Baseline", "Regresión logística", m, t, cv_auc_de(LogisticRegression(max_iter=1000)), "C = 1.0, class_weight = None")

    # Exp 2: regresión logística ajustada (regularización L2 + pesos de clase)
    log("[INFO] Exp 2 - Ajustando Regresión Logística (GridSearchCV: C, class_weight)...")
    gs = GridSearchCV(LogisticRegression(max_iter=2000), {"C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                                                          "class_weight": [None, "balanced"]},
                      scoring="roc_auc", cv=cv, n_jobs=-1)
    gs.fit(Xtr, ytr)
    m = LogisticRegression(max_iter=2000, **gs.best_params_)
    t = ajustar(m)
    registrar("Exp 2", "Ajuste A", "Regresión logística", m, t, gs.best_score_,
              f"C = {gs.best_params_['C']}, class_weight = {gs.best_params_['class_weight']}")

    # Exp 3: Random Forest por defecto (árboles sin límite de profundidad)
    log("[INFO] Exp 3 - Entrenando Random Forest por defecto (100 árboles, sin poda)...")
    m = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    t = ajustar(m)
    registrar("Exp 3", "Alternativo (defecto)", "Random Forest", m, t,
              cv_auc_de(RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)),
              "n_estimators = 100, max_depth = None, min_samples_leaf = 1")

    # Exp 4: Random Forest ajustado (poda y regularización por hojas mínimas)
    log(f"[INFO] Exp 4 - Ajustando Random Forest (RandomizedSearchCV, {a.rf_iter} combinaciones x 5 folds)...")
    espacio = {"n_estimators": [200, 300, 500], "max_depth": [3, 4, 5, 6, 8, 10],
               "min_samples_leaf": [5, 10, 20, 40], "max_features": ["sqrt", 0.5, 0.8],
               "class_weight": [None, "balanced_subsample"]}
    rs = RandomizedSearchCV(RandomForestClassifier(random_state=SEED, n_jobs=1), espacio, n_iter=a.rf_iter,
                            scoring="roc_auc", cv=cv, n_jobs=-1, random_state=SEED)
    rs.fit(Xtr, ytr)
    bp = rs.best_params_
    m = RandomForestClassifier(random_state=SEED, n_jobs=-1, **bp)
    t = ajustar(m)
    registrar("Exp 4", "Ajuste B (optimizado)", "Random Forest", m, t, rs.best_score_,
              f"n_estimators = {bp['n_estimators']}, max_depth = {bp['max_depth']}, "
              f"min_samples_leaf = {bp['min_samples_leaf']}, max_features = {bp['max_features']}, "
              f"class_weight = {bp['class_weight']}")

    # ---------------------------------------------------------------- evaluación sobre el mismo Test Set
    filas = []
    for e in exps:
        # umbral operativo: elegido con predicciones OOF de train (sin mirar test)
        clon = type(e["modelo"])(**e["modelo"].get_params())
        oof = cross_val_predict(clon, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
        u = umbral_operativo(ytr, oof)
        r = evaluar(e["modelo"], Xtr, ytr, Xte, yte, u)
        r["diag"] = diagnostico(r["auc_train"], r["auc_test"])
        e["res"] = r
        filas.append({"experimento": e["id"], "variante": e["nombre"], "modelo": e["familia"], "hiperparametros": e["hp"],
                      "auc_cv_train": round(e["cv_auc"], 4), "auc_train": round(r["auc_train"], 4),
                      "auc_test": round(r["auc_test"], 4), "pr_auc_test": round(r["pr_auc_test"], 4),
                      "umbral_operativo": round(u, 3), "recall_test": round(r["recall_test"], 4),
                      "precision_test": round(r["precision_test"], 4), "f1_test": round(r["f1_test"], 4),
                      "t_entrenamiento_s": round(e["t_train"], 3), "t_inferencia_test_s": round(r["t_inferencia_s"], 4),
                      "diagnostico": r["diag"]})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(rep / "experimentos.csv", index=False)

    # ganador: mejor AUC de validación cruzada (criterio independiente del Test Set)
    gan = max(exps, key=lambda e: e["cv_auc"])
    base = exps[0]
    log("[RESULTADOS EN TEST SET] (mismo conjunto de prueba para los 4 experimentos)")
    log(tabla[["experimento", "modelo", "auc_train", "auc_test", "pr_auc_test", "recall_test", "f1_test", "diagnostico"]].to_string(index=False))
    log(f"[INFO] Ganador (mejor AUC en validación cruzada): {gan['id']} - {gan['familia']} ({gan['hp']})")

    r = gan["res"]
    pred = (r["p_test"] >= r["umbral"]).astype(int)
    log(f"[INFO] Umbral operativo (elegido con CV en train para recall >= {META_RECALL:.0%}): {r['umbral']:.3f}")
    log("[REPORTE DE CLASIFICACION DEL GANADOR EN TEST]")
    log(classification_report(yte, pred, target_names=["0 Cliente fiel", "1 Abandono"], digits=2))
    cm = confusion_matrix(yte, pred)
    log(f"[MATRIZ DE CONFUSION] TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
    cumple = r["auc_test"] >= META_AUC and r["recall_test"] >= META_RECALL
    log(f"[META O3] AUC-ROC >= {META_AUC}: {'SI' if r['auc_test'] >= META_AUC else 'NO'} ({r['auc_test']:.3f}) | "
        f"Recall >= {META_RECALL:.0%}: {'SI' if r['recall_test'] >= META_RECALL else 'NO'} ({r['recall_test']:.1%})")

    # ---------------------------------------------------------------- ¿la diferencia entre familias es real o ruido?
    # Bootstrap pareado sobre el Test Set: mejor regresión logística vs. mejor Random Forest (por AUC de CV).
    mejor_lr = max((e for e in exps if e["familia"] == "Regresión logística"), key=lambda e: e["cv_auc"])
    mejor_rf = max((e for e in exps if e["familia"] == "Random Forest"), key=lambda e: e["cv_auc"])
    rng = np.random.default_rng(SEED)
    difs = []
    for _ in range(1000):
        i = rng.integers(0, len(yte), len(yte))
        if yte[i].min() == yte[i].max():
            continue
        difs.append(roc_auc_score(yte[i], mejor_lr["res"]["p_test"][i]) - roc_auc_score(yte[i], mejor_rf["res"]["p_test"][i]))
    lo_, hi_ = np.percentile(difs, [2.5, 97.5])
    log(f"[INFO] Diferencia de AUC en Test ({mejor_lr['id']} - {mejor_rf['id']}) = "
        f"{mejor_lr['res']['auc_test'] - mejor_rf['res']['auc_test']:+.4f} | IC 95% bootstrap [{lo_:+.4f}, {hi_:+.4f}]"
        f" -> {'diferencia significativa' if lo_ > 0 or hi_ < 0 else 'diferencia NO significativa (empate técnico)'}")
    comparacion = {"mejor_lr": mejor_lr["id"], "mejor_rf": mejor_rf["id"],
                   "dif_auc_test": mejor_lr["res"]["auc_test"] - mejor_rf["res"]["auc_test"], "ic95": [lo_, hi_]}

    # ---------------------------------------------------------------- importancia de variables (ganador, permutación en test)
    pi = permutation_importance(gan["modelo"], Xte, yte, scoring="roc_auc", n_repeats=15, random_state=SEED, n_jobs=1)
    imp = pd.Series(pi.importances_mean, index=feats).sort_values(ascending=False)
    log("[INFO] Importancia por permutación (caída de AUC en test): " +
        ", ".join(f"{k}={v:.3f}" for k, v in imp.head(5).items()))

    # ---------------------------------------------------------------- serialización
    joblib.dump(gan["modelo"], salida / "modelo_candidato.joblib")
    joblib.dump(base["modelo"], salida / "modelo_baseline.joblib")
    cfg = {"modelo_candidato": f"{gan['id']} - {gan['familia']}", "hiperparametros": gan["hp"], "umbral": r["umbral"],
           "features": feats, "auc_test": r["auc_test"], "recall_test": r["recall_test"],
           "importancia_variables": imp.to_dict(), "semilla": SEED}
    (salida / "config_modelo.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    log(f"[SUCCESS] Modelo candidato serializado en: {salida / 'modelo_candidato.joblib'}")
    log(f"[SUCCESS] Modelo baseline serializado en: {salida / 'modelo_baseline.joblib'}")
    log(f"[SUCCESS] Escalador (ajustado solo con train) en: {salida / 'scaler.joblib'} | umbral y metadatos: {salida / 'config_modelo.json'}")

    # ---------------------------------------------------------------- gráficas
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=16,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Fiel", "Abandono"]); ax.set_yticks([0, 1], ["Fiel", "Abandono"])
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real"); ax.set_title(f"Matriz de confusión (Test)\n{gan['id']} · {gan['familia']}")
    fig.tight_layout(); fig.savefig(rep / "figures" / "matriz_confusion.png", dpi=170); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for e in exps:
        fpr, tpr, _ = roc_curve(yte, e["res"]["p_test"]); pr, rc, _ = precision_recall_curve(yte, e["res"]["p_test"])
        et = f"{e['id']} {e['familia']} (AUC {e['res']['auc_test']:.3f})"
        axs[0].plot(fpr, tpr, label=et, lw=2 if e is gan else 1.2); axs[1].plot(rc, pr, label=et, lw=2 if e is gan else 1.2)
    axs[0].plot([0, 1], [0, 1], "k--", lw=.8); axs[0].set_xlabel("Tasa de falsos positivos"); axs[0].set_ylabel("Recall"); axs[0].set_title("Curva ROC (Test)")
    axs[1].axhline(yte.mean(), color="k", ls="--", lw=.8); axs[1].set_xlabel("Recall"); axs[1].set_ylabel("Precisión"); axs[1].set_title("Curva precisión-recall (Test)")
    axs[0].legend(fontsize=7, loc="lower right"); fig.tight_layout(); fig.savefig(rep / "figures" / "curvas_roc_pr.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    imp.sort_values().plot.barh(ax=ax, color="#0568cc"); ax.set_xlabel("Caída de AUC al permutar la variable (Test)")
    ax.set_title("Importancia de variables · modelo candidato"); fig.tight_layout()
    fig.savefig(rep / "figures" / "importancia_variables.png", dpi=170); plt.close(fig)

    (rep / "log_entrenamiento.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    metricas = {"ganador": f"{gan['id']} {gan['familia']}", "comparacion_lr_vs_rf": comparacion,
                "matriz_confusion": {"TN": int(cm[0, 0]), "FP": int(cm[0, 1]), "FN": int(cm[1, 0]), "TP": int(cm[1, 1])},
                "experimentos": json.loads(tabla.to_json(orient="records"))}
    (rep / "metricas.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False))
    log(f"[SUCCESS] Reportes: {rep / 'experimentos.csv'}, {rep / 'metricas.json'}, gráficas en {rep / 'figures'}")
    (rep / "log_entrenamiento.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
