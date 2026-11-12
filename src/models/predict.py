"""Función de inferencia desacoplada del SPRVC (contrato de interfaz con el dashboard de la Entrega 4).

No depende de rutas absolutas: los artefactos se buscan relativos a la raíz del repositorio
(carpeta models/real/, o la indicada en la variable de entorno SPRVC_MODELOS).

Uso desde Streamlit / otro módulo:
    from src.models.predict import predecir
    resultado = predecir({"antiguedad_meses": 14, "num_compras": 6, ...})

Uso desde terminal (demo):  python src/models/predict.py
"""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
CARPETA = RAIZ / os.environ.get("SPRVC_MODELOS", "models/real")

# 1. Carga de artefactos serializados (una sola vez en memoria)
MODELO = joblib.load(CARPETA / "modelo_candidato.joblib")
SCALER = joblib.load(CARPETA / "scaler.joblib")
META = json.loads((CARPETA / "meta.json").read_text(encoding="utf-8"))
CONFIG = json.loads((CARPETA / "config_modelo.json").read_text(encoding="utf-8"))
UMBRAL = CONFIG["umbral"]                      # umbral operativo elegido por validación cruzada (recall >= 60%)
FEATURES = META["features"]

NUMERICAS = {  # variable: (mínimo, máximo)
    "antiguedad_meses": (0, 600), "num_compras": (0, 10000), "ticket_promedio_usd": (0, 100000),
    "gasto_mensual_usd": (0, 100000), "quejas": (0, 1000), "devoluciones": (0, 1000),
    "satisfaccion": (1, 5), "tasa_apertura_correos": (0, 1), "tasa_clics": (0, 1), "tasa_conversion": (0, 1)}
CATEGORICAS = {"canal": "canal", "dispositivo": "dispositivo", "lealtad": "lealtad"}  # campo de entrada -> prefijo
OBLIGATORIAS = list(NUMERICAS)


def _validar(entrada: dict) -> str | None:
    """Devuelve un mensaje de error legible o None si la entrada es válida."""
    if not isinstance(entrada, dict):
        return "La entrada debe ser un diccionario."
    faltan = [c for c in OBLIGATORIAS if c not in entrada or entrada[c] is None]
    if faltan:
        return "Faltan variables obligatorias: " + ", ".join(faltan)
    for c, (lo, hi) in NUMERICAS.items():
        try:
            v = float(entrada[c])
        except (TypeError, ValueError):
            return f"La variable '{c}' debe ser numérica."
        if not np.isfinite(v) or not lo <= v <= hi:
            return f"La variable '{c}' debe estar entre {lo} y {hi}."
    return None


def _fila(entrada: dict) -> pd.DataFrame:
    """Construye la fila de 26 variables igual que en preprocess_real.py (razones y one-hot)."""
    n = max(float(entrada["num_compras"]), 1.0)
    f = {c: float(entrada[c]) for c in NUMERICAS}
    f["quejas_por_compra"] = float(entrada["quejas"]) / n
    f["devoluciones_por_compra"] = float(entrada["devoluciones"]) / n
    for campo, pref in CATEGORICAS.items():
        valor = entrada.get(campo) or "None"
        f[f"{pref}_{valor}"] = 1.0
    return pd.DataFrame([{c: f.get(c, 0.0) for c in FEATURES}])


def _factores(x_scaled: np.ndarray, top: int = 3) -> list[str]:
    """Variables que más empujan la predicción hacia abandono (coeficiente x valor, o importancia x desviación)."""
    if hasattr(MODELO, "coef_"):
        aporte = MODELO.coef_[0] * x_scaled[0]
    else:
        aporte = MODELO.feature_importances_ * np.abs(x_scaled[0])
    orden = np.argsort(aporte)[::-1][:top]
    return [FEATURES[i] for i in orden if aporte[i] > 0]


def predecir(entrada: dict) -> dict:
    """Recibe las variables de un cliente (dict plano del formulario) y devuelve la inferencia estructurada.

    Entrada: antiguedad_meses, num_compras, ticket_promedio_usd, gasto_mensual_usd, quejas, devoluciones,
             satisfaccion (1-5), tasa_apertura_correos, tasa_clics, tasa_conversion (0-1) y, opcionales,
             canal ("Online", "In-Store", "Mobile App", "Marketplace"), dispositivo ("Desktop", "Mobile",
             "Tablet", "Multiple") y lealtad ("Bronze", "Silver", "Gold", "Platinum", "Diamond" o ausente).
    Salida:  dict con clase_predicha (0/1), etiqueta, probabilidad_abandono (%), nivel_riesgo, factores_principales,
             umbral_usado, advertencia y status ("success" | "error" + mensaje).
    """
    error = _validar(entrada)
    if error:
        return {"status": "error", "mensaje": error}
    try:
        x_scaled = SCALER.transform(_fila(entrada))                 # mismo preprocesamiento que en entrenamiento
        prob = float(MODELO.predict_proba(x_scaled)[0][1])
        clase = int(prob >= UMBRAL)
        return {
            "clase_predicha": clase,
            "etiqueta": "Riesgo de Cancelación" if clase == 1 else "Cliente Fiel",
            "probabilidad_abandono": round(prob * 100, 2),
            "nivel_riesgo": "Alto" if prob >= 0.35 else "Medio" if prob >= UMBRAL else "Bajo",
            "factores_principales": _factores(x_scaled) if clase == 1 else [],
            "umbral_usado": round(UMBRAL, 3),
            "advertencia": f"Modelo con AUC en test {CONFIG['auc_test']:.2f}: la señal de comportamiento es débil.",
            "status": "success",
        }
    except Exception as exc:  # el dashboard nunca debe romperse por una entrada inesperada
        return {"status": "error", "mensaje": f"No se pudo calcular la predicción: {exc}"}


if __name__ == "__main__":
    ejemplo = {"antiguedad_meses": 14, "num_compras": 6, "ticket_promedio_usd": 85.0, "gasto_mensual_usd": 36.4,
               "quejas": 3, "devoluciones": 2, "satisfaccion": 2.1, "tasa_apertura_correos": 0.05,
               "tasa_clics": 0.01, "tasa_conversion": 0.02, "canal": "Online", "dispositivo": "Mobile", "lealtad": "Bronze"}
    print("Entrada:", json.dumps(ejemplo, ensure_ascii=False))
    print("Salida: ", json.dumps(predecir(ejemplo), indent=2, ensure_ascii=False))
    print("Error controlado:", predecir({"antiguedad_meses": 14}))
