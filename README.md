# SPRVC · Motor de IA (Entrega 3)

Predicción de abandono (churn) con los datos reales `E-commerce_Customer_Segmentation_2026.csv` (50,000 clientes).

**Hallazgo clave:** `activity_status` (etiqueta) es una regla fija de `days_since_last_purchase` (Inactive = 90 o más días).
Con esa variable el AUC sería 1.00 (fuga de información); sin ella, las 26 variables de comportamiento dan AUC ≈ 0.51.
Por eso esas variables se excluyen y la meta O3 (AUC ≥ 0.75) no se cumple de forma honesta. Ver `reports/real/control_fuga.txt`.

## Reproducir desde cero (Python >= 3.10)
```bash
pip install -r requirements.txt
python src/data/preprocess_real.py --raw data/real/ecommerce_real.csv --out data/processed/dataset_real_limpio.csv --models models/real/
python src/models/control_fuga.py --data data/processed/dataset_real_limpio.csv --models models/real/
python src/models/train.py --data data/processed/dataset_real_limpio.csv --output models/real/ --reports reports/real/ --rf-iter 5
python src/models/predict.py                                             # demo de inferencia
```
Comando de una línea para el entrenamiento: `python src/models/train.py --data data/processed/dataset_real_limpio.csv --output models/real/ --reports reports/real/ --rf-iter 5`

## Estructura
- `src/data/preprocess_real.py` limpieza, seudonimización, 26 variables, partición 80/20 estratificada, escalado (solo train)
- `src/models/train.py` 4 experimentos (regresión logística y Random Forest, por defecto y ajustados)
- `src/models/control_fuga.py` demuestra la fuga de información de `days_since_last_purchase`
- `src/models/predict.py` función `predecir(dict) -> dict` para el dashboard de la Entrega 4
- `models/real/` modelos y escalador serializados · `reports/real/` log, tabla de experimentos y gráficas
## Datos
El CSV real no se sube al repositorio (datos de clientes). Colócalo en `data/real/ecommerce_real.csv`.
