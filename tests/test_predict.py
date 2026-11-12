"""Pruebas de la función predecir() del SPRVC.

Ejecutar desde la raíz del repositorio:
    python -m unittest tests/test_predict.py -v
o con pytest:
    pytest tests/test_predict.py -v
"""
import sys
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.models.predict import predecir  # noqa: E402

CLIENTE_VALIDO = {
    "antiguedad_meses": 14, "num_compras": 6, "ticket_promedio_usd": 85.0, "gasto_mensual_usd": 36.4,
    "quejas": 3, "devoluciones": 2, "satisfaccion": 2.1, "tasa_apertura_correos": 0.05,
    "tasa_clics": 0.01, "tasa_conversion": 0.02, "canal": "Online", "dispositivo": "Mobile", "lealtad": "Bronze",
}

CLAVES_SALIDA = {"clase_predicha", "etiqueta", "probabilidad_abandono", "nivel_riesgo",
                 "factores_principales", "umbral_usado", "advertencia", "status"}


def _cliente(**cambios):
    datos = dict(CLIENTE_VALIDO)
    datos.update(cambios)
    return datos


class TestEntradaValida(unittest.TestCase):
    def test_devuelve_success_con_todas_las_claves(self):
        r = predecir(CLIENTE_VALIDO)
        self.assertEqual(r["status"], "success")
        self.assertEqual(set(r), CLAVES_SALIDA)

    def test_tipos_y_rangos_de_la_salida(self):
        r = predecir(CLIENTE_VALIDO)
        self.assertIn(r["clase_predicha"], (0, 1))
        self.assertIn(r["etiqueta"], ("Riesgo de Cancelación", "Cliente Fiel"))
        self.assertIn(r["nivel_riesgo"], ("Alto", "Medio", "Bajo"))
        self.assertGreaterEqual(r["probabilidad_abandono"], 0)
        self.assertLessEqual(r["probabilidad_abandono"], 100)
        self.assertIsInstance(r["factores_principales"], list)

    def test_etiqueta_coherente_con_clase(self):
        r = predecir(CLIENTE_VALIDO)
        esperada = "Riesgo de Cancelación" if r["clase_predicha"] == 1 else "Cliente Fiel"
        self.assertEqual(r["etiqueta"], esperada)

    def test_clase_coherente_con_umbral(self):
        r = predecir(CLIENTE_VALIDO)
        prob = r["probabilidad_abandono"] / 100
        self.assertEqual(r["clase_predicha"], int(prob >= r["umbral_usado"] - 0.001))

    def test_es_determinista(self):
        self.assertEqual(predecir(CLIENTE_VALIDO), predecir(CLIENTE_VALIDO))

    def test_categoricas_opcionales(self):
        datos = {k: v for k, v in CLIENTE_VALIDO.items() if k not in ("canal", "dispositivo", "lealtad")}
        self.assertEqual(predecir(datos)["status"], "success")

    def test_numeros_como_texto(self):
        self.assertEqual(predecir(_cliente(num_compras="6", satisfaccion="2.1"))["status"], "success")

    def test_num_compras_cero_no_divide_entre_cero(self):
        self.assertEqual(predecir(_cliente(num_compras=0))["status"], "success")


class TestFaltantes(unittest.TestCase):
    def test_falta_una_variable(self):
        datos = dict(CLIENTE_VALIDO)
        del datos["quejas"]
        r = predecir(datos)
        self.assertEqual(r["status"], "error")
        self.assertIn("quejas", r["mensaje"])

    def test_variable_en_none(self):
        r = predecir(_cliente(satisfaccion=None))
        self.assertEqual(r["status"], "error")
        self.assertIn("satisfaccion", r["mensaje"])

    def test_diccionario_vacio(self):
        self.assertEqual(predecir({})["status"], "error")


class TestRangos(unittest.TestCase):
    def test_satisfaccion_fuera_de_rango(self):
        for valor in (0, 6):
            r = predecir(_cliente(satisfaccion=valor))
            self.assertEqual(r["status"], "error")
            self.assertIn("satisfaccion", r["mensaje"])

    def test_tasa_mayor_a_uno(self):
        self.assertEqual(predecir(_cliente(tasa_clics=1.5))["status"], "error")

    def test_valores_negativos(self):
        self.assertEqual(predecir(_cliente(quejas=-1))["status"], "error")

    def test_limites_exactos_son_validos(self):
        r = predecir(_cliente(satisfaccion=5, tasa_conversion=1, tasa_clics=0))
        self.assertEqual(r["status"], "success")

    def test_infinito_y_nan(self):
        self.assertEqual(predecir(_cliente(gasto_mensual_usd=float("inf")))["status"], "error")
        self.assertEqual(predecir(_cliente(gasto_mensual_usd=float("nan")))["status"], "error")


class TestErrores(unittest.TestCase):
    def test_entrada_no_es_diccionario(self):
        for entrada in (None, [], "texto", 5):
            self.assertEqual(predecir(entrada)["status"], "error")

    def test_valor_no_numerico(self):
        r = predecir(_cliente(antiguedad_meses="catorce"))
        self.assertEqual(r["status"], "error")
        self.assertIn("antiguedad_meses", r["mensaje"])

    def test_categoria_desconocida_no_rompe(self):
        self.assertEqual(predecir(_cliente(canal="Telefono"))["status"], "success")

    def test_error_trae_mensaje(self):
        r = predecir({})
        self.assertIn("mensaje", r)
        self.assertTrue(r["mensaje"])


if __name__ == "__main__":
    unittest.main()
