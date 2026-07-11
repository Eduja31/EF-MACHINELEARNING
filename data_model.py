"""
data_model.py
==============================================================
Modelo de datos del Sistema de Gestión Hídrica Inteligente.

Contiene:
- La clase Zona (estado de cada zona de cultivo)
- El historial de lecturas (simulando lo que vendría de Firebase)
- El módulo de análisis estadístico (numpy / pandas)
==============================================================
"""

import random
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
#  CONSTANTES DE NEGOCIO (umbrales del sistema de riego)
# ------------------------------------------------------------------
UMBRAL_CRITICO = 30     # < este valor -> riego urgente / automático
UMBRAL_BAJO = 45        # < este valor -> riego recomendado
UMBRAL_PARAR = 75       # > este valor -> detener riego
UMBRAL_EXCESO = 88      # > este valor -> exceso de agua


@dataclass
class Zona:
    id: int
    nombre: str
    cultivo: str
    humedad: float
    temperatura: float
    color: tuple          # color RGB (0-1) para OpenGL
    pos: tuple            # posición (x, z) en el campo 3D
    tam: tuple = (10, 9)  # tamaño (ancho, profundidad)

    def estado(self) -> str:
        if self.humedad < UMBRAL_CRITICO:
            return "Crítico"
        elif self.humedad < UMBRAL_BAJO:
            return "Bajo"
        elif self.humedad < UMBRAL_PARAR:
            return "Óptimo"
        else:
            return "Alto"

    def color_estado(self) -> tuple:
        """Color RGB normalizado según el estado de humedad."""
        return {
            "Crítico": (0.96, 0.26, 0.21),
            "Bajo": (1.00, 0.60, 0.00),
            "Óptimo": (0.00, 0.90, 0.46),
            "Alto": (0.00, 0.69, 1.00),
        }[self.estado()]


def crear_zonas_iniciales():
    """Crea las 4 zonas de cultivo con sus valores iniciales."""
    return [
        Zona(0, "Zona Norte", "Maíz amarillo", 67.0, 23.1, (0.0, 0.9, 1.0), (-8, -8)),
        Zona(1, "Zona Sur", "Papa nativa", 43.0, 25.8, (1.0, 0.6, 0.0), (8, -8)),
        Zona(2, "Zona Este", "Espárrago", 78.0, 22.4, (0.0, 0.9, 0.46), (8, 8)),
        Zona(3, "Zona Oeste", "Quinua", 31.0, 26.9, (0.96, 0.26, 0.21), (-8, 8)),
    ]


# ==================================================================
#  HISTORIAL DE LECTURAS  (simula lo que en el proyecto real
#  vendría de Firebase, cada 15 minutos durante varias semanas)
# ==================================================================
class HistorialDatos:
    """
    Guarda lecturas de sensores en memoria y permite:
      - registrar nuevas lecturas en cada ciclo de simulación
      - exportarlas a un DataFrame de pandas
      - calcular estadísticas descriptivas
      - detectar anomalías simples (outliers) con z-score
      - generar un pronóstico simple por regresión lineal (numpy)
    """

    def __init__(self, zonas):
        self.zonas = zonas
        self.registros = []   # lista de dicts: timestamp, zona, humedad, temp
        self.t0 = time.time()
        self._semilla_historica()

    # --------------------------------------------------------------
    def _semilla_historica(self, horas=72, paso_min=15):
        """
        Genera un historial sintético de varias horas atrás,
        como si fueran lecturas reales guardadas en la nube.
        """
        pasos = int(horas * 60 / paso_min)
        ahora = time.time()
        for z in self.zonas:
            humedad = z.humedad
            for i in range(pasos, 0, -1):
                ts = ahora - i * paso_min * 60
                # camina aleatoriamente alrededor del valor base (random walk)
                humedad += random.uniform(-1.2, 1.1)
                humedad = float(np.clip(humedad, 15, 95))
                temp = z.temperatura + random.uniform(-1.5, 1.5)
                self.registros.append({
                    "timestamp": ts,
                    "zona_id": z.id,
                    "zona": z.nombre,
                    "humedad": round(humedad, 2),
                    "temperatura": round(temp, 2),
                })

    # --------------------------------------------------------------
    def registrar_lectura(self):
        """Agrega la lectura actual de todas las zonas al historial."""
        ts = time.time()
        for z in self.zonas:
            self.registros.append({
                "timestamp": ts,
                "zona_id": z.id,
                "zona": z.nombre,
                "humedad": round(z.humedad, 2),
                "temperatura": round(z.temperatura, 2),
            })
        # limita el tamaño en memoria (deja máx. ~5000 registros)
        if len(self.registros) > 5000:
            self.registros = self.registros[-5000:]

    # --------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.registros)
        if df.empty:
            return df
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
        return df

    # --------------------------------------------------------------
    def estadisticas_por_zona(self) -> pd.DataFrame:
        """
        Devuelve un resumen estadístico (media, desviación, min, max,
        percentiles) de humedad y temperatura agrupado por zona.
        Este es el "análisis por código" pedido para el proyecto.
        """
        df = self.to_dataframe()
        if df.empty:
            return pd.DataFrame()

        resumen = df.groupby("zona").agg(
            humedad_media=("humedad", "mean"),
            humedad_std=("humedad", "std"),
            humedad_min=("humedad", "min"),
            humedad_max=("humedad", "max"),
            temp_media=("temperatura", "mean"),
            n_lecturas=("humedad", "count"),
        ).round(2)

        # % de tiempo en estado crítico/bajo
        pct_critico = (
            df[df["humedad"] < UMBRAL_CRITICO]
            .groupby("zona")["humedad"].count()
            / df.groupby("zona")["humedad"].count() * 100
        ).round(1).reindex(resumen.index).fillna(0)
        resumen["pct_tiempo_critico"] = pct_critico

        return resumen

    # --------------------------------------------------------------
    def detectar_anomalias(self, zona_id: int, umbral_z=2.0) -> pd.DataFrame:
        """
        Detecta lecturas anómalas de humedad en una zona usando z-score.
        Una lectura es anómala si se aleja más de `umbral_z` desviaciones
        estándar de la media histórica de esa zona.
        """
        df = self.to_dataframe()
        if df.empty:
            return df
        sub = df[df["zona_id"] == zona_id].copy()
        if len(sub) < 5:
            return sub.iloc[0:0]

        media = sub["humedad"].mean()
        std = sub["humedad"].std() or 1e-6
        sub["z_score"] = (sub["humedad"] - media) / std
        anomalas = sub[sub["z_score"].abs() > umbral_z]
        return anomalas[["datetime", "zona", "humedad", "z_score"]]

    # --------------------------------------------------------------
    def pronostico_lineal(self, zona_id: int, horas_adelante=(6, 12, 24, 36, 48)):
        """
        Pronóstico simple de humedad futura usando regresión lineal
        (numpy.polyfit) sobre las últimas lecturas de la zona.
        Simula lo que haría el modelo de Machine Learning del proyecto.
        Devuelve una lista de tuples (horas, humedad_estimada).
        """
        df = self.to_dataframe()
        if df.empty:
            return []
        sub = df[df["zona_id"] == zona_id].sort_values("timestamp")
        if len(sub) < 5:
            return [(h, sub["humedad"].iloc[-1] if len(sub) else 50.0) for h in horas_adelante]

        # usa las últimas 24 lecturas (6 horas aprox a 15 min) para la tendencia
        ventana = sub.tail(24)
        x = np.arange(len(ventana))
        y = ventana["humedad"].to_numpy()

        # regresión lineal: humedad = a*x + b
        a, b = np.polyfit(x, y, 1)

        resultado = []
        ultimo_x = len(ventana) - 1
        paso_por_hora = 4  # 4 lecturas de 15 min = 1 hora
        for h in horas_adelante:
            x_futuro = ultimo_x + h * paso_por_hora
            estimado = a * x_futuro + b
            estimado = float(np.clip(estimado, 0, 100))
            resultado.append((h, round(estimado, 1)))
        return resultado

    # --------------------------------------------------------------
    def precision_modelo_simulada(self) -> float:
        """
        Calcula un valor de 'precisión' simulado del modelo ML
        comparando el pronóstico de hace 24h con el valor real actual,
        usando error porcentual absoluto medio (MAPE) invertido.
        """
        errores = []
        for z in self.zonas:
            pron = self.pronostico_lineal(z.id, horas_adelante=(0,))
            if pron:
                estimado = pron[0][1]
                real = z.humedad
                error = abs(estimado - real) / max(real, 1) * 100
                errores.append(error)
        if not errores:
            return 75.0
        mape = float(np.mean(errores))
        precision = max(50.0, min(99.0, 100 - mape))
        return round(precision, 1)
