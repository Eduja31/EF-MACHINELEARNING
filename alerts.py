"""
alerts.py
==============================================================
Sistema de alertas del proyecto.
Decide cuándo avisar al usuario que debe regar, detener el riego,
o si el sistema lo hizo automáticamente.
==============================================================
"""

import time
from dataclasses import dataclass, field
from data_model import UMBRAL_CRITICO, UMBRAL_BAJO, UMBRAL_PARAR, UMBRAL_EXCESO


@dataclass
class Alerta:
    tipo: str        # "critical" | "warning" | "ok" | "info" | "auto"
    icono: str
    titulo: str
    mensaje: str
    creada_en: float = field(default_factory=time.time)
    duracion: float = 6.0   # segundos visibles en pantalla

    def expirada(self) -> bool:
        return (time.time() - self.creada_en) > self.duracion

    def progreso(self) -> float:
        """Devuelve 0.0 -> 1.0 según cuánto ha pasado de su duración (para la barrita)."""
        t = (time.time() - self.creada_en) / self.duracion
        return max(0.0, min(1.0, t))

    def color(self):
        return {
            "critical": (0.96, 0.26, 0.21),
            "warning": (1.00, 0.60, 0.00),
            "ok": (0.00, 0.90, 0.46),
            "info": (0.00, 0.90, 1.00),
            "auto": (0.49, 0.30, 1.00),
        }[self.tipo]


class GestorAlertas:
    """
    Administra la cola de alertas visibles en pantalla y la lógica
    de cuándo deben dispararse según el estado de cada zona.
    Evita repetir la misma alerta una y otra vez (lo hace solo al
    cruzar el umbral, no en cada frame).
    """

    def __init__(self):
        self.alertas_activas = []
        self.ya_mostradas = set()   # claves "tipo_zonaId" ya disparadas
        self.modo_auto = False
        self.zonas_riego_auto = set()
        self.eventos_log = []       # historial textual (lado derecho de la UI)

    # --------------------------------------------------------------
    def lanzar(self, tipo, icono, titulo, mensaje, duracion=6.0):
        alerta = Alerta(tipo, icono, titulo, mensaje, duracion=duracion)
        self.alertas_activas.append(alerta)
        self.registrar_evento(f"{icono} {titulo}")

    def registrar_evento(self, texto):
        self.eventos_log.insert(0, (time.strftime("%H:%M:%S"), texto))
        self.eventos_log = self.eventos_log[:10]

    def limpiar_expiradas(self):
        self.alertas_activas = [a for a in self.alertas_activas if not a.expirada()]

    # --------------------------------------------------------------
    def evaluar_zonas(self, zonas, riego_manual_activo):
        """
        Revisa cada zona y decide si debe lanzar una alerta nueva.
        Se llama periódicamente (ej. cada 2 segundos) desde el loop principal.
        """
        for z in zonas:
            k_critico = f"critico_{z.id}"
            k_bajo = f"bajo_{z.id}"
            k_parar = f"parar_{z.id}"
            k_exceso = f"exceso_{z.id}"

            # 1) CRÍTICO -> riego automático o alerta urgente
            if z.humedad < UMBRAL_CRITICO:
                if k_critico not in self.ya_mostradas:
                    self.ya_mostradas.add(k_critico)
                    self.ya_mostradas.discard(k_bajo)
                    if self.modo_auto:
                        self.zonas_riego_auto.add(z.id)
                        self.lanzar(
                            "auto", "🤖", f"AUTO-RIEGO: {z.nombre}",
                            f"Humedad crítica ({z.humedad:.0f}%). Electroválvulas abiertas automáticamente.",
                            duracion=8,
                        )
                    else:
                        self.lanzar(
                            "critical", "🚨", f"¡RIEGO URGENTE! — {z.nombre}",
                            f"Humedad en {z.humedad:.0f}%. {z.cultivo} en estrés hídrico severo. Riega ahora.",
                            duracion=9,
                        )
            else:
                self.ya_mostradas.discard(k_critico)
                self.zonas_riego_auto.discard(z.id)

            # 2) BAJO -> recomendación
            if UMBRAL_CRITICO <= z.humedad < UMBRAL_BAJO:
                if k_bajo not in self.ya_mostradas:
                    self.ya_mostradas.add(k_bajo)
                    self.lanzar(
                        "warning", "⚠️", f"Riego Recomendado — {z.nombre}",
                        f"Humedad en {z.humedad:.0f}%. Se recomienda regar pronto.",
                        duracion=7,
                    )
            else:
                self.ya_mostradas.discard(k_bajo)

            # 3) PARAR -> humedad recuperada
            riega_esta_zona = riego_manual_activo or z.id in self.zonas_riego_auto
            if z.humedad >= UMBRAL_PARAR and riega_esta_zona:
                if k_parar not in self.ya_mostradas:
                    self.ya_mostradas.add(k_parar)
                    self.ya_mostradas.discard(k_exceso)
                    self.lanzar(
                        "ok", "✅", f"Detener Riego — {z.nombre}",
                        f"Humedad en {z.humedad:.0f}%. Nivel óptimo alcanzado.",
                        duracion=6,
                    )
            if z.humedad < UMBRAL_PARAR:
                self.ya_mostradas.discard(k_parar)

            # 4) EXCESO
            if z.humedad > UMBRAL_EXCESO:
                if k_exceso not in self.ya_mostradas:
                    self.ya_mostradas.add(k_exceso)
                    self.lanzar(
                        "warning", "💦", f"Exceso de Agua — {z.nombre}",
                        f"Humedad en {z.humedad:.0f}%. Riesgo de saturación del suelo.",
                        duracion=7,
                    )
            else:
                self.ya_mostradas.discard(k_exceso)

    # --------------------------------------------------------------
    def estado_global(self, zonas):
        if any(z.humedad < UMBRAL_CRITICO for z in zonas):
            return "CRÍTICO", (0.96, 0.26, 0.21)
        elif any(z.humedad < UMBRAL_BAJO for z in zonas):
            return "ALERTA", (1.00, 0.60, 0.00)
        return "NORMAL", (0.00, 0.90, 0.46)
