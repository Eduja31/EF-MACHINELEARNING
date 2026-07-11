"""
app.py  —  Sistema de Gestión Hídrica Inteligente
Backend Flask con API REST y Server-Sent Events (SSE) para tiempo real.

Rutas:
  GET  /                  → Sirve el portal web (index.html)
  GET  /api/estado        → JSON con el estado actual de todas las zonas
  GET  /api/estadisticas  → JSON con el análisis estadístico (pandas)
  GET  /api/anomalias/<id>→ JSON con anomalías detectadas por zona
  GET  /api/pronostico/<id>→JSON con pronóstico ML de la zona
  POST /api/accion        → Ejecutar una acción (riego, modo auto, etc.)
  GET  /stream            → Server-Sent Events (actualizaciones en tiempo real)
"""

import json
import random
import time
import threading
from flask import Flask, render_template, jsonify, request, Response, stream_with_context

from data_model import crear_zonas_iniciales, HistorialDatos, UMBRAL_CRITICO, UMBRAL_BAJO, UMBRAL_PARAR, UMBRAL_EXCESO
from alerts import GestorAlertas

app = Flask(__name__)

# ── Estado global del sistema ──────────────────────────────────────────
zonas          = crear_zonas_iniciales()
historial      = HistorialDatos(zonas)
alertas        = GestorAlertas()
riego_manual   = False
modo_auto      = False
agua_ahorrada  = 0.0
sync_count     = 0
_lock          = threading.Lock()


# ── Hilo de simulación en background ──────────────────────────────────
def hilo_simulacion():
    global riego_manual, modo_auto, agua_ahorrada, sync_count
    while True:
        time.sleep(2)
        with _lock:
            sync_count += 1
            for z in zonas:
                drift = random.uniform(-0.45, 0.38)
                z.humedad = max(15, min(95, z.humedad + drift))
                z.temperatura += random.uniform(-0.1, 0.1)
                z.temperatura = round(max(18, min(35, z.temperatura)), 1)

                riega = riego_manual or (modo_auto and z.id in alertas.zonas_riego_auto)
                if riega:
                    z.humedad = min(95, z.humedad + 0.5)

            if riego_manual or alertas.zonas_riego_auto:
                agua_ahorrada += 0.05

            historial.registrar_lectura()
            alertas.evaluar_zonas(zonas, riego_manual)
            alertas.limpiar_expiradas()

thread = threading.Thread(target=hilo_simulacion, daemon=True)
thread.start()


# ── Helper: serializar zona ────────────────────────────────────────────
def zona_a_dict(z):
    return {
        "id":          z.id,
        "nombre":      z.nombre,
        "cultivo":     z.cultivo,
        "humedad":     round(z.humedad, 1),
        "temperatura": round(z.temperatura, 1),
        "estado":      z.estado(),
        "color":       z.color_estado(),
        "valvula":     riego_manual or z.id in alertas.zonas_riego_auto,
    }


# ── Rutas API ──────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/estado")
def api_estado():
    with _lock:
        estado_txt, estado_col = alertas.estado_global(zonas)
        avg_hum = sum(z.humedad for z in zonas) / len(zonas)
        return jsonify({
            "zonas":         [zona_a_dict(z) for z in zonas],
            "estado_global": estado_txt,
            "estado_color":  estado_col,
            "avg_humedad":   round(avg_hum, 1),
            "riego_manual":  riego_manual,
            "modo_auto":     modo_auto,
            "agua_ahorrada": round(agua_ahorrada, 1),
            "sync_count":    sync_count,
            "alertas": [
                {
                    "tipo":    a.tipo,
                    "icono":   a.icono,
                    "titulo":  a.titulo,
                    "mensaje": a.mensaje,
                    "progreso": round(1 - a.progreso(), 3),
                    "color":   a.color(),
                }
                for a in alertas.alertas_activas[-5:]
            ],
            "eventos": alertas.eventos_log[:10],
        })


@app.route("/api/estadisticas")
def api_estadisticas():
    with _lock:
        df = historial.estadisticas_por_zona()
        if df.empty:
            return jsonify([])
        registros = []
        for zona_nombre, row in df.iterrows():
            registros.append({
                "zona":               zona_nombre,
                "humedad_media":      row["humedad_media"],
                "humedad_std":        row["humedad_std"],
                "humedad_min":        row["humedad_min"],
                "humedad_max":        row["humedad_max"],
                "temp_media":         row["temp_media"],
                "n_lecturas":         int(row["n_lecturas"]),
                "pct_tiempo_critico": row["pct_tiempo_critico"],
            })
        return jsonify(registros)


@app.route("/api/anomalias/<int:zona_id>")
def api_anomalias(zona_id):
    with _lock:
        df = historial.detectar_anomalias(zona_id)
        if df.empty:
            return jsonify([])
        result = []
        for _, row in df.tail(8).iterrows():
            result.append({
                "datetime": row["datetime"].strftime("%d/%m %H:%M"),
                "zona":     row["zona"],
                "humedad":  row["humedad"],
                "z_score":  round(row["z_score"], 2),
            })
        return jsonify(result)


@app.route("/api/pronostico/<int:zona_id>")
def api_pronostico(zona_id):
    with _lock:
        pron = historial.pronostico_lineal(zona_id)
        return jsonify([{"horas": h, "valor": v} for h, v in pron])


@app.route("/api/precision")
def api_precision():
    with _lock:
        p = historial.precision_modelo_simulada()
        return jsonify({"precision": p})


@app.route("/api/accion", methods=["POST"])
def api_accion():
    global riego_manual, modo_auto
    data = request.get_json()
    accion = data.get("accion", "")
    with _lock:
        if accion == "toggle_riego":
            riego_manual = not riego_manual
            msg = "Riego manual activado" if riego_manual else "Riego manual detenido"
            alertas.lanzar(
                "info" if riego_manual else "ok",
                "💧" if riego_manual else "🛑",
                msg, "", duracion=4
            )
        elif accion == "toggle_auto":
            modo_auto = not modo_auto
            alertas.modo_auto = modo_auto
            if modo_auto:
                alertas.lanzar("auto", "🤖", "Modo Automático Activado",
                                f"El sistema regará si la humedad cae bajo {UMBRAL_CRITICO}%.", duracion=6)
            else:
                alertas.zonas_riego_auto.clear()
                alertas.lanzar("info", "👤", "Modo Manual Activado",
                                "El riego requiere intervención manual.", duracion=4)
        elif accion == "prediccion_ml":
            for z in zonas:
                z.humedad = max(20, min(95, z.humedad + random.uniform(-5, 5)))
            prec = historial.precision_modelo_simulada()
            alertas.lanzar("info", "🤖", "Predicción ML Ejecutada",
                            f"Modelo ejecutado · Precisión: {prec}%", duracion=5)
        elif accion == "estres_hidrico":
            zid = random.randint(0, len(zonas)-1)
            zonas[zid].humedad = 22
            alertas.ya_mostradas.discard(f"critico_{zid}")
            alertas.lanzar("critical", "🚨", f"Estrés Hídrico — {zonas[zid].nombre}",
                            "Humedad al 22%. Intervención inmediata.", duracion=8)
        elif accion == "ajustar_humedad":
            zid = data.get("zona_id")
            val = data.get("valor")
            if zid is not None and val is not None:
                zonas[zid].humedad = float(val)
        return jsonify({"ok": True, "riego_manual": riego_manual, "modo_auto": modo_auto})


# ── Server-Sent Events: push de estado cada 2 s ───────────────────────
@app.route("/stream")
def stream():
    def generar():
        while True:
            time.sleep(2)
            with _lock:
                estado_txt, estado_col = alertas.estado_global(zonas)
                avg_hum = sum(z.humedad for z in zonas) / len(zonas)
                payload = {
                    "zonas":         [zona_a_dict(z) for z in zonas],
                    "estado_global": estado_txt,
                    "avg_humedad":   round(avg_hum, 1),
                    "riego_manual":  riego_manual,
                    "modo_auto":     modo_auto,
                    "agua_ahorrada": round(agua_ahorrada, 1),
                    "alertas": [
                        {
                            "tipo":    a.tipo,
                            "icono":   a.icono,
                            "titulo":  a.titulo,
                            "mensaje": a.mensaje,
                            "progreso": round(1 - a.progreso(), 3),
                            "color":   a.color(),
                        }
                        for a in alertas.alertas_activas[-5:]
                    ],
                    "eventos": alertas.eventos_log[:10],
                }
            yield f"data: {json.dumps(payload)}\n\n"
    return Response(stream_with_context(generar()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)
