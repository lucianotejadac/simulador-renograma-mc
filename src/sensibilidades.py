"""Paso 3: imagen planar por compartimento (cuentas por MBq·s) en vista posterior y anterior, por Monte Carlo.

Cada compartimento del modelo tiene una distribución espacial fija (regiones del fantoma con pesos); se
normaliza a 1 MBq y se simula una vez por vista. Un renograma de cualquier caso es después una suma de estas
imágenes ponderadas por la actividad integrada en cada cuadro: los casos nuevos no requieren Monte Carlo.

Salida: salida/sensibilidades.npz con S[vista][comp] (128 × 128, cuentas por MBq·s) y la geometría.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import montecarlo as mc
from modelo_mag3 import COMPARTIMENTOS

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# pesos relativos por etiqueta de región (ver regiones.py)
PESOS = {
    "plasma": {11: 1.0, 9: 0.25, 10: 0.30, 4: 0.25, 6: 0.25, 5: 0.05, 7: 0.05, 12: 0.03, 1: 0.04, 2: 0.04, 8: 0.02, 13: 0.02},
    "intersticio": {1: 0.12, 9: 0.15, 10: 0.15, 4: 0.15, 6: 0.15, 12: 0.12, 2: 0.05},
    "corteza_d": {4: 1.0}, "corteza_i": {6: 1.0}, "pelvis_d": {5: 1.0}, "pelvis_i": {7: 1.0},
    "vejiga": {8: 1.0}, "higado": {9: 1.0}, "intestino": {13: 1.0, 12: 0.01},
}
MATRIZ, PIXEL_MM = 128, 4.8


def mapa(reg, pesos, vox_ml):
    a = np.zeros(reg.shape, np.float32)
    for et, w in pesos.items():
        a[reg == et] = w
    tot = a.sum() * vox_ml
    return a * (1000.0 / tot) if tot > 0 else a           # kBq/mL que suman 1 MBq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--historias", type=int, default=3_000_000)
    ap.add_argument("--semilla", type=int, default=11)
    a = ap.parse_args()
    f = np.load(os.path.join(RAIZ, "salida", "fantoma.npz"))
    mu, hu, iso = f["mu"], f["hu"], float(f["iso"])
    reg = np.load(os.path.join(RAIZ, "salida", "regiones.npz"))["reg"]
    vox_cm = iso / 10.0
    yy = np.nonzero((mu > 0.02).any(axis=(0, 2)))[0]
    cy = mu.shape[1] * vox_cm / 2.0
    dist = {"posterior": round(float((yy.max() + 1) * vox_cm - cy) + 2.0, 1), "anterior": round(float(cy - yy.min() * vox_cm) + 2.0, 1)}
    angulo = {"posterior": 90.0, "anterior": 270.0}           # normal del detector: +y es posterior en LPS
    S = {}
    info = {"distancia_cm": dist, "matriz": MATRIZ, "pixel_mm": PIXEL_MM, "historias": a.historias, "compartimentos": COMPARTIMENTOS, "segundos": {}}
    t0 = time.time()
    for vista in ("posterior", "anterior"):
        cam = mc.Camara(1, 360.0, angulo[vista], dist[vista], MATRIZ, PIXEL_MM, 1.0)
        for i, comp in enumerate(COMPARTIMENTOS):
            act = mapa(reg, PESOS[comp], vox_cm ** 3)
            if act.sum() == 0:
                S[f"{vista}_{comp}"] = np.zeros((MATRIZ, MATRIZ), np.float32)
                continue
            t1 = time.time()
            esp, _, est = mc.simular(act, mu, hu, vox_cm, cam, n_hist=a.historias, semilla=a.semilla + 100 * i + (0 if vista == "posterior" else 50), poisson=False)
            S[f"{vista}_{comp}"] = esp[0].astype(np.float32)          # cuentas por MBq·s
            info["segundos"][f"{vista}_{comp}"] = round(time.time() - t1, 1)
            print(f"{vista:9s} {comp:12s}: {esp[0].sum():8.1f} cuentas por MBq·s ({time.time() - t1:4.1f} s)", flush=True)
    np.savez_compressed(os.path.join(RAIZ, "salida", "sensibilidades.npz"), **S)
    json.dump(info, open(os.path.join(RAIZ, "salida", "sensibilidades.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"listo en {time.time() - t0:.0f} s; distancias {dist}")


if __name__ == "__main__":
    main()
