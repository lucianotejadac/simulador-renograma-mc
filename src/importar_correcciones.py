"""Aplica un archivo de correcciones de contornos descargado de la página a las regiones del fantoma.

El archivo trae, por corte del CT original, solo los píxeles cambiados como corridas [inicio, largo, etiqueta]
sobre la imagen de etiquetas de ese corte (misma grilla en el plano que el fantoma de 2 mm). Cada corte del CT
(3.27 mm) cubre uno o dos cortes del fantoma (2 mm): la corrección se aplica a todos los cortes del fantoma
cuyo centro cae en su espesor.

Uso:
    python src/importar_correcciones.py contornos-renograma-v1.json            # solo aplica y resume
    python src/importar_correcciones.py contornos-renograma-v1.json --rehacer  # y rehace todo el flujo
La versión anterior de las regiones queda en salida/regiones_antes_de_<fecha>.npz.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys

import numpy as np

from regiones import NOMBRES

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archivo")
    ap.add_argument("--rehacer", action="store_true")
    a = ap.parse_args()
    d = json.load(open(a.archivo, encoding="utf-8"))
    if d.get("formato") != "contornos-editados-v1" or d.get("meta", {}).get("pagina") != "simulador-renograma-mc":
        sys.exit("el archivo no es de correcciones del renograma")
    meta = d["meta"]
    f = np.load(os.path.join(RAIZ, "salida", "fantoma.npz"))
    iso, oz = float(f["iso"]), float(f["origen"][2])
    ruta = os.path.join(RAIZ, "salida", "regiones.npz")
    reg = np.load(ruta)["reg"]
    assert reg.shape[1:] == (meta["etiquetas_alto"], meta["etiquetas_ancho"]), "la grilla del archivo no coincide con la del fantoma"
    z = np.array(meta["z_mm"])
    dz = float(np.median(np.diff(z)))
    zf = oz + np.arange(reg.shape[0]) * iso
    antes = {n: int((reg == i).sum()) for i, n in enumerate(NOMBRES)}
    nuevo = reg.copy()
    cortes_f = set()
    for k, corridas in d["cortes"].items():
        k = int(k)
        plano = reg[int(np.argmin(np.abs(zf - z[k])))].ravel().copy()
        for ini, largo, et in corridas:
            plano[ini:ini + largo] = et
        plano = plano.reshape(reg.shape[1:])
        sel = np.nonzero(np.abs(zf - z[k]) <= dz / 2)[0]
        if len(sel) == 0:
            sel = [int(np.argmin(np.abs(zf - z[k])))]
        cambiados = plano != reg[int(np.argmin(np.abs(zf - z[k])))]
        for pz in sel:
            nuevo[pz][cambiados] = plano[cambiados]
            cortes_f.add(int(pz))
    copia = os.path.join(RAIZ, "salida", f"regiones_antes_de_{dt.datetime.now():%Y%m%d_%H%M%S}.npz")
    shutil.copy(ruta, copia)
    np.savez_compressed(ruta, reg=nuevo)
    ml = iso ** 3 / 1000.0
    print(f"{len(d['cortes'])} cortes del CT corregidos -> {len(cortes_f)} cortes del fantoma; copia anterior en {copia}")
    for i, n in enumerate(NOMBRES):
        dv = (int((nuevo == i).sum()) - antes[n]) * ml
        if abs(dv) > 0.001:
            print(f"   {n:24s} {antes[n] * ml:8.1f} mL -> {antes[n] * ml + dv:8.1f} mL ({dv:+.1f})")
    if a.rehacer:
        py = sys.executable
        for paso in ("sensibilidades.py", "dinamico.py", "exportar.py", "cortes_web.py"):
            print("--", paso, flush=True)
            subprocess.run([py, os.path.join(RAIZ, "src", paso)], check=True)


if __name__ == "__main__":
    main()
