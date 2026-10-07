"""Cortes axiales del CT original (0.98 mm) con las etiquetas de las regiones, para la página.

Por corte del tomógrafo dentro del fantoma abdominal escribe:
  docs/datos/ct/NNN.png         CT recortado al cuerpo, ventana de abdomen (40/400), 8 bits
  docs/datos/etiquetas/NNN.png  etiquetas de regiones.npz (grilla de 2 mm) en ese corte, valor = número de región
  docs/datos/cortes.json        cantidad, tamaños, posición z de cada corte y nombres de las regiones
El fantoma es un remuestreo del mismo recorte, así que ambas imágenes cubren la misma extensión física.
"""
from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image

import fantoma
from regiones import NOMBRES

from paciente import SALIDA, DOCS_DATOS, DOCS_DICOM  # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    meta = json.load(open(os.path.join(RAIZ, SALIDA, "fantoma.json"), encoding="utf-8"))
    reg = np.load(os.path.join(RAIZ, SALIDA, "regiones.npz"))["reg"]
    iso = meta["iso_mm"]
    if "ct" in meta:                                     # paciente TCIA: cortes del CT original
        hu, z, ps, origen, _ = fantoma.leer_ct(meta["ct"])
        sub, (a, b, y0, y1, x0, x1) = fantoma.recortar(hu, z, ps, meta["z_sup"], meta["z_inf"])
        assert [a, b, y0, y1, x0, x1] == meta["recorte_indices"]
        zs = z[a:b]
    else:                                                # fantoma ICRP: no hay CT de tomógrafo; se usan sus propios cortes (pseudo-HU)
        f = np.load(os.path.join(RAIZ, SALIDA, "fantoma.npz"))
        sub = f["hu"].astype(np.float32)
        ps = [iso, iso]
        zs = meta["origen_mm"][2] + (np.arange(sub.shape[0]) + 0.5) * iso
    d_ct = os.path.join(RAIZ, DOCS_DATOS, "ct")
    d_et = os.path.join(RAIZ, DOCS_DATOS, "etiquetas")
    os.makedirs(d_ct, exist_ok=True)
    os.makedirs(d_et, exist_ok=True)
    for k in range(sub.shape[0]):
        g = np.clip((sub[k] - (40 - 200)) / 400.0 * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(g).save(os.path.join(d_ct, f"{k:03d}.png"), optimize=True)
        pz = int(round((zs[k] - zs[0]) / iso))
        pz = min(max(pz, 0), reg.shape[0] - 1)
        Image.fromarray(reg[pz].astype(np.uint8)).save(os.path.join(d_et, f"{k:03d}.png"), optimize=True)
    cuenta = [int(np.isin(np.asarray(Image.open(os.path.join(d_et, f"{k:03d}.png"))), [4, 5, 6, 7]).sum()) for k in range(sub.shape[0])]
    corte_rinones = int(np.argmax(cuenta))
    tam = sum(os.path.getsize(os.path.join(d, f)) for d in (d_ct, d_et) for f in os.listdir(d))
    json.dump({"n": int(sub.shape[0]), "alto": int(sub.shape[1]), "ancho": int(sub.shape[2]), "pixel_mm": ps[0], "espaciado_z_mm": float(np.median(np.diff(zs))),
               "z_mm": [round(float(v), 1) for v in zs], "etiquetas_alto": int(reg.shape[1]), "etiquetas_ancho": int(reg.shape[2]), "nombres": NOMBRES, "corte_rinones": corte_rinones},
              open(os.path.join(RAIZ, DOCS_DATOS, "cortes.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print(f"{sub.shape[0]} cortes de {sub.shape[1]} × {sub.shape[2]} a {ps[0]:.2f} mm; {tam / 1e6:.1f} MB en PNG")


if __name__ == "__main__":
    main()
