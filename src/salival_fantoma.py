"""Fantoma de cabeza y cuello para la cintigrafía de glándulas salivales: hombre de referencia ICRP 145, a 2 mm.

Recorte: del vértice del cráneo a 3 cm bajo la tiroides. Reutiliza la lectura y la voxelización de icrp145.py.
El fantoma trae una glándula salival por lado (órganos 12000 izquierda y 12100 derecha); en la malla cada una
viene en piezas no conectadas, que se separan por posición: la pieza más craneal y posterior es la parótida,
la más caudal y anterior la submaxilar (las piezas menores se asignan a la más cercana).

Regiones: 0 aire, 1 tejido blando, 2 hueso, 3 aire interno, 4 parótida D, 5 parótida I, 6 submaxilar D,
7 submaxilar I, 8 tiroides, 9 boca (mucosa oral y lengua), 10 vasos, 11 cerebro.
Salida en salida_salival/: fantoma.npz, fantoma.json, regiones.npz, regiones.json, organos.npz.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
from scipy import ndimage

import icrp145 as ic
from fantoma import hu_a_mu

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALIDA = os.path.join(RAIZ, "salida_salival")
NOMBRES = ["aire", "tejido blando", "hueso", "aire interno", "parótida derecha", "parótida izquierda", "submaxilar derecha",
           "submaxilar izquierda", "tiroides", "boca", "vasos", "cerebro"]


def separar(glandula, iso):
    """Piezas de una glándula -> (parótida, submaxilar). Devuelve máscaras y el detalle de las piezas."""
    et, n = ndimage.label(glandula)
    piezas = []
    for i in range(1, n + 1):
        m = et == i
        z, y, x = np.nonzero(m)
        piezas.append({"i": i, "vol_ml": round(m.sum() * (iso / 10) ** 3, 2), "centro_zyx": [float(z.mean()), float(y.mean()), float(x.mean())]})
    piezas.sort(key=lambda p: -p["vol_ml"])
    if len(piezas) < 2:
        return glandula, np.zeros_like(glandula), piezas
    a, b = piezas[0], piezas[1]
    par, sub = (a, b) if a["centro_zyx"][0] > b["centro_zyx"][0] else (b, a)     # la parótida es más craneal
    mpar, msub = et == par["i"], et == sub["i"]
    for p in piezas[2:]:                                                          # piezas chicas: a la más cercana
        c = np.array(p["centro_zyx"])
        dpar = np.linalg.norm(c - np.array(par["centro_zyx"]))
        dsub = np.linalg.norm(c - np.array(sub["centro_zyx"]))
        (mpar if dpar < dsub else msub)[et == p["i"]] = True
    return mpar, msub, piezas


def main():
    os.makedirs(SALIDA, exist_ok=True)
    nodos, tets, ids = ic.leer()
    def centro(org):
        sel = ids == org
        return nodos[tets[sel].ravel()].mean(axis=0)
    ri, rd, vej, lum, cab = centro(8900), centro(9200), centro(13800), centro(5100), centro(6100)
    eje_lr = int(np.argmax(np.abs(ri - rd)))
    s_x = np.sign(ri[eje_lr] - rd[eje_lr])
    resto = [e for e in range(3) if e != eje_lr]
    eje_si = max(resto, key=lambda e: abs(cab[e] - vej[e]))
    s_z = np.sign(cab[eje_si] - vej[eje_si])
    eje_ap = [e for e in resto if e != eje_si][0]
    s_y = np.sign(lum[eje_ap] - vej[eje_ap])
    lps = np.stack([nodos[:, eje_lr] * s_x, nodos[:, eje_ap] * s_y, nodos[:, eje_si] * s_z], axis=1)
    def caja(orgs):
        p = lps[tets[np.isin(ids, orgs)].ravel()]
        return p.min(axis=0), p.max(axis=0)
    _, cmax = caja([2600, 2700, 6100])
    tmin, _ = caja([13200])
    zmin, zmax = tmin[2] - 3.0, cmax[2] + 1.0
    sel = (lps[:, 2] >= zmin) & (lps[:, 2] <= zmax)
    lo = lps[sel].min(axis=0) - 1.0
    hi = lps[sel].max(axis=0) + 1.0
    lo[2], hi[2] = zmin, zmax
    # solo la cabeza y el cuello en x: el recorte en z también corta los hombros; limitar a ±13 cm del centro de la cabeza
    cxh = cab[eje_lr] * s_x
    lo[0], hi[0] = max(lo[0], cxh - 13.0), min(hi[0], cxh + 13.0)
    n = np.ceil((hi - lo) / ic.ISO_CM).astype(int)
    vol = np.zeros((n[2], n[1], n[0]), np.uint16)
    t0 = time.time()
    ic._voxelizar(np.ascontiguousarray(lps), np.ascontiguousarray(tets), ids.astype(np.uint16), lo.astype(np.float64), ic.ISO_CM, n[2], n[1], n[0], vol)
    print(f"voxelizado {vol.shape} en {time.time() - t0:.0f} s", flush=True)
    orgs = np.unique(vol)
    rho = np.zeros(int(orgs.max()) + 1, np.float32)
    for o_ in orgs:
        rho[o_] = ic.densidad(int(o_)) if o_ else 0.0012
    dens = rho[vol]
    hu = np.round(1000.0 * (dens - 1.0)).astype(np.int16)
    hu[vol == 0] = -1000
    mu = hu_a_mu(hu)
    reg = np.zeros(vol.shape, np.uint8)
    reg[vol > 0] = 1
    reg[dens > 1.15] = 2
    reg[(dens < 0.5) & (vol > 0)] = 3
    reg[np.isin(vol, [500, 501, 600, 13300, 13301])] = 9
    reg[np.isin(vol, [900, 910])] = 10
    reg[vol == 6100] = 11
    reg[vol == 13200] = 8
    detalle = {}
    for org, (lpar, lsub), lado in ((12100, (4, 6), "derecha"), (12000, (5, 7), "izquierda")):
        mpar, msub, piezas = separar(vol == org, ic.ISO_CM * 10)
        reg[mpar] = lpar
        reg[msub] = lsub
        detalle[lado] = piezas
    iso = ic.ISO_CM * 10
    np.savez_compressed(os.path.join(SALIDA, "fantoma.npz"), hu=hu, mu=mu, iso=iso, origen=np.array(lo * 10.0))
    np.savez_compressed(os.path.join(SALIDA, "regiones.npz"), reg=reg)
    np.savez_compressed(os.path.join(SALIDA, "organos.npz"), organo=vol)
    vml = ic.ISO_CM ** 3
    volumen = {NOMBRES[i]: round(float((reg == i).sum()) * vml, 2) for i in range(len(NOMBRES))}
    json.dump({"fuente": "ICRP Publication 145, MRCP_AM, cabeza y cuello a 2 mm", "iso_mm": iso, "forma_zyx": list(vol.shape),
               "origen_mm": (lo * 10.0).tolist(), "sexo": "M", "edad": ""}, open(os.path.join(SALIDA, "fantoma.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump({"volumen_ml": volumen, "nombres": NOMBRES, "piezas_glandulas": detalle}, open(os.path.join(SALIDA, "regiones.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("volúmenes (mL):", volumen)
    print("piezas de las glándulas:", json.dumps(detalle, ensure_ascii=False))


if __name__ == "__main__":
    main()
