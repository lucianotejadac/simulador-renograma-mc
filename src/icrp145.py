"""Paciente estándar: fantoma masculino de referencia de la ICRP (Publicación 145, malla tetraédrica) a la grilla de 2 mm.

Fuente: ICRP Publication 145, archivos electrónicos (https://www.icrp.org/docs/P145%20Electronic%20files.zip):
MRCP_AM.node (nodos, cm) y MRCP_AM.ele (tetraedros con número de órgano). De acceso libre desde 2023.

Voxelización: un vóxel toma el órgano del tetraedro que contiene su centro (Numba, recorriendo cada tetraedro
sobre su caja de vóxeles). Densidades por órgano según la tabla de medios de la publicación; se guarda un
pseudo-HU = 1000 (ρ − 1) para reutilizar la conversión a μ y la bandera de hueso del motor (μ a 140 keV con
error < 2 % frente a ρ · (μ/ρ) del material).

Coordenadas del fantoma: se detectan los ejes con la anatomía (riñón izquierdo hacia +x LPS, columna posterior,
cabeza arriba) y se pasan a LPS. Recorte: de 3 cm bajo la vejiga a 10 cm sobre los riñones.

Salida (RENO_PACIENTE=icrp → salida_icrp/): fantoma.npz, fantoma.json, regiones.npz, regiones.json, organos.npz
(número de órgano ICRP por vóxel, uint16).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from numba import njit, prange

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATOS = os.environ.get("ICRP145_DIR", r"C:\Users\lucia\Documents\icrp145\Phantom_data\MRCP_AM")
SALIDA = os.path.join(RAIZ, "salida_icrp")
ISO_CM = 0.2

CORTICAL = {1300, 1900, 2200, 2400, 2600, 2800, 3400, 3700, 3900, 4100, 4300, 4500, 4700, 4900, 5100, 5300, 5500}
ESPONJOSA = {1400: 1.233, 1700: 1.109, 2000: 1.109, 2300: 1.109, 2500: 1.157, 2700: 1.165, 2900: 1.125, 3200: 1.109, 3500: 1.109,
             3800: 1.109, 4000: 1.271, 4200: 1.121, 4400: 1.170, 4600: 1.201, 4800: 1.049, 5000: 1.070, 5200: 1.108, 5400: 1.033, 5600: 1.041}
OTRAS = {1500: 0.981, 2100: 0.981, 3000: 0.981, 3600: 0.981, 5700: 1.099, 5800: 1.099, 9700: 0.415, 9900: 0.415, 14000: 0.001,
         11600: 0.939, 6200: 0.953, 6400: 0.953, 10600: 1.050, 9500: 1.060, 12700: 1.060, 900: 1.060, 910: 1.060, 8800: 1.060,
         8900: 1.053, 9000: 1.053, 9100: 1.053, 9200: 1.053, 9300: 1.053, 9400: 1.053, 13800: 1.040, 13700: 1.040, 12200: 1.089,
         12800: 2.688, 7100: 1.030, 7300: 1.040, 7500: 1.040, 7700: 1.040, 7900: 1.040, 8100: 1.040, 8300: 1.040, 8500: 1.040}
# órgano ICRP -> región del renograma (ver regiones.py: 4/5 corteza/pelvis D, 6/7 I, 8 vejiga, 9 hígado, 10 bazo,
# 11 vasos, 12 intestino, 13 vesícula, 14/15 uréter D/I)
REGION = {9200: 4, 9300: 4, 9400: 5, 8900: 6, 9000: 6, 9100: 7, 13800: 8, 9500: 9, 12700: 10, 900: 11, 910: 11, 8800: 11, 8700: 11,
          7000: 13, 7100: 13, 13600: 14, 13500: 15}
INTESTINO = range(7200, 8700)


def densidad(org):
    if org in CORTICAL:
        return 1.904
    if org in ESPONJOSA:
        return ESPONJOSA[org]
    return OTRAS.get(org, 1.04)


@njit(cache=True, parallel=True)
def _voxelizar(nodos, tets, ids, o, iso, nz, ny, nx, salida):
    for t in prange(tets.shape[0]):
        a, b, c, d = nodos[tets[t, 0]], nodos[tets[t, 1]], nodos[tets[t, 2]], nodos[tets[t, 3]]
        mn = np.minimum(np.minimum(a, b), np.minimum(c, d))
        mx = np.maximum(np.maximum(a, b), np.maximum(c, d))
        i0 = max(0, int(np.ceil((mn[0] - o[0]) / iso - 0.5))); i1 = min(nx - 1, int(np.floor((mx[0] - o[0]) / iso - 0.5)))
        j0 = max(0, int(np.ceil((mn[1] - o[1]) / iso - 0.5))); j1 = min(ny - 1, int(np.floor((mx[1] - o[1]) / iso - 0.5)))
        k0 = max(0, int(np.ceil((mn[2] - o[2]) / iso - 0.5))); k1 = min(nz - 1, int(np.floor((mx[2] - o[2]) / iso - 0.5)))
        if i0 > i1 or j0 > j1 or k0 > k1:
            continue
        # matriz de baricéntricas: p = a + M λ
        m00 = b[0] - a[0]; m01 = c[0] - a[0]; m02 = d[0] - a[0]
        m10 = b[1] - a[1]; m11 = c[1] - a[1]; m12 = d[1] - a[1]
        m20 = b[2] - a[2]; m21 = c[2] - a[2]; m22 = d[2] - a[2]
        det = m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20) + m02 * (m10 * m21 - m11 * m20)
        if abs(det) < 1e-12:
            continue
        inv = 1.0 / det
        for k in range(k0, k1 + 1):
            pz = o[2] + (k + 0.5) * iso - a[2]
            for j in range(j0, j1 + 1):
                py = o[1] + (j + 0.5) * iso - a[1]
                for i in range(i0, i1 + 1):
                    px = o[0] + (i + 0.5) * iso - a[0]
                    l1 = ((m11 * m22 - m12 * m21) * px - (m01 * m22 - m02 * m21) * py + (m01 * m12 - m02 * m11) * pz) * inv
                    l2 = (-(m10 * m22 - m12 * m20) * px + (m00 * m22 - m02 * m20) * py - (m00 * m12 - m02 * m10) * pz) * inv
                    l3 = ((m10 * m21 - m11 * m20) * px - (m00 * m21 - m01 * m20) * py + (m00 * m11 - m01 * m10) * pz) * inv
                    if l1 >= 0 and l2 >= 0 and l3 >= 0 and l1 + l2 + l3 <= 1:
                        salida[k, j, i] = ids[t]


def leer():
    import pandas as pd
    t0 = time.time()
    nodos = pd.read_csv(os.path.join(DATOS, "MRCP_AM.node"), sep=r"\s+", skiprows=1, header=None, comment="#", usecols=[1, 2, 3], dtype=np.float64).to_numpy()
    ele = pd.read_csv(os.path.join(DATOS, "MRCP_AM.ele"), sep=r"\s+", skiprows=1, header=None, comment="#", usecols=[1, 2, 3, 4, 5], dtype=np.int64).to_numpy()
    print(f"leídos {len(nodos)} nodos y {len(ele)} tetraedros en {time.time() - t0:.0f} s", flush=True)
    return nodos, ele[:, :4], ele[:, 4]


def main():
    os.makedirs(SALIDA, exist_ok=True)
    nodos, tets, ids = leer()
    def centro(org):
        sel = ids == org
        return nodos[tets[sel].ravel()].mean(axis=0)
    # ejes: riñón izquierdo hacia +x (LPS), columna (vértebras lumbares) posterior = +y, cabeza = +z
    ri, rd, vej, lum = centro(8900), centro(9200), centro(13800), centro(5100)
    eje_lr = int(np.argmax(np.abs(ri - rd)))
    s_x = np.sign(ri[eje_lr] - rd[eje_lr])
    resto = [e for e in range(3) if e != eje_lr]
    cab = centro(6100)                                   # cerebro
    eje_si = max(resto, key=lambda e: abs(cab[e] - vej[e]))
    s_z = np.sign(cab[eje_si] - vej[eje_si])
    eje_ap = [e for e in resto if e != eje_si][0]
    s_y = np.sign(lum[eje_ap] - vej[eje_ap])            # la columna es posterior a la vejiga
    lps = np.stack([nodos[:, eje_lr] * s_x, nodos[:, eje_ap] * s_y, nodos[:, eje_si] * s_z], axis=1)
    print(f"ejes del fantoma -> LPS: x = {'+-'[s_x < 0]}{'xyz'[eje_lr]}, y = {'+-'[s_y < 0]}{'xyz'[eje_ap]}, z = {'+-'[s_z < 0]}{'xyz'[eje_si]}")
    def caja(orgs):
        sel = np.isin(ids, orgs)
        p = lps[tets[sel].ravel()]
        return p.min(axis=0), p.max(axis=0)
    vmin, _ = caja([13700, 13800])
    _, rmax = caja([8900, 9200])
    zmin, zmax = vmin[2] - 3.0, rmax[2] + 10.0
    sel_tronco = (lps[:, 2] >= zmin) & (lps[:, 2] <= zmax)
    lo = lps[sel_tronco].min(axis=0) - 1.0
    hi = lps[sel_tronco].max(axis=0) + 1.0
    lo[2], hi[2] = zmin, zmax
    n = np.ceil((hi - lo) / ISO_CM).astype(int)          # (nx, ny, nz)
    vol = np.zeros((n[2], n[1], n[0]), np.uint16)
    t0 = time.time()
    _voxelizar(np.ascontiguousarray(lps), np.ascontiguousarray(tets), ids.astype(np.uint16), lo.astype(np.float64), ISO_CM, n[2], n[1], n[0], vol)
    print(f"voxelizado {vol.shape} en {time.time() - t0:.0f} s; vóxeles con órgano: {int((vol > 0).sum())}", flush=True)
    orgs = np.unique(vol)
    rho = np.zeros(int(orgs.max()) + 1, np.float32)
    for o_ in orgs:
        rho[o_] = densidad(int(o_)) if o_ else 0.0012
    dens = rho[vol]
    hu = np.round(1000.0 * (dens - 1.0)).astype(np.int16)
    hu[vol == 0] = -1000
    sys.path.insert(0, os.path.join(RAIZ, "src"))
    from fantoma import hu_a_mu
    mu = hu_a_mu(hu)
    reg = np.zeros(vol.shape, np.uint8)
    reg[vol > 0] = 1
    reg[(dens > 1.15)] = 2                               # hueso (cortical y esponjosa densa)
    reg[(dens < 0.5) & (vol > 0)] = 3                    # pulmón y aire
    for o_, r_ in REGION.items():
        reg[vol == o_] = r_
    reg[np.isin(vol, [o_ for o_ in INTESTINO])] = 12
    origen_mm = (lo * 10.0).tolist()
    np.savez_compressed(os.path.join(SALIDA, "fantoma.npz"), hu=hu, mu=mu, iso=ISO_CM * 10.0, origen=np.array(origen_mm))
    np.savez_compressed(os.path.join(SALIDA, "regiones.npz"), reg=reg)
    np.savez_compressed(os.path.join(SALIDA, "organos.npz"), organo=vol)
    from regiones import NOMBRES
    vml = (ISO_CM ** 3)
    volumen = {NOMBRES[i]: round(float((reg == i).sum()) * vml, 1) for i in range(len(NOMBRES))}
    json.dump({"fuente": "ICRP Publication 145, MRCP_AM (malla tetraédrica), voxelizado a 2 mm", "iso_mm": ISO_CM * 10, "forma_zyx": list(vol.shape),
               "origen_mm": origen_mm, "sexo": "M", "edad": "", "z_sup": float(zmax * 10), "z_inf": float(zmin * 10)},
              open(os.path.join(SALIDA, "fantoma.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump({"volumen_ml": volumen, "nombres": NOMBRES}, open(os.path.join(SALIDA, "regiones.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    masas = {k: round(float((vol == k).sum()) * vml * densidad(k), 1) for k in (8900, 9000, 9100, 9200, 9300, 9400, 13500, 13600, 13800, 9500)}
    print("volúmenes por región (mL):", volumen)
    print("masas por órgano ICRP (g):", masas)


if __name__ == "__main__":
    main()
