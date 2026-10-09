"""Cintigrama óseo trifásico: fantoma de cuerpo entero (hombre de referencia ICRP 145) a 3 mm y recorte de piernas y pies a 2 mm.

Etiquetas: 0 aire, 1 tejido blando, 2 hueso cortical, 3 hueso esponjoso, 4 cavidad medular, 5 pulmón/aire,
6 riñones, 7 contenido vesical, 8 sangre de grandes vasos y corazón, 9 hígado, 10 bazo, 11 lesión ósea,
12 partes blandas de la lesión (hiperemia y edema).

Lesión (un solo caso, osteomielitis): esfera de 9 mm de radio en la cabeza del primer metatarsiano izquierdo, ubicada
desde la anatomía del pie (el fantoma trae todos los huesos del tobillo y el pie como un solo órgano): a lo largo del
eje del pie, a 70 % del talón a la punta, en el borde medial y a la altura de la cara plantar del antepié. Las partes
blandas dentro de 16 mm del centro forman la zona inflamada.

Salida en salida_osea/: cuerpo.npz y piernas.npz (hu, mu, reg, iso, origen) y lesion.json.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

import icrp145 as ic
from fantoma import hu_a_mu

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALIDA = os.path.join(RAIZ, "salida_osea")
NOMBRES = ["aire", "tejido blando", "hueso cortical", "hueso esponjoso", "cavidad medular", "pulmón/aire", "riñones", "vejiga",
           "sangre (grandes vasos y corazón)", "hígado", "bazo", "lesión ósea", "lesión: partes blandas"]
ESPONJOSA_IDS = set(ic.ESPONJOSA) | {2900, 3200, 3500, 3800}
MEDULAR_IDS = {1500, 2100, 3000, 3600}


def orientar():
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
    return lps, tets, ids


def voxelizar(lps, tets, ids, lo, hi, iso_cm):
    n = np.ceil((hi - lo) / iso_cm).astype(int)
    vol = np.zeros((n[2], n[1], n[0]), np.uint16)
    ic._voxelizar(np.ascontiguousarray(lps), np.ascontiguousarray(tets), ids.astype(np.uint16), lo.astype(np.float64), iso_cm, n[2], n[1], n[0], vol)
    return vol


def etiquetar(vol):
    orgs = np.unique(vol)
    rho = np.zeros(int(orgs.max()) + 1, np.float32)
    for o_ in orgs:
        rho[o_] = ic.densidad(int(o_)) if o_ else 0.0012
    dens = rho[vol]
    hu = np.round(1000.0 * (dens - 1.0)).astype(np.int16)
    hu[vol == 0] = -1000
    reg = np.zeros(vol.shape, np.uint8)
    reg[vol > 0] = 1
    reg[np.isin(vol, list(ic.CORTICAL))] = 2
    reg[np.isin(vol, list(ESPONJOSA_IDS))] = 3
    reg[np.isin(vol, list(MEDULAR_IDS))] = 4
    reg[(dens < 0.5) & (vol > 0)] = 5
    reg[np.isin(vol, [8900, 9000, 9100, 9200, 9300, 9400])] = 6
    reg[vol == 13800] = 7
    reg[np.isin(vol, [900, 910, 8800])] = 8
    reg[vol == 9500] = 9
    reg[vol == 12700] = 10
    return hu, hu_a_mu(hu), reg


def ubicar_lesion(reg, vol, iso_mm):
    """Cabeza del primer metatarsiano izquierdo en la grilla de piernas (índices z, y, x)."""
    pie = np.isin(vol, [3700, 3800])
    zs, ys, xs = np.nonzero(pie)
    izq = xs > np.median(xs)                                  # pie izquierdo: x mayor (LPS)
    zs, ys, xs = zs[izq], ys[izq], xs[izq]
    talon, punta = ys.max(), ys.min()                         # y crece hacia posterior
    y_c = talon - 0.70 * (talon - punta)
    banda = np.abs(ys - y_c) <= 4
    x_med = np.percentile(xs[banda], 12)                      # borde medial del pie izquierdo: x menor
    sel = banda & (np.abs(xs - x_med) <= 5)
    z_c = np.percentile(zs[sel], 35)
    return np.array([z_c, y_c, x_med], float)


def main():
    os.makedirs(SALIDA, exist_ok=True)
    lps, tets, ids = orientar()
    t0 = time.time()
    lo_c, hi_c = lps.min(axis=0) - 1.0, lps.max(axis=0) + 1.0
    cuerpo = voxelizar(lps, tets, ids, lo_c, hi_c, 0.3)
    print(f"cuerpo entero {cuerpo.shape} a 3 mm ({time.time() - t0:.0f} s)", flush=True)
    zmin = lps[:, 2].min()
    lo_p, hi_p = lo_c.copy(), hi_c.copy()
    lo_p[2], hi_p[2] = zmin - 1.0, zmin + 45.0
    selp = lps[:, 2] <= zmin + 45.0
    lo_p[:2], hi_p[:2] = lps[selp, :2].min(axis=0) - 1.5, lps[selp, :2].max(axis=0) + 1.5
    piernas = voxelizar(lps, tets, ids, lo_p, hi_p, 0.2)
    print(f"piernas y pies {piernas.shape} a 2 mm", flush=True)
    hu_p, mu_p, reg_p = etiquetar(piernas)
    c = ubicar_lesion(reg_p, piernas, 2.0)
    z, y, x = np.ogrid[:reg_p.shape[0], :reg_p.shape[1], :reg_p.shape[2]]
    d2 = (z - c[0]) ** 2 + (y - c[1]) ** 2 + (x - c[2]) ** 2
    hueso = np.isin(reg_p, [2, 3, 4])
    reg_p[(d2 <= (9 / 2.0) ** 2) & hueso] = 11
    reg_p[(d2 <= (16 / 2.0) ** 2) & (reg_p == 1)] = 12
    hu_c, mu_c, reg_c = etiquetar(cuerpo)
    # la lesión también en el cuerpo entero (3 mm): mismas coordenadas físicas
    c_mm = lo_p[::-1] * 10 + (c + 0.5) * 2.0                  # (z, y, x) en mm
    c_c = (c_mm - lo_c[::-1] * 10) / 3.0 - 0.5
    z, y, x = np.ogrid[:reg_c.shape[0], :reg_c.shape[1], :reg_c.shape[2]]
    d2c = (z - c_c[0]) ** 2 + (y - c_c[1]) ** 2 + (x - c_c[2]) ** 2
    reg_c[(d2c <= (9 / 3.0) ** 2) & np.isin(reg_c, [2, 3, 4])] = 11
    reg_c[(d2c <= (16 / 3.0) ** 2) & (reg_c == 1)] = 12
    for nombre, hu, mu, reg, lo, iso in (("cuerpo", hu_c, mu_c, reg_c, lo_c, 3.0), ("piernas", hu_p, mu_p, reg_p, lo_p, 2.0)):
        np.savez_compressed(os.path.join(SALIDA, f"{nombre}.npz"), hu=hu, mu=mu, reg=reg, iso=iso, origen=np.array(lo * 10.0))
        vml = (iso / 10) ** 3
        print(nombre, {NOMBRES[i]: round(float((reg == i).sum()) * vml, 1) for i in range(len(NOMBRES))})
    json.dump({"caso": "osteomielitis", "descripcion": "Osteomielitis de la cabeza del primer metatarsiano izquierdo.",
               "centro_piernas_zyx": c.tolist(), "centro_cuerpo_zyx": c_c.tolist(), "centro_mm_zyx": c_mm.tolist(),
               "radio_oseo_mm": 9, "radio_inflamacion_mm": 16, "nombres": NOMBRES},
              open(os.path.join(SALIDA, "lesion.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("lesión en piernas (z, y, x):", np.round(c, 1))


if __name__ == "__main__":
    main()
