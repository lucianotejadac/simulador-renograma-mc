"""Paso 4: renograma dinámico de un caso: cuadros con ruido, regiones de interés y curvas.

Cuadro f (vista v) = Poisson( Σ_c  I[c, f] · S[v, c] ), con I la actividad integrada del compartimento en el
cuadro (MBq·s, modelo_mag3) y S la imagen por MBq·s del Monte Carlo (sensibilidades.npz).

Regiones de interés en la vista posterior: riñón = píxeles donde la imagen de la corteza supera el 20 % de su
máximo, ampliada 1 píxel; fondo = media luna inferolateral de 2 píxeles, separada 2 píxeles del riñón.
Mediciones sobre los cuadros con ruido, como en la clínica: función relativa por la integral de 1 a 2.5 min
corregida por fondo (normalizada por área), tiempo al máximo, T½ tras el máximo y C20/máx.

Salida en salida/casos/<caso>/: cuadros.npz (posterior y anterior, uint16), rois.npz, curvas.json y montaje.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy import ndimage

import modelo_mag3 as mm

from paciente import SALIDA, DOCS_DATOS, DOCS_DICOM  # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rois(S):
    out = {}
    for lado, comps in (("derecho", ("corteza_d", "pelvis_d")), ("izquierdo", ("corteza_i", "pelvis_i"))):
        img = S[f"posterior_{comps[0]}"]                      # la corteza define el contorno (la pelvis por MBq es un punto en el hilio)
        r = img > 0.20 * img.max()
        et, n = ndimage.label(r)
        if n > 1:
            r = et == (1 + int(np.argmax(ndimage.sum(r, et, range(1, n + 1)))))
        r = ndimage.binary_fill_holes(r)
        r = ndimage.binary_dilation(r, iterations=1)           # incluye el borde borroso del riñón
        afuera = ndimage.binary_dilation(r, iterations=4) & ~ndimage.binary_dilation(r, iterations=2)
        filas = np.nonzero(r.any(axis=1))[0]
        centro_v = filas.mean()
        cols = np.nonzero(r.any(axis=0))[0]
        centro_u = cols.mean()
        vv, uu = np.mgrid[:r.shape[0], :r.shape[1]]
        # media luna inferolateral: por debajo del centro (v menor = caudal) y hacia el lado lateral
        lateral = (uu - 64) * (uu - centro_u) >= 0
        fondo = afuera & ((vv <= centro_v) | lateral)
        out[lado] = r
        out[f"fondo_{lado}"] = fondo
    return out


def medir(curva_rinon, curva_fondo, area_r, area_f, t_centro, duraciones):
    """Curva corregida por fondo (cuentas/s) y parámetros clínicos."""
    neta = curva_rinon / duraciones - curva_fondo / duraciones * (area_r / max(area_f, 1))
    ventana = (t_centro >= 60) & (t_centro <= 150)
    integral = float((neta[ventana] * duraciones[ventana]).sum())
    suave = np.convolve(neta, np.ones(3) / 3, mode="same")
    tardio = t_centro > 60
    imax = int(np.argmax(np.where(tardio, suave, -np.inf)))
    pico = suave[imax]
    tras = np.nonzero(suave[imax:] <= pico / 2)[0]
    i20 = int(np.argmin(np.abs(t_centro - 1200)))
    return neta, {"integral_1_2_5": integral, "tmax_min": round(float(t_centro[imax] / 60), 1),
                  "t_medio_min": round(float((t_centro[imax + tras[0]] - t_centro[imax]) / 60), 1) if len(tras) else None,
                  "c20_sobre_max": round(float(suave[i20] / pico), 2) if pico > 0 else None}


def generar(nombre: str, semilla: int = 1):
    caso = mm.casos()[nombre]
    S = dict(np.load(os.path.join(RAIZ, SALIDA, "sensibilidades.npz")))
    dur = mm.protocolo()
    t, A = mm.simular_curvas(caso, t_fin_min=dur.sum() / 60.0)
    I = mm.integrar_cuadros(t, A, dur)                       # (comp, cuadro) MBq·s
    rng = np.random.default_rng(semilla)
    cuadros = {}
    for vista in ("posterior", "anterior"):
        esp = np.zeros((len(dur), 128, 128), np.float64)
        for c, comp in enumerate(mm.COMPARTIMENTOS):
            esp += I[c][:, None, None] * S[f"{vista}_{comp}"][None]
        cuadros[vista] = rng.poisson(esp).astype(np.uint16)
        cuadros[f"{vista}_esperado"] = esp.astype(np.float32)
    R = rois(S)
    t_centro = np.cumsum(dur) - dur / 2
    curvas, medidos = {}, {}
    for lado in ("derecho", "izquierdo"):
        r, fo = R[lado], R[f"fondo_{lado}"]
        cr = cuadros["posterior"][:, r].sum(axis=1).astype(float)
        cf = cuadros["posterior"][:, fo].sum(axis=1).astype(float)
        neta, med = medir(cr, cf, r.sum(), fo.sum(), t_centro, dur)
        curvas[lado] = [round(float(v), 2) for v in neta]
        curvas[f"bruta_{lado}"] = [int(v) for v in cr]
        medidos[lado] = med
    tot = sum(max(0.0, medidos[l]["integral_1_2_5"]) for l in ("derecho", "izquierdo"))
    for l in ("derecho", "izquierdo"):
        medidos[l]["funcion_relativa"] = round(100 * max(0.0, medidos[l]["integral_1_2_5"]) / tot, 1) if tot > 0 else None
        if medidos[l]["funcion_relativa"] is not None and medidos[l]["funcion_relativa"] < 5:      # sin captación: la curva es pool sanguíneo
            medidos[l].update({"tmax_min": None, "t_medio_min": None, "c20_sobre_max": None})
    vej = cuadros["posterior"].sum(axis=0)
    verdad = mm.verdad_clinica(caso, t, A)
    i_t = (np.arange(0, len(t), int(10 / (t[1] - t[0]))))       # curvas verdaderas cada 10 s
    carpeta = os.path.join(RAIZ, SALIDA, "casos", nombre)
    os.makedirs(carpeta, exist_ok=True)
    np.savez_compressed(os.path.join(carpeta, "cuadros.npz"), posterior=cuadros["posterior"], anterior=cuadros["anterior"], duraciones_s=dur)
    np.savez_compressed(os.path.join(carpeta, "rois.npz"), **{k: v.astype(np.uint8) for k, v in R.items()})
    json.dump({"caso": nombre, "descripcion": caso.descripcion, "parametros": {"derecho": vars(caso.derecho), "izquierdo": vars(caso.izquierdo),
               "actividad_MBq": caso.actividad_MBq, "t_furosemida_min": caso.t_furo_min},
               "duraciones_s": dur.tolist(), "t_centro_s": t_centro.tolist(), "curvas_medidas": curvas, "medidos": medidos, "verdad": verdad,
               "curvas_verdaderas_MBq": {"t_s": t[i_t].tolist(), **{comp: [round(float(v), 3) for v in A[c, i_t]] for c, comp in enumerate(mm.COMPARTIMENTOS)}},
               "cuentas_totales_posterior": int(cuadros["posterior"].sum()), "semilla": semilla},
              open(os.path.join(carpeta, "curvas.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    montaje(cuadros["posterior"], dur, R, os.path.join(carpeta, "montaje.png"))
    return medidos, verdad


def montaje(post, dur, R, ruta):
    from PIL import Image
    t_fin = np.cumsum(dur)
    tiempos = [60, 120, 180, 300, 600, 900, 1200, 1800]
    ims = []
    for tt in tiempos:
        sel = (t_fin > tt - 60) & (t_fin <= tt) if tt > 60 else (t_fin <= 60)
        img = post[sel].sum(axis=0).astype(float)[::-1]
        g = np.clip(img / (np.percentile(img, 99.5) or 1) * 255, 0, 255)
        rgb = np.stack([g, g, g], -1)
        for k, col in (("derecho", (80, 220, 255)), ("izquierdo", (255, 200, 60))):
            borde = R[k] & ~ndimage.binary_erosion(R[k])
            rgb[borde[::-1]] = col
        ims.append(Image.fromarray(rgb.astype(np.uint8)).resize((256, 256), Image.NEAREST))
    L = Image.new("RGB", (4 * 260, 2 * 260))
    for i, im in enumerate(ims):
        L.paste(im, ((i % 4) * 260, (i // 4) * 260))
    L.save(ruta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--casos", default=",".join(mm.casos()))
    ap.add_argument("--semilla", type=int, default=1)
    a = ap.parse_args()
    for nombre in a.casos.split(","):
        med, ver = generar(nombre, a.semilla)
        f = lambda d: {k: d[k] for k in ("funcion_relativa", "tmax_min", "t_medio_min", "c20_sobre_max")}
        print(f"{nombre}:\n   medido  D {f(med['derecho'])}  I {f(med['izquierdo'])}\n   verdad  D {ver['derecho']}  I {ver['izquierdo']}")


if __name__ == "__main__":
    main()
