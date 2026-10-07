"""Paso 2: regiones del fantoma abdominal con TotalSegmentator (Wasserthal et al., 2023) y umbral.

Corre TotalSegmentator en CPU sobre el fantoma exportado a NIfTI con su geometría exacta (la salida cae en la
misma grilla). Usa el ejecutable indicado en la variable TOTALSEG_EXE o el del entorno actual.

Etiquetas: 0 aire, 1 tejido blando, 2 hueso, 3 gas/pulmón, 4 corteza renal derecha, 5 pelvis renal derecha,
6 corteza renal izquierda, 7 pelvis renal izquierda, 8 vejiga, 9 hígado, 10 bazo, 11 vasos (aorta, cava,
ilíacas, corazón), 12 intestino (duodeno, delgado, colon, estómago), 13 vesícula.

Pelvis renal: dentro de cada riñón, los vóxeles de densidad de orina (−10 a 18 HU) del lado medial (hilio). En CT
sin contraste un sistema colector normal no se distingue: si da menos de 2 mL se usa una pelvis geométrica
(riñón a ≤ 13 mm del hilio).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOMBRES = ["aire", "tejido blando", "hueso", "gas/pulmón", "corteza renal derecha", "pelvis renal derecha", "corteza renal izquierda",
           "pelvis renal izquierda", "vejiga", "hígado", "bazo", "vasos", "intestino", "vesícula"]
ESTRUCTURAS = ["kidney_left", "kidney_right", "urinary_bladder", "liver", "spleen", "gallbladder", "stomach", "pancreas", "duodenum",
               "small_bowel", "colon", "aorta", "inferior_vena_cava", "heart", "iliac_artery_left", "iliac_artery_right",
               "iliac_vena_left", "iliac_vena_right", "adrenal_gland_left", "adrenal_gland_right",
               "lung_lower_lobe_left", "lung_lower_lobe_right"]


def correr_totalseg(hu, iso, origen, d):
    os.makedirs(d, exist_ok=True)
    aff = np.array([[-iso, 0, 0, -origen[0]], [0, -iso, 0, -origen[1]], [0, 0, iso, origen[2]], [0, 0, 0, 1]], float)
    nib.save(nib.Nifti1Image(np.transpose(hu.astype(np.int16), (2, 1, 0)), aff), os.path.join(d, "abdomen_ct.nii.gz"))
    exe = os.environ.get("TOTALSEG_EXE", os.path.join(os.path.dirname(sys.executable), "TotalSegmentator"))
    r = subprocess.run([exe, "-i", os.path.join(d, "abdomen_ct.nii.gz"), "-o", os.path.join(d, "total.nii.gz"), "--ml", "-d", "cpu",
                        "--roi_subset"] + ESTRUCTURAS, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-1500:])
        sys.exit(1)
    return np.transpose(np.asarray(nib.load(os.path.join(d, "total.nii.gz")).dataobj).astype(np.uint8), (2, 1, 0))


def pelvis(rinon, hu, iso, lado_medial):
    """lado_medial: +1 si el hilio está hacia x creciente (riñón derecho: la línea media queda a la izquierda del paciente)."""
    zs, ys, xs = np.nonzero(rinon)
    cx = xs.mean()
    medial = np.zeros_like(rinon)
    medial[zs, ys, xs] = ((xs - cx) * lado_medial) > 0
    p = rinon & medial & (hu > -10) & (hu < 18)        # orina; la grasa del seno renal (< -50 HU) queda fuera
    p = ndimage.binary_opening(p, iterations=1)
    et, n = ndimage.label(p)
    if n:
        tam = ndimage.sum(p, et, range(1, n + 1))
        p = et == (1 + int(np.argmax(tam)))
    if p.sum() * iso ** 3 / 1000.0 < 2.0:
        # sin contraste el sistema colector no dilatado no se ve: pelvis geométrica = riñón a ≤ 13 mm del hilio
        # (hilio = centro de los vóxeles más mediales del tercio medio del riñón). Unos 6–10 mL.
        zmed = (zs >= np.percentile(zs, 33)) & (zs <= np.percentile(zs, 67))
        prof = (xs - cx) * lado_medial
        sel = zmed & (prof >= np.percentile(prof[zmed], 90))
        h = (zs[sel].mean(), ys[sel].mean(), xs[sel].mean())
        r = 13.0 / iso
        p = np.zeros_like(rinon)
        cerca = (zs - h[0]) ** 2 + (ys - h[1]) ** 2 + (xs - h[2]) ** 2 <= r * r
        p[zs[cerca], ys[cerca], xs[cerca]] = True
    return p


def main():
    f = np.load(os.path.join(RAIZ, "salida", "fantoma.npz"))
    hu, iso, origen = f["hu"], float(f["iso"]), [float(v) for v in f["origen"]]
    d = os.path.join(RAIZ, "salida", "totalseg")
    ruta_et = os.path.join(d, "etiquetas.npz")
    if os.path.exists(ruta_et) and np.load(ruta_et)["et"].shape == hu.shape:
        et = np.load(ruta_et)["et"]
    else:
        et = correr_totalseg(hu, iso, origen, d)
        np.savez_compressed(ruta_et, et=et)
    from totalsegmentator.map_to_binary import class_map
    nom = {v: int(k) for k, v in class_map["total"].items()}

    def m(*ns):
        return np.isin(et, [nom[n] for n in ns if n in nom])

    cuerpo = np.stack([ndimage.binary_fill_holes(c) for c in (hu > -300)])
    reg = np.zeros(hu.shape, np.uint8)
    reg[cuerpo] = 1
    reg[cuerpo & (hu < -300)] = 3
    reg[cuerpo & (hu > 200)] = 2
    reg[m("aorta", "inferior_vena_cava", "heart", "iliac_artery_left", "iliac_artery_right", "iliac_vena_left", "iliac_vena_right")] = 11
    reg[m("duodenum", "small_bowel", "colon", "stomach") & (hu > -300)] = 12
    reg[m("liver")] = 9
    reg[m("spleen")] = 10
    reg[m("gallbladder")] = 13
    reg[m("urinary_bladder")] = 8
    # riñón derecho del paciente: x menor (LPS: x crece hacia la izquierda del paciente); su hilio mira a x creciente
    for nombre, cort, pel, lado in (("kidney_right", 4, 5, +1), ("kidney_left", 6, 7, -1)):
        r = m(nombre)
        if not r.any():
            continue
        p = pelvis(r, hu, iso, lado)
        reg[r] = cort
        reg[p] = pel
    np.savez_compressed(os.path.join(RAIZ, "salida", "regiones.npz"), reg=reg)
    vol = {NOMBRES[i]: round(float((reg == i).sum()) * iso ** 3 / 1000.0, 1) for i in range(len(NOMBRES))}
    json.dump({"volumen_ml": vol, "nombres": NOMBRES}, open(os.path.join(RAIZ, "salida", "regiones.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("volúmenes (mL):", vol)


if __name__ == "__main__":
    main()
