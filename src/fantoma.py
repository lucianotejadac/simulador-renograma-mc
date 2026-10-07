"""Paso 1: fantoma abdominal de atenuación desde el CT público (TCIA, caso 2 de la entrega docente PET/CT).

Recorta el CT entre z_sup y z_inf (por defecto de la cúpula hepática a bajo la vejiga), quita la camilla
(componente conexa mayor), remuestrea a vóxeles isotrópicos (borde por vecino más cercano: sin paredes de
0 HU) y convierte HU a mu a 140 keV con la conversión bilineal. Mismo método que simulador-paratiroides-mc.

Salida: salida/fantoma.npz (hu int16, mu float32 1/cm, iso, origen) y salida/fantoma.json.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pydicom
from scipy import ndimage

from paciente import SALIDA, DOCS_DATOS, DOCS_DICOM  # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CT_POR_DEFECTO = r"C:\Users\lucia\Downloads\PET CT\ENTREGA PET\PET Magdalena\Caso 2\CT"
MU_AGUA_140, MU_HUESO_140, HU_HUESO_REF = 0.1537, 0.2860, 1000.0


def hu_a_mu(hu):
    hu = hu.astype(np.float32)
    mu = np.where(hu <= 0.0, MU_AGUA_140 * (1.0 + hu / 1000.0), MU_AGUA_140 + (MU_HUESO_140 - MU_AGUA_140) * hu / HU_HUESO_REF)
    return np.clip(mu, 0.0, 0.6).astype(np.float32)


def leer_ct(carpeta):
    cortes = []
    for a in os.listdir(carpeta):
        d = pydicom.dcmread(os.path.join(carpeta, a), force=True)
        if getattr(d, "Modality", "") == "CT" and hasattr(d, "PixelData"):
            cortes.append(d)
    cortes.sort(key=lambda d: float(d.ImagePositionPatient[2]))
    hu = np.stack([d.pixel_array.astype(np.float32) * float(d.RescaleSlope) + float(d.RescaleIntercept) for d in cortes])
    z = np.array([float(d.ImagePositionPatient[2]) for d in cortes])
    return hu, z, [float(x) for x in cortes[0].PixelSpacing], [float(v) for v in cortes[0].ImagePositionPatient], cortes


def recortar(hu, z, ps, z_sup, z_inf):
    a = int(np.searchsorted(z, z_inf))
    b = int(np.searchsorted(z, z_sup, side="right"))
    sub = hu[a:b]
    cuerpo = ndimage.binary_opening(sub > -300, iterations=2)
    et, k = ndimage.label(cuerpo)
    if k > 1:
        tam = ndimage.sum(cuerpo, et, range(1, k + 1))
        cuerpo = et == (1 + int(np.argmax(tam)))
    cuerpo = np.stack([ndimage.binary_fill_holes(c) for c in cuerpo])
    ys, xs = np.nonzero(cuerpo.any(axis=0))
    m = int(round(15.0 / ps[0]))
    y0, y1 = max(0, ys.min() - m), min(sub.shape[1], ys.max() + m + 1)
    x0, x1 = max(0, xs.min() - m), min(sub.shape[2], xs.max() + m + 1)
    return np.where(cuerpo, sub, -1000.0)[:, y0:y1, x0:x1], (a, b, int(y0), int(y1), int(x0), int(x1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct", default=CT_POR_DEFECTO)
    ap.add_argument("--iso", type=float, default=2.0)
    ap.add_argument("--z-sup", type=float, default=-440.0, help="mm (LPS), borde craneal")
    ap.add_argument("--z-inf", type=float, default=-1000.0, help="mm (LPS), borde caudal")
    a = ap.parse_args()
    os.makedirs(os.path.join(RAIZ, SALIDA), exist_ok=True)
    hu, z, ps, origen, cortes = leer_ct(a.ct)
    dz = float(np.median(np.diff(z)))
    sub, (i_a, i_b, y0, y1, x0, x1) = recortar(hu, z, ps, a.z_sup, a.z_inf)
    iso = ndimage.zoom(sub, (dz / a.iso, ps[0] / a.iso, ps[1] / a.iso), order=1, mode="nearest").astype(np.float32)
    mu = hu_a_mu(iso)
    org = [origen[0] + x0 * ps[1], origen[1] + y0 * ps[0], float(z[i_a])]
    np.savez_compressed(os.path.join(RAIZ, SALIDA, "fantoma.npz"), hu=np.round(iso).astype(np.int16), mu=mu, iso=a.iso, origen=np.array(org))
    json.dump({"ct": a.ct, "iso_mm": a.iso, "forma_zyx": list(iso.shape), "origen_mm": org, "recorte_indices": [i_a, i_b, y0, y1, x0, x1],
               "ct_espaciado_mm": [dz, ps[0], ps[1]], "z_sup": a.z_sup, "z_inf": a.z_inf,
               "sexo": str(getattr(cortes[0], "PatientSex", "")), "edad": str(getattr(cortes[0], "PatientAge", ""))},
              open(os.path.join(RAIZ, SALIDA, "fantoma.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"fantoma {iso.shape} a {a.iso} mm ({iso.shape[0] * a.iso / 10:.0f} x {iso.shape[1] * a.iso / 10:.0f} x {iso.shape[2] * a.iso / 10:.0f} cm); "
          f"bordes: {[int(((iso > -300)[:, :, k]).sum()) for k in (0, -1)]} vóxeles de cuerpo en las columnas extremas")


if __name__ == "__main__":
    main()
