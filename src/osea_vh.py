"""Cintigrama óseo trifásico sobre el CT del Visible Human masculino (National Library of Medicine, dominio público).

Base: CT del cadáver congelado (frozenCT), 1877 cortes de 1 mm de la cabeza a los pies, brazos a los lados, en PNG de
16 bits (HU + 1024) con una cabecera GE por corte. Campo de visión 270 mm en la cabeza, 400 y 480 mm en el resto, y tres
sesiones de escaneo (cortes 1006–1849, 1850–2658 y 2659–2882): cada corte se lleva a una grilla común de 480 mm con su
propio campo y centro, y los saltos entre sesiones se corrigen por correlación de fase de los cortes vecinos.
El tejido congelado tiene HU más bajos que el fresco (hielo ~ −80 HU): se corrige el tejido blando llevando su moda al
valor del músculo fresco antes de calcular atenuaciones.

Produce en salida_osea_vh/ los mismos archivos que osea_fantoma.py (cuerpo.npz, piernas.npz, lesion.json), con las
mismas etiquetas:
  0 aire, 1 tejido blando, 2 hueso cortical, 3 hueso esponjoso, 4 cavidad medular, 5 pulmón/gas, 6 riñones,
  7 vejiga, 8 sangre (corazón y grandes vasos), 9 hígado, 10 bazo, 11 lesión ósea, 12 lesión: partes blandas
Hueso por HU (≥ 150; cortical ≥ 600; médula = interior del contorno óseo de cada corte); órganos con TotalSegmentator.

Pasos: python src/osea_vh.py leer | organos | etiquetar
Cita requerida: «Imágenes del Visible Human Project, U.S. National Library of Medicine».
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys

import numpy as np
from scipy import ndimage as ndi

from fantoma import hu_a_mu

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALIDA = os.path.join(RAIZ, "salida_osea_vh")
VH = os.environ.get("VH_DIR", r"C:\Users\lucia\Documents\visible-human\male")
NOMBRES = ["aire", "tejido blando", "hueso cortical", "hueso esponjoso", "cavidad medular", "pulmón/gas", "riñones", "vejiga",
           "sangre (grandes vasos y corazón)", "hígado", "bazo", "lesión ósea", "lesión: partes blandas"]
ORGANOS_TS = {"kidney_left": 6, "kidney_right": 6, "urinary_bladder": 7, "heart": 8, "aorta": 8, "inferior_vena_cava": 8,
              "superior_vena_cava": 8, "portal_vein_and_splenic_vein": 8, "iliac_artery_left": 8, "iliac_artery_right": 8,
              "iliac_vena_left": 8, "iliac_vena_right": 8, "pulmonary_vein": 8, "liver": 9, "spleen": 10}
FOV_COMUN, N_COMUN = 480.0, 512
PX = FOV_COMUN / N_COMUN
PIE_MM = 120.0                                       # el pie en flexión plantar: del dedo más bajo al tobillo


def geometria():
    filas = {}
    for f in glob.glob(os.path.join(VH, "frozenCTHeaders", "c_vm*.fro.txt")):
        t = open(f, "rb").read().replace(b"\0", b"").decode("latin1")
        g = lambda p: re.search(p, t).group(1)
        n = int(re.findall(r"\d+", os.path.basename(f))[0])
        filas[n] = {"serie": g(r"CT Recon (HSUC/\d+/\d+)/"), "fov": float(g(r"Display Field of view - X \(mm\)\.*: ([0-9.]+)")),
                    "R": float(g(r"Center R coord of plane image\.*: ([-0-9.e+]+)")),
                    "A": float(g(r"Center A coord of plane image\.*: ([-0-9.e+]+)"))}
    return filas


def corte_comun(n, g):
    """Corte n en la grilla común (LPS: columnas hacia la izquierda del paciente, filas hacia posterior)."""
    from PIL import Image
    a = np.array(Image.open(os.path.join(VH, "frozenCT", f"cvm{n}f.png"))).astype(np.float32) - 1024.0
    px = g["fov"] / a.shape[1]
    c = (np.arange(N_COMUN) - (N_COMUN - 1) / 2) * PX
    yy, xx = np.meshgrid(c, c, indexing="ij")
    j = (xx + g["R"]) / px + (a.shape[1] - 1) / 2
    i = (yy + g["A"]) / px + (a.shape[0] - 1) / 2
    return ndi.map_coordinates(a, [i, j], order=1, cval=-1024.0)


def desplazamiento(a, b):
    """Traslación (filas, columnas) que lleva b sobre a, por correlación de fase de las máscaras de hueso y cuerpo."""
    fa = np.fft.fft2((a > 150).astype(float) + 0.3 * (a > -400))
    fb = np.fft.fft2((b > 150).astype(float) + 0.3 * (b > -400))
    q = fa * np.conj(fb)
    r = np.fft.ifft2(q / (np.abs(q) + 1e-9)).real
    i, j = np.unravel_index(np.argmax(r), r.shape)
    return [int(i if i < a.shape[0] // 2 else i - a.shape[0]), int(j if j < a.shape[1] // 2 else j - a.shape[1])]


def paso_leer():
    os.makedirs(SALIDA, exist_ok=True)
    geo = geometria()
    ns = sorted(geo)
    assert ns == list(range(ns[0], ns[-1] + 1)), "faltan cortes"
    vol = np.empty((len(ns), N_COMUN, N_COMUN), np.int16)
    corr, saltos, previo = [0, 0], [], None
    for k, n in enumerate(ns):
        s = corte_comun(n, geo[n])
        if previo is not None and geo[n]["serie"] != geo[ns[k - 1]]["serie"]:
            d = desplazamiento(previo, s)
            corr = [corr[0] + d[0], corr[1] + d[1]]
            saltos.append({"corte": n, "desplazamiento_px": d})
            print(f"cambio de sesión en {n}: desplazamiento {d} px", flush=True)
        if corr != [0, 0]:
            s = ndi.shift(s, corr, order=0, cval=-1024.0)
        previo = s
        vol[k] = np.round(s).astype(np.int16)
        if k % 300 == 0:
            print("corte", n, flush=True)
    vol = vol[::-1]                                       # índice 0 = pies (los cortes se numeran desde la cabeza)
    cuerpo = vol > -400
    cuerpo = ndi.binary_opening(cuerpo, structure=np.ones((1, 5, 5), bool))
    et, nn = ndi.label(cuerpo)
    tam = ndi.sum(cuerpo, et, range(1, nn + 1))
    cuerpo = et == (1 + int(np.argmax(tam)))
    for k in range(cuerpo.shape[0]):
        cuerpo[k] = ndi.binary_fill_holes(cuerpo[k])
    # objetos ajenos al cuerpo (un soporte denso junto a la rodilla izquierda): componentes de cada corte cuya
    # mediana supera 150 HU; solo bajo los hombros, para no tocar la calota, que en el vértice es casi solo hueso
    quitados, huella, zq = 0, np.zeros(cuerpo.shape[1:], bool), []
    for k in range(int(0.7 * cuerpo.shape[0])):
        et, nn = ndi.label(cuerpo[k])
        for i in range(1, nn + 1):
            m_i = et == i
            if np.median(vol[k][m_i]) > 150:
                cuerpo[k][m_i] = False
                quitados += int(m_i.sum())
                huella |= m_i
                zq.append(k)
    if zq:
        # donde el objeto toca la pierna queda unido a ella: dentro de su huella se conserva solo lo que también es
        # cuerpo 5 mm más arriba y más abajo
        nucleo = huella.copy()
        ref = cuerpo.copy()
        for k in range(max(5, min(zq) - 20), min(cuerpo.shape[0] - 5, max(zq) + 21)):
            fuera = nucleo & ~(ref[k - 5] & ref[k + 5])
            if fuera.any():
                quitados += int((cuerpo[k] & fuera).sum())
                cuerpo[k] &= ~fuera
        # restos delgados del objeto: lo que cae casi entero en su huella
        huella = ndi.binary_dilation(huella, iterations=5)
        for k in range(max(0, min(zq) - 20), min(cuerpo.shape[0], max(zq) + 21)):
            et, nn = ndi.label(cuerpo[k])
            for i in range(1, nn + 1):
                m_i = et == i
                if huella[m_i].mean() >= 0.8:
                    cuerpo[k][m_i] = False
                    quitados += int(m_i.sum())
    print(f"objetos ajenos quitados: {quitados * PX * PX / 1000:.0f} mL", flush=True)
    hu = np.where(cuerpo, vol, -1000).astype(np.float32)
    # tejido congelado -> fresco: moda del tejido blando llevada a 40 HU, sin tocar el hueso
    sel = cuerpo & (hu > -250) & (hu < 150)
    h, b = np.histogram(hu[sel], bins=400, range=(-250, 150))
    moda = float(b[np.argmax(h)] + 0.5)
    dlt = 40.0 - moda
    hu[sel] += dlt * np.clip((150 - hu[sel]) / 100.0, 0, 1)
    zs, ys, xs = [np.nonzero(cuerpo.any(axis=ax))[0] for ax in ((1, 2), (0, 2), (0, 1))]
    m = 10
    z0, z1 = max(0, zs[0] - m), min(hu.shape[0], zs[-1] + 1 + m)
    y0, y1 = max(0, ys[0] - m), min(hu.shape[1], ys[-1] + 1 + m)
    x0, x1 = max(0, xs[0] - m), min(hu.shape[2], xs[-1] + 1 + m)
    hu = hu[z0:z1, y0:y1, x0:x1]
    info = {"base": "Visible Human masculino, CT congelado (NLM)", "cortes": len(ns), "pixel_mm": PX, "paso_mm": 1.0,
            "saltos_entre_sesiones": saltos, "moda_blando_congelado_HU": moda, "correccion_HU": dlt,
            "largo_cuerpo_mm": float(zs[-1] - zs[0] + 1), "recorte_zyx": [[int(z0), int(z1)], [int(y0), int(y1)], [int(x0), int(x1)]]}
    print(info, hu.shape, flush=True)
    c = ndi.zoom(hu, (1 / 3, PX / 3, PX / 3), order=1, mode="nearest", prefilter=False)
    p = ndi.zoom(hu[:450], (1 / 2, PX / 2, PX / 2), order=1, mode="nearest", prefilter=False)
    origen = np.zeros(3)
    np.savez_compressed(os.path.join(SALIDA, "hu_3mm.npz"), hu=np.round(c).astype(np.int16), origen=origen)
    np.savez_compressed(os.path.join(SALIDA, "hu_2mm.npz"), hu=np.round(p).astype(np.int16), origen=origen)
    json.dump(info, open(os.path.join(SALIDA, "serie.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    import nibabel as nib
    aff = np.diag([-3.0, -3.0, 3.0, 1.0])
    nib.save(nib.Nifti1Image(np.transpose(np.round(c).astype(np.int16), (2, 1, 0)), aff), os.path.join(SALIDA, "ct_3mm.nii.gz"))
    print("3 mm", c.shape, "· 2 mm", p.shape, flush=True)


def paso_organos():
    # interfaz de Python: el Control de aplicaciones de Windows bloquea el ejecutable TotalSegmentator.exe
    from totalsegmentator.python_api import totalsegmentator
    totalsegmentator(os.path.join(SALIDA, "ct_3mm.nii.gz"), os.path.join(SALIDA, "organos.nii.gz"), ml=True, fast=True,
                     device="cpu", roi_subset=list(ORGANOS_TS), quiet=True)
    print("TotalSegmentator listo")


def geometria_fresco():
    filas = {}
    for f in glob.glob(os.path.join(VH, "normalCTHeaders", "c_vm*.fre.txt")):
        t = open(f, "rb").read().replace(b"\0", b"").decode("latin1")
        g = lambda p: re.search(p, t).group(1)
        n = int(re.findall(r"\d+", os.path.basename(f))[0])
        filas[n] = {"serie": g(r"CT Recon (HSUC/\d+/\d+)/"), "fov": float(g(r"Display Field of view - X \(mm\)\.*: ([0-9.]+)")),
                    "R": float(g(r"Center R coord of plane image\.*: ([-0-9.e+]+)")),
                    "A": float(g(r"Center A coord of plane image\.*: ([-0-9.e+]+)"))}
    return filas


def paso_organos_fresco():
    """Órganos sobre el CT fresco (mejor contraste de partes blandas) y traslado al congelado.

    El CT fresco del mismo cadáver cubre de la cabeza al muslo en su primera serie (cortes 1013–1957, con pasos de 1, 3 y
    5 mm). Se lleva a la misma grilla recortada de 1 mm que el congelado (la numeración de los cortes es la misma en
    milímetros), se interpola en z, se remuestrea a 3 mm y se alinea con el congelado por correlación de fase 3D del
    hueso. TotalSegmentator corre sobre el fresco y las etiquetas, desplazadas, quedan en la grilla del congelado.
    """
    import nibabel as nib
    from PIL import Image
    from totalsegmentator.python_api import totalsegmentator
    info = json.load(open(os.path.join(SALIDA, "serie.json"), encoding="utf-8"))
    (z0, z1), (y0, y1), (x0, x1) = info["recorte_zyx"]
    n_ult = 2882                                          # último corte del congelado (índice 0 tras invertir)
    geo = geometria_fresco()
    ns = sorted(n for n in geo if geo[n]["serie"] == "HSUC/1174/2" and os.path.exists(os.path.join(VH, "normalCT", f"cvm{n}f.png")))
    cortes = {}
    for n in ns:
        a = np.array(Image.open(os.path.join(VH, "normalCT", f"cvm{n}f.png"))).astype(np.float32) - 1024.0
        g = geo[n]
        px = g["fov"] / a.shape[1]
        c = (np.arange(N_COMUN) - (N_COMUN - 1) / 2) * PX
        yy, xx = np.meshgrid(c[y0:y1], c[x0:x1], indexing="ij")
        j = (xx + g["R"]) / px + (a.shape[1] - 1) / 2
        i = (yy + g["A"]) / px + (a.shape[0] - 1) / 2
        cortes[n] = ndi.map_coordinates(a, [i, j], order=1, cval=-1024.0)
    # z de 1 mm en la grilla recortada del congelado: índice = n_ult − n − z0
    zi = np.array([n_ult - n - z0 for n in ns], float)
    orden = np.argsort(zi)
    zi = zi[orden]
    pila = np.stack([cortes[ns[k]] for k in orden])
    nz = z1 - z0
    fres = np.full((nz, y1 - y0, x1 - x0), -1024.0, np.float32)
    for z in range(int(np.ceil(zi[0])), int(np.floor(zi[-1])) + 1):
        k = int(np.searchsorted(zi, z))
        if k < len(zi) and zi[k] == z:
            fres[z] = pila[k]
        else:
            w = (z - zi[k - 1]) / (zi[k] - zi[k - 1])
            fres[z] = (1 - w) * pila[k - 1] + w * pila[k]
    f3 = ndi.zoom(fres, (1 / 3, PX / 3, PX / 3), order=1, mode="nearest", prefilter=False)
    c3 = np.load(os.path.join(SALIDA, "hu_3mm.npz"))["hu"].astype(np.float32)
    f3 = f3[:c3.shape[0], :c3.shape[1], :c3.shape[2]]
    f3 = np.pad(f3, [(0, a - b) for a, b in zip(c3.shape, f3.shape)], constant_values=-1024)
    # alineación por el hueso (tramo cubierto por el fresco)
    zz = np.nonzero((f3 > -500).any(axis=(1, 2)))[0]
    sl = slice(zz.min(), zz.max() + 1)
    A = np.fft.fftn((c3[sl] > 200).astype(np.float32))
    Bf = np.fft.fftn((f3[sl] > 200).astype(np.float32))
    q = A * np.conj(Bf)
    r = np.fft.ifftn(q / (np.abs(q) + 1e-9)).real
    d = np.array(np.unravel_index(np.argmax(r), r.shape))
    d = np.where(d > np.array(r.shape) // 2, d - np.array(r.shape), d)
    print("desplazamiento fresco -> congelado (z, y, x) en vóxeles de 3 mm:", d.tolist(), flush=True)
    aff = np.diag([-3.0, -3.0, 3.0, 1.0])
    ruta = os.path.join(SALIDA, "fresco_3mm.nii.gz")
    nib.save(nib.Nifti1Image(np.transpose(np.round(f3).astype(np.int16), (2, 1, 0)), aff), ruta)
    totalsegmentator(ruta, os.path.join(SALIDA, "organos_fresco.nii.gz"), ml=True, fast=True, device="cpu",
                     roi_subset=list(ORGANOS_TS), quiet=True)
    et = np.asarray(nib.load(os.path.join(SALIDA, "organos_fresco.nii.gz")).dataobj).astype(np.uint8)
    et = np.transpose(et, (2, 1, 0))
    et = ndi.shift(et, d, order=0, cval=0)
    nib.save(nib.Nifti1Image(np.transpose(et, (2, 1, 0)), aff), os.path.join(SALIDA, "organos.nii.gz"))
    info["fresco_a_congelado_vox3mm"] = d.tolist()
    json.dump(info, open(os.path.join(SALIDA, "serie.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    np.savez_compressed(os.path.join(SALIDA, "fresco_3mm.npz"), hu=np.round(f3).astype(np.int16))
    print("órganos del fresco trasladados al congelado")


def huesos(hu, cuerpo, iso_mm=3.0):
    hueso = (hu >= 150) & cuerpo
    hueso = ndi.binary_opening(hueso, structure=np.ones((1, 2, 2), bool)) | (hu >= 400) & cuerpo
    # los brazos tocan el borde del campo de 480 mm: el truncamiento sube el tejido blando a 150–400 HU y deja un
    # reborde brillante en las últimas columnas. En una franja de 30 mm junto al borde lateral solo cuenta como hueso lo
    # que tiene 400 HU o más y está junto a la cortical del húmero (700 HU o más); el reborde de 2 columnas se descarta.
    franja = np.zeros(hu.shape[2], bool)
    n = int(round(30 / iso_mm))
    franja[:n] = franja[-n:] = True
    nucleo = (hu >= 700) & cuerpo
    nucleo[:, :, :2] = nucleo[:, :, -2:] = False
    nucleo = ndi.binary_dilation(nucleo, structure=np.ones((1, 3, 3), bool)) & (hu >= 400)
    hueso[:, :, franja] &= nucleo[:, :, franja]
    relleno = np.zeros_like(hueso)
    for k in range(hueso.shape[0]):
        relleno[k] = ndi.binary_fill_holes(hueso[k])
    return hueso, relleno & ~hueso


def etiquetar(hu, organos=None, iso_mm=3.0):
    cuerpo = hu > -500
    for k in range(cuerpo.shape[0]):
        cuerpo[k] = ndi.binary_fill_holes(cuerpo[k])
    reg = np.zeros(hu.shape, np.uint8)
    reg[cuerpo] = 1
    reg[cuerpo & (hu < -400)] = 5
    if organos is not None:
        for v in (8, 9, 10, 6, 7):                        # orden: los órganos chicos pisan a los grandes
            reg[organos == v] = v
    hueso, medula = huesos(hu, cuerpo, iso_mm)
    libre = ~np.isin(reg, [6, 7, 8, 9, 10])
    reg[medula & libre] = 4
    reg[hueso & libre & (hu < 600)] = 3
    reg[hueso & libre & (hu >= 600)] = 2
    return reg


def ubicar_lesion(reg, iso_mm):
    """Cabeza del primer metatarsiano izquierdo (índices z, y, x) en la grilla de 2 mm.

    Pie = hueso por debajo del tobillo (los 9 cm inferiores del hueso de cada pierna). Izquierdo: x mayor (LPS).
    Eje del pie: del punto óseo más posterior (talón) al más anterior (punta de los dedos), en el plano y–z.
    """
    hueso = np.isin(reg, [2, 3, 4])
    zs, ys, xs = np.nonzero(hueso)
    xm = np.median(xs[zs < zs.min() + PIE_MM / iso_mm])
    izq = (xs > xm) & (zs < zs[xs > xm].min() + PIE_MM / iso_mm)
    zs, ys, xs = zs[izq], ys[izq], xs[izq]
    i_t, i_p = int(np.argmax(ys)), int(np.argmin(ys))
    talon = np.array([zs[i_t], ys[i_t]], float)
    punta = np.array([zs[i_p], ys[i_p]], float)
    eje = punta - talon
    largo = float(np.linalg.norm(eje))
    s = ((np.stack([zs, ys], 1) - talon) @ eje) / largo ** 2      # 0 talón, 1 punta
    banda = np.abs(s - 0.70) <= 0.04                              # ±4 % del largo del pie
    x_med = np.percentile(xs[banda], 12)                          # borde medial del pie izquierdo: x menor
    sel = banda & (np.abs(xs - x_med) <= 5)
    zc, yc = np.median(zs[sel]), np.median(ys[sel])
    return np.array([zc, yc, x_med], float), {"largo_pie_mm": round(largo * iso_mm, 1), "talon_zy": talon.tolist(), "punta_zy": punta.tolist()}


def paso_etiquetar():
    import nibabel as nib
    from totalsegmentator.map_to_binary import class_map
    c = np.load(os.path.join(SALIDA, "hu_3mm.npz"))
    p = np.load(os.path.join(SALIDA, "hu_2mm.npz"))
    hu_c, hu_p, origen = c["hu"].astype(np.float32), p["hu"].astype(np.float32), c["origen"]
    ts = np.transpose(np.asarray(nib.load(os.path.join(SALIDA, "organos.nii.gz")).dataobj).astype(np.uint8), (2, 1, 0))
    assert ts.shape == hu_c.shape, (ts.shape, hu_c.shape)
    nombres = class_map["total"]
    org = np.zeros(ts.shape, np.uint8)
    for i in np.unique(ts):
        if i and nombres[int(i)] in ORGANOS_TS:
            org[ts == i] = ORGANOS_TS[nombres[int(i)]]
    reg_c = etiquetar(hu_c, org)
    # órganos en la grilla de 2 mm (vecino más cercano); en las piernas solo importan los vasos ilíacos, si llegan
    zf = np.array(hu_p.shape) / np.array([hu_p.shape[0] * 2 / 3, hu_c.shape[1], hu_c.shape[2]])
    org_p = ndi.zoom(org[:int(round(hu_p.shape[0] * 2 / 3))], zf, order=0, mode="nearest")[:hu_p.shape[0], :hu_p.shape[1], :hu_p.shape[2]]
    pad = [(0, a - b) for a, b in zip(hu_p.shape, org_p.shape)]
    org_p = np.pad(org_p, pad, mode="edge")
    reg_p = etiquetar(hu_p, org_p, 2.0)
    centro, pie = ubicar_lesion(reg_p, 2.0)
    z, y, x = np.ogrid[:reg_p.shape[0], :reg_p.shape[1], :reg_p.shape[2]]
    d2 = (z - centro[0]) ** 2 + (y - centro[1]) ** 2 + (x - centro[2]) ** 2
    reg_p[(d2 <= (9 / 2.0) ** 2) & np.isin(reg_p, [2, 3, 4])] = 11
    reg_p[(d2 <= (16 / 2.0) ** 2) & (reg_p == 1)] = 12
    c_mm = (centro + 0.5) * 2.0                                   # (z, y, x) mm desde el origen del recorte
    c_c = c_mm / 3.0 - 0.5
    z, y, x = np.ogrid[:reg_c.shape[0], :reg_c.shape[1], :reg_c.shape[2]]
    d2c = (z - c_c[0]) ** 2 + (y - c_c[1]) ** 2 + (x - c_c[2]) ** 2
    reg_c[(d2c <= (9 / 3.0) ** 2) & np.isin(reg_c, [2, 3, 4])] = 11
    reg_c[(d2c <= (16 / 3.0) ** 2) & (reg_c == 1)] = 12
    for nombre, hu, reg, iso in (("cuerpo", hu_c, reg_c, 3.0), ("piernas", hu_p, reg_p, 2.0)):
        hu16 = np.round(hu).astype(np.int16)
        np.savez_compressed(os.path.join(SALIDA, f"{nombre}.npz"), hu=hu16, mu=hu_a_mu(hu16), reg=reg, iso=iso, origen=origen)
        vml = (iso / 10) ** 3
        print(nombre, reg.shape, {NOMBRES[i]: round(float((reg == i).sum()) * vml, 1) for i in range(len(NOMBRES))})
    json.dump({"caso": "osteomielitis", "descripcion": "Osteomielitis de la cabeza del primer metatarsiano izquierdo.",
               "base": "CT congelado del Visible Human masculino (NLM, dominio público)",
               "centro_piernas_zyx": centro.tolist(), "centro_cuerpo_zyx": c_c.tolist(), "centro_mm_zyx": c_mm.tolist(),
               "pie": pie, "radio_oseo_mm": 9, "radio_inflamacion_mm": 16, "nombres": NOMBRES},
              open(os.path.join(SALIDA, "lesion.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("lesión en piernas (z, y, x):", np.round(centro, 1), pie)


if __name__ == "__main__":
    {"leer": paso_leer, "organos": paso_organos, "organos_fresco": paso_organos_fresco, "etiquetar": paso_etiquetar}[sys.argv[1]]()
