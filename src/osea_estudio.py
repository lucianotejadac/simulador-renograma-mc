"""Cintigrama óseo trifásico con Tc-99m MDP (un caso: osteomielitis del primer metatarsiano izquierdo), fantoma ICRP 145.

Modelo global (MBq): plasma P, extracelular E, hueso H, riñones K, vejiga V.
    P' = bolo − (kpe + kh + kr)·P + kep·E,   E' = kpe·P − kep·E,   H' = kh·P,   K' = kr·P − K/tk,   V' = K/tk
    micción a los 150 min (queda 10 %). Primer paso arterial en las piernas: curva gamma (llegada 10 s, máximo ~20 s)
    que suma, en la distribución vascular, 2.5 veces la actividad sanguínea de ese instante.
Calibración (MDP): ~50 % en hueso y ~35 % en orina a las 3 h, sangre ~3 %.
Lesión (osteomielitis): flujo y volumen sanguíneo ×3, extracelular ×2.5 y fijación ósea ×5 respecto del hueso esponjoso.

Imágenes (Monte Carlo una vez por compartimento y vista; plasma, extracelular y hueso escalados por la fracción del
total corporal que cae en el campo):
  fase 1  perfusión: 30 cuadros de 2 s, vista plantar de piernas y pies (128 × 128, 3.5 mm)
  fase 2  pool sanguíneo: estática de 300 s entre los 3 y los 8 min, plantar
  fase 3  tardía a las 3 h: estática de 300 s plantar, y cuerpo entero anterior y posterior (768 × 768, 2.4 mm,
          160 s por punto, equivalente a un barrido de 15 cm/min)
Salida: salida_osea/ (imágenes, DICOM) y docs/icrp/osea/ (página local).
"""
from __future__ import annotations

import json
import math
import os
import time
import zipfile

import numpy as np

import montecarlo as mc

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALIDA = os.path.join(RAIZ, "salida_osea")
DOCS = os.path.join(RAIZ, "docs", "icrp", "osea")
T_MEDIO = 6.0067 * 3600.0
VOL_SANGRE, VOL_EXTRA = 5300.0, 14000.0
# pesos por etiqueta (ver osea_fantoma.py)
VASC = {1: 0.03, 2: 0.02, 3: 0.05, 4: 0.05, 5: 0.10, 6: 0.25, 8: 1.0, 9: 0.25, 10: 0.30, 11: 0.15, 12: 0.09}
EXTRA = {1: 0.15, 3: 0.08, 4: 0.10, 6: 0.15, 9: 0.15, 10: 0.15, 11: 0.20, 12: 0.375}
HUESO = {2: 0.5, 3: 1.0, 4: 0.1, 11: 5.0}
PARAM = {"kpe": 0.25, "kep": 0.08, "kh": 0.035, "kr": 0.025, "tk": 6.0, "A0": 740.0, "miccion_min": 150.0}


def curvas(t_s):
    """Actividad (MBq) de cada compartimento global en los tiempos t_s (s)."""
    dt = 0.5
    tf = float(t_s.max())
    n = int(tf / dt) + 2
    P = E = H = K = V = 0.0
    out = np.zeros((5, n))
    dec = math.exp(-math.log(2) / T_MEDIO * dt)
    p = PARAM
    micciono = False
    for j in range(n):
        tt = j * dt
        out[:, j] = (P, E, H, K, V)
        bolo = p["A0"] / 10.0 if 2.0 <= tt < 12.0 else 0.0          # MBq/s durante 10 s
        m = 1 / 60.0
        dP = bolo - (p["kpe"] + p["kh"] + p["kr"]) * P * m + p["kep"] * E * m
        dE = (p["kpe"] * P - p["kep"] * E) * m
        dH = p["kh"] * P * m
        dK = (p["kr"] * P - K / p["tk"]) * m
        dV = K / p["tk"] * m
        P, E, H, K, V = [(v + dt * d) * dec for v, d in ((P, dP), (E, dE), (H, dH), (K, dK), (V, dV))]
        if not micciono and tt >= p["miccion_min"] * 60:
            V *= 0.1
            micciono = True
    tg = np.arange(n) * dt
    return np.stack([np.interp(t_s, tg, out[i]) for i in range(5)])


def primer_paso(t_s, P_t):
    """Actividad adicional de primer paso arterial en la distribución vascular (MBq equivalentes de sangre)."""
    t0, a, b = 10.0, 3.0, 3.0
    x = np.clip(t_s - t0, 0, None)
    g = (x / (a * b)) ** a * np.exp(a - x / b)                  # máximo 1 en t0 + a·b
    i_pico = int(np.argmin(np.abs(t_s - (t0 + a * b))))
    return 2.5 * P_t[i_pico] * g


def mapa(reg, pesos, vox_ml):
    a = np.zeros(reg.shape, np.float32)
    for et, w in pesos.items():
        a[reg == et] = w
    return a, float(a.sum() * vox_ml)


def plantar(arr):
    """Volumen de piernas (z, y, x) -> (y, z_invertido, x): las plantas quedan hacia +y, frente a un detector 'posterior'."""
    return np.ascontiguousarray(np.transpose(arr, (1, 0, 2))[:, ::-1, :])


def sensibilidades(historias):
    pc = np.load(os.path.join(SALIDA, "cuerpo.npz"))
    pp = np.load(os.path.join(SALIDA, "piernas.npz"))
    totales = {}
    vml_c = (float(pc["iso"]) / 10) ** 3
    for nombre, pesos in (("vasc", VASC), ("extra", EXTRA), ("hueso", HUESO)):
        totales[nombre] = mapa(pc["reg"], pesos, vml_c)[1]
    S = {}
    # piernas, vista plantar
    reg, mu, hu = plantar(pp["reg"]), plantar(pp["mu"]), plantar(pp["hu"])
    iso = float(pp["iso"]) / 10
    yy = np.nonzero((mu > 0.02).any(axis=(0, 2)))[0]
    dist = round(float((yy.max() + 1) * iso - mu.shape[1] * iso / 2) + 1.0, 1)
    cam = mc.Camara(1, 360.0, 90.0, dist, 128, 3.5, 1.0)
    for nombre, pesos, total in (("vasc", VASC, VOL_SANGRE), ("extra", EXTRA, VOL_EXTRA), ("hueso", HUESO, None)):
        a, suma = mapa(reg, pesos, iso ** 3)
        act = a * (1000.0 / suma)
        esp, _, _ = mc.simular(act, mu, hu, iso, cam, n_hist=historias, semilla=51 + len(S), poisson=False)
        frac = suma / (total if total else totales["hueso"])
        if nombre == "vasc":
            frac = suma / VOL_SANGRE
        elif nombre == "extra":
            frac = suma / VOL_EXTRA
        S[f"pies_{nombre}"] = (esp[0] * min(1.0, frac)).astype(np.float32)
        print(f"pies {nombre:6s}: fracción en campo {frac:.3f}", flush=True)
    # cuerpo entero, anterior y posterior
    reg, mu, hu = pc["reg"], pc["mu"], pc["hu"]
    iso = float(pc["iso"]) / 10
    yy = np.nonzero((mu > 0.02).any(axis=(0, 2)))[0]
    cy = mu.shape[1] * iso / 2
    for vista, ang, d in (("ant", 270.0, cy - yy.min() * iso + 2.0), ("post", 90.0, (yy.max() + 1) * iso - cy + 2.0)):
        cam = mc.Camara(1, 360.0, ang, round(float(d), 1), 768, 2.4, 1.0)
        for nombre, pesos in (("vasc", VASC), ("extra", EXTRA), ("hueso", HUESO), ("rinon", {6: 1.0}), ("vejiga", {7: 1.0})):
            a, suma = mapa(reg, pesos, iso ** 3)
            act = a * (1000.0 / suma)
            t0 = time.time()
            esp, _, _ = mc.simular(act, mu, hu, iso, cam, n_hist=historias, semilla=71 + len(S), poisson=False)
            frac = {"vasc": suma / VOL_SANGRE, "extra": suma / VOL_EXTRA}.get(nombre, 1.0)
            S[f"ce_{vista}_{nombre}"] = (esp[0] * min(1.0, frac)).astype(np.float32)
            print(f"cuerpo {vista:4s} {nombre:6s}: {esp[0].sum():7.1f} cuentas/MBq·s, fracción {min(1.0, frac):.3f} ({time.time() - t0:.0f} s)", flush=True)
    np.savez_compressed(os.path.join(SALIDA, "sensibilidades.npz"), **S)


def integrar(t0, t1, n=400):
    t = np.linspace(t0, t1, n)
    C = curvas(np.linspace(0, t1, int(t1 / 0.5) + 2))
    tt = np.linspace(0, t1, C.shape[1])
    Ci = np.stack([np.interp(t, tt, C[i]) for i in range(5)])
    return Ci, t


def estudio(semilla=3):
    S = dict(np.load(os.path.join(SALIDA, "sensibilidades.npz")))
    rng = np.random.default_rng(semilla)
    # fase 1: 30 cuadros de 2 s
    tf = np.arange(0, 61, 0.25)
    C = curvas(tf)
    fp = primer_paso(tf, C[0])
    f1 = []
    for k in range(30):
        sel = (tf >= 2 * k) & (tf < 2 * k + 2)
        Iv = np.trapezoid((C[0] + fp)[sel], tf[sel]) if sel.sum() > 1 else 0.0
        Ie = np.trapezoid(C[1][sel], tf[sel])
        Ih = np.trapezoid(C[2][sel], tf[sel])
        esp = Iv * S["pies_vasc"] + Ie * S["pies_extra"] + Ih * S["pies_hueso"]
        f1.append(rng.poisson(esp))
    f1 = np.array(f1, np.uint16)
    # fase 2: 180–480 s ; fase 3: 10800–11100 s
    def estatica(t0, t1, claves, sens):
        tt = np.linspace(0, t1, int(t1 / 1.0) + 1)
        Cc = curvas(tt)
        sel = tt >= t0
        I = [np.trapezoid(Cc[i][sel], tt[sel]) for i in range(5)]
        esp = sum(I[i] * sens[c] for i, c in claves if c in sens)
        return rng.poisson(esp).astype(np.uint16), I
    f2, _ = estatica(180, 480, [(0, "pies_vasc"), (1, "pies_extra"), (2, "pies_hueso")], S)
    f3, I3 = estatica(10800, 11100, [(0, "pies_vasc"), (1, "pies_extra"), (2, "pies_hueso")], S)
    ce = {}
    for v in ("ant", "post"):
        ce[v], _ = estatica(10800, 10960, [(0, f"ce_{v}_vasc"), (1, f"ce_{v}_extra"), (2, f"ce_{v}_hueso"), (3, f"ce_{v}_rinon"), (4, f"ce_{v}_vejiga")], S)
    C3 = curvas(np.array([10800.0]))[:, 0]
    reparto = {n: round(100 * v / PARAM["A0"] / math.exp(-math.log(2) * 10800 / T_MEDIO), 1) for n, v in zip(("sangre", "extracelular", "hueso", "riñones", "vejiga"), C3)}
    np.savez_compressed(os.path.join(SALIDA, "imagenes.npz"), fase1=f1, fase2=f2, fase3=f3, ce_ant=ce["ant"], ce_post=ce["post"])
    print("reparto a las 3 h (% de lo inyectado, corregido por decaimiento):", reparto)
    print("cuentas: fase 1 (total)", int(f1.sum()), "fase 2", int(f2.sum()), "fase 3", int(f3.sum()), "cuerpo entero ant/post", int(ce["ant"].sum()), int(ce["post"].sum()))
    return f1, f2, f3, ce, reparto


def main():
    import sys
    if "--solo-imagenes" not in sys.argv:
        sensibilidades(3_000_000)
    estudio()


def posiciones_lesion():
    """Píxel (u, v) de la lesión en la vista plantar (128) y en el cuerpo entero anterior y posterior (768)."""
    les = json.load(open(os.path.join(SALIDA, "lesion.json"), encoding="utf-8"))
    pp = np.load(os.path.join(SALIDA, "piernas.npz"))
    pc = np.load(os.path.join(SALIDA, "cuerpo.npz"))
    z, y, x = les["centro_piernas_zyx"]
    nz, ny, nx = pp["reg"].shape
    isp = float(pp["iso"])
    zr = y                                  # eje 0 del volumen plantar = y original
    u_pl = -((x + 0.5) * isp - nx * isp / 2) / 3.5 + 64
    v_pl = ((zr + 0.5) * isp - ny * isp / 2) / 3.5 + 64
    zc, yc, xc = les["centro_cuerpo_zyx"]
    nzc, nyc, nxc = pc["reg"].shape
    isc = float(pc["iso"])
    u_ant = ((xc + 0.5) * isc - nxc * isc / 2) / 2.4 + 384
    v = ((zc + 0.5) * isc - nzc * isc / 2) / 2.4 + 384
    return {"plantar_uv": [round(u_pl, 1), round(v_pl, 1)], "ant_uv": [round(u_ant, 1), round(v, 1)], "post_uv": [round(768 - u_ant, 1), round(v, 1)]}


def exportar():
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.sequence import Sequence
    from pydicom.uid import ExplicitVRLittleEndian
    d = np.load(os.path.join(SALIDA, "imagenes.npz"))
    raiz = "1.2.826.0.1.3680043.10.1245.12."
    os.makedirs(os.path.join(DOCS, "datos"), exist_ok=True)
    os.makedirs(os.path.join(SALIDA, "dicom"), exist_ok=True)
    def nm(nombre, cuadros, pix, desc, tipo, serie, dur_ms, vista):
        fm = FileMetaDataset()
        fm.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.20"
        fm.MediaStorageSOPInstanceUID = raiz + f"1.{serie}"
        fm.TransferSyntaxUID = ExplicitVRLittleEndian
        fm.ImplementationClassUID = raiz + "0"
        ds = FileDataset("", {}, file_meta=fm, preamble=b"\0" * 128)
        ds.is_little_endian, ds.is_implicit_VR = True, False
        ds.SOPClassUID, ds.SOPInstanceUID = fm.MediaStorageSOPClassUID, fm.MediaStorageSOPInstanceUID
        ds.SpecificCharacterSet = "ISO_IR 100"
        ds.PatientName, ds.PatientID, ds.IssuerOfPatientID, ds.PatientSex = "SIM^OSEO TRIFASICO", "SIM-OSEO-ICRP-01", "SIM", "M"
        ds.PatientIdentityRemoved = "YES"
        ds.DeidentificationMethod = "Fantoma de referencia ICRP Pub. 145 (MRCP_AM); actividad simulada"
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = raiz + "10", raiz + f"11.{serie}", raiz + "12"
        ds.StudyDate = ds.SeriesDate = "20261009"
        ds.StudyTime = ds.SeriesTime = "090000"
        ds.StudyID, ds.AccessionNumber, ds.ReferringPhysicianName = "SIMOSEO", "", ""
        ds.StudyDescription, ds.SeriesDescription = "CINTIGRAMA OSEO TRIFASICO (SIMULADO)", desc
        ds.SeriesNumber = ds.InstanceNumber = serie
        ds.Modality, ds.Manufacturer, ds.ManufacturerModelName = "NM", "simulador-renograma-mc", "Monte Carlo LEHR"
        ds.ImageType = ["ORIGINAL", "PRIMARY", tipo, "EMISSION"]
        nf, filas, cols = cuadros.shape
        ds.Rows, ds.Columns, ds.NumberOfFrames = filas, cols, nf
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
        ds.PixelSpacing = [f"{pix:.4f}", f"{pix:.4f}"]
        ds.Units, ds.CountsSource = "CNTS", "EMISSION"
        if tipo == "DYNAMIC":
            ds.FrameIncrementPointer = [(0x0054, 0x0010), (0x0054, 0x0020), (0x0054, 0x0030)]
            ds.PhaseVector = [1] * nf
            ph = Dataset()
            ph.PhaseDelay, ph.ActualFrameDuration, ph.PauseBetweenFrames, ph.NumberOfFramesInPhase = 0, dur_ms, 0, nf
            ds.PhaseInformationSequence = Sequence([ph])
            ds.NumberOfPhases = 1
        else:
            ds.FrameIncrementPointer = [(0x0054, 0x0010), (0x0054, 0x0020)]
            ds.ActualFrameDuration = str(dur_ms)
        ds.EnergyWindowVector, ds.DetectorVector = [1] * nf, [1] * nf
        ds.NumberOfEnergyWindows, ds.NumberOfDetectors = 1, 1
        ew = Dataset()
        ew.EnergyWindowName = "Tc99m 20%"
        rr = Dataset()
        rr.EnergyWindowLowerLimit, rr.EnergyWindowUpperLimit = "126.45", "154.55"
        ew.EnergyWindowRangeSequence = Sequence([rr])
        ds.EnergyWindowInformationSequence = Sequence([ew])
        rf = Dataset()
        rf.Radiopharmaceutical, rf.RadionuclideTotalDose = "Tc-99m MDP", "740000000"
        ds.RadiopharmaceuticalInformationSequence = Sequence([rf])
        det = Dataset()
        det.CollimatorGridName, det.CollimatorType = "LEHR", "PARA"
        ds.DetectorInformationSequence = Sequence([det])
        ds.PixelData = np.ascontiguousarray(cuadros).astype(np.uint16).tobytes()
        ruta = os.path.join(SALIDA, "dicom", nombre)
        ds.save_as(ruta, enforce_file_format=True)
        return ruta
    # vista plantar como en la clínica: dedos arriba (sin invertir filas) y pie izquierdo a la derecha (columnas invertidas)
    plantar = lambda a: a[..., ::-1]
    ce = lambda a: a[::-1, 250:518]                 # cabeza arriba, recorte lateral del campo de 768
    rutas = [nm("FASE1_PERFUSION.dcm", plantar(d["fase1"]), 3.5, "FASE 1 PERFUSION PLANTAR", "DYNAMIC", 1, 2000, "plantar"),
             nm("FASE2_POOL.dcm", plantar(d["fase2"])[None], 3.5, "FASE 2 POOL PLANTAR", "STATIC", 2, 300000, "plantar"),
             nm("FASE3_TARDIA.dcm", plantar(d["fase3"])[None], 3.5, "FASE 3 TARDIA PLANTAR", "STATIC", 3, 300000, "plantar"),
             nm("CUERPO_ENTERO.dcm", np.stack([ce(d["ce_ant"]), ce(d["ce_post"])[:, ::-1]]), 2.4, "CUERPO ENTERO ANT Y POST", "WHOLE BODY", 4, 160000, "")]
    with zipfile.ZipFile(os.path.join(DOCS, "osteomielitis.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for r in rutas:
            z.write(r, os.path.basename(r))
    pos = posiciones_lesion()
    u, v = pos["plantar_uv"]
    pos["plantar_uv"] = [round(127 - u, 1), round(v, 1)]                           # columnas invertidas
    pos["ant_uv"] = [round(pos["ant_uv"][0] - 250, 1), round(767 - pos["ant_uv"][1], 1)]
    a_u = posiciones_lesion()["ant_uv"]          # sin recortar ni invertir
    pos["post_uv"] = [round(268 - 1 - (a_u[0] - 250), 1), round(767 - a_u[1], 1)]   # posterior vista desde atrás: izquierda del paciente a la izquierda
    for nombre, arr in (("fase1", plantar(d["fase1"])), ("fase2", plantar(d["fase2"])), ("fase3", plantar(d["fase3"])),
                        ("ce_ant", ce(d["ce_ant"])), ("ce_post", ce(d["ce_post"])[:, ::-1])):
        np.ascontiguousarray(arr).astype(np.uint16).tofile(os.path.join(DOCS, "datos", f"{nombre}.bin"))
    les = json.load(open(os.path.join(SALIDA, "lesion.json"), encoding="utf-8"))
    json.dump({"descripcion": les["descripcion"], "lesion": pos, "fase1": {"cuadros": 30, "dur_s": 2, "matriz": 128},
               "fase2": {"inicio_s": 180, "fin_s": 480}, "fase3": {"inicio_min": 180, "dur_s": 300},
               "cuerpo_entero": {"filas": 768, "columnas": 268, "pixel_mm": 2.4}, "actividad_MBq": 740},
              open(os.path.join(DOCS, "datos", "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("exportado; lesión en:", pos)


if __name__ == "__main__":
    main()
    exportar()
