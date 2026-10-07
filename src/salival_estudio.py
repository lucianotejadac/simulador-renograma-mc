"""Cintigrafía dinámica de glándulas salivales (pertecnetato, estímulo ácido) sobre el fantoma ICRP de cabeza y cuello.

1. Monte Carlo una vez por compartimento en vista anterior (128 × 128 de 3 mm, LEHR): imágenes por MBq·s.
2. Por caso: cuadros = Poisson(Σ actividad integrada · imagen), 80 cuadros de 30 s (40 min), limón a los 20 min.
3. Regiones por glándula (imagen de su compartimento > 30 % del máximo, ampliada 1 píxel; cada píxel a la
   glándula que más aporta) y fondo en anillo alrededor de cada una (sin otras glándulas, boca ni tiroides).
4. Mediciones como en la clínica sobre los cuadros con ruido: tiempo al máximo antes del estímulo, captación
   relativa (glándula/fondo por píxel antes del limón) y fracción de excreción (caída en los 5 min tras el limón).
5. DICOM NM dinámico y datos para la página local (docs/icrp/salival/), paciente sintético SIM-SALIV-ICRP-nn.

Uso: python src/salival_estudio.py [--historias 3000000] [--solo-casos]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
import zipfile

import numpy as np
from scipy import ndimage

import montecarlo as mc
import salival_modelo as sm

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALIDA = os.path.join(RAIZ, "salida_salival")
DOCS = os.path.join(RAIZ, "docs", "icrp", "salival")
MATRIZ, PIXEL_MM = 128, 3.0
ETIQUETA = {"parotida_d": 4, "parotida_i": 5, "submax_d": 6, "submax_i": 7, "tiroides": 8, "boca": 9}
PESOS = {
    "plasma": {10: 1.0, 11: 0.04, 1: 0.04, 4: 0.10, 5: 0.10, 6: 0.10, 7: 0.10, 8: 0.20, 9: 0.06, 2: 0.03},
    "intersticio": {1: 0.15, 4: 0.15, 5: 0.15, 6: 0.15, 7: 0.15, 8: 0.15, 9: 0.15, 2: 0.05},
    **{k: {v: 1.0} for k, v in ETIQUETA.items()},
    "estomago": {},
}

# El plasma y el intersticio ocupan todo el cuerpo: dentro del campo solo cae una fracción. Cada imagen por MBq de
# esos compartimentos se escala por (volumen sanguíneo o extracelular en el campo) / (volumen corporal total) del
# hombre de referencia (sangre 5.3 L, líquido extracelular 14 L). Sin esto, el fondo queda varias veces más alto.
VOL_SANGRE_ML, VOL_EXTRACELULAR_ML = 5300.0, 14000.0


def fraccion_en_campo(reg, pesos, vox_ml, total_ml):
    a = 0.0
    for et, w in pesos.items():
        a += float((reg == et).sum()) * w * vox_ml
    return min(1.0, a / total_ml)


def mapa(reg, pesos, vox_ml):
    a = np.zeros(reg.shape, np.float32)
    for et, w in pesos.items():
        a[reg == et] = w
    tot = a.sum() * vox_ml
    return a * (1000.0 / tot) if tot > 0 else a


def sensibilidades(historias):
    f = np.load(os.path.join(SALIDA, "fantoma.npz"))
    mu, hu, iso = f["mu"], f["hu"], float(f["iso"])
    reg = np.load(os.path.join(SALIDA, "regiones.npz"))["reg"]
    vox = iso / 10.0
    yy = np.nonzero((mu > 0.02).any(axis=(0, 2)))[0]
    dist = round(float(mu.shape[1] * vox / 2 - yy.min() * vox) + 2.0, 1)
    cam = mc.Camara(1, 360.0, 270.0, dist, MATRIZ, PIXEL_MM, 1.0)
    S = {}
    for i, comp in enumerate(sm.COMPARTIMENTOS):
        act = mapa(reg, PESOS[comp], vox ** 3)
        if act.sum() == 0:
            S[comp] = np.zeros((MATRIZ, MATRIZ), np.float32)
            continue
        t0 = time.time()
        esp, _, _ = mc.simular(act, mu, hu, vox, cam, n_hist=historias, semilla=31 + 7 * i, poisson=False)
        S[comp] = esp[0].astype(np.float32)
        if comp in ("plasma", "intersticio"):
            f_ = fraccion_en_campo(reg, PESOS[comp], vox ** 3, VOL_SANGRE_ML if comp == "plasma" else VOL_EXTRACELULAR_ML)
            S[comp] *= f_
            print(f"   {comp}: fracción del volumen corporal dentro del campo = {f_:.3f}", flush=True)
        print(f"{comp:12s}: {esp[0].sum():7.1f} cuentas por MBq·s ({time.time() - t0:.1f} s)", flush=True)
    np.savez_compressed(os.path.join(SALIDA, "sensibilidades.npz"), **S)
    json.dump({"distancia_cm": dist, "matriz": MATRIZ, "pixel_mm": PIXEL_MM, "historias": historias},
              open(os.path.join(SALIDA, "sensibilidades.json"), "w", encoding="utf-8"), indent=2)


def regiones_interes(S):
    gl = sm.GLANDULAS
    pila = np.stack([S[g] / S[g].max() for g in gl])
    dueno = np.argmax(pila, axis=0)
    R = {}
    for i, g in enumerate(gl):
        r = (S[g] > 0.30 * S[g].max()) & (dueno == i)
        et, n = ndimage.label(r)
        if n > 1:
            r = et == (1 + int(np.argmax(ndimage.sum(r, et, range(1, n + 1)))))
        R[g] = ndimage.binary_dilation(r, iterations=1) & ((dueno == i) | (pila.max(axis=0) < 0.3))
    ocupado = np.zeros_like(R[gl[0]])
    for g in gl:
        ocupado |= ndimage.binary_dilation(R[g], iterations=1)
    for extra in ("boca", "tiroides"):
        ocupado |= S[extra] > 0.15 * S[extra].max()
    for g in gl:
        R[f"fondo_{g}"] = ndimage.binary_dilation(R[g], iterations=4) & ~ndimage.binary_dilation(R[g], iterations=2) & ~ocupado
    return R


def medir(cuadros, R, dur, c: sm.Caso):
    tc = np.cumsum(dur) - dur / 2
    i_pre = np.nonzero(tc < c.t_estimulo_min * 60)[0]
    i_post = np.nonzero((tc > c.t_estimulo_min * 60) & (tc <= (c.t_estimulo_min + 5) * 60))[0]
    curvas, medidos = {}, {}
    for g in sm.GLANDULAS:
        r, fo = R[g], R[f"fondo_{g}"]
        cr = cuadros[:, r].sum(axis=1).astype(float)
        cf = cuadros[:, fo].sum(axis=1).astype(float) * (r.sum() / max(fo.sum(), 1))
        neta = (cr - cf) / dur
        suave = np.convolve(neta, np.ones(3) / 3, mode="same")
        pre = suave[i_pre[-2:]].mean()
        post_min = suave[i_post].min()
        tmax = float(tc[i_pre[int(np.argmax(suave[i_pre]))]] / 60)
        fondo_px = cf[i_pre[-2:]].mean() / max(r.sum(), 1)
        bruta_px = cr[i_pre[-2:]].mean() / max(r.sum(), 1)
        curvas[g] = [round(float(v), 2) for v in neta]
        senal = pre > 3 * np.sqrt(max(cf[i_pre[-2:]].mean(), 1)) / dur[0]
        medidos[g] = {"tmax_min": round(tmax, 1) if senal else None,
                      "captacion_relativa": round(float(bruta_px / max(fondo_px, 1e-9)), 2),
                      "fraccion_excrecion_pct": round(float(100 * (pre - post_min) / pre), 1) if senal and pre > 0 else None}
    return curvas, medidos


def dicom(cuadros, dur, caso, n, ruta):
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.sequence import Sequence
    from pydicom.uid import ExplicitVRLittleEndian
    raiz = "1.2.826.0.1.3680043.10.1245.11."
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.20"
    fm.MediaStorageSOPInstanceUID = raiz + f"{n}.1"
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    fm.ImplementationClassUID = raiz + "0"
    ds = FileDataset("", {}, file_meta=fm, preamble=b"\0" * 128)
    ds.is_little_endian, ds.is_implicit_VR = True, False
    ds.SOPClassUID, ds.SOPInstanceUID = fm.MediaStorageSOPClassUID, fm.MediaStorageSOPInstanceUID
    hoy = dt.datetime(2026, 10, 7, 11, 0, 0)
    ds.SpecificCharacterSet = "ISO_IR 100"
    ds.PatientName, ds.PatientID, ds.IssuerOfPatientID = f"SIM^SALIVALES {caso.upper()}", f"SIM-SALIV-ICRP-{n:02d}", "SIM"
    ds.PatientSex, ds.PatientIdentityRemoved = "M", "YES"
    ds.DeidentificationMethod = "Fantoma de referencia ICRP Pub. 145 (MRCP_AM); actividad simulada"
    ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.FrameOfReferenceUID = raiz + f"{n}.10", raiz + f"{n}.11", raiz + f"{n}.12"
    ds.StudyDate = ds.SeriesDate = hoy.strftime("%Y%m%d")
    ds.StudyTime = ds.SeriesTime = hoy.strftime("%H%M%S")
    ds.StudyID, ds.AccessionNumber, ds.ReferringPhysicianName = "SIMSALIV", "", ""
    ds.StudyDescription, ds.SeriesDescription = "CINTIGRAFIA GLANDULAS SALIVALES (SIMULADO)", "DINAMICO ANTERIOR"
    ds.SeriesNumber = ds.InstanceNumber = 1
    ds.Modality, ds.Manufacturer, ds.ManufacturerModelName = "NM", "simulador-renograma-mc", "Monte Carlo LEHR"
    ds.ImageType = ["ORIGINAL", "PRIMARY", "DYNAMIC", "EMISSION"]
    nf = cuadros.shape[0]
    ds.Rows, ds.Columns, ds.NumberOfFrames = MATRIZ, MATRIZ, nf
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.PixelSpacing = [f"{PIXEL_MM:.4f}", f"{PIXEL_MM:.4f}"]
    ds.Units, ds.CountsSource = "CNTS", "EMISSION"
    ds.FrameIncrementPointer = [(0x0054, 0x0010), (0x0054, 0x0020), (0x0054, 0x0030)]
    ds.EnergyWindowVector, ds.DetectorVector, ds.PhaseVector = [1] * nf, [1] * nf, [1] * nf
    ds.NumberOfEnergyWindows, ds.NumberOfDetectors, ds.NumberOfPhases = 1, 1, 1
    ph = Dataset()
    ph.PhaseDelay, ph.ActualFrameDuration, ph.PauseBetweenFrames, ph.NumberOfFramesInPhase = 0, int(dur[0] * 1000), 0, nf
    ds.PhaseInformationSequence = Sequence([ph])
    ew = Dataset()
    ew.EnergyWindowName = "Tc99m 20%"
    rr = Dataset()
    rr.EnergyWindowLowerLimit, rr.EnergyWindowUpperLimit = "126.45", "154.55"
    ew.EnergyWindowRangeSequence = Sequence([rr])
    ds.EnergyWindowInformationSequence = Sequence([ew])
    rf = Dataset()
    rf.Radiopharmaceutical, rf.RadionuclideTotalDose = "Tc-99m pertechnetate", "185000000"
    ds.RadiopharmaceuticalInformationSequence = Sequence([rf])
    det = Dataset()
    det.CollimatorGridName, det.CollimatorType = "LEHR", "PARA"
    ds.DetectorInformationSequence = Sequence([det])
    vc = Dataset()
    vc.CodeValue, vc.CodingSchemeDesignator, vc.CodeMeaning = "R-10206", "SRT", "anterior"
    ds.ViewCodeSequence = Sequence([vc])
    ds.PatientOrientation = ["L", "F"]
    ds.PixelData = np.ascontiguousarray(cuadros[:, ::-1, :]).astype(np.uint16).tobytes()
    ds.save_as(ruta, enforce_file_format=True)


def casos(semilla=1):
    S = dict(np.load(os.path.join(SALIDA, "sensibilidades.npz")))
    R = regiones_interes(S)
    dur = sm.protocolo()
    rng = np.random.default_rng(semilla)
    os.makedirs(os.path.join(DOCS, "dicom"), exist_ok=True)
    indice = []
    for n, (nombre, c) in enumerate(sm.casos().items(), start=1):
        t, A = sm.simular_curvas(c, t_fin_min=dur.sum() / 60)
        bordes = np.concatenate([[0.0], np.cumsum(dur)])
        acum = np.concatenate([np.zeros((A.shape[0], 1)), np.cumsum((A[:, 1:] + A[:, :-1]) * 0.5 * np.diff(t), axis=1)], axis=1)
        I = np.stack([np.diff(np.interp(bordes, t, acum[i])) for i in range(A.shape[0])])
        esp = np.zeros((len(dur), MATRIZ, MATRIZ))
        for i, comp in enumerate(sm.COMPARTIMENTOS):
            esp += I[i][:, None, None] * S[comp][None]
        cuadros = rng.poisson(esp).astype(np.uint16)
        curvas, medidos = medir(cuadros, R, dur, c)
        verdad = sm.verdad_clinica(c, t, A)
        carpeta = os.path.join(SALIDA, "casos", nombre)
        os.makedirs(carpeta, exist_ok=True)
        np.savez_compressed(os.path.join(carpeta, "cuadros.npz"), anterior=cuadros, duraciones_s=dur)
        ruta_dcm = os.path.join(carpeta, "SALIVALES_ANT.dcm")
        dicom(cuadros, dur, nombre, n, ruta_dcm)
        with zipfile.ZipFile(os.path.join(DOCS, "dicom", f"{nombre}.zip"), "w", zipfile.ZIP_DEFLATED) as z:
            z.write(ruta_dcm, "SALIVALES_ANT.dcm")
        d = os.path.join(DOCS, "datos", nombre)
        os.makedirs(d, exist_ok=True)
        cuadros.tofile(os.path.join(d, "ant.bin"))
        i_t = np.arange(0, len(t), int(10 / (t[1] - t[0])))
        json.dump({"caso": nombre, "descripcion": c.descripcion, "duraciones_s": dur.tolist(), "t_centro_s": (np.cumsum(dur) - dur / 2).tolist(),
                   "t_estimulo_min": c.t_estimulo_min, "curvas_medidas": curvas, "medidos": medidos, "verdad": verdad,
                   "parametros": {g: vars(c.glandulas[g]) for g in sm.GLANDULAS},
                   "curvas_verdaderas_MBq": {"t_s": t[i_t].tolist(), **{comp: [round(float(v), 4) for v in A[i, i_t]] for i, comp in enumerate(sm.COMPARTIMENTOS)}},
                   "rois_indices": {k: np.nonzero(v.ravel())[0].tolist() for k, v in R.items()}, "matriz": MATRIZ, "pixel_mm": PIXEL_MM,
                   "n_cuadros": int(len(dur)), "cuentas_totales": int(cuadros.sum())},
                  open(os.path.join(d, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False)
        indice.append({"caso": nombre})
        f = lambda m: {g[:9]: (m[g]["tmax_min"], m[g].get("fraccion_excrecion_pct")) for g in sm.GLANDULAS}
        print(f"{nombre}:\n   medido  (Tmax, EF%) {f(medidos)}\n   verdad  (Tmax, EF%) {f(verdad)}", flush=True)
    json.dump({"casos": indice}, open(os.path.join(DOCS, "datos", "indice.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print("ROIs (píxeles):", {k: int(v.sum()) for k, v in R.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--historias", type=int, default=3_000_000)
    ap.add_argument("--solo-casos", action="store_true")
    a = ap.parse_args()
    if not a.solo_casos:
        sensibilidades(a.historias)
    casos()


if __name__ == "__main__":
    main()
