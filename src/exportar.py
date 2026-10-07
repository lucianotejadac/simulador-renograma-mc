"""Paso 5: DICOM NM dinámico (vista posterior y anterior) y datos para la página (docs/).

DICOM por caso en salida/dicom/<caso>/: RENOGRAMA_POST.dcm y RENOGRAMA_ANT.dcm, multicuadro NM DYNAMIC con
dos fases (30 × 2 s y 87 × 20 s), un detector por serie, paciente sintético SIM-RENO-nn.
Página: docs/datos/<caso>/post.bin (uint16, cuadros × 128 × 128), meta.json (curvas, ROIs, verdad) y
docs/dicom/<caso>.zip.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import zipfile

import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian

import modelo_mag3 as mm

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAIZ_UID = "1.2.826.0.1.3680043.10.1245.9."


def _archivo(sop_uid):
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.20"
    fm.MediaStorageSOPInstanceUID = sop_uid
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    fm.ImplementationClassUID = RAIZ_UID + "1"
    ds = FileDataset("", {}, file_meta=fm, preamble=b"\0" * 128)
    ds.is_little_endian, ds.is_implicit_VR = True, False
    ds.SOPClassUID, ds.SOPInstanceUID = fm.MediaStorageSOPClassUID, sop_uid
    return ds


def dicom_dinamico(cuadros, dur, vista, caso, n, meta_f, ruta):
    nf, filas, cols = cuadros.shape
    ds = _archivo(RAIZ_UID + f"{n}.{1 if vista == 'posterior' else 2}")
    hoy = dt.datetime(2026, 10, 7, 10, 0, 0)
    ds.SpecificCharacterSet = "ISO_IR 100"
    ds.PatientName = f"SIM^RENOGRAMA {caso.upper()}"
    ds.PatientID = f"SIM-RENO-{n:02d}"
    ds.IssuerOfPatientID = "SIM"
    ds.PatientSex = meta_f.get("sexo", "") or "O"
    ds.PatientAge = meta_f.get("edad", "") or ""
    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = "Fantoma de CT publico TCIA (PS3.15 AnnexE); actividad simulada"
    ds.StudyInstanceUID = RAIZ_UID + f"{n}.10"
    ds.SeriesInstanceUID = RAIZ_UID + f"{n}.{11 if vista == 'posterior' else 12}"
    ds.FrameOfReferenceUID = RAIZ_UID + f"{n}.13"
    ds.StudyDate = ds.SeriesDate = hoy.strftime("%Y%m%d")
    ds.StudyTime = ds.SeriesTime = hoy.strftime("%H%M%S")
    ds.StudyID, ds.AccessionNumber, ds.ReferringPhysicianName = "SIMRENO", "", ""
    ds.StudyDescription = "RENOGRAMA MAG3 (SIMULADO)"
    ds.SeriesDescription = f"RENOGRAMA {vista.upper()}"
    ds.SeriesNumber, ds.InstanceNumber = (1 if vista == "posterior" else 2), 1
    ds.Modality = "NM"
    ds.Manufacturer, ds.ManufacturerModelName = "simulador-renograma-mc", "Monte Carlo LEHR"
    ds.ImageType = ["ORIGINAL", "PRIMARY", "DYNAMIC", "EMISSION"]
    ds.Rows, ds.Columns, ds.NumberOfFrames = filas, cols, nf
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.PixelSpacing = ["4.8000", "4.8000"]
    ds.Units, ds.CountsSource = "CNTS", "EMISSION"
    ds.FrameIncrementPointer = [(0x0054, 0x0010), (0x0054, 0x0020), (0x0054, 0x0030)]
    ds.EnergyWindowVector = [1] * nf
    ds.DetectorVector = [1] * nf
    ds.NumberOfEnergyWindows, ds.NumberOfDetectors = 1, 1
    fases, ini = [], 0
    for d in sorted(set(dur.tolist()), key=lambda v: list(dur).index(v)):
        k = int((dur == d).sum())
        ph = Dataset()
        ph.PhaseDelay = int(sum(dur[:ini]) * 1000)
        ph.ActualFrameDuration = int(d * 1000)
        ph.PauseBetweenFrames = 0
        ph.NumberOfFramesInPhase = k
        fases.append(ph)
        ini += k
    ds.PhaseInformationSequence = Sequence(fases)
    ds.NumberOfPhases = len(fases)
    ds.PhaseVector = [i + 1 for i, ph in enumerate(fases) for _ in range(int(ph.NumberOfFramesInPhase))]
    ew = Dataset()
    ew.EnergyWindowName = "Tc99m 20%"
    r = Dataset()
    r.EnergyWindowLowerLimit, r.EnergyWindowUpperLimit = "126.45", "154.55"
    ew.EnergyWindowRangeSequence = Sequence([r])
    ds.EnergyWindowInformationSequence = Sequence([ew])
    rf = Dataset()
    rf.Radiopharmaceutical = "Tc-99m MAG3"
    rf.RadionuclideTotalDose = "185000000"
    ds.RadiopharmaceuticalInformationSequence = Sequence([rf])
    det = Dataset()
    det.CollimatorGridName, det.CollimatorType = "LEHR", "PARA"
    ds.DetectorInformationSequence = Sequence([det])
    vc = Dataset()
    vc.CodeValue, vc.CodingSchemeDesignator, vc.CodeMeaning = ("R-10214", "SRT", "posterior") if vista == "posterior" else ("R-10206", "SRT", "anterior")
    ds.ViewCodeSequence = Sequence([vc])
    ds.PatientOrientation = ["R", "F"] if vista == "posterior" else ["L", "F"]
    ds.PixelData = np.ascontiguousarray(cuadros[:, ::-1, :]).astype(np.uint16).tobytes()   # fila 0 = cabeza
    ds.save_as(ruta, enforce_file_format=True)


def main():
    meta_f = json.load(open(os.path.join(RAIZ, "salida", "fantoma.json"), encoding="utf-8"))
    indice = []
    os.makedirs(os.path.join(RAIZ, "docs", "dicom"), exist_ok=True)
    for n, nombre in enumerate(mm.casos(), start=1):
        c = os.path.join(RAIZ, "salida", "casos", nombre)
        cu = np.load(os.path.join(c, "cuadros.npz"))
        ro = np.load(os.path.join(c, "rois.npz"))
        cur = json.load(open(os.path.join(c, "curvas.json"), encoding="utf-8"))
        dur = cu["duraciones_s"]
        sal = os.path.join(RAIZ, "salida", "dicom", nombre)
        os.makedirs(sal, exist_ok=True)
        for vista in ("posterior", "anterior"):
            dicom_dinamico(cu[vista], dur, vista, nombre, n, meta_f, os.path.join(sal, f"RENOGRAMA_{vista[:4].upper()}.dcm"))
        with zipfile.ZipFile(os.path.join(RAIZ, "docs", "dicom", f"{nombre}.zip"), "w", zipfile.ZIP_DEFLATED) as z:
            for a in os.listdir(sal):
                z.write(os.path.join(sal, a), a)
        d = os.path.join(RAIZ, "docs", "datos", nombre)
        os.makedirs(d, exist_ok=True)
        np.ascontiguousarray(cu["posterior"]).astype(np.uint16).tofile(os.path.join(d, "post.bin"))
        rois = {k: np.nonzero(ro[k].ravel())[0].tolist() for k in ro.files}
        cur.update({"rois_indices": rois, "matriz": 128, "pixel_mm": 4.8, "n_cuadros": int(cu["posterior"].shape[0])})
        json.dump(cur, open(os.path.join(d, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False)
        indice.append({"caso": nombre, "zip_mb": round(os.path.getsize(os.path.join(RAIZ, "docs", "dicom", f"{nombre}.zip")) / 1e6, 1)})
        print(f"{nombre}: {cu['posterior'].shape} cuadros, zip {indice[-1]['zip_mb']} MB")
    json.dump({"casos": indice}, open(os.path.join(RAIZ, "docs", "datos", "indice.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
