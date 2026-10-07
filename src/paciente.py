"""Paciente sobre el que trabaja el flujo: RENO_PACIENTE=tcia (por defecto, CT del caso 2 de TCIA) o icrp
(fantoma masculino de referencia de la ICRP, Publicación 145). Cada uno tiene sus carpetas de salida y de página."""
import os

PACIENTE = os.environ.get("RENO_PACIENTE", "tcia")
assert PACIENTE in ("tcia", "icrp"), PACIENTE
SALIDA = "salida" if PACIENTE == "tcia" else "salida_icrp"
DOCS_DATOS = os.path.join("docs", "datos") if PACIENTE == "tcia" else os.path.join("docs", "icrp", "datos")
DOCS_DICOM = os.path.join("docs", "dicom") if PACIENTE == "tcia" else os.path.join("docs", "icrp", "dicom")
