"""Pruebas del modelo de MAG3 y del armado de cuadros."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import modelo_mag3 as mm  # noqa: E402


def test_conserva_actividad_con_decaimiento():
    c = mm.casos()["normal"]
    t, A = mm.simular_curvas(c, t_fin_min=30.0)
    total = A[:, -1].sum()
    esperado = c.actividad_MBq * math.exp(-math.log(2) / mm.T_MEDIO_TC99M_MIN * (t[-1] / 60.0 - (c.t_inyeccion_s + c.duracion_bolo_s / 2) / 60.0))
    assert abs(total - esperado) / esperado < 0.01, (total, esperado)


def test_normal_dentro_de_rangos_clinicos():
    c = mm.casos()["normal"]
    t, A = mm.simular_curvas(c)
    v = mm.verdad_clinica(c, t, A)
    for lado in ("derecho", "izquierdo"):
        assert 2.0 <= v[lado]["tmax_min"] <= 5.5          # máximo renal de 2 a 5 min
        assert v[lado]["c20_sobre_max"] < 0.3              # retención cortical normal
        assert v[lado]["funcion_relativa"] == 50.0


def test_obstruccion_no_vacia_y_dilatacion_responde():
    casos = mm.casos()
    t, A = mm.simular_curvas(casos["obstruccion-der"])
    assert mm.verdad_clinica(casos["obstruccion-der"], t, A)["derecho"]["c20_sobre_max"] > 0.9
    t, A = mm.simular_curvas(casos["dilatacion-der"])
    i22, i30 = np.searchsorted(t, 22 * 60), len(t) - 1
    renal = A[2] + A[4]
    assert renal[i30] < 0.5 * renal[i22]                  # vacía tras la furosemida


def test_integracion_por_cuadros():
    c = mm.casos()["normal"]
    t, A = mm.simular_curvas(c)
    dur = mm.protocolo()
    assert len(dur) == 117 and abs(dur.sum() - 1800) < 1e-9
    I = mm.integrar_cuadros(t, A, dur)
    total = np.trapezoid(A.sum(axis=0), t)
    assert abs(I.sum() - total) / total < 1e-3
