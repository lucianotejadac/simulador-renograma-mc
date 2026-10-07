"""Modelo de compartimentos del Tc-99m MAG3 (curvas de actividad en MBq por compartimento).

Compartimentos: plasma (B), intersticio (T), corteza derecha/izquierda (Cd, Ci), pelvis derecha/izquierda
(Pd, Pi), vejiga (V), hígado (H) e intestino (G). Constantes en 1/min, tiempos en min.

    B' = entrada - (kbt + kd + ki + kh) B + ktb T
    T' = kbt B - ktb T
    C' = k B - C / tc            (secreción tubular y tránsito parenquimatoso)
    P' = C / tc - P / tp         (tránsito pelvicalicial; la furosemida acorta tp desde t_furo + 2 min)
    V' = Pd / tpd + Pi / tpi
    H' = kh B - H / th,   G' = H / th

Valores de referencia (orden de magnitud clínico): aclaramiento renal de MAG3 ~ 300 mL/min sobre ~3 L de
plasma (k total ≈ 0.10–0.15/min), tránsito cortical medio 2–3 min (máximo renal a los 3–5 min), vaciamiento
pélvico normal 3–5 min, excreción hepatobiliar ~3 %. Decaimiento físico del Tc-99m incluido.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import numpy as np

T_MEDIO_TC99M_MIN = 6.0067 * 60.0
COMPARTIMENTOS = ["plasma", "intersticio", "corteza_d", "corteza_i", "pelvis_d", "pelvis_i", "vejiga", "higado", "intestino", "ureter_d", "ureter_i"]


@dataclass
class Rinon:
    k: float = 0.08           # 1/min: extracción desde el plasma (función)
    tc: float = 3.0           # min: tránsito parenquimatoso medio (cadena de 3 etapas)
    tp: float = 2.0           # min: vaciamiento pélvico
    tp_furo: float | None = None   # min: vaciamiento pélvico tras la furosemida (None = sin respuesta, igual a tp)
    tu: float = 1.0           # min: tránsito ureteral (pelvis -> uréter -> vejiga)
    tu_furo: float | None = None   # min: tránsito ureteral tras la furosemida (None = sin respuesta)


@dataclass
class Caso:
    nombre: str
    descripcion: str
    derecho: Rinon = field(default_factory=Rinon)
    izquierdo: Rinon = field(default_factory=Rinon)
    actividad_MBq: float = 185.0
    kbt: float = 0.08
    ktb: float = 0.12
    kh: float = 0.004
    th: float = 30.0
    t_furo_min: float | None = None
    t_inyeccion_s: float = 5.0
    duracion_bolo_s: float = 10.0


def casos() -> dict[str, Caso]:
    import os
    if os.environ.get("RENO_PACIENTE", "tcia") == "icrp":
        return casos_icrp()
    return {
        "normal": Caso("normal", "Función y drenaje normales en ambos riñones.", Rinon(), Rinon()),
        "funcion-reducida-izq": Caso("funcion-reducida-izq", "Riñón izquierdo con función reducida (aporte relativo 30 %) y tránsito lento.",
                                     Rinon(k=0.08), Rinon(k=0.034, tc=4.5, tp=3.0)),
        "obstruccion-der": Caso("obstruccion-der", "Obstrucción pieloureteral derecha que no responde a la furosemida.",
                                Rinon(tp=300.0, tp_furo=None), Rinon(), t_furo_min=20.0),
        "dilatacion-der": Caso("dilatacion-der", "Pelvis derecha dilatada sin obstrucción: retiene y vacía tras la furosemida.",
                               Rinon(tp=60.0, tp_furo=2.0), Rinon(), t_furo_min=20.0),
        "no-funcionante-izq": Caso("no-funcionante-izq", "Riñón izquierdo no funcionante: solo pool sanguíneo.",
                                   Rinon(k=0.08), Rinon(k=0.0)),
    }


ETAPAS_CORTEZA = 3      # tránsito parenquimatoso como cadena de 3 etapas (Erlang): la salida empieza tras ~1 min


def casos_icrp() -> dict[str, Caso]:
    """Casos del paciente estándar ICRP 145, que tiene uréteres: patología ureteral además de la renal."""
    return {
        "normal": Caso("normal", "Función y drenaje normales; los uréteres se ven en tránsito hacia la vejiga.", Rinon(), Rinon()),
        "obstruccion-ureterovesical-izq": Caso("obstruccion-ureterovesical-izq",
            "Obstrucción de la unión ureterovesical izquierda: uréter y pelvis izquierdos se llenan y no vacían, ni con furosemida.",
            Rinon(), Rinon(tp=20.0, tu=400.0), t_furo_min=20.0),     # la contrapresión también enlentece la pelvis
        "megaureter-der": Caso("megaureter-der", "Megauréter derecho no obstructivo: el uréter retiene y vacía tras la furosemida.",
            Rinon(tp=4.0, tu=45.0, tu_furo=2.0), Rinon(), t_furo_min=20.0),
        "obstruccion-pieloureteral-der": Caso("obstruccion-pieloureteral-der",
            "Obstrucción pieloureteral derecha: la pelvis retiene y el uréter derecho no se ve; no responde a la furosemida.",
            Rinon(tp=300.0), Rinon(), t_furo_min=20.0),
        "funcion-reducida-izq": Caso("funcion-reducida-izq", "Riñón izquierdo con función reducida (aporte relativo 30 %) y tránsito lento.",
            Rinon(k=0.08), Rinon(k=0.034, tc=4.5, tp=3.0)),
    }


def simular_curvas(caso: Caso, t_fin_min: float = 30.0, dt_s: float = 0.25):
    """Devuelve (t_s, A) con A[c, i] en MBq para cada compartimento c (orden COMPARTIMENTOS).
    La corteza de cada riñón es una cadena de ETAPAS_CORTEZA subcompartimentos de tc/ETAPAS cada uno (tránsito
    con tiempo mínimo, como el real); A informa su suma."""
    n = int(round(t_fin_min * 60.0 / dt_s)) + 1
    t = np.arange(n) * dt_s
    A = np.zeros((len(COMPARTIMENTOS), n))
    E = ETAPAS_CORTEZA
    B = T = Pd = Pi = V = H = G = Ud = Ui = 0.0
    Cd = np.zeros(E)
    Ci = np.zeros(E)
    dtm = dt_s / 60.0
    decae = math.exp(-math.log(2) / T_MEDIO_TC99M_MIN * dtm)
    d, i_ = caso.derecho, caso.izquierdo
    tasa_bolo = caso.actividad_MBq / (caso.duracion_bolo_s / 60.0)
    for j in range(n):
        tmin = t[j] / 60.0
        A[:, j] = (B, T, Cd.sum(), Ci.sum(), Pd, Pi, V, H, G, Ud, Ui)
        entrada = tasa_bolo if caso.t_inyeccion_s <= t[j] < caso.t_inyeccion_s + caso.duracion_bolo_s else 0.0
        furo = caso.t_furo_min is not None and tmin >= caso.t_furo_min + 2.0
        tpd = d.tp_furo if (furo and d.tp_furo) else d.tp
        tpi = i_.tp_furo if (furo and i_.tp_furo) else i_.tp
        tud = d.tu_furo if (furo and d.tu_furo) else d.tu
        tui = i_.tu_furo if (furo and i_.tu_furo) else i_.tu
        kd_, ki_ = E / d.tc, E / i_.tc
        sal_d, sal_i = Cd[-1] * kd_, Ci[-1] * ki_
        dCd = np.empty(E); dCi = np.empty(E)
        dCd[0] = d.k * B - Cd[0] * kd_
        dCi[0] = i_.k * B - Ci[0] * ki_
        dCd[1:] = Cd[:-1] * kd_ - Cd[1:] * kd_
        dCi[1:] = Ci[:-1] * ki_ - Ci[1:] * ki_
        dB = entrada - (caso.kbt + d.k + i_.k + caso.kh) * B + caso.ktb * T
        dT = caso.kbt * B - caso.ktb * T
        dPd = sal_d - Pd / tpd
        dPi = sal_i - Pi / tpi
        dUd = Pd / tpd - Ud / tud
        dUi = Pi / tpi - Ui / tui
        dV = Ud / tud + Ui / tui
        dH = caso.kh * B - H / caso.th
        dG = H / caso.th
        B, T = (B + dtm * dB) * decae, (T + dtm * dT) * decae
        Cd, Ci = (Cd + dtm * dCd) * decae, (Ci + dtm * dCi) * decae
        Pd, Pi = (Pd + dtm * dPd) * decae, (Pi + dtm * dPi) * decae
        V, H, G = (V + dtm * dV) * decae, (H + dtm * dH) * decae, (G + dtm * dG) * decae
        Ud, Ui = (Ud + dtm * dUd) * decae, (Ui + dtm * dUi) * decae
    return t, A


def integrar_cuadros(t, A, duraciones_s):
    """Actividad integrada por cuadro (MBq·s) para cada compartimento: (n_comp, n_cuadros)."""
    bordes = np.concatenate([[0.0], np.cumsum(duraciones_s)])
    acum = np.concatenate([np.zeros((A.shape[0], 1)), np.cumsum((A[:, 1:] + A[:, :-1]) * 0.5 * np.diff(t), axis=1)], axis=1)
    out = np.empty((A.shape[0], len(duraciones_s)))
    for c in range(A.shape[0]):
        v = np.interp(bordes, t, acum[c])
        out[c] = np.diff(v)
    return out


def protocolo():
    """Cuadros: 30 de 2 s (fase vascular) y 87 de 20 s, total 30 min."""
    return np.array([2.0] * 30 + [20.0] * 87)


def verdad_clinica(caso: Caso, t, A):
    """Parámetros verdaderos por riñón (sin fondo, sin ruido): función relativa, tiempo al máximo, T1/2 y C20/máx."""
    res = {}
    tot = caso.derecho.k + caso.izquierdo.k
    for lado, ic, ip, r in (("derecho", 2, 4, caso.derecho), ("izquierdo", 3, 5, caso.izquierdo)):
        curva = A[ic] + A[ip]
        if r.k <= 0 or curva.max() <= 0:
            res[lado] = {"funcion_relativa": 0.0, "tmax_min": None, "t_medio_min": None, "c20_sobre_max": None}
            continue
        imax = int(np.argmax(curva))
        tmax = float(t[imax] / 60.0)
        mitad = np.nonzero(curva[imax:] <= curva[imax] / 2)[0]
        t_medio = float(t[imax + mitad[0]] / 60.0 - tmax) if len(mitad) else None
        i20 = int(np.searchsorted(t, 20 * 60))
        res[lado] = {"funcion_relativa": round(100 * r.k / tot, 1) if tot > 0 else 0.0, "tmax_min": round(tmax, 1),
                     "t_medio_min": round(t_medio, 1) if t_medio is not None else None,
                     "c20_sobre_max": round(float(curva[min(i20, len(curva) - 1)] / curva[imax]), 2) if curva[imax] > 0 else None}
    return res


if __name__ == "__main__":
    for nombre, c in casos().items():
        t, A = simular_curvas(c)
        print(nombre, verdad_clinica(c, t, A), "| vejiga a 30 min:", round(A[6, -1], 1), "MBq; plasma a 30 min:", round(A[0, -1], 1))
