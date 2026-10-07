"""Modelo de compartimentos del Tc-99m pertecnetato para la cintigrafía de glándulas salivales con estímulo ácido.

Compartimentos: plasma, intersticio, parótida D/I, submaxilar D/I, tiroides, boca, estómago (MBq).
    glándula' = k·B − (kr + s(t))·G          k captación (NIS), kr retorno al plasma, s secreción a la boca
    s(t) = s_basal, y s_estimulo durante dur_estimulo minutos desde el jugo de limón
    boca' = Σ s(t)·G − boca / t_trago,  estómago' = boca / t_trago + k_est·B
    tiroides' = k_t·B − kr_t·Tir
    B' = bolo − (kbt + k_renal + Σk + k_t + k_est)·B + ktb·T + Σ kr·G + kr_t·Tir,  T' = kbt·B − ktb·T
Calibración (valores clínicos habituales): máximo glandular a los 15–20 min sin estímulo, captación de una
parótida normal ~0.5–0.8 % de la actividad inyectada a los 20 min, fracción de excreción tras el limón
~50–65 % en la parótida y ~40–55 % en la submaxilar, tiroides ~1 % a los 20 min. Con kr 0.03, s 0.02 y
k 0.002/min la parótida capta 0.7 %, llega al máximo a los ~15 min y excreta ~55 % con el limón. Decaimiento del Tc-99m incluido.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

T_MEDIO_MIN = 6.0067 * 60.0
COMPARTIMENTOS = ["plasma", "intersticio", "parotida_d", "parotida_i", "submax_d", "submax_i", "tiroides", "boca", "estomago"]
GLANDULAS = ["parotida_d", "parotida_i", "submax_d", "submax_i"]


@dataclass
class Glandula:
    k: float               # 1/min
    kr: float = 0.03       # 1/min
    s: float = 0.02        # 1/min, secreción basal
    s_est: float = 0.32    # 1/min, secreción durante el estímulo


PAROTIDA = Glandula(k=0.0020)
SUBMAX = Glandula(k=0.0012, s_est=0.22)


@dataclass
class Caso:
    nombre: str
    descripcion: str
    glandulas: dict = field(default_factory=lambda: {"parotida_d": PAROTIDA, "parotida_i": PAROTIDA, "submax_d": SUBMAX, "submax_i": SUBMAX})
    actividad_MBq: float = 185.0
    kbt: float = 0.20
    ktb: float = 0.05
    k_renal: float = 0.010
    k_t: float = 0.0020
    kr_t: float = 0.02
    k_est: float = 0.002
    t_trago: float = 3.0
    t_estimulo_min: float = 20.0
    dur_estimulo: float = 3.0
    t_inyeccion_s: float = 5.0
    duracion_bolo_s: float = 10.0


def casos():
    def g(**kw):
        base = {"parotida_d": PAROTIDA, "parotida_i": PAROTIDA, "submax_d": SUBMAX, "submax_i": SUBMAX}
        base.update(kw)
        return base
    return {
        "normal": Caso("normal", "Captación y excreción normales en las cuatro glándulas."),
        "sjogren": Caso("sjogren", "Síndrome de Sjögren: captación y excreción disminuidas en las cuatro glándulas.",
                        g(parotida_d=Glandula(k=0.0007, s_est=0.09), parotida_i=Glandula(k=0.0007, s_est=0.09),
                          submax_d=Glandula(k=0.0004, s_est=0.06), submax_i=Glandula(k=0.0004, s_est=0.06))),
        "litiasis-parotida-der": Caso("litiasis-parotida-der", "Obstrucción del conducto de la parótida derecha (litiasis): capta y no se vacía con el limón.",
                                      g(parotida_d=Glandula(k=0.0020, s=0.002, s_est=0.004))),
        "dano-radioyodo": Caso("dano-radioyodo", "Daño tras radioyodo: parótidas afectadas en forma desigual, submaxilares menos.",
                               g(parotida_d=Glandula(k=0.0011, s_est=0.16), parotida_i=Glandula(k=0.0006, s_est=0.08),
                                 submax_d=Glandula(k=0.0010, s_est=0.17), submax_i=Glandula(k=0.0010, s_est=0.16))),
        "submaxilar-izq-ausente": Caso("submaxilar-izq-ausente", "Submaxilar izquierda sin función (resecada o atrófica).",
                                       g(submax_i=Glandula(k=0.0, s=0.0, s_est=0.0))),
    }


def simular_curvas(c: Caso, t_fin_min: float = 40.0, dt_s: float = 0.5):
    n = int(round(t_fin_min * 60 / dt_s)) + 1
    t = np.arange(n) * dt_s
    A = np.zeros((len(COMPARTIMENTOS), n))
    x = dict.fromkeys(COMPARTIMENTOS, 0.0)
    dtm = dt_s / 60.0
    decae = math.exp(-math.log(2) / T_MEDIO_MIN * dtm)
    bolo = c.actividad_MBq / (c.duracion_bolo_s / 60.0)
    G = c.glandulas
    ksum = sum(G[gl].k for gl in GLANDULAS)
    for j in range(n):
        tm = t[j] / 60.0
        for i, comp in enumerate(COMPARTIMENTOS):
            A[i, j] = x[comp]
        est = c.t_estimulo_min <= tm < c.t_estimulo_min + c.dur_estimulo
        B, T = x["plasma"], x["intersticio"]
        d = {}
        retorno = 0.0
        secrecion = 0.0
        for gl in GLANDULAS:
            p = G[gl]
            s = p.s_est if est else p.s
            d[gl] = p.k * B - (p.kr + s) * x[gl]
            retorno += p.kr * x[gl]
            secrecion += s * x[gl]
        entrada = bolo if c.t_inyeccion_s <= t[j] < c.t_inyeccion_s + c.duracion_bolo_s else 0.0
        d["plasma"] = entrada - (c.kbt + c.k_renal + ksum + c.k_t + c.k_est) * B + c.ktb * T + retorno + c.kr_t * x["tiroides"]
        d["intersticio"] = c.kbt * B - c.ktb * T
        d["tiroides"] = c.k_t * B - c.kr_t * x["tiroides"]
        d["boca"] = secrecion - x["boca"] / c.t_trago
        d["estomago"] = x["boca"] / c.t_trago + c.k_est * B
        for comp in COMPARTIMENTOS:
            x[comp] = (x[comp] + dtm * d[comp]) * decae
    return t, A


def verdad_clinica(c: Caso, t, A):
    res = {}
    i_pre = int(np.searchsorted(t, c.t_estimulo_min * 60)) - 1
    i_fin = int(np.searchsorted(t, (c.t_estimulo_min + 5) * 60))
    for gl in GLANDULAS:
        g = A[COMPARTIMENTOS.index(gl)]
        if g.max() <= 0:
            res[gl] = {"captacion_pct": 0.0, "tmax_min": None, "fraccion_excrecion_pct": None}
            continue
        pre = g[i_pre]
        res[gl] = {"captacion_pct": round(100 * pre / c.actividad_MBq, 2), "tmax_min": round(float(t[int(np.argmax(g[:i_pre + 1]))] / 60), 1),
                   "fraccion_excrecion_pct": round(100 * (pre - g[i_pre:i_fin].min()) / pre, 1)}
    return res


def protocolo():
    return np.array([30.0] * 80)            # 80 cuadros de 30 s = 40 min


if __name__ == "__main__":
    for n, c in casos().items():
        t, A = simular_curvas(c)
        v = verdad_clinica(c, t, A)
        tir = A[COMPARTIMENTOS.index("tiroides"), int(np.searchsorted(t, 1200))] / c.actividad_MBq * 100
        print(n, {k: (v[k]["captacion_pct"], v[k]["tmax_min"], v[k]["fraccion_excrecion_pct"]) for k in v}, f"tiroides 20 min {tir:.2f} %")
