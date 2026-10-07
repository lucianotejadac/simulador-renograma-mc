# Simulador Monte Carlo de renograma con MAG3

Renogramas dinámicos simulados con Tc-99m MAG3 sobre un CT real: la física de la cámara (atenuación,
dispersión Compton, colimador LEHR, ventana de energía, ruido de Poisson) se simula fotón a fotón y la
cinética del radiofármaco con un modelo de compartimentos. Cada caso tiene la verdad conocida.

Página con los cinco casos (cine, regiones de interés, renograma, verdad oculta y DICOM descargables):
<https://lucianotejadac.github.io/simulador-renograma-mc/>

## Cómo funciona

1. `src/fantoma.py`: el CT del caso 2 de la entrega docente PET/CT de TCIA, de la cúpula hepática a la pelvis,
   remuestreado a 2 mm; HU → μ a 140 keV.
2. `src/regiones.py`: riñones, vejiga, hígado, bazo, vesícula, vasos e intestino con
   [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) en CPU; corteza y pelvis renal separadas
   (pelvis geométrica en el hilio, porque sin contraste el sistema colector normal no se ve).
3. `src/modelo_mag3.py`: plasma, intersticio, corteza (cadena de tres etapas, tránsito con tiempo mínimo),
   pelvis, vejiga, hígado e intestino. Parámetros por riñón: extracción, tránsito cortical, vaciamiento
   pélvico y respuesta a la furosemida.
4. `src/sensibilidades.py`: Monte Carlo una vez por compartimento y vista (posterior y anterior), 128 × 128
   de 4.8 mm. El motor (`src/montecarlo.py`) es el de
   [simulador-paratiroides-mc](https://github.com/lucianotejadac/simulador-paratiroides-mc), con sus pruebas.
5. `src/dinamico.py`: cada cuadro es la suma de las imágenes por compartimento ponderadas por la actividad
   integrada en ese cuadro, con ruido de Poisson. Regiones automáticas desde la anatomía, fondo en media luna,
   y los parámetros clínicos medidos como en la práctica.
6. `src/exportar.py`: DICOM NM dinámico (dos fases: 30 × 2 s y 87 × 20 s) y datos de la página.
7. `src/cortes_web.py`: cortes axiales del CT original con los contornos, para la página.

```bash
python src/fantoma.py
TOTALSEG_EXE=... python src/regiones.py
python src/sensibilidades.py     # ~30 s en 16 hilos
python src/dinamico.py           # los cinco casos, segundos
python src/exportar.py
python -m pytest -q tests
```

Requisitos: numpy, scipy, pydicom, pillow, nibabel, numba==0.61.2 y, para el paso 2, totalsegmentator con
torch==2.8.0 y torchvision==0.23.0 de CPU, pandas==2.2.3 y connected-components-3d==3.18.0 (versiones que
pasan el Control de aplicaciones de Windows).

## Casos

| caso | qué tiene |
|---|---|
| 1 | normal |
| 2 | riñón izquierdo con función reducida (30 %) y tránsito lento |
| 3 | obstrucción pieloureteral derecha sin respuesta a la furosemida |
| 4 | pelvis derecha dilatada no obstructiva, que vacía tras la furosemida |
| 5 | riñón izquierdo no funcionante |

Detalles, calibración y trampas en [BITACORA.md](BITACORA.md). Código MIT; los DICOM llevan paciente
sintético `SIM-RENO-nn` y conservan la anatomía de un paciente anónimo de TCIA.

## Editar contornos

En la página, «Editar contornos» bajo el corte axial: pincel, borrador, deshacer y copiar del corte vecino.
Las correcciones se descargan como JSON; `python src/importar_correcciones.py archivo.json --rehacer` las
aplica a la grilla del fantoma y rehace la simulación.
