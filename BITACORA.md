# Bitácora de decisiones

---

## 0001 · 2026-10-07 · Renograma con MAG3 por Monte Carlo y compartimentos

**Pedido.** Simular imágenes planares dinámicas de MAG3, después de ampliar la cintigrafía de paratiroides
hasta el corazón (ver simulador-paratiroides-mc, BITACORA 0004).

**Decisiones** (acordadas antes de construir):
- Mismo paciente que las paratiroides: el CT del caso 2 de TCIA cubre de la base del cráneo a la pelvis.
  Fantoma de 56 × 29 × 50 cm a 2 mm, de la cúpula hepática a bajo la vejiga.
- Repositorio aparte (`simulador-renograma-mc`) con el motor Monte Carlo copiado del de paratiroides, ya
  validado (sus pruebas también corren aquí).
- Monte Carlo una vez por compartimento y vista; los cuadros se arman por suma ponderada. Los 18 Monte
  Carlo tardan 32 s; cada caso nuevo, segundos.
- Protocolo: 185 MBq en bolo, 30 cuadros de 2 s y 87 de 20 s, furosemida a los 20 min cuando corresponde.

**Calibración del modelo contra valores clínicos de MAG3.** Normal: máximo renal a 4.7 min, T½ 6.7 min,
C20/máx 0.19 (referencias habituales: máximo de 2 a 5 min, C20/máx < 0.3). Extracción 0.08/min por riñón,
tránsito cortical 3 min en tres etapas, vaciamiento pélvico 2 min, intercambio con el intersticio 0.08 y
0.12/min, 0.4 % hepatobiliar por minuto.

**Trampas.**
- **La grasa del seno renal parecía pelvis.** El umbral de orina (< 18 HU) tomaba también la grasa
  (< −50 HU): pelvis de 53–59 mL. Con −10 a 18 HU no queda casi nada, porque en CT sin contraste un sistema
  colector normal no se distingue; la regla de respaldo (cuarto medial del riñón, 50 mL) era peor. Se usa
  una pelvis geométrica: el riñón a ≤ 13 mm del hilio (7.6 mL por lado).
- **El tránsito exponencial sesgaba la función relativa.** Con la corteza como un solo compartimento, la
  actividad empieza a salir apenas entra, y la integral de 1 a 2.5 min ya pierde parte de lo captado. La
  corteza pasó a ser una cadena de tres etapas (tránsito con tiempo mínimo, como el real).
- **Las regiones de interés salieron del hilio, no del riñón.** Se definieron sobre la suma de las imágenes
  por MBq de corteza y pelvis; la pelvis concentra 1 MBq en 7.6 mL y su pico dominaba: regiones de 20
  píxeles en vez de 300, con el fondo cayendo sobre el propio riñón y restándole su actividad. La función
  relativa salía 92/8 donde la verdad era 70/30. Se separaron las fuentes de error con las imágenes sin
  ruido (renal puro frente a región menos fondo) y se corrigió: contorno desde la corteza (20 % del máximo,
  más un píxel) y fondo separado 2 píxeles.

**Resultado** (medido sobre los cuadros con ruido / verdad):

| caso | función relativa D/I | máximo D / I (min) | C20/máx D / I |
|---|---|---|---|
| normal | 49.9/50.1 · 50/50 | 4.5 / 4.5 · 4.7 / 4.7 | 0.18 / 0.19 · 0.19 / 0.19 |
| función reducida izq. | 66.6/33.4 · 70.2/29.8 | 4.8 / 6.5 · 5.1 / 6.8 | 0.27 / 0.37 · 0.28 / 0.40 |
| obstrucción der. | 50.1/49.9 · 50/50 | 26.5 / 4.5 · 30 / 4.7 | 0.97 / 0.19 · 0.96 / 0.19 |
| dilatación der. | 50/50 · 50/50 | 15.8 / 4.2 · 18.1 / 4.7 | 0.99 / 0.19 · 1.0 / 0.19 |
| no funcionante izq. | 90.6/9.4 · 100/0 | 5.2 / — · 5.5 / — | 0.37 / — · 0.38 / — |

El riñón no funcionante mide 9.4 % por el pool sanguíneo que el fondo no alcanza a restar, como en la
práctica. La diferencia entre medido y verdad es docente: muestra cuánto pesan el fondo y la superposición.

**Abierto.** Uréteres (hoy la pelvis vacía directo a la vejiga), media geométrica con la vista anterior en
la página, regiones dibujadas por el estudiante, Patlak-Rutland, y validación contra SIMIND.

**Agregado (mismo día): cortes axiales con contornos.** `src/cortes_web.py` exporta los 171 cortes del CT
original del abdomen (0.98 mm, ventana 40/400) y las etiquetas de regiones del mismo corte como PNG (8.6 MB
en total, se cargan corte a corte). La página dibuja el borde de cada estructura con un interruptor por
estructura y abre en el corte de los hilios. Revisado a ojo: pelvis en el hilio de cada riñón, cava a la
derecha y aorta a la izquierda delante de la columna.
