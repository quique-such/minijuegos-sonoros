# Minijuegos controlados por voz

[![Tecnologías](https://skillicons.dev/icons?i=py)](https://skillicons.dev)

Cuatro minijuegos que se controlan con la voz: el micrófono hace de mando y el juego responde al volumen, al tono y a la frecuencia.

![Minijuegos controlados por voz](docs/preview.jpg)

## Qué hace

- Menú principal desde el que se lanza cada juego: Flappy, Pesca, Invasores y Simón.
- Análisis del audio en tiempo real: volumen (RMS y dB), espectro con FFT y detección de notas.
- Indicadores en pantalla, como la barra de volumen en tiempo real.

## Cómo ejecutarlo

Hace falta un micrófono.

```bash
pip install pygame sounddevice pyaudio numpy scipy matplotlib
python main.py
```

## Contenido

| Fichero | Qué es |
|---|---|
| `main.py` | Menú principal |
| `flappy.py, pesca.py, invasores.py, simon.py` | Un fichero por juego |
| `ABRE ESTO.ipynb` | Guía del proyecto: cómo se juega y qué procesado de audio usa cada juego |

---

Proyecto en equipo del Grado en Tecnología Digital y Multimedia (UPV). Forma parte de mi [portfolio](https://quique-such.github.io/portafolio/).
