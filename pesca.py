# --- IMPORTACIONES ---
import pygame
import numpy as np
import pyaudio
import pygame.mixer
import threading
import queue
import time
import random
import math
import os
from collections import deque
from scipy.fft import rfft, rfftfreq

# --- INICIALIZACIÓN ---
ANCHURA_PANTALLA = 1080
ALTURA_PANTALLA = 600
FOTOGRAMAS_POR_SEGUNDO = 30  # FPS

# Audio y detección de tono
FRECUENCIA_MAXIMA_DETECTABLE = 2093.00 # Hz (C7)
FRECUENCIA_MINIMA_DETECTABLE = 65.41 # Hz (C2)
UMBRAL_DECIBELIOS_SILENCIO = -50.0 # dB (Silencio)
VOLUMEN_MAXIMO_EXTENSION_TOTAL_CANA = -10.0 # dB

# --- CONFIGURACIÓN DE ENTRADA DE SONIDO ---
TASA_MUESTREO_AUDIO = 44100 # Hertz
TAMANO_FRAGMENTO_AUDIO = 4096 # Número de muestras de audio por bloque (chunk)
FORMATO_DATOS_AUDIO = pyaudio.paFloat32 # Formato de las muestras de audio
NUMERO_CANALES_AUDIO = 1 # Mono

# Colores
COLOR_CANA_PESCAR = (101, 67, 33)
NEGRO = (0, 0, 0)
VERDE = (0, 200, 0)
BLANCO = (255, 255, 255)

# Caña de Pescar
LONGITUD_CANA_POR_DEFECTO = 100 # Longitud inicial en silencio
LONGITUD_MAXIMA_CANA = 590 # Máxima extensión de la caña
LONGITUD_MINIMA_CANA = 50  # Mínima extensión de la caña (cuando hay sonido pero es bajo)
RADIO_ANZUELO_CIRCULO = 5 # Radio del círculo que representa el anzuelo

# --- NOTAS MUSICALES ---
FRECUENCIAS_NOTAS_MUSICALES = {
    'C2': 65.41, 'C#2': 69.30, 'D2': 73.42, 'D#2': 77.78, 'E2': 82.41,
    'F2': 87.31, 'F#2': 92.50, 'G2': 98.00, 'G#2': 103.83, 'A2': 110.00,
    'A#2': 116.54, 'B2': 123.47,

    'C3': 130.81, 'C#3': 138.59, 'D3': 146.83, 'D#3': 155.56, 'E3': 164.81,
    'F3': 174.61, 'F#3': 185.00, 'G3': 196.00, 'G#3': 207.65, 'A3': 220.00,
    'A#3': 233.08, 'B3': 246.94,

    'C4': 261.63, 'C#4': 277.18, 'D4': 293.66, 'D#4': 311.13, 'E4': 329.63,
    'F4': 349.23, 'F#4': 369.99, 'G4': 392.00, 'G#4': 415.30, 'A4': 440.00,
    'A#4': 466.16, 'B4': 493.88,

    'C5': 523.25, 'C#5': 554.37, 'D5': 587.33, 'D#5': 622.25, 'E5': 659.25,
    'F5': 698.46, 'F#5': 739.99, 'G5': 783.99, 'G#5': 830.61, 'A5': 880.00,
    'A#5': 932.33, 'B5': 987.77,

    'C6': 1046.50, 'C#6': 1108.73, 'D6': 1174.66, 'D#6': 1244.51, 'E6': 1318.51,
    'F6': 1396.91, 'F#6': 1479.98, 'G6': 1567.98, 'G#6': 1661.22, 'A6': 1760.00,
    'A#6': 1864.66, 'B6': 1975.53,
}

def detectar_tono_fundamental(datos_audio_crudo, tasa_muestreo_hz=TASA_MUESTREO_AUDIO):
    """
    Analiza una muestra de datos de audio para estimar su frecuencia fundamental (tono).
    Utiliza la Transformada Rápida de Fourier (RFFT) y busca el pico de mayor amplitud
    dentro del rango de frecuencias definido por FRECUENCIA_MINIMA_DETECTABLE y
    FRECUENCIA_MAXIMA_DETECTABLE. Aplica un leve suavizado espectral y una
    interpolación cuadrática para mejorar la precisión de la frecuencia detectada.
    """
    ventana_hanning = np.hanning(len(datos_audio_crudo))
    datos_audio_ventaneados = datos_audio_crudo * ventana_hanning
    amplitudes_fft = np.abs(rfft(datos_audio_ventaneados))
    frecuencias_fft = rfftfreq(len(datos_audio_crudo), 1 / tasa_muestreo_hz)

    # Filtrar frecuencias fuera del rango global de detección (C2-C7)
    indices_en_rango_frecuencias = np.where((frecuencias_fft >= FRECUENCIA_MINIMA_DETECTABLE) & (frecuencias_fft <= FRECUENCIA_MAXIMA_DETECTABLE))[0]

    if len(indices_en_rango_frecuencias) == 0:
        return 0.0 # No hay componentes de frecuencia en el rango detectable

    amplitudes_fft_en_rango = amplitudes_fft[indices_en_rango_frecuencias]

    if len(amplitudes_fft_en_rango) == 0:
        return 0.0

    # Suavizado espectral leve para mejorar la selección del pico en señales ruidosas o débiles
    amplitudes_procesadas_para_pico = amplitudes_fft_en_rango
    if len(amplitudes_fft_en_rango) >= 5: # Aplicar solo si hay suficientes puntos para el kernel
        tamano_kernel = 5
        filtro_suavizado = np.ones(tamano_kernel) / tamano_kernel
        amplitudes_procesadas_para_pico = np.convolve(amplitudes_fft_en_rango, filtro_suavizado, mode='same')
    
    # Encontrar el pico en el espectro (posiblemente suavizado)
    indice_pico_en_rango_filtrado = np.argmax(amplitudes_procesadas_para_pico)
    indice_pico_espectro_completo = indices_en_rango_frecuencias[indice_pico_en_rango_filtrado]

    # Usar la amplitud del pico del espectro *original* (no suavizado) para el umbral de detección
    if amplitudes_fft[indice_pico_espectro_completo] < 0.15: # Umbral de amplitud mínima para considerar un tono
        return 0.0

    frecuencia_fundamental_estimada = frecuencias_fft[indice_pico_espectro_completo]

    # Interpolación cuadrática para afinar la estimación de la frecuencia del pico
    if 1 < indice_pico_espectro_completo < len(amplitudes_fft) - 1:
        # Amplitudes de los puntos adyacentes al pico en el espectro original
        alpha = amplitudes_fft[indice_pico_espectro_completo - 1]
        beta = amplitudes_fft[indice_pico_espectro_completo] # Amplitud del pico
        gamma = amplitudes_fft[indice_pico_espectro_completo + 1]
        
        denominador_interpolacion = alpha - 2 * beta + gamma
        if denominador_interpolacion != 0:
            correccion_interpolacion_cuadratica = 0.5 * (alpha - gamma) / denominador_interpolacion
            # Solo aplicar corrección si es pequeña (dentro de un bin de frecuencia)
            if abs(correccion_interpolacion_cuadratica) < 0.5:
                diferencia_frecuencia_entre_bins_fft = frecuencias_fft[1] - frecuencias_fft[0]
                frecuencia_fundamental_estimada += correccion_interpolacion_cuadratica * diferencia_frecuencia_entre_bins_fft
    
    # Asegurar que la frecuencia detectada esté dentro de los límites globales
    frecuencia_fundamental_estimada = max(FRECUENCIA_MINIMA_DETECTABLE, min(frecuencia_fundamental_estimada, FRECUENCIA_MAXIMA_DETECTABLE))
    
    return frecuencia_fundamental_estimada

# --- PECES ---
class Pez:
    """
    Representa un pez en el juego, con su posición, velocidad, tamaño, imagen y puntuación.
    """
    def __init__(self, x, y, imagen_pez=None, puntos_por_captura=1):
        self.x = x  # Posición horizontal actual
        self.y = y  # Posición vertical actual
        self.velocidad = random.uniform(1, 3) # Velocidad de nado (píxeles por fotograma)
        self.direccion_movimiento = random.choice([-1, 1]) # -1 para izquierda, 1 para derecha
        self.tamano_visual = random.randint(20, 40) 
        self.imagen_pez = imagen_pez # Imagen del pez
        self.puntos_por_captura = puntos_por_captura # Puntos otorgados al pescar este pez

        if self.imagen_pez:
            # Escalar la imagen del pez a un tamaño estándar para consistencia visual
            self.imagen_pez = pygame.transform.scale(self.imagen_pez, (80, 48))

    def actualizar_posicion(self): 
        """
        Actualiza la posición horizontal del pez.
        Si el pez alcanza los bordes laterales de la pantalla, invierte su dirección de movimiento.
        """
        self.x += self.velocidad * self.direccion_movimiento
        if self.x < 0 or self.x > ANCHURA_PANTALLA:
            self.direccion_movimiento *= -1 # Invertir dirección

    def dibujar_en_pantalla(self, superficie_destino):
        """
        Dibuja el pez en la superficie de Pygame especificada (normalmente la pantalla principal).
        Si el pez tiene una imagen asignada, la dibuja. De lo contrario, dibuja una elipse.
        """
        # Voltear la imagen horizontalmente si el pez se mueve hacia la izquierda
        imagen_a_renderizar = pygame.transform.flip(self.imagen_pez, self.direccion_movimiento < 0, False)
        rectangulo_imagen = imagen_a_renderizar.get_rect(center=(self.x, self.y))
        superficie_destino.blit(imagen_a_renderizar, rectangulo_imagen)

    def comprobar_colision_con_anzuelo(self, anzuelo_pos_x, anzuelo_pos_y):
        """
        Verifica si el anzuelo (representado por un punto y un radio) ha colisionado con el pez.
        La colisión se detecta de forma diferente si el pez tiene una imagen (rectángulo)
        o si es una elipse (aproximada como círculo).
        """
        if self.imagen_pez:
            # Colisión círculo (anzuelo) vs rectángulo (imagen del pez)
            rect_pez = self.imagen_pez.get_rect(center=(self.x, self.y))
            
            # Encontrar el punto en el rectángulo del pez más cercano al centro del anzuelo
            punto_mas_cercano_x = max(rect_pez.left, min(anzuelo_pos_x, rect_pez.right))
            punto_mas_cercano_y = max(rect_pez.top, min(anzuelo_pos_y, rect_pez.bottom))
            
            # Calcular la distancia al cuadrado entre el centro del anzuelo y este punto cercano
            distancia_al_cuadrado = (anzuelo_pos_x - punto_mas_cercano_x)**2 + (anzuelo_pos_y - punto_mas_cercano_y)**2
            
            # Hay colisión si la distancia es menor que el radio del anzuelo al cuadrado
            return distancia_al_cuadrado < (RADIO_ANZUELO_CIRCULO**2)
        else:
            # Colisión círculo (anzuelo) vs elipse (pez sin imagen), aproximada como círculo-círculo
            # Se usa el radio horizontal de la elipse (tamano_visual / 2) como radio aproximado del pez.
            radio_aproximado_pez = self.tamano_visual / 2
            distancia_entre_centros = math.sqrt((anzuelo_pos_x - self.x)**2 + (anzuelo_pos_y - self.y)**2)
            
            # Colisión si la distancia entre centros es menor que la suma de los radios (pez + anzuelo)
            return distancia_entre_centros < (radio_aproximado_pez + RADIO_ANZUELO_CIRCULO)

# --- PROCESADOR DE AUDIO ---
class ProcesadorAudio(threading.Thread):
    """
    Gestiona la captura y procesamiento de audio en un hilo separado.
    Detecta el tono (frecuencia fundamental) y el volumen (en dB) de la entrada del micrófono.
    Comunica los resultados al hilo principal del juego a través de una cola.
    """
    def __init__(self, cola_datos_procesados_audio, indice_dispositivo_microfono=None):
        super().__init__()
        self.daemon = True # El hilo terminará cuando el programa principal termine
        self.cola_datos_procesados_audio = cola_datos_procesados_audio
        self.indice_dispositivo_microfono = indice_dispositivo_microfono

        # Colas para suavizar las lecturas de tono y volumen
        self.historial_tonos_reales_hz = deque(maxlen=7) # Almacena las últimas N detecciones de frecuencia
        self.historial_volumenes_db = deque(maxlen=3)  # Almacena las últimas N mediciones de volumen

        self.en_ejecucion = True 

        # Estado del audio detectado
        self.frecuencia_fundamental_cantada_hz = 0.0 # Frecuencia promedio suavizada
        self.nota_musical_cantada = ""               # Nota musical más cercana a la frecuencia
        self.tono_mapeado_para_juego_hz = 0.0        # Tono ajustado a la octava de juego actual
        self.estado_tono_en_octava_juego = "silencio" # Indica si el tono está en, por encima, o por debajo de la octava de juego
        self.volumen_actual_db = UMBRAL_DECIBELIOS_SILENCIO # Volumen promedio suavizado

        # Límites de frecuencia para la octava de juego activa (establecidos desde el hilo principal)
        self.frecuencia_minima_octava_juego_hz = 0
        self.frecuencia_maxima_octava_juego_hz = 0

    def forzar_reevaluacion_estado_en_octava(self):
        """
        Indica que la frecuencia actual debe ser reevaluada contra los límites de la octava de juego.
        Útil cuando los límites de la octava de juego cambian dinámicamente.
        """
        self._ultima_frecuencia_evaluada_para_estado = -1

    def _actualizar_estado_y_tono_para_juego(self):
        """
        Lógica interna para actualizar self.tono_mapeado_para_juego_hz y
        self.estado_tono_en_octava_juego, basándose en la frecuencia cantada
        y los límites de la octava de juego actual.
        """
        if not self.nota_musical_cantada and self.frecuencia_fundamental_cantada_hz == 0:
            self.estado_tono_en_octava_juego = "silencio"
            return

        # Usar la frecuencia canónica de la nota real si está disponible; si no, la frecuencia detectada.
        frecuencia_a_comparar = FRECUENCIAS_NOTAS_MUSICALES.get(self.nota_musical_cantada, self.frecuencia_fundamental_cantada_hz)
        if frecuencia_a_comparar == 0: # Fallback si la nota no está en el diccionario y la frecuencia real es 0
            frecuencia_a_comparar = self.frecuencia_fundamental_cantada_hz

        if self.frecuencia_minima_octava_juego_hz <= 0 or self.frecuencia_maxima_octava_juego_hz <= 0:
            self.estado_tono_en_octava_juego = "silencio" # O un estado de error/advertencia
            self.tono_mapeado_para_juego_hz = 0
            return

        # Comparar la frecuencia con los límites de la octava de juego
        if frecuencia_a_comparar < self.frecuencia_minima_octava_juego_hz:
            self.tono_mapeado_para_juego_hz = self.frecuencia_minima_octava_juego_hz
            self.estado_tono_en_octava_juego = "debajo_octava_activa"
        elif frecuencia_a_comparar > self.frecuencia_maxima_octava_juego_hz:
            self.tono_mapeado_para_juego_hz = self.frecuencia_maxima_octava_juego_hz
            self.estado_tono_en_octava_juego = "encima_octava_activa"
        else:
            # La frecuencia está dentro de la octava de juego
            self.tono_mapeado_para_juego_hz = frecuencia_a_comparar
            # Asegurar que tono_mapeado_para_juego_hz esté estrictamente dentro de los límites (clamp)
            self.tono_mapeado_para_juego_hz = max(self.frecuencia_minima_octava_juego_hz,
                                                min(self.tono_mapeado_para_juego_hz, self.frecuencia_maxima_octava_juego_hz))
            self.estado_tono_en_octava_juego = "en_octava_activa"
        
        self._ultima_frecuencia_evaluada_para_estado = self.frecuencia_fundamental_cantada_hz

    def run(self):
        """
        Bucle principal del hilo de procesamiento de audio.
        Lee continuamente datos del micrófono, calcula el tono y el volumen,
        y envía los resultados a la cola para el hilo principal del juego.
        """
        gestor_audio = pyaudio.PyAudio()
        flujo_audio = gestor_audio.open(format=FORMATO_DATOS_AUDIO,
                                    channels=NUMERO_CANALES_AUDIO,
                                    rate=TASA_MUESTREO_AUDIO,
                                    input=True,
                                    frames_per_buffer=TAMANO_FRAGMENTO_AUDIO,
                                    input_device_index=self.indice_dispositivo_microfono)
        try:
            while self.en_ejecucion:
                buffer_datos_crudos = flujo_audio.read(TAMANO_FRAGMENTO_AUDIO, exception_on_overflow=False)
                datos_audio_numpy = np.frombuffer(buffer_datos_crudos, dtype=np.float32)
                
                # Calcular volumen (RMS convertido a dB)
                valor_rms = np.sqrt(np.mean(datos_audio_numpy**2)) if np.any(datos_audio_numpy) else 0
                volumen_decibelios = 20 * np.log10(valor_rms) if valor_rms > 0 else UMBRAL_DECIBELIOS_SILENCIO
                
                # Suavizar volumen
                self.historial_volumenes_db.append(volumen_decibelios)
                self.volumen_actual_db = sum(self.historial_volumenes_db) / len(self.historial_volumenes_db)

                if self.volumen_actual_db > UMBRAL_DECIBELIOS_SILENCIO:
                    frecuencia_detectada_cruda = detectar_tono_fundamental(datos_audio_numpy, TASA_MUESTREO_AUDIO)

                    if frecuencia_detectada_cruda > 0:
                        # Suavizar frecuencia
                        self.historial_tonos_reales_hz.append(frecuencia_detectada_cruda)
                        nueva_frecuencia_cantada_promedio = sum(self.historial_tonos_reales_hz) / len(self.historial_tonos_reales_hz)
                        
                        # Actualizar estado solo si la frecuencia cambió significativamente o se forzó reevaluación
                        if abs(nueva_frecuencia_cantada_promedio - self.frecuencia_fundamental_cantada_hz) > 0.1 or \
                        self._ultima_frecuencia_evaluada_para_estado != nueva_frecuencia_cantada_promedio:
                            self.frecuencia_fundamental_cantada_hz = nueva_frecuencia_cantada_promedio
                            self.nota_musical_cantada = self.obtener_nota_musical_cercana(self.frecuencia_fundamental_cantada_hz)
                            self._actualizar_estado_y_tono_para_juego()
                    else:
                        # Si detectar_tono devuelve 0 (sin tono claro), considerar silencio para la nota
                        if self.nota_musical_cantada != "" or self.frecuencia_fundamental_cantada_hz != 0: # Si antes había una nota
                            self.historial_tonos_reales_hz.clear()
                            self.frecuencia_fundamental_cantada_hz = 0
                            self.nota_musical_cantada = ""
                            self._actualizar_estado_y_tono_para_juego()
                else:
                    # Silencio detectado por bajo volumen
                    if self.nota_musical_cantada != "" or self.frecuencia_fundamental_cantada_hz != 0: # Si antes había una nota
                        self.historial_tonos_reales_hz.clear()
                        self.frecuencia_fundamental_cantada_hz = 0
                        self.nota_musical_cantada = ""
                        self._actualizar_estado_y_tono_para_juego()
                
                # Enviar datos procesados al hilo principal del juego
                try:
                    self.cola_datos_procesados_audio.put_nowait((
                        self.tono_mapeado_para_juego_hz,
                        self.volumen_actual_db,
                        self.nota_musical_cantada,
                        self.frecuencia_fundamental_cantada_hz,
                        self.estado_tono_en_octava_juego
                    ))
                except queue.Full:
                    pass # No hay nuevos datos de audio esta vez
                time.sleep(0.01)
        finally:
            flujo_audio.stop_stream()
            flujo_audio.close()
            gestor_audio.terminate()

    def obtener_nota_musical_cercana(self, frecuencia_hz):
        """
        Encuentra la nota musical (ej: "A4", "C#3") cuya frecuencia fundamental
        está logarítmicamente más cerca a la frecuencia_hz proporcionada.
        """
        if frecuencia_hz == 0:
            return ""
        
        nombre_nota_cercana = min(
            FRECUENCIAS_NOTAS_MUSICALES.keys(),
            key=lambda nota: abs(math.log2(frecuencia_hz / FRECUENCIAS_NOTAS_MUSICALES[nota]))
        )
        return nombre_nota_cercana

    def detener_procesamiento(self):
        """
        Establece la bandera self.en_ejecucion a False para que el bucle principal
        del hilo termine de forma segura en la próxima iteración.
        """
        self.en_ejecucion = False

# --- FUNCIÓN PRINCIPAL ---
def main(pantalla_principal, indice_microfono_seleccionado):
    """
    Función principal que contiene el bucle del juego y toda la lógica de la pesca por voz.
    """
    pygame.display.set_caption('Minijuego - Pesca por Voz')
    reloj_juego = pygame.time.Clock()

    # Fuente
    ruta_archivo_fuente = os.path.join("fonts", "minecraft.ttf")
    fuente_textos_juego = pygame.font.Font(ruta_archivo_fuente, 16)
    fuente_simbolos_botones = pygame.font.Font(ruta_archivo_fuente, 26)

    pygame.mixer.init()
    sonido_captura_pez = pygame.mixer.Sound(os.path.join("pesca", "pop.wav")) # Sonido de captura de pez

    # Fondo
    imagen_fondo_juego = pygame.image.load(os.path.join("fondos", "pesca.png")).convert()
    imagen_fondo_juego = pygame.transform.scale(imagen_fondo_juego, (ANCHURA_PANTALLA, ALTURA_PANTALLA))
    
    # --- PECES ---
    # Peces normales
    nombres_archivos_peces_normales = ["pez.png", "pez2.png", "pez3.png", "pez4.png", "pez5.png"]
    imagenes_peces_normales_cargadas = []
    for nombre_img_pez in nombres_archivos_peces_normales:
        ruta_completa_img_pez = os.path.join("pesca", nombre_img_pez)
        imagen_cargada = pygame.image.load(ruta_completa_img_pez).convert_alpha()
        imagenes_peces_normales_cargadas.append(imagen_cargada)
            
    # Pez especial
    imagen_pez_especial_cargada = pygame.image.load(os.path.join("pesca", "tortuga.png")).convert_alpha()

    # Lista inicial de peces
    lista_peces_en_pantalla = []
    for _ in range(8): # Número inicial de peces normales
        imagen_aleatoria_pez = random.choice(imagenes_peces_normales_cargadas) if imagenes_peces_normales_cargadas else None
        nuevo_pez = Pez(random.randint(50, ANCHURA_PANTALLA - 50), 
                        random.randint(ALTURA_PANTALLA // 2, ALTURA_PANTALLA - 50), 
                        imagen_pez=imagen_aleatoria_pez, puntos_por_captura=1)
        lista_peces_en_pantalla.append(nuevo_pez)

    # --- OCTAVAS ---
    # Cada diccionario define una octava que el jugador puede seleccionar para controlar el anzuelo
    OCTAVAS_DE_JUEGO_CONFIG = [
        {"nombre": "C2 - B2", "notas": ['C2', 'C#2', 'D2', 'D#2', 'E2', 'F2', 'F#2', 'G2', 'G#2', 'A2', 'A#2', 'B2'], "freq_min": FRECUENCIAS_NOTAS_MUSICALES.get('C2'), "freq_max": FRECUENCIAS_NOTAS_MUSICALES.get('B2')},
        {"nombre": "C3 - B3", "notas": ['C3', 'C#3', 'D3', 'D#3', 'E3', 'F3', 'F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3'], "freq_min": FRECUENCIAS_NOTAS_MUSICALES.get('C3'), "freq_max": FRECUENCIAS_NOTAS_MUSICALES.get('B3')},
        {"nombre": "C4 - B4", "notas": ['C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4'], "freq_min": FRECUENCIAS_NOTAS_MUSICALES.get('C4'), "freq_max": FRECUENCIAS_NOTAS_MUSICALES.get('B4')}, # Octava por defecto
        {"nombre": "C5 - B5", "notas": ['C5', 'C#5', 'D5', 'D#5', 'E5', 'F5', 'F#5', 'G5', 'G#5', 'A5', 'A#5', 'B5'], "freq_min": FRECUENCIAS_NOTAS_MUSICALES.get('C5'), "freq_max": FRECUENCIAS_NOTAS_MUSICALES.get('B5')},
        {"nombre": "C6 - B6", "notas": ['C6', 'C#6', 'D6', 'D#6', 'E6', 'F6', 'F#6', 'G6', 'G#6', 'A6', 'A#6', 'B6'], "freq_min": FRECUENCIAS_NOTAS_MUSICALES.get('C6'), "freq_max": FRECUENCIAS_NOTAS_MUSICALES.get('B6')}
    ]
    indice_octava_juego_seleccionada = 2 # Por defecto C4-B4

    # Barra de octavas
    LISTA_COMPLETA_NOTAS_VISUALES = [
        'C2', 'C#2', 'D2', 'D#2', 'E2', 'F2', 'F#2', 'G2', 'G#2', 'A2', 'A#2', 'B2',
        'C3', 'C#3', 'D3', 'D#3', 'E3', 'F3', 'F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3',
        'C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4',
        'C5', 'C#5', 'D5', 'D#5', 'E5', 'F5', 'F#5', 'G5', 'G#5', 'A5', 'A#5', 'B5',
        'C6', 'C#6', 'D6', 'D#6', 'E6', 'F6', 'F#6', 'G6', 'G#6', 'A6', 'A#6', 'B6',
        'C7' 
    ]
    NUM_TOTAL_SEMITONOS_VISUALIZADOS = len(LISTA_COMPLETA_NOTAS_VISUALES)

    # Octava actual
    notas_octava_juego_activa = []
    nombre_octava_juego_actual_str = ""
    num_notas_en_octava_juego_activa = 0
    anchura_segmento_nota_juego_px = ANCHURA_PANTALLA

    # Límites de frecuencia de la octava de juego activa (para normalización del anzuelo)
    frecuencia_minima_octava_juego_activa_hz = 0
    frecuencia_maxima_octava_juego_activa_hz = 0

    # Variables de estado del audio (recibidas del ProcesadorAudio)
    tono_mapeado_anzuelo_hz = frecuencia_minima_octava_juego_activa_hz 
    volumen_detectado_db = UMBRAL_DECIBELIOS_SILENCIO
    nota_cantada_real_str = ""  
    frecuencia_cantada_real_hz = 0.0
    estado_tono_en_rango_juego_str = "silencio"   

    # --- BOTONES ---
    anchura_boton_control_px = 50
    altura_boton_control_px = 50
    
    imagen_burbuja_para_botones = pygame.image.load(os.path.join("pesca", "burbuja.png")).convert_alpha()
    imagen_burbuja_para_botones = pygame.transform.scale(imagen_burbuja_para_botones, (anchura_boton_control_px, altura_boton_control_px))

    margen_controles_borde_x_px = 20
    margen_controles_borde_y_px = 20
    espacio_entre_botones_x_px = 10
    espacio_texto_velocidad_botones_y_px = 5

    pos_y_botones_velocidad = ALTURA_PANTALLA - margen_controles_borde_y_px - altura_boton_control_px
    pos_x_boton_mas_velocidad = ANCHURA_PANTALLA - margen_controles_borde_x_px - anchura_boton_control_px
    pos_x_boton_menos_velocidad = pos_x_boton_mas_velocidad - espacio_entre_botones_x_px - anchura_boton_control_px

    rect_boton_mas_velocidad = pygame.Rect(pos_x_boton_mas_velocidad, pos_y_botones_velocidad, anchura_boton_control_px, altura_boton_control_px)
    rect_boton_menos_velocidad = pygame.Rect(pos_x_boton_menos_velocidad, pos_y_botones_velocidad, anchura_boton_control_px, altura_boton_control_px)
    
    desplazamiento_img_burbuja_x = -3
    desplazamiento_img_burbuja_y = 3

    rect_boton_mas_octava = pygame.Rect(0,0,0,0)
    rect_boton_menos_octava = pygame.Rect(0,0,0,0)

    def configurar_octava_juego_activa(indice_octava_nueva, instancia_procesador_audio):
        '''
        Configura la octava de juego activa.
        '''
        nonlocal indice_octava_juego_seleccionada, notas_octava_juego_activa, nombre_octava_juego_actual_str
        nonlocal num_notas_en_octava_juego_activa, anchura_segmento_nota_juego_px
        nonlocal frecuencia_minima_octava_juego_activa_hz, frecuencia_maxima_octava_juego_activa_hz

        # Asegurar que el índice esté dentro de los límites de las octavas disponibles
        indice_octava_juego_seleccionada = max(0, min(indice_octava_nueva, len(OCTAVAS_DE_JUEGO_CONFIG) - 1))
        
        config_octava_seleccionada = OCTAVAS_DE_JUEGO_CONFIG[indice_octava_juego_seleccionada]
        
        notas_octava_juego_activa = config_octava_seleccionada["notas"]
        nombre_octava_juego_actual_str = config_octava_seleccionada["nombre"]
        frecuencia_minima_octava_juego_activa_hz = config_octava_seleccionada["freq_min"]
        frecuencia_maxima_octava_juego_activa_hz = config_octava_seleccionada["freq_max"]
        
        num_notas_en_octava_juego_activa = len(notas_octava_juego_activa)
        if num_notas_en_octava_juego_activa > 0:
            anchura_segmento_nota_juego_px = ANCHURA_PANTALLA / num_notas_en_octava_juego_activa
        else:
            anchura_segmento_nota_juego_px = ANCHURA_PANTALLA

        # Actualizar límites en el procesador de audio
        if instancia_procesador_audio:
            instancia_procesador_audio.frecuencia_minima_octava_juego_hz = frecuencia_minima_octava_juego_activa_hz
            instancia_procesador_audio.frecuencia_maxima_octava_juego_hz = frecuencia_maxima_octava_juego_activa_hz
            # Forzar una reevaluación del estado del rango la próxima vez que el procesador de audio procese un tono
            instancia_procesador_audio.forzar_reevaluacion_estado_en_octava()

    # --- CONFIGURACIÓN DE ENTRADA DE SONIDO ---
    cola_para_datos_de_audio = queue.Queue(maxsize=10) # Cola para comunicar datos del hilo de audio al hilo de juego
    instancia_procesador_audio = ProcesadorAudio(cola_para_datos_de_audio, indice_dispositivo_microfono=indice_microfono_seleccionado)
    instancia_procesador_audio.start() # Iniciar el hilo de procesamiento de audio

    # Establecer la octava inicial del juego
    configurar_octava_juego_activa(indice_octava_juego_seleccionada, instancia_procesador_audio)

    # Variables del estado del juego
    posicion_x_anzuelo_px = ANCHURA_PANTALLA // 2
    longitud_actual_sedal_px = LONGITUD_CANA_POR_DEFECTO
    puntos_juego = 0
    juego_en_pausa = False
    factor_velocidad_peces = 1.0
    FACTOR_VELOCIDAD_MINIMA_PECES = 0.5
    FACTOR_VELOCIDAD_MAXIMA_PECES = 5.0

    # Pez especial (inicialmente no existe)
    pez_especial_activo = None

    def modificar_velocidad_peces(factor_actual, multiplicador_cambio, lista_peces, obj_pez_especial, vel_min, vel_max):
        """
        Actualiza la velocidad de todos los peces (normales y especial) según un multiplicador.
        Asegura que el factor de velocidad global se mantenga dentro de los límites.
        Retorna el nuevo factor de velocidad global.
        """
        nuevo_factor_velocidad = factor_actual * multiplicador_cambio
        nuevo_factor_velocidad = max(vel_min, min(nuevo_factor_velocidad, vel_max))

        if nuevo_factor_velocidad != factor_actual: # Solo aplicar si hay cambio real
            if factor_actual != 0: # Evitar división por cero si la velocidad era 0
                proporcion_cambio_velocidad = nuevo_factor_velocidad / factor_actual
            for pez_iter in lista_peces:
                    pez_iter.velocidad *= proporcion_cambio_velocidad
            if obj_pez_especial:
                    obj_pez_especial.velocidad *= proporcion_cambio_velocidad
            else: # Si la velocidad era 0, y ahora no, hay que reasignar una base
                pass
            return nuevo_factor_velocidad
        return factor_actual


    # --- BUCLE PRINCIPAL ---
    running = True
    try:
        while running:
            # --- EVENTOS (TECLADO, RATÓN, CERRAR VENTANA) ---
            for evento_pygame in pygame.event.get():
                if evento_pygame.type == pygame.QUIT:
                    running = False
                if evento_pygame.type == pygame.KEYDOWN:
                    if evento_pygame.key == pygame.K_ESCAPE:
                        running = False
                    if evento_pygame.key == pygame.K_p: # Tecla 'P' para pausar/reanudar
                        juego_en_pausa = not juego_en_pausa
                    # Controles de velocidad con teclado (+/-)
                    if evento_pygame.key in (pygame.K_PLUS, pygame.K_KP_PLUS):
                        factor_velocidad_peces = modificar_velocidad_peces(factor_velocidad_peces, 1.2, lista_peces_en_pantalla, pez_especial_activo, FACTOR_VELOCIDAD_MINIMA_PECES, FACTOR_VELOCIDAD_MAXIMA_PECES)
                    if evento_pygame.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        factor_velocidad_peces = modificar_velocidad_peces(factor_velocidad_peces, 0.8, lista_peces_en_pantalla, pez_especial_activo, FACTOR_VELOCIDAD_MINIMA_PECES, FACTOR_VELOCIDAD_MAXIMA_PECES)
                    # Atajos de teclado para cambio de octava (flechas arriba/abajo)
                    if evento_pygame.key == pygame.K_UP:
                        configurar_octava_juego_activa(indice_octava_juego_seleccionada + 1, instancia_procesador_audio)
                    if evento_pygame.key == pygame.K_DOWN:
                        configurar_octava_juego_activa(indice_octava_juego_seleccionada - 1, instancia_procesador_audio)
                
                if evento_pygame.type == pygame.MOUSEBUTTONDOWN:
                    # Clics en botones de control de velocidad
                    if rect_boton_mas_velocidad.collidepoint(evento_pygame.pos):
                        factor_velocidad_peces = modificar_velocidad_peces(factor_velocidad_peces, 1.2, lista_peces_en_pantalla, pez_especial_activo, FACTOR_VELOCIDAD_MINIMA_PECES, FACTOR_VELOCIDAD_MAXIMA_PECES)
                    elif rect_boton_menos_velocidad.collidepoint(evento_pygame.pos):
                        factor_velocidad_peces = modificar_velocidad_peces(factor_velocidad_peces, 0.8, lista_peces_en_pantalla, pez_especial_activo, FACTOR_VELOCIDAD_MINIMA_PECES, FACTOR_VELOCIDAD_MAXIMA_PECES)
                    # Clics en botones de cambio de octava (rects se actualizan en cada frame)
                    elif rect_boton_mas_octava.collidepoint(evento_pygame.pos):
                        configurar_octava_juego_activa(indice_octava_juego_seleccionada + 1, instancia_procesador_audio)
                    elif rect_boton_menos_octava.collidepoint(evento_pygame.pos):
                        configurar_octava_juego_activa(indice_octava_juego_seleccionada - 1, instancia_procesador_audio)

            if juego_en_pausa:
                # Mostrar mensaje de pausa
                texto_pausa_renderizado = fuente_textos_juego.render("PAUSA", True, BLANCO)
                rect_texto_pausa = texto_pausa_renderizado.get_rect(center=(ANCHURA_PANTALLA // 2, ALTURA_PANTALLA // 2))
                pantalla_principal.blit(texto_pausa_renderizado, rect_texto_pausa)
                pygame.display.flip()
                reloj_juego.tick(FOTOGRAMAS_POR_SEGUNDO)
                continue 

            # Obtener datos de audio del procesador (de la cola)
            try:
                while not cola_para_datos_de_audio.empty(): # Procesar todos los datos disponibles en la cola
                    (tono_mapeado_anzuelo_hz, volumen_detectado_db, nota_cantada_real_str, frecuencia_cantada_real_hz, estado_tono_en_rango_juego_str) = cola_para_datos_de_audio.get_nowait()
            except queue.Empty:
                pass # No hay nuevos datos de audio esta vez

            # Normalizar volumen para controlar la longitud de la caña/sedal
            volumen_normalizado_para_sedal = 0.0 # Por defecto, caña recogida
            if volumen_detectado_db > UMBRAL_DECIBELIOS_SILENCIO:
                # El rango dinámico para la extensión del sedal va desde UMBRAL_DECIBELIOS_SILENCIO hasta VOLUMEN_MAXIMO_EXTENSION_TOTAL_CANA
                rango_dinamico_volumen_db = VOLUMEN_MAXIMO_EXTENSION_TOTAL_CANA - UMBRAL_DECIBELIOS_SILENCIO
                if rango_dinamico_volumen_db <= 0: # Evitar división por cero o rango inválido
                    rango_dinamico_volumen_db = 1 
                
                # Calcular qué tan "dentro" de este rango está el volumen actual
                volumen_relativo_al_umbral = volumen_detectado_db - UMBRAL_DECIBELIOS_SILENCIO
                volumen_normalizado_para_sedal = volumen_relativo_al_umbral / rango_dinamico_volumen_db
                volumen_normalizado_para_sedal = max(0.0, min(1.0, volumen_normalizado_para_sedal)) # Asegurar entre 0 y 1
            
            # --- ACTUALIZAR POSICIÓN DEL ANZUELO ---
            if estado_tono_en_rango_juego_str != "silencio" and tono_mapeado_anzuelo_hz > 0:
                # tono_mapeado_anzuelo_hz ya está en el rango de la octava de juego activa
                # Normalizar este tono para la posición del anzuelo en pantalla
                rango_frecuencias_octava_juego = frecuencia_maxima_octava_juego_activa_hz - frecuencia_minima_octava_juego_activa_hz
                if rango_frecuencias_octava_juego <= 0: rango_frecuencias_octava_juego = 1 # Evitar división por cero
                
                # Calcular la posición normalizada dentro de la octava de juego (0.0 a 1.0)
                pos_normalizada_en_octava_juego = (tono_mapeado_anzuelo_hz - frecuencia_minima_octava_juego_activa_hz) / rango_frecuencias_octava_juego
                pos_normalizada_en_octava_juego = max(0.0, min(1.0, pos_normalizada_en_octava_juego))

                # Mapear la nota más cercana DENTRO de la octava de juego para centrar el anzuelo en su segmento
                nota_mas_cercana_en_octava_juego = ""
                min_diferencia_logaritmica = float('inf')
                
                if tono_mapeado_anzuelo_hz > 0 and notas_octava_juego_activa:
                    for nombre_nota_juego in notas_octava_juego_activa:
                        frec_nota_actual_juego = FRECUENCIAS_NOTAS_MUSICALES.get(nombre_nota_juego, 0)
                        if frec_nota_actual_juego > 0:
                            diferencia_log = abs(math.log2(tono_mapeado_anzuelo_hz / frec_nota_actual_juego))
                            if diferencia_log < min_diferencia_logaritmica:
                                min_diferencia_logaritmica = diferencia_log
                                nota_mas_cercana_en_octava_juego = nombre_nota_juego
                
                if nota_mas_cercana_en_octava_juego:
                    try:
                        indice_nota_en_octava = notas_octava_juego_activa.index(nota_mas_cercana_en_octava_juego)
                        # Centrar el anzuelo en el medio del segmento de esta nota
                        posicion_x_anzuelo_px = int((indice_nota_en_octava + 0.5) * anchura_segmento_nota_juego_px)
                    except ValueError:
                        # Fallback: si la nota no se encuentra, usar la posición normalizada general
                        posicion_x_anzuelo_px = int(pos_normalizada_en_octava_juego * ANCHURA_PANTALLA)
                else: 
                    # Fallback si no se pudo mapear a una nota discreta (ej. octava vacía o tono fuera de notas definidas)
                    posicion_x_anzuelo_px = int(pos_normalizada_en_octava_juego * ANCHURA_PANTALLA)
            
            # Asegurar que el anzuelo no se salga de la pantalla
            posicion_x_anzuelo_px = max(0, min(ANCHURA_PANTALLA - 1, posicion_x_anzuelo_px))

            # Actualizar longitud del sedal (profundidad del anzuelo) basada en el volumen_normalizado_para_sedal
            if volumen_detectado_db > UMBRAL_DECIBELIOS_SILENCIO:
                longitud_actual_sedal_px = LONGITUD_MINIMA_CANA + volumen_normalizado_para_sedal * (LONGITUD_MAXIMA_CANA - LONGITUD_MINIMA_CANA)
            else: # Silencio
                longitud_actual_sedal_px = LONGITUD_CANA_POR_DEFECTO

            # La punta de la caña (anzuelo) en el eje Y
            posicion_y_anzuelo_px = int(longitud_actual_sedal_px)

            # --- COLISIONES PARA PECES ---
            for pez_actual in lista_peces_en_pantalla[:]: # Iterar sobre una copia para poder modificar la lista original
                pez_actual.actualizar_posicion()
                if pez_actual.comprobar_colision_con_anzuelo(posicion_x_anzuelo_px, posicion_y_anzuelo_px):
                    if sonido_captura_pez: sonido_captura_pez.play()
                    puntos_juego += pez_actual.puntos_por_captura
                    lista_peces_en_pantalla.remove(pez_actual)
                    # Añadir un nuevo pez para mantener la población
                    img_nuevo_pez = random.choice(imagenes_peces_normales_cargadas) if imagenes_peces_normales_cargadas else None
                    nuevo_pez_reemplazo = Pez(random.randint(50, ANCHURA_PANTALLA - 50), 
                                             random.randint(ALTURA_PANTALLA // 2, ALTURA_PANTALLA - 50), 
                                            imagen_pez=img_nuevo_pez, puntos_por_captura=1)
                    # Ajustar velocidad del nuevo pez según el factor global
                    nuevo_pez_reemplazo.velocidad *= factor_velocidad_peces
                    lista_peces_en_pantalla.append(nuevo_pez_reemplazo)

            # --- PECES ESPECIALES ---
            # Aparece si se tienen 3+ puntos y no hay uno ya en pantalla.
            if puntos_juego >= 3 and pez_especial_activo is None and imagen_pez_especial_cargada:
                pez_especial_activo = Pez(
                    random.randint(50, ANCHURA_PANTALLA - 50),
                    random.randint(ALTURA_PANTALLA // 2, ALTURA_PANTALLA - 50),
                    imagen_pez=imagen_pez_especial_cargada,
                    puntos_por_captura = 3 # Vale más puntos
                )
                pez_especial_activo.velocidad = 2.0 * factor_velocidad_peces

                pez_especial_activo.tamano = 40

            if pez_especial_activo:
                pez_especial_activo.actualizar_posicion()
                if pez_especial_activo.comprobar_colision_con_anzuelo(posicion_x_anzuelo_px, posicion_y_anzuelo_px):
                    if sonido_captura_pez: sonido_captura_pez.play()
                    puntos_juego += pez_especial_activo.puntos_por_captura
                    pez_especial_activo = None # Desaparece al ser pescado

            # --- DIBUJO ---
            pantalla_principal.blit(imagen_fondo_juego, (0, 0))

            # Barra de octavas
            altura_barra_octavas_px = 30
            pos_y_barra_octavas_px = 10
            anchura_total_barra_octavas_px = ANCHURA_PANTALLA - 20
            pos_x_barra_octavas_px = 10

            pygame.draw.rect(pantalla_principal, (200,200,200), (pos_x_barra_octavas_px, pos_y_barra_octavas_px, anchura_total_barra_octavas_px, altura_barra_octavas_px))
            anchura_semitono_visual_px = anchura_total_barra_octavas_px / NUM_TOTAL_SEMITONOS_VISUALIZADOS

            colores_octavas_visuales = {
                'C2-B2': ('C2', 'B2', (220,220,220)), 'C3-B3': ('C3', 'B3', (200,200,200)),
                'C4-B4': ('C4', 'B4', (190,190,190)), 'C5-B5': ('C5', 'B5', (180,180,180)),
                'C6-B6': ('C6', 'B6', (160,160,160)),
            }

            y_final_barra_con_etiquetas_c = pos_y_barra_octavas_px + altura_barra_octavas_px + 2
            hay_etiquetas_c = False
            for i, nota_barra_visual in enumerate(LISTA_COMPLETA_NOTAS_VISUALES):
                x_semitono_actual_px = pos_x_barra_octavas_px + i * anchura_semitono_visual_px
                color_segmento_semitono = (50,50,50) # Color base por si no coincide ninguna octava (gris oscuro)
                for nombre_oct, (nota_inicio_oct, nota_fin_oct, color_oct) in colores_octavas_visuales.items():
                    frec_nota_visual_actual = FRECUENCIAS_NOTAS_MUSICALES.get(nota_barra_visual, 0)
                    frec_nota_inicio_oct = FRECUENCIAS_NOTAS_MUSICALES.get(nota_inicio_oct, 0)
                    frec_nota_fin_oct = FRECUENCIAS_NOTAS_MUSICALES.get(nota_fin_oct, 99999)
                    if frec_nota_visual_actual > 0 and frec_nota_inicio_oct > 0 and frec_nota_visual_actual >= frec_nota_inicio_oct and frec_nota_visual_actual <= frec_nota_fin_oct:
                        color_segmento_semitono = color_oct
                        break # Salir del bucle interno una vez encontrado el color
                
                # Segmento de color del semitono
                pygame.draw.rect(pantalla_principal, color_segmento_semitono, (x_semitono_actual_px, pos_y_barra_octavas_px, anchura_semitono_visual_px , altura_barra_octavas_px))
                
                # Línea divisoria negra a la derecha del segmento
                if i < NUM_TOTAL_SEMITONOS_VISUALIZADOS - 1:
                    x_linea_divisoria = x_semitono_actual_px + anchura_semitono_visual_px -1 # -1 para que esté justo en el borde
                    pygame.draw.line(pantalla_principal, NEGRO, 
                                    (x_linea_divisoria, pos_y_barra_octavas_px), 
                                    (x_linea_divisoria, pos_y_barra_octavas_px + altura_barra_octavas_px), 1)

                # Resaltar la octava de juego seleccionada
                config_octava_juego_sel = OCTAVAS_DE_JUEGO_CONFIG[indice_octava_juego_seleccionada]
                nota_inicio_oct_juego_sel = config_octava_juego_sel["notas"][0]
                nota_fin_oct_juego_sel = config_octava_juego_sel["notas"][-1]
                try:
                    idx_visual_actual = LISTA_COMPLETA_NOTAS_VISUALES.index(nota_barra_visual)
                    idx_inicio_oct_juego_sel_en_visual = LISTA_COMPLETA_NOTAS_VISUALES.index(nota_inicio_oct_juego_sel)
                    idx_fin_oct_juego_sel_en_visual = LISTA_COMPLETA_NOTAS_VISUALES.index(nota_fin_oct_juego_sel)
                    if idx_inicio_oct_juego_sel_en_visual <= idx_visual_actual <= idx_fin_oct_juego_sel_en_visual:
                        pygame.draw.rect(pantalla_principal, (255, 255, 0), (x_semitono_actual_px, pos_y_barra_octavas_px, anchura_semitono_visual_px -1, altura_barra_octavas_px), 2)
                except ValueError: pass
                if nota_barra_visual.startswith('C') and (len(nota_barra_visual) == 2 or nota_barra_visual[1] != '#'):
                    texto_etiqueta_nota_c = fuente_textos_juego.render(nota_barra_visual, True, NEGRO)
                    rect_etiqueta_nota_c = texto_etiqueta_nota_c.get_rect(topleft=(x_semitono_actual_px + 2, pos_y_barra_octavas_px + altura_barra_octavas_px + 2))
                    pantalla_principal.blit(texto_etiqueta_nota_c, rect_etiqueta_nota_c)
                    hay_etiquetas_c = True
            if hay_etiquetas_c:
                y_final_barra_con_etiquetas_c += fuente_textos_juego.get_height()
            
            if nota_cantada_real_str and nota_cantada_real_str in LISTA_COMPLETA_NOTAS_VISUALES:
                try:
                    indice_nota_cantada_en_barra_visual = LISTA_COMPLETA_NOTAS_VISUALES.index(nota_cantada_real_str)
                    x_indicador_nota_cantada_px = pos_x_barra_octavas_px + (indice_nota_cantada_en_barra_visual + 0.5) * anchura_semitono_visual_px
                    pygame.draw.line(pantalla_principal, BLANCO, (x_indicador_nota_cantada_px, pos_y_barra_octavas_px), (x_indicador_nota_cantada_px, pos_y_barra_octavas_px + altura_barra_octavas_px), 3)
                except ValueError: pass
            pygame.draw.rect(pantalla_principal, NEGRO, (pos_x_barra_octavas_px, pos_y_barra_octavas_px, anchura_total_barra_octavas_px, altura_barra_octavas_px), 2)

            # Botones de cambio de octava
            padding_botones_octava_px = 10
            pos_y_botones_octava = y_final_barra_con_etiquetas_c + padding_botones_octava_px
            rect_boton_menos_octava = pygame.Rect(margen_controles_borde_x_px, pos_y_botones_octava, anchura_boton_control_px, altura_boton_control_px)
            rect_boton_mas_octava = pygame.Rect(margen_controles_borde_x_px + anchura_boton_control_px + espacio_entre_botones_x_px, pos_y_botones_octava, anchura_boton_control_px, altura_boton_control_px)
            
            pos_y_texto_octava_actual = pos_y_botones_octava + altura_boton_control_px + 5
            texto_octava_juego_str_render = f"Octava Actual: {nombre_octava_juego_actual_str}"
            superficie_texto_octava_juego = fuente_textos_juego.render(texto_octava_juego_str_render, True, NEGRO)
            pantalla_principal.blit(superficie_texto_octava_juego, (10, pos_y_texto_octava_actual))
            
            # Caña, anzuelo y peces
            pygame.draw.line(pantalla_principal, COLOR_CANA_PESCAR, (posicion_x_anzuelo_px, 0), (posicion_x_anzuelo_px, posicion_y_anzuelo_px), 3)
            pygame.draw.circle(pantalla_principal, (150, 150, 150), (posicion_x_anzuelo_px, posicion_y_anzuelo_px), RADIO_ANZUELO_CIRCULO)
            for pez_en_lista in lista_peces_en_pantalla:
                pez_en_lista.dibujar_en_pantalla(pantalla_principal)
            if pez_especial_activo:
                pez_especial_activo.dibujar_en_pantalla(pantalla_principal)

            # Texto
            pos_y_actual_info_general = pos_y_texto_octava_actual + superficie_texto_octava_juego.get_height() + 10

            mensaje_estado_tono_juego = ""
            if estado_tono_en_rango_juego_str == "debajo_octava_activa": 
                mensaje_estado_tono_juego = f"Canta MAS AGUDO para {nombre_octava_juego_actual_str}"
            elif estado_tono_en_rango_juego_str == "encima_octava_activa": 
                mensaje_estado_tono_juego = f"Canta MAS GRAVE para {nombre_octava_juego_actual_str}"
            elif estado_tono_en_rango_juego_str == "en_octava_activa" and nota_cantada_real_str: 
                mensaje_estado_tono_juego = f"{nota_cantada_real_str} (en rango {nombre_octava_juego_actual_str})"
            elif estado_tono_en_rango_juego_str == "silencio":
                mensaje_estado_tono_juego = "(Silencio detectado)"

            if mensaje_estado_tono_juego:
                superficie_mensaje_estado = fuente_textos_juego.render(mensaje_estado_tono_juego, True, NEGRO)
                rect_mensaje_estado = superficie_mensaje_estado.get_rect(centerx=ANCHURA_PANTALLA // 2, top=pos_y_actual_info_general)
                pantalla_principal.blit(superficie_mensaje_estado, rect_mensaje_estado)
                pos_y_actual_info_general += superficie_mensaje_estado.get_height() + 5 
            
            pantalla_principal.blit(fuente_textos_juego.render(f"Volumen: {volumen_detectado_db:.1f} dB", True, NEGRO), (10, pos_y_actual_info_general))
            pos_y_actual_info_general += fuente_textos_juego.get_height() + 5
            pantalla_principal.blit(fuente_textos_juego.render(f"Nota cantada: {nota_cantada_real_str}", True, NEGRO), (10, pos_y_actual_info_general))
            pos_y_actual_info_general += fuente_textos_juego.get_height() + 5
            pantalla_principal.blit(fuente_textos_juego.render(f"Frecuencia: {frecuencia_cantada_real_hz:.1f} Hz", True, NEGRO), (10, pos_y_actual_info_general))
            
            # Puntuación
            superficie_texto_puntuacion = fuente_textos_juego.render(f"Peces: {puntos_juego}", True, NEGRO)
            # Usar pos_y_texto_octava_actual para la alineación vertical de la puntuación
            rect_texto_puntuacion = superficie_texto_puntuacion.get_rect(topright=(ANCHURA_PANTALLA - 10, pos_y_texto_octava_actual))
            pantalla_principal.blit(superficie_texto_puntuacion, rect_texto_puntuacion)
            
            # Barra de volumen
            anchura_total_barra_vol_px = 200
            altura_total_barra_vol_px = 20
            pos_x_barra_vol_px = 10
            pos_y_barra_vol_px = ALTURA_PANTALLA - altura_total_barra_vol_px - 10 
            anchura_relleno_barra_vol_px = int(volumen_normalizado_para_sedal * anchura_total_barra_vol_px)
            pygame.draw.rect(pantalla_principal, NEGRO, (pos_x_barra_vol_px, pos_y_barra_vol_px, anchura_total_barra_vol_px, altura_total_barra_vol_px), 2) 
            pygame.draw.rect(pantalla_principal, VERDE, (pos_x_barra_vol_px, pos_y_barra_vol_px, anchura_relleno_barra_vol_px, altura_total_barra_vol_px)) 

            # Velocidad
            texto_velocidad_render = f"Vel.: {factor_velocidad_peces:.2f}x"
            superficie_texto_velocidad = fuente_textos_juego.render(texto_velocidad_render, True, NEGRO)
            rect_texto_velocidad = superficie_texto_velocidad.get_rect()
            
            rect_texto_velocidad.centerx = (pos_x_boton_menos_velocidad + rect_boton_menos_velocidad.width // 2 + pos_x_boton_mas_velocidad + rect_boton_mas_velocidad.width // 2) // 2
            rect_texto_velocidad.bottom = pos_y_botones_velocidad - espacio_texto_velocidad_botones_y_px
            pantalla_principal.blit(superficie_texto_velocidad, rect_texto_velocidad)

            # Botones
            if imagen_burbuja_para_botones:
                # Botones de octava
                pos_img_burbuja_mas_oct = (rect_boton_mas_octava.x + desplazamiento_img_burbuja_x, rect_boton_mas_octava.y + desplazamiento_img_burbuja_y)
                pos_img_burbuja_menos_oct = (rect_boton_menos_octava.x + desplazamiento_img_burbuja_x, rect_boton_menos_octava.y + desplazamiento_img_burbuja_y)
                pantalla_principal.blit(imagen_burbuja_para_botones, pos_img_burbuja_mas_oct)
                pantalla_principal.blit(imagen_burbuja_para_botones, pos_img_burbuja_menos_oct)
                # Botones de velocidad
                pos_img_burbuja_mas_vel = (rect_boton_mas_velocidad.x + desplazamiento_img_burbuja_x, rect_boton_mas_velocidad.y + desplazamiento_img_burbuja_y)
                pos_img_burbuja_menos_vel = (rect_boton_menos_velocidad.x + desplazamiento_img_burbuja_x, rect_boton_menos_velocidad.y + desplazamiento_img_burbuja_y)
                pantalla_principal.blit(imagen_burbuja_para_botones, pos_img_burbuja_mas_vel)
                pantalla_principal.blit(imagen_burbuja_para_botones, pos_img_burbuja_menos_vel)
            else: 
                pygame.draw.rect(pantalla_principal, (200, 200, 200), rect_boton_mas_octava)
                pygame.draw.rect(pantalla_principal, (200, 200, 200), rect_boton_menos_octava)
                pygame.draw.rect(pantalla_principal, (200, 200, 200), rect_boton_mas_velocidad)
                pygame.draw.rect(pantalla_principal, (200, 200, 200), rect_boton_menos_velocidad)

            # Símbolos "+" y "-" para botones de octava
            superficie_simbolo_mas_oct = fuente_simbolos_botones.render("+", True, NEGRO) 
            superficie_simbolo_menos_oct = fuente_simbolos_botones.render("-", True, NEGRO) 
            rect_simbolo_mas_oct = superficie_simbolo_mas_oct.get_rect(center=rect_boton_mas_octava.center)
            rect_simbolo_menos_oct = superficie_simbolo_menos_oct.get_rect(center=rect_boton_menos_octava.center)
            pantalla_principal.blit(superficie_simbolo_mas_oct, rect_simbolo_mas_oct)
            pantalla_principal.blit(superficie_simbolo_menos_oct, rect_simbolo_menos_oct)
            
            # Símbolos "+" y "-" para botones de velocidad
            superficie_simbolo_mas_vel = fuente_simbolos_botones.render("+", True, NEGRO)
            superficie_simbolo_menos_vel = fuente_simbolos_botones.render("-", True, NEGRO)
            rect_simbolo_mas_vel = superficie_simbolo_mas_vel.get_rect(center=rect_boton_mas_velocidad.center)
            rect_simbolo_menos_vel = superficie_simbolo_menos_vel.get_rect(center=rect_boton_menos_velocidad.center)
            pantalla_principal.blit(superficie_simbolo_mas_vel, rect_simbolo_mas_vel)
            pantalla_principal.blit(superficie_simbolo_menos_vel, rect_simbolo_menos_vel)

            pygame.display.flip()
            reloj_juego.tick(FOTOGRAMAS_POR_SEGUNDO)
    finally:
        # Asegurarse de que el hilo de audio se detenga correctamente al salir del juego
        print("Deteniendo el hilo de procesamiento de audio...")
        instancia_procesador_audio.detener_procesamiento()
        instancia_procesador_audio.join(timeout=2.0) # Esperar un máximo de 2 segundos a que el hilo termine
        print("Hilo de audio detenido.")

# --- PRUEBAS SIN EL MAIN.PY ---
if __name__ == "__main__":
    pygame.init() # Inicializar todos los módulos de Pygame necesarios
    
    # Crear la ventana principal del juego
    pantalla_juego_principal = pygame.display.set_mode((ANCHURA_PANTALLA, ALTURA_PANTALLA))
    
    indice_microfono_a_usar = None # Por defecto, PyAudio elegirá el micrófono predeterminado del sistema

    # Intento de seleccionar un micrófono de forma más explícita usando 'sounddevice'
    try:
        import sounddevice as sd
        lista_dispositivos_audio = sd.query_devices()
        print("Dispositivos de audio encontrados por 'sounddevice':")
        for i, dispositivo in enumerate(lista_dispositivos_audio):
            print(f"  Índice {i}: {dispositivo['name']}, Canales de entrada: {dispositivo['max_input_channels']}")
            # Criterios para un micrófono: más de 0 canales de entrada y una latencia baja por defecto razonable
            if dispositivo['max_input_channels'] > 0 and dispositivo['default_low_input_latency'] > 0:
                # Tomar el primer dispositivo que parezca un micrófono válido y no haya sido seleccionado aún
                if indice_microfono_a_usar is None:
                    indice_microfono_a_usar = i
                    print(f"Micrófono tentativo seleccionado (índice sounddevice {i}): {dispositivo['name']}")
        
        # Intentar usar el dispositivo de entrada predeterminado de 'sounddevice'.
        if indice_microfono_a_usar is None and len(lista_dispositivos_audio) > 0:
            try:
                indice_entrada_predeterminado_sd = sd.default.device[0] # Índice del dispositivo de entrada por defecto
                if indice_entrada_predeterminado_sd != -1: # -1 significa que no hay dispositivo de entrada por defecto
                # Verificar que el dispositivo por defecto sea realmente de entrada
                    if lista_dispositivos_audio[indice_entrada_predeterminado_sd]['max_input_channels'] > 0:
                        indice_microfono_a_usar = indice_entrada_predeterminado_sd
                        print(f"Usando micrófono predeterminado de 'sounddevice' (índice {indice_microfono_a_usar}): {lista_dispositivos_audio[indice_microfono_a_usar]['name']}")
                    else:
                        print(f"El dispositivo predeterminado de 'sounddevice' (índice {indice_entrada_predeterminado_sd}) no es de entrada. PyAudio usará su predeterminado.")
                else:
                    print("'sounddevice' no tiene un dispositivo de entrada predeterminado (el índice es -1). PyAudio usará su predeterminado.")
            except AttributeError: # Por si sd.default.device no es accesible como se espera
                print("'sounddevice' no pudo determinar un dispositivo de entrada predeterminado (AttributeError). PyAudio usará su predeterminado.")
            except Exception as e_sd_default:
                print(f"Error al obtener el dispositivo predeterminado de 'sounddevice': {e_sd_default}. PyAudio usará su predeterminado.")

        elif indice_microfono_a_usar is None: # Si después de todo, no se seleccionó ninguno
            print("No se encontró un micrófono adecuado con 'sounddevice' o no hay dispositivos. PyAudio usará su predeterminado.")

    except ImportError:
        print("Módulo 'sounddevice' no encontrado. Se usará el micrófono predeterminado de PyAudio.")
    except Exception as e_sd_general:
        print(f"Error al consultar 'sounddevice' para el índice del micrófono: {e_sd_general}. PyAudio usará su predeterminado.")

    # Ejecutar la función principal del juego
    main(pantalla_juego_principal, indice_microfono_a_usar)
    
    pygame.quit()
    import sys
    sys.exit()
