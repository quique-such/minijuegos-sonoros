import pygame
import os
import numpy as np
import sounddevice as sd
from scipy.fft import rfft, rfftfreq
import random
import time

# --- FUNCIÓN PRINCIPAL ---
def main(pantalla, indice_microfono):
    pygame.mixer.init(frequency=44100, size=-16, channels=1)
    # Establecer el título de la ventana del juego
    pygame.display.set_caption("Minijuego - Simon Dice")
    
    # Fuentes
    RUTA_FUENTE = os.path.join("fonts", "minecraft.ttf") # fuente personalizada
    COLOR_TEXTO = (0, 0, 0)

    # Verificar si el archivo de la fuente existe
    if not os.path.exists(RUTA_FUENTE):
        raise FileNotFoundError(f"No se encontró la fuente: {RUTA_FUENTE}")

    # Cargar la fuente desde el archivo .ttf (fuente)
    fuente_principal = pygame.font.Font(RUTA_FUENTE, 38) # Para mensajes principales
    fuente_pequena = pygame.font.Font(RUTA_FUENTE, 16)   # Para texto secundario
    
    # Fondos
    imagen_fondo = pygame.image.load(os.path.join("fondos", "simon.png")).convert()
    imagen_fondo = pygame.transform.scale(imagen_fondo, (pantalla.get_width(), pantalla.get_height())) # fondo tamaño de la pantalla
    pantalla.blit(imagen_fondo, (0, 0)) # dibujamos la imagen en la pantalla
    
    # Parámetros
    ANCHO, ALTO = pantalla.get_size() # Dimensiones de la pantalla
    nombres_notas = ["Do4", "Fa4", "La4", "Do5"] # notas
    frecuencias_notas = {"Do4": 261.63, "Fa4": 349.228, "La4": 440, "Do5": 523.25} # frecuencias de las notas
    colores_notas = {"Do4": (217, 180, 56), "Fa4": (77, 157, 79), "La4": (119, 140, 191), "Do5": (201, 58, 36)}
    
    # Dimensiones y posiciones cuadrados
    tamano_cuadrado = 150 # Lado del cuadrado
    centro_x, centro_y = ANCHO // 2, ALTO // 2 # Centro de la pantalla
    rectangulos_notas = {
        "Do4": pygame.Rect(centro_x - tamano_cuadrado - 10, centro_y - tamano_cuadrado - 10, tamano_cuadrado, tamano_cuadrado),
        "Fa4": pygame.Rect(centro_x + 10, centro_y - tamano_cuadrado - 10, tamano_cuadrado, tamano_cuadrado),
        "La4": pygame.Rect(centro_x - tamano_cuadrado - 10, centro_y + 10, tamano_cuadrado, tamano_cuadrado),
        "Do5": pygame.Rect(centro_x + 10, centro_y + 10, tamano_cuadrado, tamano_cuadrado)
    }

    # Sonidos de notas
    def generar_sonido_nota(frecuencia, duracion_sonido=1, volumen_sonido=0.5):
        tasa_muestreo_sonido = 44100 # Tasa de muestreo estándar para audio
        tiempo_array = np.linspace(0, duracion_sonido, int(tasa_muestreo_sonido * duracion_sonido), False) # Crear un array de tiempo
        tono_onda = np.sin(frecuencia * tiempo_array * 2 * np.pi) # Generar la onda sinusoidal para el tono
        datos_audio_nota = (tono_onda * volumen_sonido * 32767).astype(np.int16) # Convertir la onda a formato de audio de 16 bits
        # Crear un objeto de sonido
        return pygame.mixer.Sound(buffer=datos_audio_nota.tobytes())

    # Sonido de fallo
    def generar_sonido_fallo():
        tasa_muestreo_sonido_fallo = 44100
        tiempo_array_fallo = np.linspace(0, 0.5, int(tasa_muestreo_sonido_fallo * 0.5), False) # Duración de 0.5s
        frecuencias_fallo = np.linspace(600, 200, len(tiempo_array_fallo))
        tono_onda_fallo = np.sin(2 * np.pi * frecuencias_fallo * tiempo_array_fallo)
        datos_audio_fallo = (tono_onda_fallo * 0.5 * 32767).astype(np.int16) # Volumen 0.5
        return pygame.mixer.Sound(buffer=datos_audio_fallo.tobytes())

    # Alamcenamos los sonidos
    sonidos_notas = {nota: generar_sonido_nota(freq) for nota, freq in frecuencias_notas.items()}
    sonido_fallo_juego = generar_sonido_fallo()

    # Micrófono
    tasa_muestreo_microfono = 44100 # Tasa de muestreo para la entrada del micrófono
    duracion_bloque_audio = 0.2  # Duración de cada bloque de audio procesado (en segundos)

    # Detección de las notas
    nota_actual_detectada_mic = "" # Nota actualmente detectada por el micrófono
    tiempo_ultima_nota_valida_mic = time.time() # Momento en que se detectó la última nota con volumen suficiente
    umbral_volumen_mic = 0.01 # Umbral mínimo de volumen
    tolerancias_afinacion = {"Do4": 2**4.5, "Fa4": 2**5, "La4": 2**5, "Do5": 2**5.5} # tolerancia y margen de error

    # Función para calcular el volumen
    def calcular_volumen(datos_audio_procesar):
        return np.sqrt(np.mean(datos_audio_procesar**2))

    # Función para detectar el tono (frecuencia fundamental) de una señal de audio
    def detectar_tono(datos_audio_procesar):
        ventana_hanning_aplicada = np.hanning(len(datos_audio_procesar)) # ventana para reducir el aliasing
        datos_audio_con_ventana = datos_audio_procesar * ventana_hanning_aplicada

        transformada_fourier_abs = np.abs(rfft(datos_audio_con_ventana)) # calculamos la fft
        transformada_fourier_abs[:10] = 0 # ignoramos componentes de baja frecuencia
        frecuencias_fft_calculadas = rfftfreq(len(datos_audio_con_ventana), 1 / tasa_muestreo_microfono) # obtenemos frecuencias calculadas
        
        # Filtrar las frecuencias para el rango donde se esperan
        indices_frecuencias_interes = np.where((frecuencias_fft_calculadas > 200) & (frecuencias_fft_calculadas < 700))[0]
        if len(indices_frecuencias_interes) == 0:
            return 0 

        # Encontrar el índice del pico de mayor amplitud en el rango filtrado
        indice_pico_amplitud = np.argmax(transformada_fourier_abs[indices_frecuencias_interes])
        indice_verdadero_pico_fft = indices_frecuencias_interes[indice_pico_amplitud]
        
        # Interpolación cuadrática para una estimación más precisa de la frecuencia del pico
        frecuencia_estimada_final = 0
        if 1 < indice_verdadero_pico_fft < len(transformada_fourier_abs) - 1:
            alpha = transformada_fourier_abs[indice_verdadero_pico_fft - 1]
            beta = transformada_fourier_abs[indice_verdadero_pico_fft]
            gamma = transformada_fourier_abs[indice_verdadero_pico_fft + 1]
            denominador = alpha - 2 * beta + gamma
            if denominador == 0: # Evitar división por cero
                p_interpolacion = 0
            else:
                p_interpolacion = 0.5 * (alpha - gamma) / denominador
            
            frecuencia_estimada_final = frecuencias_fft_calculadas[indice_verdadero_pico_fft] + \
                                      p_interpolacion * (frecuencias_fft_calculadas[1] - frecuencias_fft_calculadas[0])
        elif indice_verdadero_pico_fft < len(frecuencias_fft_calculadas): # Caso sin interpolación
            frecuencia_estimada_final = frecuencias_fft_calculadas[indice_verdadero_pico_fft]
        else: # no debería ocurrir si indices_frecuencias_interes no está vacío
            return 0

        return frecuencia_estimada_final

    # Función para convertir una frecuencia a nota musical
    def frecuencia_a_nota(frecuencia_entrada):
        if frecuencia_entrada == 0:
            return "" 
        
        nota_mas_cercana_encontrada = ""
        minima_diferencia_frecuencia = float("inf")
        # Comparar la frecuencia de entrada con las frecuencias de las notas de referencia
        for nombre_nota_ref, frecuencia_de_nota_referencia_actual in frecuencias_notas.items():
            diferencia_frecuencia_actual = abs(frecuencia_entrada - frecuencia_de_nota_referencia_actual)
            if diferencia_frecuencia_actual < minima_diferencia_frecuencia:
                minima_diferencia_frecuencia = diferencia_frecuencia_actual
                nota_mas_cercana_encontrada = nombre_nota_ref
        
        # Verificar si la nota más cercana está dentro de la tolerancia de afinación permitida
        if nota_mas_cercana_encontrada and minima_diferencia_frecuencia < tolerancias_afinacion[nota_mas_cercana_encontrada]:
            return nota_mas_cercana_encontrada
        return "" 

    # ---ENTRADA DE AUDIO---
    def procesar_audio_microfono_callback(datos_entrada_microfono_actual, numero_frames_actual, info_tiempo_actual, estado_stream_actual):
        nonlocal nota_actual_detectada_mic, tiempo_ultima_nota_valida_mic 
        
        datos_audio_canal_izquierdo = datos_entrada_microfono_actual[:, 0] # Usar solo un canal
        volumen_detectado_actual = calcular_volumen(datos_audio_canal_izquierdo)
        
        if volumen_detectado_actual < umbral_volumen_mic:
            if time.time() - tiempo_ultima_nota_valida_mic > 0.5: 
                nota_actual_detectada_mic = "" # Si el volumen es bajo y ha pasado tiempo, limpiar la nota detectada
            return # No procesar si el volumen es muy bajo
        
        frecuencia_detectada_actual = detectar_tono(datos_audio_canal_izquierdo)
        nota_detectada_audio_actual = frecuencia_a_nota(frecuencia_detectada_actual)
        
        if nota_detectada_audio_actual != "":
            nota_actual_detectada_mic = nota_detectada_audio_actual
            tiempo_ultima_nota_valida_mic = time.time() # Actualizar tiempo de última nota válida

    # Iniciar el flujo de entrada de audio desde el micrófono
    flujo_audio_microfono = sd.InputStream(callback=procesar_audio_microfono_callback, channels=1, samplerate=tasa_muestreo_microfono,
                                       blocksize=int(tasa_muestreo_microfono * duracion_bloque_audio), device=indice_microfono)
    flujo_audio_microfono.start()

    # --- DIBUJO ---
    def dibujar_cuadrados_notas(nota_activa_visual_actual=None): # Dibujar los cuadrados de las notas en la pantalla
        pantalla.blit(imagen_fondo, (0, 0)) # limpiar
        for nombre_nota, rectangulo_nota in rectangulos_notas.items():
            color_cuadrado_actual = colores_notas[nombre_nota]
            # Si la nota es la "activa" iluminarla
            if nombre_nota == nota_activa_visual_actual:
                color_cuadrado_actual = tuple(min(255, componente_color_rgb + 100) for componente_color_rgb in color_cuadrado_actual)
            pygame.draw.rect(pantalla, color_cuadrado_actual, rectangulo_nota)

    # Texto centrado
    def mostrar_texto_centrado(cadena_texto_mostrar, posicion_y_texto_actual, fuente_usada=fuente_principal):
        superficie_texto_renderizado = fuente_usada.render(cadena_texto_mostrar, True, COLOR_TEXTO)
        pantalla.blit(superficie_texto_renderizado, (ANCHO // 2 - superficie_texto_renderizado.get_width() // 2, posicion_y_texto_actual))

    # Reproducir (audio y visualmente) una nota
    def reproducir_nota_visual_auditiva(nota_a_reproducir_actual, retraso_visual_nota_actual=800):
        dibujar_cuadrados_notas(nota_activa_visual_actual=nota_a_reproducir_actual) # Iluminar
        pygame.display.flip()
        sonidos_notas[nota_a_reproducir_actual].play()
        pygame.time.delay(int(retraso_visual_nota_actual)) # Convertir a int
        dibujar_cuadrados_notas() # Apagar iluminación
        pygame.display.flip()
        pygame.time.delay(200) # Pequeña pausa

    # Secuencia de notas
    def reproducir_secuencia_para_jugador(secuencia_notas_actual):
        dibujar_cuadrados_notas() # dibujamos los cuadrados
        mostrar_texto_centrado("Escucha la secuencia", 30)
        pygame.display.flip()
        pygame.time.delay(1000) # dejamos tiempo para leer mensaje
        retraso_entre_notas_secuencia = max(300, 1000 - len(secuencia_notas_actual) * 80)
        for nota_en_secuencia in secuencia_notas_actual:
            reproducir_nota_visual_auditiva(nota_en_secuencia, retraso_entre_notas_secuencia)
            pygame.time.delay(200) # Pausa entre notas de la secuencia

    # --- ESTADO DEL JUEGO ---
    secuencia_actual_juego = [random.choice(nombres_notas)] # Secuencia de notas que el jugador debe repetir
    indice_jugador_en_secuencia_actual = 0 # nota de la secuencia que está respondiendo el jugador
    estado_actual_juego = "reproduciendo_secuencia" 
    nota_registrada_correctamente_por_jugador = False # True si el jugador ya cantó correctamente la nota actual de la secuencia
    nota_anterior_detectada_mic_buffer = "" # detectar cambios en la nota del micrófono
    reloj_pygame_control = pygame.time.Clock() # controlar los FPS
    puntuacion_maxima_lograda = 0 # Récord
    pistas_restantes_jugador = 3 # Número de pistas que le quedan al jugador
    pista_usada_para_nota_actual_secuencia = False # True si ya se usó una pista para la nota actual de la secuencia

    # --- INICIALIZACIÓN ---
    reproducir_secuencia_para_jugador(secuencia_actual_juego)
    estado_actual_juego = "esperando_respuesta_jugador" # estado para poder iniciar

    # --- BUCLE PRINCIPAL ---
    juego_en_marcha_activo = True
    while juego_en_marcha_activo:
        nota_clicada_por_jugador = None 

        # Pantalla
        dibujar_cuadrados_notas() # Dibuja los cuadrados
        mostrar_texto_centrado("Nota detectada: " + nota_actual_detectada_mic, 30, fuente_usada=fuente_principal)
        
        if estado_actual_juego == "esperando_respuesta_jugador":
            mostrar_texto_centrado(f"Canta la nota {indice_jugador_en_secuencia_actual + 1} de {len(secuencia_actual_juego)}", ALTO - 120, fuente_usada=fuente_principal)

        # Puntuación 
        puntuacion_actual_partida = len(secuencia_actual_juego) - 1 if len(secuencia_actual_juego) > 1 else 0
        if estado_actual_juego == "fallo": # Si acaba de fallar, la puntuación es la que tenía antes del fallo
            puntuacion_actual_partida = max(0, len(secuencia_actual_juego) -1 -1) # se guardará como réecord si se supera este
        elif indice_jugador_en_secuencia_actual > 0 : # Si ha acertado alguna nota en la secuencia actual
            puntuacion_actual_partida = len(secuencia_actual_juego) -1
        elif len(secuencia_actual_juego) == 1 and indice_jugador_en_secuencia_actual == 0 : # primera nota
            puntuacion_actual_partida = 0
            
        texto_puntuacion_partida_renderizado = fuente_pequena.render(f"Puntuacion: {puntuacion_actual_partida}", True, COLOR_TEXTO)
        pantalla.blit(texto_puntuacion_partida_renderizado, (10, 10)) # Puntuación (posición)
        
        texto_record_renderizado = fuente_pequena.render(f"Record: {puntuacion_maxima_lograda}", True, COLOR_TEXTO)
        pantalla.blit(texto_record_renderizado, (10, 40)) # Récord posició(posición)
        
        superficie_texto_pistas_renderizado = fuente_pequena.render(f"Pistas: {pistas_restantes_jugador}", True, COLOR_TEXTO)
        pantalla.blit(superficie_texto_pistas_renderizado, (10, 70)) # Pistas (posición)

        pygame.display.flip() # Actualizar la pantalla completa

        # --- EVENTOS ---
        for evento_actual in pygame.event.get():
            if evento_actual.type == pygame.QUIT: # Si el jugador cierra la ventana
                juego_en_marcha_activo = False
            
            # Clic para pista
            if evento_actual.type == pygame.MOUSEBUTTONDOWN and estado_actual_juego == "esperando_respuesta_jugador":
                posicion_mouse_clic = pygame.mouse.get_pos()
                for nombre_nota, rectangulo_nota in rectangulos_notas.items():
                    if rectangulo_nota.collidepoint(posicion_mouse_clic):
                        nota_clicada_por_jugador = nombre_nota # Se asigna si hay clic en un rectángulo
                        break
                
                if nota_clicada_por_jugador:
                    # Si se clica la nota correcta de la secuencia
                    if nota_clicada_por_jugador == secuencia_actual_juego[indice_jugador_en_secuencia_actual]:
                        # si no se ha usado pista para esta nota
                        if not pista_usada_para_nota_actual_secuencia and pistas_restantes_jugador > 0:
                            sonidos_notas[nota_clicada_por_jugador].play()
                            pista_usada_para_nota_actual_secuencia = True 
                            pistas_restantes_jugador -= 1
                            pygame.time.delay(1200) # Pausa para que suene la pista
                            # Limpiar detección actual y esperar que el jugador cante de nuevo
                            nota_actual_detectada_mic = ""       
                            nota_anterior_detectada_mic_buffer = ""     
                            nota_registrada_correctamente_por_jugador = False 
                    else:
                        # Si se clica una nota incorrecta, el jugador pierde
                        estado_actual_juego = "fallo" # Cambiar a estado de fallo


        # --- LÓGICA DEL JUEGO ---
        if estado_actual_juego == "esperando_respuesta_jugador":
            # Si se detecta una nota nueva y no ha sido registrada aún para la posición actual:
            if nota_actual_detectada_mic != "" and nota_actual_detectada_mic != nota_anterior_detectada_mic_buffer and not nota_registrada_correctamente_por_jugador:
                # Si la nota detectada es la correcta en la secuencia:
                if nota_actual_detectada_mic == secuencia_actual_juego[indice_jugador_en_secuencia_actual]:
                    indice_jugador_en_secuencia_actual += 1 # Avanzar en la secuencia
                    nota_registrada_correctamente_por_jugador = True # Marcar como registrada para esta posición
                    pista_usada_para_nota_actual_secuencia = False # Permitir pista para la siguiente nota de la secuencia
                    time.sleep(0.5) # Pausa breve tras acierto
                    
                    # Si se completó la secuencia actual:
                    if indice_jugador_en_secuencia_actual == len(secuencia_actual_juego):
                        secuencia_actual_juego.append(random.choice(nombres_notas)) # Añadir nueva nota
                        indice_jugador_en_secuencia_actual = 0 # Reiniciar índice del jugador
                        estado_actual_juego = "reproduciendo_secuencia" # Pasar a reproducir nueva secuencia
                        pygame.time.delay(700) # Pausa antes de la nueva secuencia
                        reproducir_secuencia_para_jugador(secuencia_actual_juego)
                        estado_actual_juego = "esperando_respuesta_jugador"
                        nota_registrada_correctamente_por_jugador = False # Preparar para la nueva nota
                else:
                    estado_actual_juego = "fallo"  # si la nota detectada es incorrecta
            
            # Si hay silencio: 
            elif nota_actual_detectada_mic == "":
                nota_registrada_correctamente_por_jugador = False
            # Actualizar la nota anterior para la detección de cambio de nota
            nota_anterior_detectada_mic_buffer = nota_actual_detectada_mic

        # Estado de "fallo"
        if estado_actual_juego == "fallo":
            puntuacion_maxima_lograda = max(puntuacion_maxima_lograda, len(secuencia_actual_juego) - 1) # Actualizar récord
            pantalla.blit(imagen_fondo, (0, 0)) # Limpiar pantalla
            if nota_clicada_por_jugador and nota_clicada_por_jugador != secuencia_actual_juego[indice_jugador_en_secuencia_actual]:
                 mostrar_texto_centrado("¡Fallaste! Tocaste la nota incorrecta.", ALTO // 2, fuente_usada=fuente_principal)
            else:
                 mostrar_texto_centrado("¡Fallaste! Reiniciando...", ALTO // 2, fuente_usada=fuente_principal)
            pygame.display.flip()
            sonido_fallo_juego.play()
            pygame.time.delay(1200) # Pausa para mensaje

            # Reiniciar variables del juego para una nueva partida
            secuencia_actual_juego = [random.choice(nombres_notas)]
            indice_jugador_en_secuencia_actual = 0
            estado_actual_juego = "reproduciendo_secuencia"
            pista_usada_para_nota_actual_secuencia = False 
            pistas_restantes_jugador = 3 # Reiniciar pistas
            nota_registrada_correctamente_por_jugador = False
            nota_actual_detectada_mic = ""
            nota_anterior_detectada_mic_buffer = ""
            nota_clicada_por_jugador = None # Resetear nota clicada
            
            pygame.time.delay(500) # Pausa antes de la nueva secuencia
            reproducir_secuencia_para_jugador(secuencia_actual_juego)
            estado_actual_juego = "esperando_respuesta_jugador"

        # Controlar los FPS del juego
        reloj_pygame_control.tick(30)

    # Detener el flujo de audio y limpiar Pygame al salir del bucle principal
    flujo_audio_microfono.stop()
    flujo_audio_microfono.close()
    pygame.mixer.quit()
