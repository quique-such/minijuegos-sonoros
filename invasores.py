# --- IMPORTACIONES ---
# Paquetes
import pygame
import random
import time
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.ndimage import gaussian_filter1d
import os

# --- INICIALIZACIÓN ---
pygame.init()

# Colores
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
VERDE = (0, 255, 0)
ROJO = (255, 0, 0)
GRIS = (180, 180, 180)
NARANJA = pygame.Color("#e58438") 
LIMA = (60/255, 180/255, 75/255)

# Audio
GANANCIA_VOLUMEN_ENTRADA = 3 # Multiplicador para la señal de audio RMS
LONGITUD_HISTORIAL_VOLUMEN = 30  # Número de muestras de volumen a guardar para análisis
UMBRAL_PICO_VOLUMEN_DISPARO = 0.35 # Nivel de volumen normalizado para considerar un pico de disparo
DELTA_UMBRAL_CAMBIO_DISPARO = 0.1 # Cambio mínimo de volumen para detectar subida/bajada rápida (disparo)
UMBRAL_SONIDO_CONTINUO_BARRERA = 0.4 # Nivel de volumen para activar/mantener la barrera
MIN_TIEMPO_ENTRE_DISPAROS_SEG = 0.25 # Cooldown mínimo en segundos entre disparos
TIEMPO_ENFRIAMIENTO_BARRERA_SEG = 10 # Segundos de cooldown para la barrera tras desactivarse
MIN_DURACION_BARRERA_ACTIVA_SEG = 1.5 # Segundos que la barrera permanecerá activa como mínimo

# --- FUNCIÓN PRINCIPAL ---
def main(pantalla_juego, indice_microfono_seleccionado):
    
    # --- CONSTANTES DEL JUEGO ---
    ANCHO_PANTALLA, ALTO_PANTALLA = pantalla_juego.get_size() # Dimensiones de la pantalla

    # Fondo
    ruta_imagen_fondo = os.path.join("fondos", "invasores.png")
    imagen_fondo_original = pygame.image.load(ruta_imagen_fondo).convert()
    imagen_fondo = pygame.transform.scale(imagen_fondo_original, (ANCHO_PANTALLA, ALTO_PANTALLA))

    # Imágenes de los elementos del juego
    imagen_arma_original = pygame.image.load(os.path.join("invasores", "ballesta.png")).convert_alpha()
    imagen_oro_original = pygame.image.load(os.path.join("invasores", "oro.png")).convert_alpha()
    imagenes_invasor_originales = [
        pygame.image.load(os.path.join("invasores", "invasor.png")).convert_alpha(),
        pygame.image.load(os.path.join("invasores", "invasor2.png")).convert_alpha(),
        pygame.image.load(os.path.join("invasores", "invasor3.png")).convert_alpha(),
    ]
    imagen_escudo_original = pygame.image.load(os.path.join("invasores", "escudo.png")).convert_alpha()
    imagen_proyectil_original = pygame.image.load(os.path.join("invasores", "flecha.png")).convert_alpha()

    # Dimensiones
    ancho_arma, alto_arma = 60, 40
    tamano_oro = 35
    tamano_invasor = 60
    ancho_escudo, alto_escudo = 100, 25
    ancho_proyectil, alto_proyectil = 15, 45

    imagen_arma = pygame.transform.scale(imagen_arma_original, (ancho_arma, alto_arma))
    imagen_oro = pygame.transform.scale(imagen_oro_original, (tamano_oro, tamano_oro))
    imagenes_invasor = [pygame.transform.scale(img, (tamano_invasor, tamano_invasor)) for img in imagenes_invasor_originales]
    imagen_escudo = pygame.transform.scale(imagen_escudo_original, (ancho_escudo, alto_escudo))
    imagen_proyectil = pygame.transform.scale(imagen_proyectil_original, (ancho_proyectil, alto_proyectil))
    
    # Posición inicial del arma
    rect_imagen_arma = imagen_arma.get_rect()
    rect_arma_inicial = pygame.Rect(ANCHO_PANTALLA // 2 - rect_imagen_arma.width // 2, ALTO_PANTALLA - rect_imagen_arma.height - 10, rect_imagen_arma.width, rect_imagen_arma.height)
    
    # Texto
    RUTA_FUENTE_MINECRAFT = os.path.join("fonts", "minecraft.ttf")
    fuente_grande = pygame.font.Font(RUTA_FUENTE_MINECRAFT, 36)
    fuente_pequena = pygame.font.Font(RUTA_FUENTE_MINECRAFT, 16)

    # --- ESTADO DEL JUEGO ---
    estado_juego = { # Diccionario para almacenar el estado del juego
        'rect_arma': rect_arma_inicial.copy(), # Rectángulo que representa la posición y tamaño del arma
        'rect_barrera': None, # Rectángulo de la barrera, None si no está activa
        'barrera_esta_activa': False, # Indica si la barrera está activa
        'lista_invasores': [], # Lista de diccionarios, cada uno representando un invasor
        'lista_proyectiles': [], # Lista de diccionarios, cada uno representando un proyectil
        'lista_oros_activos': [], # Lista de diccionarios, cada uno representando un oro
        'contador_eliminaciones': 0, # Número de invasores eliminado
        'cantidad_oro_actual': 0, # Cantidad de oro que posee el jugador
        'nivel_actual': 1, # Nivel actual del juego
        'total_oro_recolectado_partida': 0, # Total de oro recolectado en la partida actual
        'volumen_actual_rms': 0, # Volumen RMS actual de la entrada de audio
        'historial_ultimos_volumenes': [], # Historial de los últimos valores de volumen
        'historial_volumen_db_grafico': [], # Historial de volumen en dB para el gráfico
        'historial_tiempo_eje_x_grafico': [], # Historial de tiempos para el eje X del gráfico
        'historial_eventos_barrera_grafico': [], # Eventos de activación/desactivación de barrera para el gráfico
        'historial_eventos_disparo_grafico': [], # Eventos de disparo para el gráfico
        'historial_eventos_revivir_grafico': [], # Eventos de uso de vida extra para el gráfico
        'disparo_esta_permitido': True, # Indica si el jugador puede disparar
        'tiempo_ultimo_disparo_seg': 0, # Momento del último disparo
        'pico_audio_detectado_para_disparo': False, # Indica si se ha detectado un pico de audio para disparo
        'tiempo_fin_enfriamiento_barrera_seg': 0, # Momento en que termina el enfriamiento de la barrera
        'tiempo_activacion_barrera_seg': 0, # Momento en que se activó la barrera
        'partida_terminada': False, # Indica si la partida ha terminado
        'grafico_final_generado': False, # Indica si el gráfico final ya se generó
        'tiempo_inicio_partida_seg': time.time(), # Momento de inicio de la partida
        'superficie_render_grafico_final': None, # Superficie de Pygame con el gráfico renderizado
        'mostrar_mensaje_vida_extra_en_pantalla': False, # Indica si se debe mostrar el mensaje de vida extra
        'tiempo_fin_mensaje_vida_extra_seg': 0 # Momento en que debe desaparecer el mensaje de vida extra
    }
    
    def procesar_audio_arma(datos_entrada, *_):
        """
        Callback para procesar los datos de entrada de audio en tiempo real.
        Calcula el volumen RMS y actualiza el estado del juego en función del audio (disparos, barrera).
        """
        # Calcula el volumen RMS (Root Mean Square) de la señal de entrada
        rms_volumen = np.sqrt(np.mean(np.square(datos_entrada)))
        # Aplica una ganancia al volumen calculado
        estado_juego['volumen_actual_rms'] = rms_volumen * GANANCIA_VOLUMEN_ENTRADA

        # Solo procesa el audio si la partida no ha terminado
        if not estado_juego['partida_terminada']:
            # Calcula el tiempo transcurrido desde el inicio de la partida
            tiempo_actual_partida_relativo = time.time() - estado_juego['tiempo_inicio_partida_seg']
            # Añade el volumen actual al historial
            estado_juego['historial_ultimos_volumenes'].append(estado_juego['volumen_actual_rms'])
            # Convierte el volumen RMS a decibelios (dB) para el gráfico
            db_actual = 20 * np.log10(estado_juego['volumen_actual_rms'] + 1e-6) # Suma 1e-6 para evitar log(0)
            estado_juego['historial_volumen_db_grafico'].append(db_actual)
            estado_juego['historial_tiempo_eje_x_grafico'].append(tiempo_actual_partida_relativo)

            # Mantiene el historial de volúmenes con una longitud máxima definida
            if len(estado_juego['historial_ultimos_volumenes']) > LONGITUD_HISTORIAL_VOLUMEN:
                estado_juego['historial_ultimos_volumenes'].pop(0)

            # Lógica para detectar un disparo (pico de sonido)
            if len(estado_juego['historial_ultimos_volumenes']) >= 3: # Necesita al menos 3 muestras para detectar un pico
                vol_anterior, vol_pico, vol_siguiente = estado_juego['historial_ultimos_volumenes'][-3:]
                # Comprueba si hay una subida rápida de volumen seguida de una bajada rápida
                es_subida_rapida = (vol_pico - vol_anterior) > DELTA_UMBRAL_CAMBIO_DISPARO
                es_bajada_rapida = (vol_siguiente - vol_pico) < -DELTA_UMBRAL_CAMBIO_DISPARO
                # Comprueba si el pico supera el umbral de disparo
                pico_supera_umbral = vol_pico > UMBRAL_PICO_VOLUMEN_DISPARO
                
                tiempo_actual_deteccion = time.time()

                # Si se cumplen las condiciones y el disparo está permitido y no hay un pico ya detectado y la barrera no está activa
                if estado_juego['disparo_esta_permitido'] and not estado_juego['pico_audio_detectado_para_disparo'] and not estado_juego['barrera_esta_activa']:
                    if es_subida_rapida and es_bajada_rapida and pico_supera_umbral:
                        estado_juego['pico_audio_detectado_para_disparo'] = True # Marca que se detectó un pico
                        estado_juego['tiempo_ultimo_disparo_seg'] = tiempo_actual_deteccion # Registra el tiempo del disparo
                        estado_juego['disparo_esta_permitido'] = False # Deshabilita el disparo temporalmente (cooldown)
                        # Crea un nuevo proyectil
                        rect_proyectil_nuevo = imagen_proyectil.get_rect(centerx=estado_juego['rect_arma'].centerx, bottom=estado_juego['rect_arma'].top)
                        estado_juego['lista_proyectiles'].append({'image': imagen_proyectil, 'rect': rect_proyectil_nuevo})
                        # Registra el evento de disparo para el gráfico
                        db_pico_registrado = 20 * np.log10(vol_pico + 1e-6)
                        estado_juego['historial_eventos_disparo_grafico'].append({'time': tiempo_actual_partida_relativo, 'db_value': db_pico_registrado})
                # Resetea la detección de pico si el volumen baja lo suficiente
                if estado_juego['pico_audio_detectado_para_disparo'] and estado_juego['volumen_actual_rms'] < (UMBRAL_PICO_VOLUMEN_DISPARO * 0.7):
                    estado_juego['pico_audio_detectado_para_disparo'] = False

                # Rehabilita el disparo después del cooldown
                if not estado_juego['disparo_esta_permitido'] and (tiempo_actual_deteccion - estado_juego['tiempo_ultimo_disparo_seg']) > MIN_TIEMPO_ENTRE_DISPAROS_SEG:
                    estado_juego['disparo_esta_permitido'] = True
            
            # Activar/desactivar la barrera
            if len(estado_juego['historial_ultimos_volumenes']) >= 10: # Necesita suficientes muestras para detectar sonido continuo
                tiempo_actual_deteccion = time.time()
                
                # Cuenta cuántas de las últimas muestras superan el umbral de sonido continuo
                muestras_altas_barrera = sum(1 for v_h in estado_juego['historial_ultimos_volumenes'][-10:] if v_h > UMBRAL_SONIDO_CONTINUO_BARRERA)
                sonido_continuo_detectado = muestras_altas_barrera >= 8 # Considera sonido continuo si 8 de 10 muestras son altas

                # Condición para desactivar la barrera: el volumen actual debe ser bajo
                volumen_actual_bajo_para_desactivar = estado_juego['volumen_actual_rms'] < UMBRAL_SONIDO_CONTINUO_BARRERA * 0.8

                # Activar la barrera si no está activa, se detecta sonido continuo y ha pasado el cooldown
                if not estado_juego['barrera_esta_activa'] and sonido_continuo_detectado and tiempo_actual_deteccion > estado_juego['tiempo_fin_enfriamiento_barrera_seg']:
                    estado_juego['rect_barrera'] = imagen_escudo.get_rect(centerx=estado_juego['rect_arma'].centerx, bottom=estado_juego['rect_arma'].top - 5)
                    estado_juego['barrera_esta_activa'] = True
                    estado_juego['tiempo_activacion_barrera_seg'] = tiempo_actual_deteccion
                    estado_juego['historial_eventos_barrera_grafico'].append({'time': tiempo_actual_partida_relativo, 'type': 'activated'})
                # Desactivar la barrera si está activa, el volumen es bajo y ha pasado la duración mínima
                elif estado_juego['barrera_esta_activa'] and volumen_actual_bajo_para_desactivar and (tiempo_actual_deteccion - estado_juego['tiempo_activacion_barrera_seg']) > MIN_DURACION_BARRERA_ACTIVA_SEG:
                    estado_juego['barrera_esta_activa'] = False
                    estado_juego['rect_barrera'] = None
                    estado_juego['tiempo_fin_enfriamiento_barrera_seg'] = tiempo_actual_deteccion + TIEMPO_ENFRIAMIENTO_BARRERA_SEG 
                    estado_juego['historial_eventos_barrera_grafico'].append({'time': tiempo_actual_partida_relativo, 'type': 'deactivated'})

    # --- CONFIGURACIÓN DE ENTRADA DE SONIDO ---
    # Inicia el stream de entrada de audio
    flujo_entrada_audio = sd.InputStream(device=indice_microfono_seleccionado, channels=1, samplerate=44100, callback=procesar_audio_arma)
    flujo_entrada_audio.start()

    reloj_juego = pygame.time.Clock()
    running = True # Bucle principal del juego.
    pygame.display.set_caption("Minijuego - Invasores") # Título de la ventana del juego

    def generar_grafico_final_partida():
        """
        Genera un gráfico del historial de audio de la partida utilizando Matplotlib
        y lo convierte en una superficie de Pygame para mostrarlo en la pantalla de Game Over.
        """
        # --- GRÁFICO ---
        if not estado_juego['historial_tiempo_eje_x_grafico'] or not estado_juego['historial_volumen_db_grafico']:
            print("No hay datos suficientes para generar el gráfico de la partida.")
            estado_juego['superficie_render_grafico_final'] = None
            return

        # Crea la figura y los ejes para el gráfico
        fig, ax = plt.subplots(figsize=(7.8, 2.8), dpi=100)
        array_db_historial = np.array(estado_juego['historial_volumen_db_grafico'])
        # Suaviza la línea de volumen si hay suficientes datos
        db_suavizados = gaussian_filter1d(array_db_historial, sigma=1.5) if array_db_historial.size > 3 else array_db_historial
        # Dibuja la línea de volumen
        ax.plot(estado_juego['historial_tiempo_eje_x_grafico'], db_suavizados, color='blue', label='Volumen (dB)', linewidth=1.5, zorder=1)

        # Dibuja los puntos de los eventos de disparo
        valores_db_disparos = [evento['db_value'] for evento in estado_juego['historial_eventos_disparo_grafico']]
        tiempos_disparos_grafico = [evento['time'] for evento in estado_juego['historial_eventos_disparo_grafico']]
        if tiempos_disparos_grafico:
            ax.scatter(tiempos_disparos_grafico, valores_db_disparos, color='red', marker='o', label='Disparos', 
                        s=70, zorder=5, edgecolors='black', linewidths=1)

        # Calcula los límites para las líneas verticales de eventos
        min_db_grafico = np.min(db_suavizados) if db_suavizados.size > 0 else -40
        max_db_grafico = np.max(db_suavizados) if db_suavizados.size > 0 else 5
        
        # Dibuja líneas verticales para los eventos de activación/desactivación de la barrera
        for evento_barrera in estado_juego['historial_eventos_barrera_grafico']:
            color_linea_barrera = 'cyan' if evento_barrera['type'] == 'activated' else 'orange'
            etiqueta_linea_barrera = 'Barrera Activada' if evento_barrera['type'] == 'activated' else 'Barrera Desactivada'
            # Añade la etiqueta a la leyenda solo si no existe para evitar duplicados
            etiquetas_existentes_leyenda = [handle.get_label() for handle in ax.get_legend_handles_labels()[0]]
            if etiqueta_linea_barrera not in etiquetas_existentes_leyenda:
                ax.vlines(x=evento_barrera['time'], ymin=min_db_grafico -1 , ymax=max_db_grafico +1, color=color_linea_barrera, linestyle='--', label=etiqueta_linea_barrera, linewidth=2, zorder=3)
            else:
                ax.vlines(x=evento_barrera['time'], ymin=min_db_grafico -1, ymax=max_db_grafico+1, color=color_linea_barrera, linestyle='--', linewidth=2, zorder=3)

        # Dibuja líneas verticales para los eventos de uso de vida extra
        tiempos_eventos_revivir = [evento['time'] for evento in estado_juego['historial_eventos_revivir_grafico']]
        if tiempos_eventos_revivir:
            etiqueta_revivir_anadida = False
            for tiempo_revivir in tiempos_eventos_revivir:
                if not etiqueta_revivir_anadida:
                    ax.vlines(x=tiempo_revivir, ymin=min_db_grafico -1, ymax=max_db_grafico +1, color=LIMA, linestyle=':', label='Vida Extra Usada', linewidth=2, zorder=4)
                    etiqueta_revivir_anadida = True
                else:
                    ax.vlines(x=tiempo_revivir, ymin=min_db_grafico -1, ymax=max_db_grafico +1, color=LIMA, linestyle=':', linewidth=2, zorder=4)

        # Configuración de títulos y etiquetas del gráfico
        ax.set_title("Historial de Audio de la Partida", fontsize=12)
        ax.set_xlabel("Tiempo (s)", fontsize=9)
        ax.set_ylabel("dB", fontsize=9)
        
        # Ajusta los límites del eje Y para que todos los datos sean visibles
        todos_valores_db_para_ylim = list(db_suavizados) + valores_db_disparos
        min_ylim_grafico = np.min(todos_valores_db_para_ylim) - 3 if todos_valores_db_para_ylim else -40
        max_ylim_grafico = np.max(todos_valores_db_para_ylim) + 3 if todos_valores_db_para_ylim else 5
        ax.set_ylim(min_ylim_grafico, max_ylim_grafico)

        ax.grid(True, linestyle=':', alpha=0.7) # Añade una rejilla al gráfico
        ax.tick_params(axis='both', which='major', labelsize=8) # Configura el tamaño de las etiquetas de los ejes
        
        # Crea la leyenda del gráfico, eliminando duplicados
        handles_leyenda, etiquetas_leyenda = ax.get_legend_handles_labels()
        leyenda_sin_duplicados = dict(zip(etiquetas_leyenda, handles_leyenda))
        leyenda_obj = ax.legend(leyenda_sin_duplicados.values(), leyenda_sin_duplicados.keys(), loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=7)
        if leyenda_obj: leyenda_obj.get_frame().set_alpha(0.8)

        fig.tight_layout(rect=[0, 0, 0.80, 1]) # Ajusta el diseño para que la leyenda no se corte
        
        # Convierte el gráfico de Matplotlib a una superficie de Pygame
        lienzo_grafico = FigureCanvasAgg(fig)
        lienzo_grafico.draw()
        datos_crudos_rgba = lienzo_grafico.buffer_rgba()
        estado_juego['superficie_render_grafico_final'] = pygame.image.frombuffer(datos_crudos_rgba, lienzo_grafico.get_width_height(), "RGBA")
        plt.close(fig) # Cierra la figura de Matplotlib para liberar memoria

        # --- BUCLE PRINCIPAL ---
    while running:
        tiempo_actual_bucle = time.time() # Tiempo actual para cálculos de cooldown, etc
        pantalla_juego.blit(imagen_fondo, (0, 0)) # Fondo

        # --- EVENTOS (TECLADO, CERRAR VENTANA) ---
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT: # Si se cierra la ventana
                running = False
            if evento.type == pygame.KEYDOWN: # Si se presiona una tecla
                # Reiniciar el juego si está terminado y se presiona 'R'
                if estado_juego['partida_terminada']:
                    # Restablece todas las variables del estado del juego a sus valores iniciales
                    estado_juego['rect_arma'].x = ANCHO_PANTALLA // 2 - rect_imagen_arma.width // 2
                    estado_juego['rect_arma'].y = ALTO_PANTALLA - rect_imagen_arma.height - 10
                    estado_juego['lista_invasores'].clear()
                    estado_juego['lista_proyectiles'].clear()
                    estado_juego['lista_oros_activos'].clear()
                    estado_juego['contador_eliminaciones'] = 0
                    estado_juego['cantidad_oro_actual'] = 0
                    estado_juego['nivel_actual'] = 1
                    estado_juego['total_oro_recolectado_partida'] = 0
                    estado_juego['partida_terminada'] = False
                    estado_juego['grafico_final_generado'] = False
                    estado_juego['superficie_render_grafico_final'] = None
                    estado_juego['historial_volumen_db_grafico'].clear()
                    estado_juego['historial_tiempo_eje_x_grafico'].clear()
                    estado_juego['historial_ultimos_volumenes'].clear()
                    estado_juego['historial_eventos_disparo_grafico'].clear()
                    estado_juego['historial_eventos_barrera_grafico'].clear()
                    estado_juego['historial_eventos_revivir_grafico'].clear()
                    estado_juego['barrera_esta_activa'] = False
                    estado_juego['rect_barrera'] = None
                    estado_juego['disparo_esta_permitido'] = True
                    estado_juego['pico_audio_detectado_para_disparo'] = False
                    estado_juego['tiempo_fin_enfriamiento_barrera_seg'] = 0
                    estado_juego['tiempo_inicio_partida_seg'] = time.time() # Reinicia el tiempo de inicio de la partida
                    estado_juego['mostrar_mensaje_vida_extra_en_pantalla'] = False
                    estado_juego['historial_eventos_revivir_grafico'].clear() # Limpia historial de revivir al reiniciar

        teclas_presionadas = pygame.key.get_pressed() # Obtiene el estado de todas las teclas

        # --- GAME OVER ---
        if estado_juego['partida_terminada']:
            if not estado_juego['grafico_final_generado']: # Genera el gráfico solo una vez
                generar_grafico_final_partida()
                estado_juego['grafico_final_generado'] = True
            # Fondo
            pantalla_juego.blit(imagen_fondo, (0,0))

            # Título "GAME OVER"
            titulo_game_over_surf = fuente_grande.render("GAME OVER", True, NARANJA)
            pantalla_juego.blit(titulo_game_over_surf, (ANCHO_PANTALLA//2 - titulo_game_over_surf.get_width()//2, 50))
            
            # Estadísticas
            mensajes_estadisticas = [
                f"Nivel alcanzado: {estado_juego['nivel_actual']}",
                f"Enemigos eliminados: {estado_juego['contador_eliminaciones']}",
                f"Total Oro Recogido: {estado_juego['total_oro_recolectado_partida']}", 
                "Presiona R para reiniciar."
            ]
            pos_y_offset_estadisticas = 100 
            for i, mensaje_est in enumerate(mensajes_estadisticas):
                color_texto_est = BLANCO
                if "Presiona R para reiniciar." in mensaje_est:
                    color_texto_est = NARANJA
                surf_mensaje_est = fuente_pequena.render(mensaje_est, True, color_texto_est) 
                pantalla_juego.blit(surf_mensaje_est, (ANCHO_PANTALLA//2 - surf_mensaje_est.get_width()//2, pos_y_offset_estadisticas + i * 25)) 

            # Gráfico
            if estado_juego['superficie_render_grafico_final']:
                pos_x_grafico = (ANCHO_PANTALLA - estado_juego['superficie_render_grafico_final'].get_width()) // 2
                pos_y_grafico = pos_y_offset_estadisticas + len(mensajes_estadisticas) * 25 + 10
                # Ajusta la posición Y del gráfico si se sale de la pantalla.
                if pos_y_grafico + estado_juego['superficie_render_grafico_final'].get_height() > ALTO_PANTALLA - 10:
                    pos_y_grafico = ALTO_PANTALLA - estado_juego['superficie_render_grafico_final'].get_height() - 10
                pantalla_juego.blit(estado_juego['superficie_render_grafico_final'], (pos_x_grafico, pos_y_grafico))
            else: # Muestra un mensaje si no hay datos para el gráfico.
                surf_no_datos_grafico = fuente_pequena.render("No hay datos de audio para mostrar.", True, BLANCO)
                pantalla_juego.blit(surf_no_datos_grafico, (ANCHO_PANTALLA//2 - surf_no_datos_grafico.get_width()//2, ALTO_PANTALLA//2))
            
            pygame.display.flip() # Actualiza la pantalla
            reloj_juego.tick(60)
            continue # Salta el resto del bucle de juego si la partida ha terminado

        # Movimiento del arma con las teclas izquierda y derecha
        if teclas_presionadas[pygame.K_LEFT] and estado_juego['rect_arma'].left > 0:
            estado_juego['rect_arma'].x -= 5
        if teclas_presionadas[pygame.K_RIGHT] and estado_juego['rect_arma'].right < ANCHO_PANTALLA:
            estado_juego['rect_arma'].x += 5
        
        # Actualiza la posición de la barrera si está activa para que siga al arma
        if estado_juego['barrera_esta_activa'] and estado_juego['rect_barrera']:
            estado_juego['rect_barrera'].centerx = estado_juego['rect_arma'].centerx

        # Generación aleatoria de invasores
        if random.random() < (0.015 + estado_juego['nivel_actual'] * 0.005): # La probabilidad aumenta con el nivel.
            imagen_invasor_elegida = random.choice(imagenes_invasor) # Elige una imagen de invasor al azar.
            max_x_aparicion_invasor = max(0, ANCHO_PANTALLA - imagen_invasor_elegida.get_width())
            rect_nuevo_invasor = imagen_invasor_elegida.get_rect(topleft=(random.randint(0, max_x_aparicion_invasor), -imagen_invasor_elegida.get_height()))
            estado_juego['lista_invasores'].append({'image': imagen_invasor_elegida, 'rect': rect_nuevo_invasor})

        # Generación aleatoria de monedas
        if random.random() < (0.003 + estado_juego['nivel_actual'] * 0.001): # La probabilidad aumenta con el nivel.
            max_x_aparicion_moneda = max(0, ANCHO_PANTALLA - imagen_oro.get_width())
            rect_nueva_moneda = imagen_oro.get_rect(topleft=(random.randint(0, max_x_aparicion_moneda), -imagen_oro.get_height()))
            estado_juego['lista_oros_activos'].append({'image': imagen_oro, 'rect': rect_nueva_moneda})

        # Lógica de los invasores (movimiento, colisiones)
        for datos_invasor in list(estado_juego['lista_invasores']):
            datos_invasor['rect'].y += (2 + estado_juego['nivel_actual'] * 0.5)
            # Colisión invasor-arma
            if datos_invasor['rect'].colliderect(estado_juego['rect_arma']):
                # Vida extra
                if estado_juego['cantidad_oro_actual'] >= 10: 
                    estado_juego['cantidad_oro_actual'] -= 10
                    # Reposiciona el arma y limpia los invasores de la pantalla
                    estado_juego['rect_arma'].x = ANCHO_PANTALLA // 2 - rect_imagen_arma.width // 2
                    estado_juego['rect_arma'].y = ALTO_PANTALLA - rect_imagen_arma.height - 10
                    estado_juego['lista_invasores'].clear() 
                    # Mensaje de "VIDA EXTRA"
                    estado_juego['mostrar_mensaje_vida_extra_en_pantalla'] = True
                    estado_juego['tiempo_fin_mensaje_vida_extra_seg'] = time.time() + 2
                    # Registra el evento de revivir para el gráfico
                    tiempo_revivir_relativo = time.time() - estado_juego['tiempo_inicio_partida_seg']
                    estado_juego['historial_eventos_revivir_grafico'].append({'time': tiempo_revivir_relativo})
                else: # Si no tiene suficiente oro, la partida termina
                    estado_juego['partida_terminada'] = True
            # Colisión invasor-barrera
            elif estado_juego['rect_barrera'] and estado_juego['barrera_esta_activa'] and datos_invasor['rect'].colliderect(estado_juego['rect_barrera']):
                estado_juego['lista_invasores'].remove(datos_invasor) # El invasor es destruido por la barrera
            # Si el invasor sale de la pantalla por abajo
            elif datos_invasor['rect'].top > ALTO_PANTALLA:
                estado_juego['lista_invasores'].remove(datos_invasor)

        # Lógica de las monedas (movimiento, recolección)
        for datos_moneda in list(estado_juego['lista_oros_activos']):
            datos_moneda['rect'].y += (3 + estado_juego['nivel_actual'] * 0.3)
            # Colisión moneda-arma (recolección)
            if datos_moneda['rect'].colliderect(estado_juego['rect_arma']):
                estado_juego['cantidad_oro_actual'] += 1
                estado_juego['total_oro_recolectado_partida'] += 1
                estado_juego['lista_oros_activos'].remove(datos_moneda)
            # Si la moneda sale de la pantalla por abajo
            elif datos_moneda['rect'].top > ALTO_PANTALLA:
                estado_juego['lista_oros_activos'].remove(datos_moneda)

        # Lógica de los proyectiles (movimiento, colisiones)
        for datos_proyectil in list(estado_juego['lista_proyectiles']):
            datos_proyectil['rect'].y -= 10
            # Si el proyectil sale de la pantalla por arriba
            if datos_proyectil['rect'].bottom < 0:
                estado_juego['lista_proyectiles'].remove(datos_proyectil)
            else:
                # Colisión proyectil-invasor
                for datos_invasor_objetivo in list(estado_juego['lista_invasores']):
                    if datos_proyectil['rect'].colliderect(datos_invasor_objetivo['rect']):
                        estado_juego['lista_proyectiles'].remove(datos_proyectil)
                        estado_juego['lista_invasores'].remove(datos_invasor_objetivo)
                        estado_juego['contador_eliminaciones'] += 1
                        # Sube de nivel al alcanzar cierto número de eliminaciones (hasta nivel 10)
                        if estado_juego['contador_eliminaciones'] >= estado_juego['nivel_actual'] * 5 and estado_juego['nivel_actual'] < 10:
                            estado_juego['nivel_actual'] += 1
                        break # El proyectil solo puede impactar a un invasor

        # Arma
        pantalla_juego.blit(imagen_arma, estado_juego['rect_arma'].topleft)

        # Muestra el mensaje de "VIDA EXTRA" si está activo
        if estado_juego['mostrar_mensaje_vida_extra_en_pantalla']:
            if time.time() < estado_juego['tiempo_fin_mensaje_vida_extra_seg']:
                surf_vida_extra_grande = fuente_grande.render("VIDA EXTRA", True, NARANJA)
                surf_menos_monedas = fuente_pequena.render("-10 Monedas", True, BLANCO)
                
                rect_vida_extra = surf_vida_extra_grande.get_rect(center=(ANCHO_PANTALLA // 2, ALTO_PANTALLA // 2 - 20))
                rect_menos_monedas = surf_menos_monedas.get_rect(center=(ANCHO_PANTALLA // 2, ALTO_PANTALLA // 2 + 15))
                
                pantalla_juego.blit(surf_vida_extra_grande, rect_vida_extra)
                pantalla_juego.blit(surf_menos_monedas, rect_menos_monedas)
            else: # Desactiva el mensaje después del tiempo establecido.
                estado_juego['mostrar_mensaje_vida_extra_en_pantalla'] = False

        # Dibuja la barrera si está activa
        if estado_juego['rect_barrera'] and estado_juego['barrera_esta_activa']:
            pantalla_juego.blit(imagen_escudo, estado_juego['rect_barrera'].topleft)
        
        # Dibuja todos los invasores activos
        for datos_inv_dibujar in estado_juego['lista_invasores']:
            pantalla_juego.blit(datos_inv_dibujar['image'], datos_inv_dibujar['rect'].topleft)

        # Dibuja todos los proyectiles activos
        for datos_proy_dibujar in estado_juego['lista_proyectiles']:
            pantalla_juego.blit(datos_proy_dibujar['image'], datos_proy_dibujar['rect'].topleft)

        # Dibuja todas las monedas activas
        for datos_mon_dibujar in estado_juego['lista_oros_activos']:
            pantalla_juego.blit(datos_mon_dibujar['image'], datos_mon_dibujar['rect'].topleft)

        # Dibuja la información del HUD (Heads-Up Display)
        textos_hud = [
            (f"Nivel: {estado_juego['nivel_actual']}", (10, 10)),
            (f"Eliminaciones: {estado_juego['contador_eliminaciones']}", (10, 35)),
            (f"Oro: {estado_juego['cantidad_oro_actual']}", (10, 60))
        ]
        for texto_hud, pos_hud in textos_hud:
            surf_texto_hud = fuente_pequena.render(texto_hud, True, BLANCO)
            pantalla_juego.blit(surf_texto_hud, pos_hud)

        # Muestra el estado de la barrera en el HUD
        texto_hud_barrera = ""
        tiempo_enfriamiento_restante_barrera = max(0, int( (estado_juego['tiempo_fin_enfriamiento_barrera_seg'] - tiempo_actual_bucle) ))
        estado_actual_barrera_hud = "Disponible" if tiempo_enfriamiento_restante_barrera == 0 else f"Cooldown: {tiempo_enfriamiento_restante_barrera}s"

        if estado_juego['barrera_esta_activa']:
            texto_hud_barrera = "Barrera: ACTIVA"
            color_hud_barrera = VERDE
        elif estado_actual_barrera_hud == "Disponible":
            texto_hud_barrera = "Barrera: Disponible"
            color_hud_barrera = NARANJA
        else: 
            texto_hud_barrera = f"Barrera: {estado_actual_barrera_hud}"
            color_hud_barrera = ROJO
        
        surf_texto_barrera_hud = fuente_pequena.render(texto_hud_barrera, True, color_hud_barrera)
        pantalla_juego.blit(surf_texto_barrera_hud, (10, 85))
        
        # Muestra un mensaje si el disparo está bloqueado por la barrera activa
        if estado_juego['barrera_esta_activa']: 
            surf_disparo_bloqueado = fuente_pequena.render("Disparo: BLOQUEADO", True, NARANJA)
            pantalla_juego.blit(surf_disparo_bloqueado, (10, 110))

        # Dibuja la barra de volumen actual en el HUD
        ancho_barra_vol, alto_barra_vol = 150, 15
        pos_x_barra_vol, pos_y_barra_vol = ANCHO_PANTALLA - ancho_barra_vol - 10, 10
        pygame.draw.rect(pantalla_juego, GRIS, (pos_x_barra_vol, pos_y_barra_vol, ancho_barra_vol, alto_barra_vol), border_radius=3)
        ancho_relleno_vol = int(ancho_barra_vol * min(max(0, estado_juego['volumen_actual_rms']), 1.0)) # Normaliza el volumen a 0-1
        # Cambia el color de la barra de volumen según el nivel
        color_relleno_vol = ROJO if estado_juego['volumen_actual_rms'] > UMBRAL_PICO_VOLUMEN_DISPARO else VERDE if estado_juego['volumen_actual_rms'] > 0.01 else GRIS
        pygame.draw.rect(pantalla_juego, color_relleno_vol, (pos_x_barra_vol, pos_y_barra_vol, ancho_relleno_vol, alto_barra_vol), border_radius=3)

        pygame.display.flip()
        reloj_juego.tick(60)

    # --- FINALIZACIÓN ---
    if flujo_entrada_audio and flujo_entrada_audio.active:
        flujo_entrada_audio.stop()
        flujo_entrada_audio.close()

# --- PRUEBAS SIN EL MAIN.PY ---
if __name__ == '__main__':
    # Configuración inicial para ejecutar el juego de forma independiente
    ANCHO_PANTALLA_STANDALONE, ALTO_PANTALLA_STANDALONE = 800, 600
    pantalla_standalone = pygame.display.set_mode((ANCHO_PANTALLA_STANDALONE, ALTO_PANTALLA_STANDALONE))
    
    # Intenta seleccionar el dispositivo de entrada de audio por defecto
    indice_microfono_elegido_standalone = None
    dispositivo_audio_por_defecto = sd.query_devices(kind='input')
    if dispositivo_audio_por_defecto and isinstance(dispositivo_audio_por_defecto, dict) and 'index' in dispositivo_audio_por_defecto:
        indice_microfono_elegido_standalone = dispositivo_audio_por_defecto['index']
    
    # Llama a la función principal del juego
    main(pantalla_standalone, indice_microfono_elegido_standalone)
    pygame.quit()
