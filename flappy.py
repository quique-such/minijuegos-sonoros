# --- IMPORTACIONES ---
# Paquetes
import pygame
import os
import sounddevice as sd
import numpy as np
import random

# --- FUNCIÓN PRINCIPAL ---
def main(pantalla, indice_microfono):
    pygame.display.set_caption("Minijuego - Flappy Voice")
    reloj = pygame.time.Clock()

    # --- CONSTANTES DEL JUEGO ---
    ANCHO_PANTALLA, ALTO_PANTALLA = pantalla.get_width(), pantalla.get_height() # Dimensiones de la pantalla

    # Fuente
    RUTA_FUENTE = os.path.join("fonts", "minecraft.ttf")
    if not os.path.exists(RUTA_FUENTE):
        raise FileNotFoundError(f"No se encontró la fuente: {RUTA_FUENTE}")
    fuente_juego = pygame.font.Font(RUTA_FUENTE, 16)

    # Colores
    ROJO = (255, 0, 0)
    VERDE = (0, 200, 0)
    NEGRO = (0, 0, 0)

    # --- CONFIGURACIÓN DEL PÁJARO ---
    TAMANO_PAJARO = 40
    pajaro_x = ANCHO_PANTALLA // 4
    pajaro_y = ALTO_PANTALLA // 2 
    velocidad_y_pajaro = 0
    gravedad = 0.5
    fuerza_aleteo = -6

    # Skins
    TAMANO_ICONO_SKIN = 30 
    RELLENO_ICONO_SKIN = 10 # Espacio entre iconos de skin

    datos_skins_pajaro = [
        {"nombre": "default", "ruta": os.path.join("flappy", "bird.png")},
        {"nombre": "blue", "ruta": os.path.join("flappy", "bird_blue.png")},
        {"nombre": "green", "ruta": os.path.join("flappy", "bird_green.png")},
        {"nombre": "white", "ruta": os.path.join("flappy", "bird_white.png")},
        {"nombre": "bee", "ruta": os.path.join("flappy", "bee.png")},
    ]

    skins_cargadas = [] # Lista para almacenar las imágenes de las skins cargadas
    for info_skin in datos_skins_pajaro:
        imagen_completa = pygame.image.load(info_skin["ruta"]).convert_alpha()
        imagen_completa = pygame.transform.scale(imagen_completa, (TAMANO_PAJARO, TAMANO_PAJARO))
        
        imagen_icono = pygame.transform.scale(imagen_completa, (TAMANO_ICONO_SKIN, TAMANO_ICONO_SKIN))
        
        skins_cargadas.append({
            "nombre": info_skin["nombre"],
            "completa": imagen_completa, # Imagen para el juego
            "icono": imagen_icono,       # Imagen para el menú de selección
            "rect_icono": None           # Rectángulo para la detección de clics
        })

    indice_skin_actual = 0 # Skin por defector
    imagen_pajaro_actual = skins_cargadas[indice_skin_actual]["completa"]

    # --- CONFIGURACIÓN DE LAS PAREDES ---
    TAMANO_HUECO_PARED= 200 # Tamaño del espacio vertical entre paredes
    ANCHO_PAREDES = 80         # Ancho de las paredes
    lista_paredes = []        # Lista para almacenar las paredes activas
    
    # Pared Superior
    imagen_muro_arriba = pygame.image.load(os.path.join("flappy", "wall_up.png")).convert_alpha()
    imagen_muro_arriba = pygame.transform.scale(imagen_muro_arriba, (ANCHO_PAREDES, ALTO_PANTALLA))

    # Pared Inferior
    imagen_muro_abajo = pygame.image.load(os.path.join("flappy", "wall.png")).convert_alpha()
    imagen_muro_abajo = pygame.transform.scale(imagen_muro_abajo, (ANCHO_PAREDES, ALTO_PANTALLA))

    # Fondo
    imagen_fondo = pygame.image.load(os.path.join("fondos", "flappy.png")).convert()
    imagen_fondo = pygame.transform.scale(imagen_fondo, (ANCHO_PANTALLA, ALTO_PANTALLA))
    posicion_x_fondo = 0 # Posición X inicial del fondo para el efecto de scroll

    # --- PUNTUACIÓN Y VELOCIDAD DEL JUEGO ---
    puntuacion = 0
    record_puntuacion = 0
    VELOCIDAD_BASE_JUEGO = 2     # Velocidad inicial del juego
    VELOCIDAD_MAXIMA_JUEGO = 6   # Velocidad máxima alcanzable
    TASA_AUMENTO_VELOCIDAD = 0.1 # Cuánto aumenta la velocidad por cada punto

    # --- FUNCIONES ---
    def obtener_velocidad_juego_actual(puntuacion_actual):
        """Calcula la velocidad del juego basada en la puntuación."""
        velocidad = VELOCIDAD_BASE_JUEGO + puntuacion_actual * TASA_AUMENTO_VELOCIDAD
        return min(velocidad, VELOCIDAD_MAXIMA_JUEGO) # Limita la velocidad

    def crear_nueva_pared():
        """Crea una nueva pared con una altura de hueco aleatoria."""
        posicion_y_hueco = random.randint(TAMANO_HUECO_PARED, ALTO_PANTALLA - TAMANO_HUECO_PARED)
        return {
            'x': ANCHO_PANTALLA,                             # Posición X inicial (fuera de la pantalla, a la derecha)
            'altura_superior': posicion_y_hueco - TAMANO_HUECO_PARED, # Altura de la parte superior de la pared
            'y_inferior': posicion_y_hueco,                  # Coordenada Y donde empieza la pared inferior
            'pasada': False                                  # Indica si el pájaro ya ha pasado esta pared
        }

    # Creación inicial de paredes
    for i in range(3):
        pared = crear_nueva_pared()
        pared['x'] += i * 300 # Espaciado inicial entre paredes
        lista_paredes.append(pared)

    # --- CONFIGURACIÓN DE ENTRADA DE SONIDO ---
    nivel_sonido = 0         # Nivel de sonido actual detectado
    UMBRAL_SONIDO_ALETEO = 15 # Nivel de sonido necesario para que el pájaro aletee
    historial_sonido = []    # Almacena los últimos niveles de sonido para suavizar la detección

    def retrollamada_audio(datos_entrada_audio, *_):
        """Función que se ejecuta cada vez que se reciben datos del micrófono."""
        nonlocal nivel_sonido # Permite modificar la variable nivel_sonido fuera del ámbito local
        
        # Calcula la norma del vector de datos de audio (amplitud/volumen)
        volumen_normalizado = np.linalg.norm(datos_entrada_audio) * 100 
        if volumen_normalizado < 10: # Ignora niveles de sonido muy bajos (ruido de fondo)
            volumen_normalizado = 0
        
        historial_sonido.append(volumen_normalizado)
        if len(historial_sonido) > 10: # Mantiene solo los últimos 10 valores
            historial_sonido.pop(0)
        
        # Calcula una media ponderada para suavizar el nivel de sonido
        pesos_sonido = np.linspace(0.5, 1, len(historial_sonido))
        nivel_sonido = np.average(historial_sonido, weights=pesos_sonido)

    # Inicia la captura de audio desde el micrófono especificado
    flujo_audio = sd.InputStream(callback=retrollamada_audio, channels=1, samplerate=44100, device=indice_microfono)
    flujo_audio.start()

    def reiniciar_partida():
        """Reinicia el estado del juego a sus valores iniciales."""
        nonlocal pajaro_y, velocidad_y_pajaro, lista_paredes, puntuacion, record_puntuacion
        
        if puntuacion > record_puntuacion: # Actualiza el récord si es necesario
            record_puntuacion = puntuacion
        
        puntuacion = 0
        pajaro_y = ALTO_PANTALLA // 2
        velocidad_y_pajaro = 0
        lista_paredes.clear()
        for i in range(3): # Vuelve a crear las paredes iniciales
            pared = crear_nueva_pared()
            pared['x'] += i * 300
            lista_paredes.append(pared)

    # --- BUCLE PRINCIPAL ---
    running = True
    while running:
        velocidad_juego_actual = obtener_velocidad_juego_actual(puntuacion)

        # --- MOVIMIENTO DEL FONDO ---
        posicion_x_fondo -= velocidad_juego_actual # Mueve el fondo hacia la izquierda
        if posicion_x_fondo <= -ANCHO_PANTALLA:    # Si el fondo se ha desplazado completamente
            posicion_x_fondo = 0                   # Lo resetea para crear un bucle infinito

        # Dibuja el fondo otra vez
        pantalla.blit(imagen_fondo, (posicion_x_fondo, 0))
        pantalla.blit(imagen_fondo, (posicion_x_fondo + ANCHO_PANTALLA, 0))

        # --- EVENTOS (TECLADO, RATÓN, CERRAR VENTANA) ---
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT: # Si se ierra la ventana
                running = False
            
            if evento.type == pygame.MOUSEBUTTONDOWN: # Si se hace clic con el ratón
                if evento.button == 1: # Botón izquierdo del ratón
                    posicion_raton = pygame.mouse.get_pos()
                    for i, datos_skin in enumerate(skins_cargadas):
                        # Comprueba si el clic fue sobre los iconos de skin y la cambia
                        if datos_skin["rect_icono"] and datos_skin["rect_icono"].collidepoint(posicion_raton):
                            indice_skin_actual = i
                            imagen_pajaro_actual = skins_cargadas[indice_skin_actual]["completa"]
                            break

        # --- MOVIMIENTO DEL PÁJARO (SALTO POR VOZ) ---
        # Salta si hay sonido y no está ya subiendo rápido
        if nivel_sonido > UMBRAL_SONIDO_ALETEO and velocidad_y_pajaro >= 0: 
            velocidad_y_pajaro = fuerza_aleteo

        # Aplica gravedad y actualiza la posición vertical del pájaro
        velocidad_y_pajaro += gravedad
        pajaro_y += velocidad_y_pajaro

        # --- PAREDES (MOVIMIENTO, COLISIONES Y PUNTUACIÓN) ---
        for pared in lista_paredes:
            pared['x'] -= velocidad_juego_actual * 2 # Las paredes se mueven más rápido que el fondo

            # Dibuja las paredes
            pantalla.blit(imagen_muro_arriba, (pared['x'], 0), area=pygame.Rect(0, 0, ANCHO_PAREDES, pared['altura_superior']))
            pantalla.blit(imagen_muro_abajo, (pared['x'], pared['y_inferior']), area=pygame.Rect(0, 0, ANCHO_PAREDES, ALTO_PANTALLA - pared['y_inferior']))

            # Rectángulos para detección de colisiones
            rect_pajaro = pygame.Rect(pajaro_x, pajaro_y, TAMANO_PAJARO, TAMANO_PAJARO)
            rect_pared_superior = pygame.Rect(pared['x'], 0, ANCHO_PAREDES, pared['altura_superior'])
            rect_pared_inferior = pygame.Rect(pared['x'], pared['y_inferior'], ANCHO_PAREDES, ALTO_PANTALLA - pared['y_inferior'])

            # Comprueba colisión con las paredes
            if rect_pajaro.colliderect(rect_pared_superior) or rect_pajaro.colliderect(rect_pared_inferior):
                reiniciar_partida()

            # Comprueba si el pájaro ha pasado la pared para sumar puntos
            if not pared['pasada'] and pared['x'] + ANCHO_PAREDES < pajaro_x:
                pared['pasada'] = True
                puntuacion += 1

        # Elimina las paredes que ya han salido de la pantalla por la izquierda
        lista_paredes = [pared for pared in lista_paredes if pared['x'] > -ANCHO_PAREDES]
        if len(lista_paredes) < 3: # Añade nuevas paredes si es necesario para mantener siempre 3 en pantalla
            lista_paredes.append(crear_nueva_pared())

        # Comprueba colisión con los límites superior e inferior de la pantalla
        if pajaro_y < 0 or pajaro_y + TAMANO_PAJARO > ALTO_PANTALLA:
            reiniciar_partida()

        # --- DIBUJADO EN PANTALLA ---
        # Pájaro
        pantalla.blit(imagen_pajaro_actual, (pajaro_x, pajaro_y))

        # Barra de volumen del micrófono
        ancho_barra_volumen = 200
        alto_barra_volumen = 20
        x_barra_volumen = 10
        y_barra_volumen = ALTO_PANTALLA - 30
        # Calcula el ancho de la parte rellena de la barra según el nivel de sonido
        ancho_relleno_barra = min(int((nivel_sonido / 50) * ancho_barra_volumen), ancho_barra_volumen)
        pygame.draw.rect(pantalla, NEGRO, (x_barra_volumen, y_barra_volumen, ancho_barra_volumen, alto_barra_volumen), 2) # Borde
        pygame.draw.rect(pantalla, VERDE, (x_barra_volumen, y_barra_volumen, ancho_relleno_barra, alto_barra_volumen))    # Relleno
        # Dibuja la línea del umbral de sonido en la barra
        posicion_umbral_barra = int((UMBRAL_SONIDO_ALETEO / 50) * ancho_barra_volumen)
        pygame.draw.line(pantalla, ROJO, (x_barra_volumen + posicion_umbral_barra, y_barra_volumen), \
                                        (x_barra_volumen + posicion_umbral_barra, y_barra_volumen + alto_barra_volumen), 2)

        # Iconos de selección de skin
        x_inicio_iconos = x_barra_volumen 
        y_iconos = y_barra_volumen - TAMANO_ICONO_SKIN - RELLENO_ICONO_SKIN

        for i, datos_skin in enumerate(skins_cargadas):
            x_icono = x_inicio_iconos + i * (TAMANO_ICONO_SKIN + RELLENO_ICONO_SKIN)
            datos_skin["rect_icono"] = pygame.Rect(x_icono, y_iconos, TAMANO_ICONO_SKIN, TAMANO_ICONO_SKIN) # Guarda el rectángulo del icono para la detección de clics
            pantalla.blit(datos_skin["icono"], datos_skin["rect_icono"].topleft)
            if i == indice_skin_actual: # Resalta la skin seleccionada
                pygame.draw.rect(pantalla, ROJO, datos_skin["rect_icono"], 2)

        # Textos
        texto_puntuacion = fuente_juego.render(f"Puntuacion : {puntuacion}", True, NEGRO)
        texto_record = fuente_juego.render(f"Record : {record_puntuacion}", True, NEGRO)
        pantalla.blit(texto_puntuacion, (10, 10))
        pantalla.blit(texto_record, (10, 40))

        pygame.display.flip() # Actualiza la pantalla
        reloj.tick(50)

    # --- FINALIZACIÓN ---
    flujo_audio.stop()
