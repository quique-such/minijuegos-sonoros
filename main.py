# --- IMPORTACIONES ---
# Paquetes
import pygame
import sys
import os
import sounddevice as sd

# Juegos
import flappy
import pesca
import invasores
import simon

# --- INICIALIZACIÓN ---
# Ventana
pygame.init()
WIDTH, HEIGHT = 1080, 600 # Dimensiones de la pantalla
screen = pygame.display.set_mode((WIDTH, HEIGHT)) # Creación de la superficie de la pantalla
pygame.display.set_caption("Inicio") # Título de la ventana

# Fuentes
FONT_PATH = os.path.join("fonts", "minecraft.ttf")
if not os.path.exists(FONT_PATH):
    raise FileNotFoundError(f"No se encontró la fuente: {FONT_PATH}")
font_title = pygame.font.Font(FONT_PATH, 72)
font_button = pygame.font.Font(FONT_PATH, 28)
font_mic = pygame.font.Font(FONT_PATH, 13)

# Variables globales micrófono
selected_mic_index = None
mic_buttons = []

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Estados del juego
STATE_MENU = "menu"
STATE_SELECT_MIC = "select_mic"
STATE_GAMES = "games"
state = STATE_MENU # Estado inicial

# Fondos e imágenes
background_portada = pygame.image.load(os.path.join("fondos", "portada.png")).convert()
background_portada = pygame.transform.scale(background_portada, (WIDTH, HEIGHT))
background_micros = pygame.image.load(os.path.join("fondos", "micros.png")).convert()
background_micros = pygame.transform.scale(background_micros, (WIDTH, HEIGHT))
img_titulo = pygame.image.load(os.path.join("fondos", "text_minijuegos.png")).convert_alpha()
texture_button = pygame.image.load(os.path.join("fondos", "textura_boton.png")).convert()

clock = pygame.time.Clock()

# --- CLASES DE BOTONES ---
# Botones Micrófono
class MicButton:
    # Método para crear un botón de micrófono
    def create(self, text, x, y, w, h, callback):
        self.rect = pygame.Rect(x, y, w, h) # Rectángulo del botón
        self.text = text # Texto del botón
        self.callback = callback # Función a llamar al hacer clic
        self.text_color = BLACK # Color del texto
        self.bg_color = (253, 248, 236) # Color de fondo

    # Método para dibujar el botón en la pantalla
    def draw(self, surface):
        pygame.draw.rect(surface, self.bg_color, self.rect)
        text_surf = font_mic.render(self.text, True, self.text_color)
        surface.blit(text_surf, text_surf.get_rect(center=self.rect.center))

    # Método para manejar eventos del botón
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.callback()

# Botones Juegos
class Button:
    # Método para crear un botón
    def create(self, text, x, y, w, h, callback, texture_button):
        self.rect = pygame.Rect(x, y, w, h) # Rectángulo del botón
        self.text = text # Texto del botón
        self.callback = callback # Función a llamar al hacer clic
        self.texture = pygame.transform.scale(texture_button, (w, h)) # Textura del botón

    # Método para dibujar el botón en la pantalla
    def draw(self, surface):
        surface.blit(self.texture, self.rect.topleft) # Dibuja la textura
        text_surf = font_button.render(self.text, True, WHITE) # Renderiza el texto
        surface.blit(text_surf, text_surf.get_rect(center=self.rect.center)) # Centra el texto
        pygame.draw.rect(surface, BLACK, self.rect, 2) # Dibuja un borde

    # Método para manejar eventos del botón
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.callback()

# --- FUNCIONES ---
# Función para quitar acentos (nombres de micrófonos fuente)
def quitar_acentos(texto):
    reemplazos = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
        'ü': 'u', 'Ü': 'U', 'ñ': 'n', 'Ñ': 'N'
    }
    for original, reemplazo in reemplazos.items():
        texto = texto.replace(original, reemplazo)
    return texto

# Función para listar los micrófonos disponibles
def list_microphones():
    global mic_buttons
    mic_buttons = [] # Limpia la lista de botones de micrófono existentes
    devices = sd.query_devices() # Obtiene la lista de dispositivos de audio

    seen_clean_names = set() # Conjunto para evitar nombres de micrófono duplicados
    input_devices = [] # Lista para almacenar los dispositivos de entrada válidos

    # Obtener índice del micrófono predeterminado
    default_input_index = sd.default.device[0]

    # Verificar micrófonos válidos
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            try:
                # Probar si se puede abrir el micrófono
                with sd.InputStream(device=i, channels=1, samplerate=int(d['default_samplerate'])):
                    name = d['name']
                    # Filtra nombres de dispositivos del sistema no deseados
                    if "@System32\\drivers" not in name:
                        # Limpiar nombre del micrófono
                        clean_name = ""
                        inside_parentheses = False
                        for char in name:
                            if char == '(':
                                inside_parentheses = True
                            elif char == ')':
                                inside_parentheses = False
                            elif not inside_parentheses:
                                clean_name += char
                        clean_name = clean_name.strip()

                        if clean_name not in seen_clean_names:
                            seen_clean_names.add(clean_name)
                            d['index'] = i  # Asegura que el índice sea el correcto
                            input_devices.append(d)
            except Exception:
                pass # Ignora los dispositivos que no se pueden abrir

    # Añadir el micrófono predeterminado a la lista si es válido
    try:
        sd.check_input_settings(device=default_input_index) # Comprueba la configuración del mic predeterminado
        mic_pred_name = "Micrófono Predeterminado"
        mic_pred = {
            'index': default_input_index,
            'name': mic_pred_name,
            'is_default': True # Marca como predeterminado
        }
        input_devices.insert(0, mic_pred) # Inserta al principio de la lista
    except Exception:
        pass

    # Crear botones en formato de cuadrícula
    cols = 2
    button_width = 375
    button_height = 35
    x_margin = 120
    y_start = 200
    y_spacing = 40

    for i, device in enumerate(input_devices):
        col = i % cols
        row = i // cols
        x = x_margin + col * (button_width + 80)
        y = y_start + row * y_spacing

        nombre = quitar_acentos(device['name'])
        btn = MicButton()
        btn.create(nombre, x, y, button_width, button_height, lambda i=device['index']: select_microphone(i))
        if device.get('is_default'):
            btn.text_color = (225, 69, 165)
        mic_buttons.append(btn)

# Función para seleccionar un micrófono
def select_microphone(index):
    global selected_mic_index, state
    selected_mic_index = index # Guarda el índice del micrófono seleccionado
    state = STATE_GAMES # Cambia el estado del juego a la selección de juegos

# Funciones para iniciar cada juego
def start_flappy():
    flappy.main(screen, selected_mic_index)

def start_pesca():
    pesca.main(screen, selected_mic_index)

def start_invasores():
    invasores.main(screen, selected_mic_index)

def start_simon():
    simon.main(screen, selected_mic_index)

# Botones de juegos
buttons = []
buttons.append(Button())
buttons[-1].create("Flappy Voice", 100, 170, 400, 75, start_flappy, texture_button)
buttons.append(Button())
buttons[-1].create("Pesca", 560, 170, 400, 75, start_pesca, texture_button)
buttons.append(Button())
buttons[-1].create("Invasores", 100, 300, 400, 75, start_invasores, texture_button)
buttons.append(Button())
buttons[-1].create("Simon Dice", 560, 300, 400, 75, start_simon, texture_button)

# --- BUCLE PRINCIPAL ---
running = True
while running:
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            running = False

        if state == STATE_MENU and event.type == pygame.MOUSEBUTTONDOWN:
            state = STATE_SELECT_MIC
            list_microphones()
        elif state == STATE_SELECT_MIC:
            for button in mic_buttons:
                button.handle_event(event)
        elif state == STATE_GAMES:
            for button in buttons:
                button.handle_event(event)

    # Dibujar según el estado
    if state == STATE_MENU:
        screen.blit(background_portada, (0, 0))
        screen.blit(img_titulo, img_titulo.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 10)))
        if (pygame.time.get_ticks() % 1000) < 800:
            text_pulsa = font_button.render("Pulsa la pantalla para empezar", True, BLACK)
            screen.blit(text_pulsa, text_pulsa.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 140)))

    elif state == STATE_SELECT_MIC:
        screen.blit(background_micros, (0, 0))
        pygame.display.set_caption("Seleccionar Micrófono")
        title = font_button.render("Selecciona tu microfono", True, WHITE)
        screen.blit(title, title.get_rect(center=(WIDTH // 2, 60)))
        subtitle = font_mic.render("(Si no sabes cual es tu microfono, elige el predeterminado)", True, WHITE)
        screen.blit(subtitle, subtitle.get_rect(center=(WIDTH // 2, 90)))
        note = font_mic.render("Te recomendamos usar auriculares para una mejor experiencia", True, WHITE)
        screen.blit(note, note.get_rect(center=(WIDTH // 2, 555)))
        for button in mic_buttons:
            button.draw(screen)

    elif state == STATE_GAMES:
        screen.blit(background_portada, (0, 0))
        pygame.display.set_caption("Selecciona un juego")
        for button in buttons:
            button.draw(screen)

    pygame.display.flip() # Actualiza toda la pantalla para mostrar los cambios
    clock.tick(60) # Limita el juego a 60 fotogramas por segundo

# --- FINALIZACIÓN ---
pygame.quit()
sys.exit()
