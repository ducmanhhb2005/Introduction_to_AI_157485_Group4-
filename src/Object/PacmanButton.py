import pygame

class PacmanButton:
    def __init__(self, x, y, width, height, screen, text="Button", onClickFunction=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.onClickFunction = onClickFunction
        self.screen = screen
        self.text = text
        self.was_hovering = False  # Track previous hover state

        # UI colors
        self.fillColors = {
            'normal': pygame.Color(255, 204, 0),    # Pac-Man yellow
            'hover': pygame.Color(255, 229, 102),   # lighter yellow
            'pressed': pygame.Color(204, 163, 0),   # darker yellow
        }
        self.borderColor = pygame.Color(0, 0, 0)    # black border

        # Prepare text
        try:
            arcade = pygame.font.match_font('arcadeclassic')
            self.font = pygame.font.Font(arcade, 32)
        except:
            self.font = pygame.font.SysFont('Arial', 32, bold=True)
        self.buttonSurf = self.font.render(self.text, True, pygame.Color(0, 0, 0))

        # Pre-create a surface for the button background (no border)
        self.buttonSurface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.buttonRect = pygame.Rect(self.x, self.y, self.width, self.height)

    def process(self):
        mousePos = pygame.mouse.get_pos()
        pressed = pygame.mouse.get_pressed()[0]
        is_hovering = self.buttonRect.collidepoint(mousePos)

        # Handle cursor change
        if is_hovering and not self.was_hovering:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND)
        elif not is_hovering and self.was_hovering:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
        
        self.was_hovering = is_hovering

        # Choose background color
        if is_hovering and pressed:
            bgColor = self.fillColors['pressed']
        elif is_hovering:
            bgColor = self.fillColors['hover']
        else:
            bgColor = self.fillColors['normal']

        border_width = 3
        radius = 6

        # Draw button background
        self.buttonSurface.fill((0, 0, 0, 0))
        pygame.draw.rect(
            self.buttonSurface,
            bgColor,
            (0, 0, self.width, self.height),
            border_radius=radius
        )

        # Blit button background
        self.screen.blit(self.buttonSurface, (self.x, self.y))

        # Draw border on screen (avoids clipping)
        pygame.draw.rect(
            self.screen,
            self.borderColor,
            self.buttonRect,
            width=border_width,
            border_radius=radius
        )

        # Blit text centered
        text_rect = self.buttonSurf.get_rect(center=self.buttonRect.center)
        self.screen.blit(self.buttonSurf, text_rect)

        # Handle click
        if is_hovering and pressed and self.onClickFunction:
            self.onClickFunction()