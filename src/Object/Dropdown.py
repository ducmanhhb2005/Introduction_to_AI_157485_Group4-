import pygame

class Dropdown:
    def __init__(self, x, y, w, h, font, options, default_index=0):
        self.rect = pygame.Rect(x, y, w, h)
        self.font = font
        self.options = options
        self.selected = default_index
        self.expanded = False

        # Pre-render texts
        self.text_surfs = [self.font.render(opt, True, pygame.Color('black')) 
                           for opt in options]
        self.bg_color      = pygame.Color(255,204,0)
        self.hover_color   = pygame.Color(255,229,102)
        self.border_color  = pygame.Color(255,255,255)
        self.text_color    = pygame.Color(255,255,255)
        self.arrow_color   = pygame.Color('black')

    def process(self, events):
        mouse_pos = pygame.mouse.get_pos()
        mouse_click = pygame.mouse.get_pressed()[0]

        # 1) Click header → toggle expanded
        if self.rect.collidepoint(mouse_pos) and mouse_click:
            self.expanded = not self.expanded

        # 2) If expanded, check clicks on each option
        if self.expanded:
            for i, text_surf in enumerate(self.text_surfs):
                opt_rect = pygame.Rect(
                    self.rect.x,
                    self.rect.y + (i+1)*self.rect.h,
                    self.rect.w,
                    self.rect.h
                )
                if opt_rect.collidepoint(mouse_pos) and mouse_click:
                    self.selected = i
                    self.expanded = False
                    break

    def draw(self, screen):
        # Draw header box
        pygame.draw.rect(screen, self.bg_color, self.rect, border_radius=4)
        pygame.draw.rect(screen, self.border_color, self.rect, width=2, border_radius=4)

        # Draw selected text
        sel_surf = self.text_surfs[self.selected]
        sel_rect = sel_surf.get_rect(midleft=(self.rect.x+10, self.rect.centery))
        screen.blit(sel_surf, sel_rect)

        # Draw arrow
        ax = self.rect.right - 15
        ay = self.rect.centery
        pts = [(ax-5, ay-3), (ax+5, ay-3), (ax, ay+4)]
        pygame.draw.polygon(screen, self.arrow_color, pts)

        # Draw options if expanded
        if self.expanded:
            for i, text_surf in enumerate(self.text_surfs):
                opt_rect = pygame.Rect(
                    self.rect.x,
                    self.rect.y + (i+1)*self.rect.h,
                    self.rect.w,
                    self.rect.h
                )
                # bg & border
                pygame.draw.rect(screen, self.bg_color, opt_rect, border_radius=4)
                pygame.draw.rect(screen, self.border_color, opt_rect, width=2, border_radius=4)
                # text
                txt_rect = text_surf.get_rect(midleft=(opt_rect.x+10, opt_rect.centery))
                screen.blit(text_surf, txt_rect)
                # tick mark for selected
                if i == self.selected:
                    pygame.draw.circle(screen, self.arrow_color,
                                       (opt_rect.right-15, opt_rect.centery), 5)