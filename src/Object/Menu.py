import os

import pygame
import sys
from Object.PacmanButton import PacmanButton

from Overall.const import *

clock = pygame.time.Clock()
bg = pygame.image.load("Images/home_bg.png")
bg = pygame.transform.scale(bg, (WIDTH, HEIGHT))
pygame.init()
font = pygame.font.SysFont('Arial', 40)
my_font = pygame.font.SysFont('Comic Sans MS', 70)

_N = _M = 0
__map = 0
SIZE_WALL = 20


class Button:
    def __init__(self, x, y, width, height, screen, buttonText="Button", onClickFunction=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.onClickFunction = onClickFunction
        self.screen = screen

        self.fillColors = {
            'normal': '#FF4500',
            'hover': '#FF6347',
            'pressed': '#FF7F50',
        }

        self.buttonSurface = pygame.Surface((self.width, self.height))
        self.buttonRect = pygame.Rect(self.x, self.y, self.width, self.height)

        self.buttonSurf = font.render(buttonText, True, (255, 255, 255))

    def process(self):
        mousePos = pygame.mouse.get_pos()
        self.buttonSurface.fill(self.fillColors['normal'])
        if self.buttonRect.collidepoint(mousePos):
            self.buttonSurface.fill(self.fillColors['hover'])
            if pygame.mouse.get_pressed()[0]:
                self.buttonSurface.fill(self.fillColors['pressed'])
                self.onClickFunction()

        self.buttonSurface.blit(self.buttonSurf, [
            self.buttonRect.width / 2 - self.buttonSurf.get_rect().width / 2,
            self.buttonRect.height / 2 - self.buttonSurf.get_rect().height / 2
        ])
        pygame.draw.rect(self.buttonSurface, BLUE, (0, 0, self.width, self.height), 5)
        self.screen.blit(self.buttonSurface, self.buttonRect)


class Menu:
    def __init__(self, screen):
        self.current_level = 0
        self.clicked = False
        self.map_name = []
        self.current_map = 0
        self.done = False
        self.current_screen = 1
        self.screen = screen
        self.btnStart = PacmanButton(WIDTH // 2 - 80, HEIGHT // 2, 150, 100, screen, "START", self.startGame)

        self.btnLevel1 = PacmanButton(WIDTH // 2 - 150, HEIGHT // 4 * 0 + 20, 300, 100, screen, "Level 1",
                                self._load_map_level_1)
        self.btnLevel2 = PacmanButton(WIDTH // 2 - 150, HEIGHT // 4 * 1 + 20, 300, 100, screen, "Level 2",
                                self._load_map_level_2)
        self.btnLevel3 = PacmanButton(WIDTH // 2 - 150, HEIGHT // 4 * 2 + 20, 300, 100, screen, "Level 3",
                                self._load_map_level_3)
        self.btnLevel4 = PacmanButton(WIDTH // 2 - 150, HEIGHT // 4 * 3 + 20, 300, 100, screen, "Level 4",
                                self._load_map_level_4)

        self.btnPrev = PacmanButton(WIDTH // 2 - 250, HEIGHT // 4 * 3 + 35, 100, 100, screen, "<", self.prevMap)
        self.btnNext = PacmanButton(WIDTH // 2 + 150, HEIGHT // 4 * 3 + 35, 100, 100, screen, ">", self.nextMap)
        self.btnConfirmMap = PacmanButton(WIDTH // 2 - 75, HEIGHT // 4 * 3 + 35, 150, 100, screen, "CONFIRM", self.selectMap)

        # self.btnBack = PacmanButton(40, HEIGHT // 4 * 3 + 35, 150, 100, screen, "BACK", self.myFunction)

        # thuật toán support
        self.algorithms      = ALGORITHMS
        self.sel_algo_idx    = 0
        self.algorithm_name  = None

        # nút sang trái/phải và xác nhận cho màn chọn thuật toán
        # bạn đặt sao cho hợp layout của bạn
        self.btnPrevAlgo     = PacmanButton( WIDTH//2 - 160, HEIGHT * 3//5,  100, 80, screen, "<", self.prevAlgo )
        self.btnNextAlgo     = PacmanButton( WIDTH//2 +  80, HEIGHT * 3//5,  100, 80, screen, ">", self.nextAlgo )
        self.btnConfirmAlgo  = PacmanButton( WIDTH//2 - 50,  HEIGHT * 3//5, 120, 80, screen, "Play", self.confirmAlgo )

    def prevAlgo(self):
        if self.clicked:
            self.sel_algo_idx = (self.sel_algo_idx - 1) % len(self.algorithms)
        self.clicked = False

    def nextAlgo(self):
        if self.clicked:
            self.sel_algo_idx = (self.sel_algo_idx + 1) % len(self.algorithms)
        self.clicked = False

    def confirmAlgo(self):
        if self.clicked:
            self.algorithm_name = self.algorithms[self.sel_algo_idx]
            self.done = True
        self.clicked = False

    def nextMap(self):
        if self.clicked:
            self.current_screen = 3
            self.current_map = (self.current_map + 1) % len(self.map_name)
        self.clicked = False

    def prevMap(self):
        if self.clicked:
            self.current_screen = 3
            self.current_map -= 1
            if self.current_map < 0:
                self.current_map += len(self.map_name)
        self.clicked = False

    def startGame(self):
        self.current_screen = 2
        self.map_name = []
        self.current_map = 0
        self.current_level = 0
        self.sel_algo_idx = 0
        self.algorithm_name = None

    def _load_map_level_1(self):
        if self.clicked:
            self.current_level = 1
            for file in os.listdir('../InputMap/Level1'):
                self.map_name.append('../InputMap/Level1/' + file)
            self.current_screen = 3
        self.clicked = False

    def _load_map_level_2(self):
        if self.clicked:
            self.current_level = 2
            for file in os.listdir('../InputMap/Level2'):
                self.map_name.append('../InputMap/Level2/' + file)
            self.current_screen = 3
        self.clicked = False

    def _load_map_level_3(self):
        if self.clicked:
            self.current_level = 3
            for file in os.listdir('../InputMap/Level3'):
                self.map_name.append('../InputMap/Level3/' + file)
            self.current_screen = 3
        self.clicked = False

    def _load_map_level_4(self):
        if self.clicked:
            self.current_level = 4
            for file in os.listdir('../InputMap/Level4'):
                self.map_name.append('../InputMap/Level4/' + file)
            self.current_screen = 3
        self.clicked = False

    def draw_map(self, fileName):
        text_surface = my_font.render(
            'LEVEL {level} - MAP {map}'.format(level=self.current_level, map=self.current_map + 1), False, WHITE)
        self.screen.blit(text_surface, (WIDTH // 2 - 270, 0))

        f = open(fileName, "r")
        x = f.readline().split()
        count_ghost = 0
        N, M = int(x[0]), int(x[1])

        MARGIN_TOP = 100
        MARGIN_LEFT = (WIDTH - M * SIZE_WALL) // 2

        for i in range(N):
            line = f.readline().split()
            for j in range(M):
                cell = int(line[j])
                if cell == WALL:
                    image = pygame.Surface([SIZE_WALL, SIZE_WALL])
                    # image.fill(color)
                    pygame.draw.rect(image, BLUE, (0, 0, SIZE_WALL, SIZE_WALL), 1)
                    image.fill(BLUE)
                    top = i * SIZE_WALL + MARGIN_TOP
                    left = j * SIZE_WALL + MARGIN_LEFT
                    self.screen.blit(image, (left, top))
                elif cell == FOOD:
                    image = pygame.Surface([SIZE_WALL // 2, SIZE_WALL // 2])
                    image.fill(WHITE)
                    image.set_colorkey(WHITE)
                    pygame.draw.ellipse(image, YELLOW, [0, 0, SIZE_WALL // 2, SIZE_WALL // 2])

                    top = i * SIZE_WALL + MARGIN_TOP + SIZE_WALL // 2 - SIZE_WALL // 4
                    left = j * SIZE_WALL + MARGIN_LEFT + SIZE_WALL // 2 - SIZE_WALL // 4
                    self.screen.blit(image, (left, top))
                elif cell == MONSTER:
                    image = pygame.image.load(IMAGE_GHOST[count_ghost]).convert_alpha()
                    image = pygame.transform.scale(image, (SIZE_WALL, SIZE_WALL))
                    top = i * SIZE_WALL + MARGIN_TOP
                    left = j * SIZE_WALL + MARGIN_LEFT
                    count_ghost = (count_ghost + 1) % len(IMAGE_GHOST)
                    self.screen.blit(image, (left, top))

        x = f.readline().split()
        image = pygame.image.load(IMAGE_PACMAN[0]).convert_alpha()
        image = pygame.transform.scale(image, (SIZE_WALL, SIZE_WALL))
        top = int(x[0]) * SIZE_WALL + MARGIN_TOP
        left = int(x[1]) * SIZE_WALL + MARGIN_LEFT
        self.screen.blit(image, (left, top))

        f.close()

    def selectMap(self):
        if self.clicked:
            # thay vì self.done = True, ta chuyển qua màn 5
            self.current_screen = 5
        self.clicked = False

    def run(self):

        while not self.done:
            self.clicked = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.clicked = True

            if self.current_screen == 1:
                self.screen.blit(bg, (0, 0))
                self.btnStart.process()

            elif self.current_screen == 2:
                self.screen.fill(BLACK)
                self.btnLevel1.process()
                self.btnLevel2.process()
                self.btnLevel3.process()
                self.btnLevel4.process()

            elif self.current_screen == 3:
                self.screen.fill(BLACK)
                self.current_screen = 4
                self.draw_map(self.map_name[self.current_map])

            elif self.current_screen == 4:
                self.btnNext.process()
                self.btnPrev.process()
                self.btnConfirmMap.process()
                # self.btnBack.process()
            elif self.current_screen == 5:
                self.screen.fill(BLACK)
                # tiêu đề
                lbl = my_font.render(f"SELECT ALGORITHM: {self.algorithms[self.sel_algo_idx]}", True, WHITE)
                self.screen.blit(lbl, (WIDTH//2 - lbl.get_width()//2, HEIGHT//2 - 100))

                self.btnPrevAlgo.process()
                self.btnNextAlgo.process()
                self.btnConfirmAlgo.process()
                # cho phép quay lại nếu cần
                # self.btnBack.process()

            pygame.display.flip()
            clock.tick(FPS)
        
        print(self.current_level, self.algorithm_name, self.map_name[self.current_map]);

        return [self.current_level, self.algorithm_name, self.map_name[self.current_map]]