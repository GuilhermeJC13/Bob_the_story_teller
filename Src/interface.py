from time import sleep
import pygame
import sys

class Mouse(pygame.sprite.Sprite):
    def __init__(self,path):
        super().__init__()
        self.image = pygame.image.load(path)
        self.rect = self.image.get_rect()

    def update(self):
        self.rect.center = pygame.mouse.get_pos()

class Interface:

    def __init__(self):
        pygame.init()
        self.width = 800
        self.height = 600

        self.clock = pygame.time.Clock()

        self.iniciate_interface()

        self.speaking = False
        self.clicking = False

    def iniciate_interface(self):
        self.screen = pygame.display.set_mode((self.width, self.height))

        self.music_theme = pygame.mixer.Sound("Data/Audio/theme.wav")

        self.background = pygame.image.load("Data\sprites\Background.png")
        self.bob = pygame.image.load("Data\sprites\Bob.png")
        self.bob_talking = pygame.image.load("Data\sprites\Bob_talking.png")

        pygame.mouse.set_visible(False)

        mouse = Mouse("Data\sprites\cursor.png")
        self.mouse_group = pygame.sprite.Group()
        self.mouse_group.add(mouse)

        mouse_click = Mouse("Data\sprites\cursor_clicked.png")
        self.mouse_click_group = pygame.sprite.Group()
        self.mouse_click_group.add(mouse_click)


    def render(self):
        if not pygame.mixer.Channel(0).get_busy():
            self.music_theme.play()

        self.clock.tick(100)

        self.screen.blit(self.background,(0,0))
        self.screen.blit(self.bob, (150,50))

        self.check()

        if(self.clicking):
            self.mouse_click_group.update()
            self.mouse_click_group.draw(self.screen)
        
        else:
            self.mouse_group.update()
            self.mouse_group.draw(self.screen)

        pygame.display.update()

    def check(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.clicking = True
            if event.type == pygame.MOUSEBUTTONUP:
                self.clicking = False

    def speak(self):
        if self.speaking:
            self.screen.blit(self.bob_talking, (150,50))
            self.speaking = False
            sleep(0.5)
        else:
            self.screen.blit(self.bob, (150,50))
            self.speaking = True
            sleep(0.5)