import pygame
import random
import time

from bird import Bird
from pipes_manager import PipeManager

pygame.init()

class GameManager:
    def __init__(self):
        self.WIDTH = 500
        self.HEIGHT = 700

        self.SCREEN = pygame.display.set_mode((self.WIDTH, self.HEIGHT))

        self.BACKGROUND_IMAGE = pygame.transform.scale(pygame.image.load('data/bird_background.png'), (self.WIDTH, self.HEIGHT))
        self.CLOUD_IMAGES = [
            pygame.transform.scale(pygame.image.load('data/cloud1.png'), (32*2, 16*2)),
            pygame.transform.scale(pygame.image.load('data/cloud2.png'), (32*2, 16*2)),
        ]
        self.clouds = []
        self.NO_OF_CLOUDS = 5
        self.CLOUD_MOVEMENT_SPEED = 10

        self.bird = Bird()

        self.pipe_manager = PipeManager(self.WIDTH, self.HEIGHT, self.bird)

        self.font = pygame.font.SysFont('Bauhaus 93', 50, bold=False)

        # Generating Clouds Randomly for the first frame only.
        for i in range(self.NO_OF_CLOUDS):
            cloud_image = random.choice(self.CLOUD_IMAGES)
            x = random.randint(0, self.WIDTH)
            y = random.randint(0, int(self.HEIGHT / 2))
            self.clouds.append((cloud_image, [x, y]))

    def generate_clouds(self, dt):
        if len(self.clouds) <= self.NO_OF_CLOUDS:
            cloud_image = random.choice(self.CLOUD_IMAGES)
            x = random.randint(self.WIDTH, self.WIDTH+150)
            y = random.randint(0, int(self.HEIGHT/2))
            self.clouds.append((cloud_image, [x, y]))

        for i in range(len(self.clouds)):
            self.clouds[i][1][0] -= self.CLOUD_MOVEMENT_SPEED*dt

        for i in range(len(self.clouds)):
            cloud_image, coords = self.clouds[i]
            self.SCREEN.blit(cloud_image, coords)

        for cloud in self.clouds:
            if cloud[1][0] <= -32*2: # Hardcoded.
                self.clouds.remove(cloud)

    def load_background(self, dt):
        self.SCREEN.blit(self.BACKGROUND_IMAGE, (0, 0))
        self.generate_clouds(dt)

    def game_over(self):
        if self.bird.is_dead:
            text = self.font.render(f'Game Over!', True, (0,0,0))
            self.SCREEN.blit(text, ((self.WIDTH/2)-120, (self.HEIGHT/2)-75))

    def score(self):
        text = self.font.render(str(self.pipe_manager.pipe_passed_score), True, (0, 0, 0))
        self.SCREEN.blit(text, ((self.WIDTH / 2) - 10, (self.HEIGHT / 2)-200))

    def restart(self):
        if self.bird.is_dead:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_r]:
                self.pipe_manager.pipes = []
                self.pipe_manager.generate_pipe()
                self.pipe_manager.scored_pipes = []
                self.pipe_manager.pipe_passed_score = 0
                self.bird.rect.x, self.bird.rect.y = 250 - (self.bird.rect.width/2), 250 - (self.bird.rect.height/2)
                self.bird.gravity_accelerator = 0
                self.bird.jump = False
                self.bird.is_dead = False

    def update(self, dt):
        self.load_background(dt)
        if self.bird.is_dead != True:
            self.pipe_manager.update_pipe(self.SCREEN, dt)
            self.bird.update(self.SCREEN, dt)
            self.score()
        self.game_over()
        self.restart()

    def run(self):
        running = True
        previous_time = time.time()
        while running:
            dt = time.time() - previous_time
            previous_time = time.time()
            self.SCREEN.fill((0, 0, 0))
            pygame.display.set_caption(f'Game with FPS: {round(1/dt) if dt > 0 else 0}')
            self.update(dt)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            if pygame.key.get_pressed()[pygame.K_ESCAPE]:
                running = False

            pygame.display.update()