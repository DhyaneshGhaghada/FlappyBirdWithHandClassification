import pygame

from classifier import Classifier

BIRD_IMAGE = pygame.transform.scale(pygame.image.load('data/bird.png'), (16*2, 16*2))

class Bird:
    def __init__(self):
        self.bird_image = BIRD_IMAGE
        self.rect = self.bird_image.get_rect(center=(250, 250))
        self.bird_pos = [self.rect.x, self.rect.y]

        self.gravity = 30
        self.gravity_accelerator = 0
        self.jump_force = -2000

        self.is_dead = False

        self.CLASSIFIER = Classifier()
        self.jump = False

    def apply_jump(self, dt):
        label_pred = self.CLASSIFIER.classify()
        if label_pred == 1:
            if self.jump == False:
                self.gravity_accelerator = self.jump_force*dt
                self.jump = True
        elif label_pred == 0:
            self.jump = False

    def apply_gravity(self, dt):
        self.gravity_accelerator += self.gravity*dt
        self.bird_pos[1] += self.gravity_accelerator*dt
        self.rect.y = self.bird_pos[1]

    def render(self, screen):
        screen.blit(self.bird_image, self.rect)

    def update(self, screen, dt):
        self.apply_jump(dt)
        self.apply_gravity(dt)
        self.render(screen)