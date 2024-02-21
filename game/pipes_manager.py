import random

import pygame

PIPE_IMAGE = pygame.transform.scale(pygame.image.load('data/pipe.png'), (64, 700))

class Pipe:
    def __init__(self, coords, inverted=False):
        self.pipe_image = PIPE_IMAGE
        self.rect = self.pipe_image.get_rect()
        self.rect.x, self.rect.y = coords

        if inverted:
            self.pipe_image = pygame.transform.flip(self.pipe_image, False, True)

    def render(self, screen):
        screen.blit(self.pipe_image, self.rect)

class PipeManager:
    def __init__(self, width, height, bird):
        self.WIDTH = width
        self.HEIGHT = height
        self.BIRD = bird

        self.open_height = 200
        self.pipe_dst = 400
        self.starting_xpos = self.WIDTH + 20
        self.pipe_movement = 60

        self.pipes = []
        # Generating First Pipe.
        self.generate_pipe()

        self.pipe_passed_score = 0 # Pipes passed by the bird.

    def generate_pipe(self):
        upper_x = self.starting_xpos
        upper_y = random.randint(int(self.HEIGHT/3), int(self.HEIGHT - (self.HEIGHT/3))) - 700
        upper_pipe = Pipe((upper_x, upper_y), True)

        lower_x = self.starting_xpos
        lower_y = (upper_y + 700) + self.open_height
        lower_pipe = Pipe((lower_x, lower_y), False)

        self.pipes.append((upper_pipe, lower_pipe))
        self.scored_pipes = [] # Storing Pipes that are already passed by the bird.

    def generate_pipes(self):
        last_pipes = self.pipes[-1]
        if abs(last_pipes[0].rect.x - self.starting_xpos) >= self.pipe_dst:
            self.generate_pipe()

    def move_pipe(self, dt):
        for i in range(len(self.pipes)):
            upper_pipe, lower_pipe = self.pipes[i]
            upper_pipe.rect.x -= self.pipe_movement*dt
            lower_pipe.rect.x -= self.pipe_movement*dt
            if upper_pipe.rect.x < 250-64: # ( Bird X Pos and Upper Pipe Width).
                if upper_pipe not in self.scored_pipes:
                    self.pipe_passed_score += 1
                    self.scored_pipes.append(upper_pipe)

    def render_pipe(self, screen):
        for i in range(len(self.pipes)):
            upper_pipe, lower_pipe = self.pipes[i]
            upper_pipe.render(screen)
            lower_pipe.render(screen)

    def remove_pipe(self):
        for upper_pipe, lower_pipe in self.pipes:
            if upper_pipe.rect.x <= -80: # Hardcoded.
                self.pipes.remove((upper_pipe, lower_pipe))
                self.scored_pipes.remove(upper_pipe)

    def bird_collision(self):
        for upper_pipe, lower_pipe in self.pipes:
            if self.BIRD.rect.colliderect(upper_pipe):
                self.BIRD.is_dead = True
            elif self.BIRD.rect.colliderect(lower_pipe):
                self.BIRD.is_dead = True

    def update_pipe(self, screen, dt):
        self.generate_pipes()
        self.move_pipe(dt)
        self.remove_pipe()
        self.bird_collision()
        self.render_pipe(screen)