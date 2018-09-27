from pygame.locals import *
from random import randint
import pygame
import math
import time
import pandas as pd
import numpy as np
import matplotlib
import random

MOVE_TO_SQUARE = .025
DEATH_SQUARE = -.025
APPLE_SQUARE = .1
BOARD_WIDTH = 20
BOARD_HEIGHT = 15
GLOB_STEP = 40
# height +2 and width +2 multiplied together
SQUARES = (BOARD_HEIGHT + 2) * (BOARD_WIDTH + 2) - 1
NUM_LAYERS = 2
NUM_NODES_IN_HIDDEN_LAYER = 50
NUM_POP = 8
NUM_GENS = 200
LIVE_SCORE = 1
EAT_SCORE = 10
MOVE_TO_APPLE = 2
SPREAD = 1
RANGE = 2
MAX_DIST = 5


class Apple:
    x = 0
    y = 0
    step = GLOB_STEP

    def __init__(self, x, y):
        self.x = x * self.step
        self.y = y * self.step

    def draw(self, surface, image):
        surface.blit(image, (self.x, self.y))


time.sleep(100.0 / 1000.0);


class Game:
    def is_collision(self, x1, y1, x2, y2, bsize):
        if x1 >= x2 and x1 <= x2 + bsize:
            if y1 >= y2 and y1 <= y2 + bsize:
                return True
        return False

    def is_list_collision(self, x1, y1, x2, y2, bsize):
        length = len(x2)
        for i in range(0, length):
            if x1 >= x2[i] and x1 <= x2[i] + bsize:
                if y1 >= y2[i] and y1 <= y2[i] + bsize:
                    return True
        return False

    def is_snake_collision(self, x1, y1, x2, y2, bsize):
        length = len(x2)
        for i in range(2, length):
            if x1 >= x2[i] and x1 <= x2[i] + bsize:
                if y1 >= y2[i] and y1 <= y2[i] + bsize:
                    return True
        return False


class Computer:
    x = [5 * GLOB_STEP]
    y = [5 * GLOB_STEP]
    step = GLOB_STEP

    updateCountMax = 2
    updateCount = 0

    def __init__(self, length):
        self.direction = 0
        self.myNetwork = []
        self.length = length
        for i in range(0, 2000):
            self.x.append(-100)
            self.y.append(-100)

        # initial positions, no collision.
        self.x[1] = 4 * self.step
        self.x[2] = 3 * self.step
        self.y[1] = 5 * self.step
        self.y[2] = 5 * self.step

        for i in range(0, NUM_POP):
            # self.myNetwork.append(network(True, 0, 0, 0))
            pass

    def update(self, apple, t_network):
        self.myNetwork[t_network].update_forward(self, apple)
        self.updateCount += 1
        if self.updateCount > self.updateCountMax:

            # update previous positions
            for i in range(self.length - 1, 0, -1):
                self.x[i] = self.x[i - 1]
                self.y[i] = self.y[i - 1]

            # update position of head of snake
            if self.direction == 0:
                self.x[0] = self.x[0] + self.step
            if self.direction == 1:
                self.x[0] = self.x[0] - self.step
            if self.direction == 2:
                self.y[0] = self.y[0] - self.step
            if self.direction == 3:
                self.y[0] = self.y[0] + self.step

            self.updateCount = 0

    def move_right(self):
        self.direction = 0

    def move_left(self):
        self.direction = 1

    def move_up(self):
        self.direction = 2

    def move_down(self):
        self.direction = 3

    def reset(self, length):
        self.length = length
        self.direction = 0
        self.x = [5 * GLOB_STEP]
        self.y = [5 * GLOB_STEP]
        for i in range(0, 2000):
            self.x.append(-100)
            self.y.append(-100)

        # initial positions, no collision.
        self.x[1] = 4 * self.step
        self.x[2] = 3 * self.step
        self.y[1] = 5 * self.step
        self.y[2] = 5 * self.step

    def draw(self, surface, image):
        for i in range(0, self.length):
            surface.blit(image, (self.x[i], self.y[i]))


class Neuron:

    def __init__(self, is_input, my_id):
        self.id = my_id
        self.input = is_input
        self.shot = (random.uniform(0, 360), random.uniform(0, 360), random.uniform(0, 360))
        self.spread = random.uniform(0, SPREAD)
        self.range = random.uniform(0, RANGE)
        self.location = (random.uniform(0, MAX_DIST), random.uniform(0, MAX_DIST), random.uniform(0, MAX_DIST))
        self.collateral = []

    def update_hit(self, list_of_neurons):
        self.collateral = []
        



class Network:
    pass


class App:
    windowWidth = GLOB_STEP * BOARD_WIDTH
    windowHeight = GLOB_STEP * BOARD_HEIGHT
    computer = 0
    apple = 0

    def __init__(self):
        self._running = True
        self._display_surf = None
        self._image_surf = None
        self._apple_surf = None
        self.notDead = True
        self.foundApple = False
        self.game = Game()
        self.computer = Computer(3)
        self.apple = Apple(randint(0, BOARD_WIDTH - 1), randint(0, BOARD_HEIGHT - 1))

    def on_death(self):
        self.notDead = False

        self.computer.reset(3)

        while True:
            self.apple.x = randint(0, BOARD_WIDTH - 1) * 40
            self.apple.y = randint(0, BOARD_HEIGHT - 1) * 40
            if not self.game.is_list_collision(self.apple.x, self.apple.y, self.computer.x, self.computer.y, 36):
                break
            print("placing apple")

        self.on_render()

    def on_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode((self.windowWidth, self.windowHeight), pygame.HWSURFACE)

        pygame.display.set_caption('my snake game')
        self._running = True
        self._image_surf = pygame.image.load("pygame.png").convert()
        self._apple_surf = pygame.image.load("food.png").convert()

    def on_event(self, event):
        if event.type == QUIT:
            self._running = False

    def on_loop(self, network):

        self.computer.update(self.apple, network)

        for i in range(0, self.computer.length):
            if self.game.is_collision(self.apple.x, self.apple.y, self.computer.x[0], self.computer.y[0], 36):
                self.foundApple = True
                while True:
                    self.apple.x = randint(0, BOARD_WIDTH - 1) * 40
                    self.apple.y = randint(0, BOARD_HEIGHT - 1) * 40
                    if not self.game.is_list_collision(self.apple.x, self.apple.y, self.computer.x, self.computer.y,
                                                       36):
                        break
                    print("placing apple")
                self.computer.length += 1
            else:
                self.foundApple = False

        if self.game.is_snake_collision(self.computer.x[0], self.computer.y[0], self.computer.x, self.computer.y, 36):
            print("You Lose! Collision: snake collided")
            self.on_death()

        elif (self.computer.x[0] > BOARD_WIDTH * 40) or (self.computer.x[0] < -1):
            print("You Lose! Collision: x collided ")
            self.on_death()

        elif (self.computer.y[0] > BOARD_HEIGHT * 40) or (self.computer.y[0] < -1):
            print("You Lose! Collision: y collided")
            self.on_death()

        pass

    def on_render(self):
        self._display_surf.fill((0, 0, 0))
        self.computer.draw(self._display_surf, self._image_surf)
        self.apple.draw(self._display_surf, self._apple_surf)
        pygame.display.flip()

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        if not self.on_init():
            self.running = False

        i = 0
        while self._running:
            for i in range(0, NUM_GENS):
                current_score = []
                for j in range(0, NUM_POP):
                    current_score.append(0)
                    self.notDead = True
                    while self.notDead:
                        pygame.event.pump()

                        keys = pygame.key.get_pressed()

                        # if (keys[K_RIGHT]):
                        #    if not self.computer.direction == 1:
                        #        self.computer.move_right()
                        # if (keys[K_LEFT]):
                        #    if not self.computer.direction == 0:
                        #        self.computer.move_left()
                        # if (keys[K_UP]):
                        #    if not self.computer.direction == 3:
                        #        self.computer.move_up()
                        # if (keys[K_DOWN]):
                        #    if not self.computer.direction == 3:
                        #        self.computer.move_down()
                        if keys[K_ESCAPE]:
                            exit(0)

                        self.on_loop(j)
                        self.on_render()
                        # current_score[j] += int(grade(self.apple, self.computer, self.notDead, self.foundApple))
                        print(current_score[j])
                        time.sleep(10.0 / 1000.0)

                    print(j)
                    print(current_score[j])

                self.computer.update_backward(current_score)
                # evolove()
            exit(0)
        self.on_cleanup()


if __name__ == "__main__":
    theApp = App()
    theApp.on_execute()
