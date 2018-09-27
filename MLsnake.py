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
SQUARES = 22 * 17 - 1
NUM_LAYERS = 2
NUM_NODES_IN_HIDDEN_LAYER = 50
NUM_POP = 8
NUM_GENS = 200
LIVE_SCORE = 1
EAT_SCORE = 10
MOVE_TO_APPLE = 2


def sigmoid(x):
    sig_num = float(1 - np.exp(-2 * x)) / (1 + np.exp(-2 * x))

    return (sig_num)


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

    def isSnakeCollision(self, x1, y1, x2, y2, bsize):
        length = len(x2)
        for i in range(2, length):
            if x1 >= x2[i] and x1 <= x2[i] + bsize:
                if y1 >= y2[i] and y1 <= y2[i] + bsize:
                    return True
        return False


# the neurons for the input layer

class InputLayer:

    def __init__(self):
        self.neurons = []

        for i in range(0, SQUARES):
            self.neurons.append(MOVE_TO_SQUARE)

        for i in range(0, BOARD_WIDTH + 1):
            self.neurons[i] = DEATH_SQUARE

        for i in range(1, BOARD_HEIGHT + 1):
            self.neurons[i * BOARD_WIDTH + 2] = DEATH_SQUARE

            self.neurons[(i * (BOARD_WIDTH + 2)) + BOARD_WIDTH + 1] = DEATH_SQUARE

        for i in range(SQUARES - (BOARD_WIDTH + 2), SQUARES):
            self.neurons[i] = -DEATH_SQUARE

    def update(self, snake, apple):

        # reset board
        for i in range(1, BOARD_HEIGHT + 1):
            for j in range(1, BOARD_WIDTH + 1):
                self.neurons[i * (BOARD_WIDTH + 2) + j] = MOVE_TO_SQUARE

        appleZ = int(apple.x / GLOB_STEP) + 1
        appleW = int(apple.y / GLOB_STEP) + 1
        appleLocation = appleZ + appleW * BOARD_WIDTH + 2
        self.neurons[appleLocation] = APPLE_SQUARE

        length = len(snake.x)
        for i in range(0, length):
            snakeZ = int(snake.x[i] / 40) + 1
            snakeW = int(snake.y[i] / 40) + 1

            snakeLoc = snakeZ + snakeW * BOARD_WIDTH + 2
            self.neurons[snakeLoc] = DEATH_SQUARE


class Neuron:

    def __init__(self, og, num_nodes, inputAxons):

        if og == True:
            self.axon = []
            for i in range(0, num_nodes):
                number = random.uniform(-1, 1)
                self.axon.append(number)
            self.activation = 0.0
        else:
            self.axon = inputAxons
            self.activation = 0.0

    def update_forward(self, prevLayor, num_nodes):
        sum = 0.0
        if not len(self.axon) == len(prevLayor.neurons):
            print(len(self.axon))
            print(len(prevLayor.neurons))
        for i in range(0, num_nodes):
            sum += self.axon[i] * prevLayor.neurons[i]
        self.activation = sigmoid(sum)

        return self.activation

    def update_backward(self):
        pass


class HiddenLayer:

    def __init__(self, og, num_nodes, inputNeurons):
        if og == True:
            self.neurons = []
            self.myNeurons = []
            for i in range(0, NUM_NODES_IN_HIDDEN_LAYER):
                self.neurons.append(0)
                self.myNeurons.append(Neuron(True, num_nodes, []))
        else:
            self.neurons = []
            for i in range(0, NUM_NODES_IN_HIDDEN_LAYER):
                self.neurons.append(0)
            self.myNeurons = inputNeurons

    def update_forward(self, prevLayer, num_nodes):
        for i in range(0, NUM_NODES_IN_HIDDEN_LAYER):
            self.neurons[i] = self.myNeurons[i].update_forward(prevLayer, num_nodes)

    def update_backward(self):
        pass


class OutputLayer:

    def __init__(self, og, inputNeurons):
        self.neurons = []
        self.myNeurons = []
        if og == True:
            for i in range(0, 4):
                self.neurons.append(0)
                myNeuron = Neuron(True, NUM_NODES_IN_HIDDEN_LAYER, [])
                self.myNeurons.append(myNeuron)
        else:
            self.neurons = []
            for i in range(0, 4):
                self.neurons.append(0)
            self.myNeurons = inputNeurons

    def move(self, computer, prevLayer):
        for i in range(0, 4):
            self.neurons[i] = self.myNeurons[i].update_forward(prevLayer, NUM_NODES_IN_HIDDEN_LAYER)

        print(self.neurons)
        if self.neurons.index(max(self.neurons)) == 0:
            computer.moveRight()
            # print("Moving Right")
        if self.neurons.index(max(self.neurons)) == 1:
            computer.moveLeft()
            # print("Moving Left")
        if self.neurons.index(max(self.neurons)) == 2:
            computer.moveUp()
            # print("Moving Up")
        if self.neurons.index(max(self.neurons)) == 3:
            computer.move_down()
            # print("Moving Down")

    def update_backwards(self):
        pass


class network:

    def __init__(self, og, hidO, hidT, out):
        if og == True:
            # print("making original")
            self.input = InputLayer()
            self.hiddenOne = HiddenLayer(True, SQUARES, [])
            self.hiddenTwo = HiddenLayer(True, NUM_NODES_IN_HIDDEN_LAYER, [])
            self.output = OutputLayer(True, [])
        else:
            # print("making child")
            self.input = InputLayer()
            self.hiddenOne = hidO
            self.hiddenTwo = hidT
            self.output = out

    def update_forward(self, snake, apple):
        self.input.update(snake, apple)
        self.hiddenOne.update_forward(self.input, SQUARES)
        self.hiddenTwo.update_forward(self.hiddenOne, NUM_NODES_IN_HIDDEN_LAYER)
        self.output.move(snake, self.hiddenTwo)


def dist(xo, yo, xt, yt):
    a = xt - xo
    b = yt - yo
    c = math.sqrt(math.pow(a, 2) + math.pow(b, 2))

    return c


def grade(apple, computer, notDead, foundApple):
    if notDead == True:
        lastDist = dist(apple.x, apple.y, computer.x[1], computer.y[1])
        currenDist = dist(apple.x, apple.y, computer.x[0], computer.y[0])
        if currenDist > lastDist:
            return MOVE_TO_APPLE
        else:
            return LIVE_SCORE

        if foundApple == True:
            return EAT_SCORE
    else:
        return 0

    pass


def breedNeurons(mom, dad):
    dists = []
    print(len(mom.axon))
    print(len(dad.axon))
    for i in range(0, len(mom.axon) - 1):
        momOrDad = random.uniform(0, -1)
        if momOrDad > .5:
            dists[i] = mom.axon[i]
        else:
            dists[i] = dad.axon[i]

    theNeuron = Neuron(False, len(dists), dists)
    return theNeuron


def breedHid(mom, dad):
    layer = []
    for i in range(0, len(mom.neurons)):
        layer.append(breedNeurons(mom.myNeurons[i], dad.myNeurons[i]))

    theLayer = HiddenLayer(False, len(layer), layer)
    return theLayer


def breedOut(mom, dad):
    layer = []
    for i in range(0, len(mom.neurons)):
        layer.append(breedNeurons(mom.myNeurons, dad.myNeurons))

    theLayer = OutputLayer(False, layer)
    return theLayer


def breedNetwork(mom, dad):
    theNewtwork = network(False, breedHid(mom.hiddenOne, dad.hiddenOne), breedHid(mom.hiddenTwo, dad.hiddenTwo),
                          breedOut(mom.output, dad.output))
    return theNewtwork


def evolove(networkOne, networkTwo):
    pass


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
            self.myNetwork.append(network(True, 0, 0, 0))

    def update(self, apple, tNetwork):
        self.myNetwork[tNetwork].update_forward(self, apple)
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

    def moveRight(self):
        self.direction = 0

    def moveLeft(self):
        self.direction = 1

    def moveUp(self):
        self.direction = 2

    def moveDown(self):
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

    def update_backward(self, grades):

        for i in range(0, int(len(grades) / 2)):
            del grades[grades.index(min(grades))]

        # start breeding randomly
        for i in range(0, len(grades)):
            mom = random.randint(0, len(grades))
            dad = random.randint(0, len(grades))

            anotherNetwork = breedNetwork(self.myNetwork[mom], self.myNetwork[dad])
            self.myNetwork.append(anotherNetwork)

    def draw(self, surface, image):
        for i in range(0, self.length):
            surface.blit(image, (self.x[i], self.y[i]))


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
                    if not self.game.is_list_collision(self.apple.x, self.apple.y, self.computer.x, self.computer.y, 36):
                        break
                    print("placing apple")
                self.computer.length += 1
            else:
                self.foundApple = False

        if self.game.isSnakeCollision(self.computer.x[0], self.computer.y[0], self.computer.x, self.computer.y, 36):
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
        if self.on_init() == False:
            self.running = False

        i = 0
        while (self._running):
            for i in range(0, NUM_GENS):
                currentScore = []
                for j in range(0, NUM_POP):
                    currentScore.append(0)
                    self.notDead = True
                    while (self.notDead):
                        pygame.event.pump()

                        keys = pygame.key.get_pressed()

                        # if (keys[K_RIGHT]):
                        #    if not self.computer.direction == 1:
                        #        self.computer.moveRight()
                        # if (keys[K_LEFT]):
                        #    if not self.computer.direction == 0:
                        #        self.computer.moveLeft()
                        # if (keys[K_UP]):
                        #    if not self.computer.direction == 3:
                        #        self.computer.moveUp()
                        # if (keys[K_DOWN]):
                        #    if not self.computer.direction == 3:
                        #        self.computer.moveDown()
                        if (keys[K_ESCAPE]):
                            exit(0)

                        self.on_loop(j)
                        self.on_render()
                        currentScore[j] += int(grade(self.apple, self.computer, self.notDead, self.foundApple))
                        print(currentScore[j])
                        time.sleep(10.0 / 1000.0)

                    print(j)
                    print(currentScore[j])

                self.computer.update_backward(currentScore)
                # evolove()
            exit(0)
        self.on_cleanup()


if __name__ == "__main__":
    theApp = App()
    theApp.on_execute()
