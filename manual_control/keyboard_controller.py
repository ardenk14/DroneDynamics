from drone import Drone
import pygame
from enum import Enum

# Get rid of self.directions and access Directions directly

class Directions(Enum):
    FORWARD = False
    BACKWARD = False
    LEFT = False
    RIGHT = False
    CLK = False
    COUNTERCLK = False
    THRUST_UP = False
    THRUST_DOWN = False

class KeyboardController:

    def __init__(self, drone):
        self.drone = drone
        pygame.init()
        self.screen = pygame.display.set_mode((649,480))

        self.FORWARD = False
        self.BACKWARD = False
        self.LEFT = False
        self.RIGHT = False
        self.CLK = False
        self.COUNTERCLK = False
        self.THRUST_UP = False
        self.THRUST_DOWN = False

    def process_input(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                kill = self.process_keydown(event.key)
                if kill is True:
                    return False
            if event.type == pygame.KEYUP:
                self.process_keyup(event.key)
        if self.THRUST_UP:
            pass
            #self.drone.inc_thrust(0x01)
        elif self.THRUST_DOWN:
            pass
            #self.drone.dec_thrust(0x01)
        return True

    def process_keydown(self, key):
        if key == pygame.K_q or key == pygame.K_ESCAPE:
            #Send neutral and clean up drone sockets
            self.drone.thrust = 0x00
            self.drone.kill()
            return True
        elif key == pygame.K_UP:
            self.drone.inc_thrust(0x05)
            self.THRUST_UP = True
            self.THRUST_DOWN = False
        elif key == pygame.K_DOWN:
            self.drone.dec_thrust(0x05)
            self.THRUST_DOWN = True
            self.THRUST_UP = False
        elif key == pygame.K_RIGHT:
            # Clockwise yaw
            self.drone.inc_clockwise(0x10)
            self.CLK = True
            self.COUNTERCLK = False
        elif key == pygame.K_LEFT:
            # Counter clockwise yaw
            self.drone.inc_counterclock(0x10)
            self.COUNTERCLK = True
            self.CLK = False
        elif key == pygame.K_KP8:
            # Forward
            self.drone.inc_forward(0x10)
            self.FORWARD = True
            self.BACKWARD = False
        elif key == pygame.K_KP2:
            # Backward
            self.drone.inc_backward(0x10)
            self.BACKWARD = True
            self.FORWARD = False
        elif key == pygame.K_KP6:
            # Right
            self.drone.inc_right(0x10)
            self.RIGHT = True
            self.LEFT = False
        elif key == pygame.K_KP4:
            # Left
            self.drone.inc_left(0x10)
            self.LEFT = True
            self.RIGHT = False
        elif key == pygame.K_KP5:
            # Neutral
            self.drone.roll = 0x80
            self.drone.pitch = 0x80
            self.drone.thrust -= 0x01
        elif key == pygame.K_KP_ENTER:
            self.drone.thrust = 0x00
        return False

    def process_keyup(self, key):
        if key == pygame.K_UP:
            if self.THRUST_UP:
                print("Stopping")
                self.THRUST_UP = False
        elif key == pygame.K_DOWN:
            if self.THRUST_DOWN:
                self.THRUST_DOWN = False
        elif key == pygame.K_RIGHT:
            pass
        elif key == pygame.K_LEFT:
            pass
        elif key == pygame.K_KP8:
            # Forward
            self.drone.inc_backward(0x10)
            self.FORWARD = False
        elif key == pygame.K_KP2:
            # Down
            self.drone.inc_forward(0x10)
            self.BACKWARD = False
        elif key == pygame.K_KP6:
            # Right
            self.drone.inc_left(0x10)
            self.RIGHT = False
        elif key == pygame.K_KP4:
            # Left
            self.drone.inc_right(0x10)
            self.LEFT = False
        elif key == pygame.K_KP5:
            # Neutral
            pass




