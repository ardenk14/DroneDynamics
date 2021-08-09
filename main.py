from drone import Drone
from keyboard_controller import KeyboardController
import pygame
import time

def main():
    drone = Drone()
    controller = KeyboardController(drone)
    while controller.process_input():
        drone.send()
        time.sleep(0.002)

if __name__ == '__main__':
    main()

