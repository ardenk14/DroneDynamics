# Simple pygame program

# Import and initialize the pygame library
import pygame
import pygame._sdl2.touch as tch
pygame.init()

# Set up the drawing window
screen = pygame.display.set_mode([500, 500])

# Run until the user asks to quit
running = True
#num_devices = tch.get_num_devices()
#print(num_devices)
#device = tch.get_device(0)
#print(device)

#pygame.joystick.init()
joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
print("JOYSTICKS: ", joysticks)
while running:
    try:
        #fingers = tch.get_num_fingers(-1)
        #print("FINGERS: ", fingers)
        #print("0: ", joysticks[0].get_axis(0))
        #print("1: ", joysticks[0].get_axis(1))
        #print("2: ", joysticks[0].get_axis(2))
        #print("3: ", joysticks[0].get_axis(3))
        print("5: ", joysticks[0].get_axis(0))
    except:
        pass

    # Did the user click the window close button?
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.FINGERDOWN:
            print("FINGER DOWN")
        elif event.type == pygame.FINGERMOTION:
            print("FINGER MOTION")
            fingers = tch.get_num_fingers(-1)
            print(fingers)
        elif event.type == pygame.FINGERUP:
            print("FINGER UP")
        elif event.type == pygame.KEYDOWN:
            print("KEY DOWN")

    # Fill the background with white
    screen.fill((255, 255, 255))

    # Draw a solid blue circle in the center
    pygame.draw.circle(screen, (0, 0, 255), (250, 250), 75)

    # Flip the display
    pygame.display.flip()

# Done! Time to quit.
pygame.quit()