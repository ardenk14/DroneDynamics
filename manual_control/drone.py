import binascii
import socket


class Drone:

    def __init__(self):
        self.thrust = 0x00  # Increase -> up       | Decrease -> down
        self.pitch = 0x80   # Increase -> forward  | Decrease -> back
        self.roll = 0x80    # Increase -> right    | Decrease -> left
        self.yaw = 0x80     # Increase -> clockwise| Decrease -> counterclockwise

        self.host = "172.16.10.1"
        self.port = 8888
        self.udp_port = 8895
        self.udp_address = (self.host, self.udp_port)

        self.send_init_packet()
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_init_packet(self):
        #Read magic packet from file
        fp = open("init-packet.txt")
        init_packet = []
        for line in fp:
            l = line.split(", ")
            for i in l:
                init_packet.append(int(i,16))

        # Create TCP connection with drone and send init packet
        s = socket.socket()
        s.connect((self.host, self.port))
        print("Sending initialization packet...")
        s.send(bytearray(init_packet))
        print("Packet sent.")
        print("Waiting for return packet...")
        byte_r = s.recv(106)
        print("Received: ")
        print(binascii.hexlify(byte_r))
        s.close()

    def send(self):
        #Create command
        cmd = self.get_cmd()
        # Send through UDP socket
        self.udp_socket.sendto(bytearray(cmd), self.udp_address)

    def kill(self):
        # Send neutral packet
        self.set_neutral()
        self.kill_thrust()
        self.send()
        # Clean up sockets
        self.udp_socket.close()

    def get_cmd(self):
        cmd = []
        cmd.append(0x66)
        cmd.append(self.roll)
        cmd.append(self.pitch)
        cmd.append(self.thrust)
        cmd.append(self.yaw)
        cmd.append(0x00)
        
        seven = cmd[1]
        for i in range(2,6):
            seven ^= cmd[i]

        cmd.append(seven)
        cmd.append(0x99)
        return cmd

    def set_neutral(self):
        self.pitch = 0x80
        self.roll = 0x80
        self.yaw = 0x80

    def set_thrust(self, level):
        if level >=  0x00 and level <= 0xFF:
            self.thrust = level

    def kill_thrust(self):
        self.thrust = 0x00

    def inc_thrust(self, amount = 1):
        if self.thrust + amount > 0xFF:
            self.thrust = 0xFF
        else:
            self.thrust += amount

    def dec_thrust(self, amount = 1):
        if self.thrust - amount < 0x00:
            self.thrust = 0x00
        else:
            self.thrust -= amount

    def inc_right(self, amount = 1):
        if self.roll + amount > 0xFF:
            self.roll = 0xFF
        else:
            self.roll += amount

    def inc_left(self, amount = 1):
        if self.roll - amount < 0x00:
            self.roll = 0x00
        else:
            self.roll -= amount

    def inc_forward(self, amount = 1):
        if self.pitch + amount > 0xFF:
            self.pitch = 0xFF
        else:
            self.pitch += amount

    def inc_backward(self, amount = 1):
        if self.pitch - amount < 0x00:
            self.pitch = 0x00
        else:
            self.pitch -= amount

    def inc_clockwise(self, amount = 1):
        if self.yaw + amount > 0xFF:
            self.yaw = 0xFF
        else:
            self.yaw += amount

    def inc_counterclock(self, amount = 1):
        if self.yaw - amount < 0x00:
            self.yaw = 0x00
        else:
            self.yaw -= amount


