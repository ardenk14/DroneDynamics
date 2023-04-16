#!/usr/bin/env python3
''' imageGrabber.py

    This is a ros node for grabbing an image stream from
    a drone and publishing the image frames to a ros topic

    Author: Arden Knoll
'''
import rospy
import socket
import codecs
import cv2
import numpy as np
import h264decoder
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

class imageGrabber():

    def __init__(self):
        rospy.init_node('image_stream', anonymous=False)   # Can have only one leader node
        self.img_pub = rospy.Publisher('/image_stream/image', Image,  queue_size=10)
        self.info_pub = rospy.Publisher('/image_stream/camera_info', CameraInfo,  queue_size=10)
        self.br = CvBridge()
        self.cam_info = CameraInfo()

        # Host and port
        HOST = '172.16.10.1' #'172.17.10.1'
        PORT = 8888
        # Setup socket
        self.s0 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("SOCKET OPEN")
        self.s0.connect((HOST, PORT))
        print("CONNECTED")
        print(self.s0.getsockname())

        # TODO: Send initial camera stream packet
        packets = [
            b'49546400000058000000808638c38d1350fd6741c2ee3689a054cae26430a3c15e40de30f6d695e030b7c2e5b7d65da8659eb2e2d5e0c2cb6c59cdcb661e7e1eb0ce8ee8df32456fa842eb20be383aab05a8c2a71f2c906d93f72a85e7356effe1b8f5af097f9147f87e'
        ]

        # looping packets and printing the hexadecimal response
        print("LOOPING THROUGH PACKETS")
        for packet in packets:
            print("NEW PACKET")
            self.s0.send(codecs.decode(packet, "hex"))
            print("SENT PACKET")
            print(codecs.encode(self.s0.recv(1024), "hex"))
            print('\n--------\n')

        rospy.loginfo('Camera stream initialized')        
        #self.pub = rospy.Publisher(topic_name, Int32, queue_size=10)

        self.cam_info.height = 576
        self.cam_info.width = 720
        self.cam_info.distortion_model = "plumb_bob"
        self.cam_info.D = [0.8337068337092105, -0.7296451172680306, 0.015187197230019187, 0.08793175446175946, 0.0]
        self.cam_info.K = [1989.1574391172464, 0.0, 554.6150636910545, 0.0, 2068.7938072844036, 280.42584157479126, 0.0, 0.0, 1.0]
        self.cam_info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.cam_info.P = [2012.7718505859375, 0.0, 562.0574733415124, 0.0, 0.0, 2135.0751953125, 281.9590617435897, 0.0, 0.0, 0.0, 1.0, 0.0]



    def run(self):
        BUFFER_SIZE = 1024
        decoder = h264decoder.H264Decoder()
        while True:
            data_in = self.s0.recv(BUFFER_SIZE)
            if not data_in:
                print("NOT IN")
                continue
            #print(data_in[0:5])
            framedatas = decoder.decode(data_in)
            for framedata in framedatas:
                (frame, w, h, ls) = framedata
                if frame is not None:
                    #print('frame size %i bytes, w %i, h %i, linesize %i' % (len(frame), w, h, ls))
                    frame = np.frombuffer(frame, dtype=np.ubyte, count=len(frame))
                    frame = frame.reshape((h, ls//3, 3))
                    frame = frame[:,:w,:] # NOTE: THIS IS IN RGB BUT OPENCV USES BGR
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    #cv2.imshow("im", frame)
                    img_msg = self.br.cv2_to_imgmsg(frame)
                    img_msg.encoding = 'bgr8'
                    img_msg.header.stamp = rospy.Time.now()
                    self.cam_info.header.stamp = img_msg.header.stamp
                    self.img_pub.publish(img_msg)
                    self.info_pub.publish(self.cam_info)
                    #print("CAM INFO PUBLISHED")
                    #if cv2.waitKey(1) & 0xFF == ord('q'):
                        #break

if __name__ == '__main__':
    stream = imageGrabber()
    
    #stream.run()
    try:
        stream.run()
    except KeyboardInterrupt:
        pass
