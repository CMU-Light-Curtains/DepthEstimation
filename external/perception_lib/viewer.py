import pyperception_lib as pylib
import numpy as np
import time
import threading

class Visualizer(threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name
        self.kill_received = False

        self.visualizer = pylib.Visualizer()
 
    def run(self):
        self.visualizer.start();

        while not self.kill_received:
            # your code
            #print self.name, "is active"
            self.visualizer.loop()
            time.sleep(0.1)

    def addCloud(self, cloud, size=1):
        self.visualizer.addCloud(cloud, size)

    def swapBuffer(self):
        self.visualizer.swapBuffer()
