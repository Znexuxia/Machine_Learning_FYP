#-------------------------------------------------------------------------------
# Name:        Ultrasonic Sensors function
# Purpose:     Create a programme function for ultrasonic sensors.
#
# Author:      Mohd Shahazureen Ikwan Bin Abdul Rahman
#
# Created:     29/03/2019
# Copyright:   (c) ILLEGEAR LAGUNA 2019
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np
from gpiozero import DistanceSensor


# from the GPIO by using DistanceSensor
ultrasonic1 = DistanceSensor (echo = 4 , trigger = 14, max_distance = 2)
ultrasonic2 = DistanceSensor (echo = 15 , trigger = 17, max_distance = 2)

ultrasonic1.distance
ultrasonic2.distance

while True:
    print ("S1: ",ultrasonic1.distance*100)
    print ("S2: ",ultrasonic2.distance*100)

