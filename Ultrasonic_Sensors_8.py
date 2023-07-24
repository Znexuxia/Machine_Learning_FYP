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
import time
import RPi.GPIO

# from the GPIO by using DistanceSensor
ultrasonic1 = DistanceSensor (echo = 4, trigger = 14,max_distance = 2)
ultrasonic2 = DistanceSensor (echo = 15, trigger = 17,max_distance = 2)
ultrasonic3 = DistanceSensor (echo = 27, trigger = 18,max_distance = 2)
ultrasonic4 = DistanceSensor (echo = 22, trigger = 23,max_distance = 2)
ultrasonic5 = DistanceSensor (echo = 10, trigger = 24,max_distance = 2)
ultrasonic6 = DistanceSensor( echo = 25, trigger = 5 ,max_distance = 2)
ultrasonic7 = DistanceSensor (echo = 8, trigger =11,max_distance = 2)
ultrasonic8 = DistanceSensor (echo = 6, trigger =12,max_distance = 2)

ultrasonic1.distance
ultrasonic2.distance
ultrasonic3.distance
ultrasonic4.distance
ultrasonic5.distance
ultrasonic6.distance
ultrasonic7.distance
ultrasonic8.distance

while True:
    time.sleep(2)
    print ("S1: ",round(ultrasonic1.distance*100,2),"cm")
    print ("S2: ",round(ultrasonic2.distance*100,2),"cm")
    print ("S3: ",round(ultrasonic3.distance*100,2),"cm")
    print ("S4: ",round(ultrasonic4.distance*100,2),"cm")
    print ("S5: ",round(ultrasonic5.distance*100,2),"cm")
    print ("S6: ",round(ultrasonic6.distance*100,2),"cm")
    print ("S7: ",round(ultrasonic7.distance*100,2),"cm")
    print ("S8: ",round(ultrasonic8.distance*100,2),"cm")


GPIO.cleanup()
