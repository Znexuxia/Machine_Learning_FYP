4#-------------------------------------------------------------------------------
# Name:        Ultrasonic Sensors function
# Purpose:     Create a programme function for ultrasonic sensors.
#
# Author:      Mohd Shahazureen Ikwan Bin Abdul Rahman
#
# Created:     29/03/2019
# Copyright:   (c) ILLEGEAR LAGUNA 2019
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import RPi.GPIO as GPIO
import time
GPIO.setmode (GPIO.BCM)

TRIGS2 = 4
ECHOS2 = 14
TRIGS3 = 15
ECHOS3 = 17
TRIGS4 = 27
ECHOS4 = 18


print ('Distance Measurement In Progress')

GPIO.setup(TRIGS2,GPIO.OUT)
GPIO.setup(ECHOS2,GPIO.IN)
GPIO.setup(TRIGS3,GPIO.OUT)
GPIO.setup(ECHOS3,GPIO.IN)
GPIO.setup(TRIGS4,GPIO.OUT)
GPIO.setup(ECHOS4,GPIO.IN)

print('Waiting For Sensor To Settle')
while True:
    GPIO.output(TRIGS2, False)
    GPIO.output(TRIGS3, False)
    GPIO.output(TRIGS4, False)


    time.sleep(2)

    GPIO.output(TRIGS2, True)
    GPIO.output(TRIGS3, True)
    GPIO.output(TRIGS4, True)


    time.sleep(0.00001)


    GPIO.output(TRIGS2, False)
    GPIO.output(TRIGS3, False)
    GPIO.output(TRIGS4, False)


    #For echo
    while GPIO.input(ECHOS2)==0:
         pulse_start2 = time.time()

    while GPIO.input(ECHOS2)==1:
        pulse_end2 = time.time()

    while GPIO.input(ECHOS3)==0:
         pulse_start3 = time.time()

    while GPIO.input(ECHOS3)==1:
        pulse_end3 = time.time()

    while GPIO.input(ECHOS4)==0:
         pulse_start4 = time.time()

    while GPIO.input(ECHOS4)==1:
        pulse_end4 = time.time()



    #Pulse Duration
    pulse_duration2 = pulse_end2 - pulse_start2
    pulse_duration3 = pulse_end3 - pulse_start3
    pulse_duration4 = pulse_end4 - pulse_start4

    #Getting Distance
    distance2 = pulse_duration2 * 17150
    distance3 = pulse_duration3 * 17150
    distance4 = pulse_duration4 * 17150

    #Round-off distance
    distance2 = round(distance2,2)
    distance3 = round(distance3,2)
    distance4 = round(distance4,2)

    #Display the distance
    print('S2: ',distance2,'cm')
    print('S3: ',distance3,'cm')
    print('S4: ',distance4,'cm')

GPIO.cleanup()


