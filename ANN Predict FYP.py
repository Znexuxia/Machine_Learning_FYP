import pickle
import numpy as np
import RPi.GPIO as GPIO
from time import sleep
from gpiozero import DistanceSensor

# setting the RPi.GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Declare the GPIO pin for motor
M1A = 24
M2A = 25
M1B = 20
M2B = 26

# Set pin as output
GPIO.setup(M1A,GPIO.OUT)
GPIO.setup(M2A,GPIO.OUT)
GPIO.setup(M1B,GPIO.OUT)
GPIO.setup(M2B,GPIO.OUT)
sleep(1)


# Define each of the movement of robot
def foward():
    print('Foward')
    GPIO.output(M1A,GPIO.HIGH)
    GPIO.output(M2A,GPIO.HIGH)
    GPIO.output(M1B,GPIO.LOW)
    GPIO.output(M2B,GPIO.LOW)
    sleep(1)

def backward():
    print('Backward')
    GPIO.output(M1A,GPIO.LOW)
    GPIO.output(M2A,GPIO.LOW)
    GPIO.output(M1B,GPIO.HIGH)
    GPIO.output(M2B,GPIO.HIGH)

    sleep(1)

def stop():
    print('stop')
    GPIO.output(M1A,GPIO.LOW)
    GPIO.output(M2A,GPIO.LOW)
    GPIO.output(M1B,GPIO.LOW)
    GPIO.output(M2B,GPIO.LOW)

def left():
    print('Left')
    GPIO.output(M1A,GPIO.LOW)
    GPIO.output(M2A,GPIO.HIGH)
    GPIO.output(M1B,GPIO.HIGH)
    GPIO.output(M2B,GPIO.LOW)
    sleep(1)

def right():
    print('Right')
    GPIO.output(M1A,GPIO.HIGH)
    GPIO.output(M2A,GPIO.LOW)
    GPIO.output(M1B,GPIO.LOW)
    GPIO.output(M2B,GPIO.HIGH)
    sleep(1)

# from the GPIO by using DistanceSensor in gpiozero
ultrasonic1 = DistanceSensor (echo = 4, trigger = 14,max_distance = 0.5)
ultrasonic2 = DistanceSensor (echo = 15, trigger = 17,max_distance = 0.5)
ultrasonic3 = DistanceSensor (echo = 27, trigger = 18,max_distance = 0.5)
ultrasonic4 = DistanceSensor (echo = 22, trigger = 23,max_distance = 0.5)
ultrasonic5 = DistanceSensor (echo = 10, trigger = 9,max_distance = 0.5)
ultrasonic6 = 50
ultrasonic7 = DistanceSensor (echo = 8, trigger =11,max_distance = 0.5)
ultrasonic8 = DistanceSensor (echo = 6, trigger =12,max_distance = 0.5)

#Load the data from the trained ANN
loaded_model = pickle.load(open('neural_network', 'rb'))

while True:
    # Getting the distance
    S1 = round(ultrasonic1.distance*100,2)
    S2 = round(ultrasonic2.distance*100,2)
    S3 = round(ultrasonic3.distance*100,2)
    S4 = round(ultrasonic4.distance*100,2)
    S5 = round(ultrasonic5.distance*100,2)
#   S6 = round(ultrasonic6.distance*100,2)
    S7 = round(ultrasonic7.distance*100,2)
    S8 = round(ultrasonic8.distance*100,2)

    #Change the aquired distance to array
    Neural = np.array ([S1,S2,S3,S4,S5,S6,S7,S8])
    Neural_reshape = Neural.reshape(1,-1)
    ANN = Neural_reshape.astype(np.float64)

    # Make prediction
    predictions = loaded_model.predict(ANN)
    sleep(1)

    if predictions == 0:
        foward()

    elif predictions == 1:
        left()

    elif predictions == 2:
        right()

    elif S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 <= 1.00 :
        stop()

    else:
        foward()



GPIO.cleanup()
