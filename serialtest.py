#!/usr/bin/env python3
import serial
import time
if __name__ == '__main__':
    ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    ser.reset_input_buffer()
    while True:
        angle = input('Let us wait for user input. \n') 
        ser.write(angle.encode())
        line = ser.readline().decode('utf-8').rstrip()
        print(line)
        time.sleep(1)