import serial
import time
import csv

ser = serial.Serial('/dev/ttyACM0')
ser.flushInput()

while True:
    ser = serial.Serial('/dev/ttyACM0', 9600, timeout=100000)
    ser.flushInput()
    ser_bytes = ser.readline()
    decoded_bytes = str(ser_bytes[0:len(ser_bytes)-2].decode("utf-8"))
    print(decoded_bytes)
    with open("practicle_testing_pool.csv","a") as f:
        writer = csv.writer(f,delimiter=",")
        writer.writerow([time.time(),decoded_bytes])
   