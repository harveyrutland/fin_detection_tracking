import serial
import time
import csv
from datetime import datetime






ser = serial.Serial('/dev/ttyACM0')
ser.flushInput()

while True:
    ser = serial.Serial('/dev/ttyACM0', 9600, timeout=100000)
    ser.flushInput()
    ser_bytes = ser.readline()
    decoded_bytes = str(ser_bytes[0:len(ser_bytes)-2].decode("utf-8"))
    print(decoded_bytes)
    timestamp = time.time()
    dt_object = datetime.fromtimestamp(timestamp)
    with open("practicle_testing_pool.csv","a") as f:
        writer = csv.writer(f,delimiter=",")
        writer.writerow([dt_object,decoded_bytes])
   