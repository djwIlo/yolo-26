import serial

PORT = "COM7"
BAUD = 115200

X_MIN = 0
X_MAX = 4800
Y_MIN = 0
Y_MAX = 3000

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)