import cv2

for i in range(3):  # try camera indexes 0, 1, 2
    cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
    ret, frame = cap.read()
    print(f"Device {i} open: {ret}")
    cap.release()
