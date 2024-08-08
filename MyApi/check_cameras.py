# import cv2

# def get_available_cameras(max_cameras=10):
#     available_cameras = []
#     for i in range(max_cameras):
#         cap = cv2.VideoCapture(i)
#         if cap.isOpened():
#             available_cameras.append(i)
#             cap.release()
#     return available_cameras

# if __name__ == "__main__":
#     cameras = get_available_cameras()
#     print(f"Available cameras: {len(cameras)}")
#     print("Camera indexes:", cameras)



#     #just for testing


import cv2
import numpy
pk=1
rtsp_url=f"rtsp://admin:android18)@192.168.20.10{pk}:554/cam/realmonitor?channel=1&subtype=0"
capture= cv2.VideoCapture(rtsp_url)
while True:
    ret,frame =capture.read()
    cv2.imshow('output',frame)
    k=cv2.waitKey(100) 
    if k==ord('q'):
        break
capture.release()
cv2.destroyAllWindows()