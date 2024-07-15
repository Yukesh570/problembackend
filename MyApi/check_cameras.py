import cv2

def get_available_cameras(max_cameras=10):
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

if __name__ == "__main__":
    cameras = get_available_cameras()
    print(f"Available cameras: {len(cameras)}")
    print("Camera indexes:", cameras)