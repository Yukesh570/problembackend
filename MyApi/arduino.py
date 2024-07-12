import serial
import time
from django.dispatch import Signal
from django.http import JsonResponse
from .views import stop_timer,start_timer,Check_timer,watch_timer
from django.shortcuts import render ,get_object_or_404
from .models import *
from decimal import Decimal
import cv2
import numpy as np
button_state_changed = Signal()
from django.http import StreamingHttpResponse

# Define a global variable to store the button state
global_button_state = ""
running = False
toople=False
def read_button_state(serial_port,request,pk):
    table=get_object_or_404(Table,tableno=pk)

    global global_button_state, running,toople
    try:
        while True:
            if serial_port.in_waiting > 0:
                button_state = serial_port.readline().decode('utf-8').strip()

        

                if button_state == "1":
                
                        # Start action
                        print("Started")
                        start_timer(pk)
                        watch_timer(pk)
                        toople=True
                    

                        
                elif button_state == "0":
                    if toople:
                        print("Stopped")
                        stop_timer(request,pk)
                        # stop_running(request,pk)
                        # table.save()
                        toople=False
                        
                        
                        break   
                

                # elif button_state==1:
                #             if button_state == "0":
                #                 if not running:
                #                     # Start action
                #                     print("Started")
                #                     toople=True
                #                 else:
                #                     # Stop action
                #                     running = False

                #                     print("Stopped")
                #                     break
                #             elif button_state == "1":
                #                 if toople:
                #                     running = True

                    # Wait for the next "1" to toggle state
                    
                print(toople)
                global_button_state = button_state
                button_state_changed.send(sender=None, button_state=button_state)
                print(f"The button state is: {button_state}")
                
    except KeyboardInterrupt:
        print("Exiting program")

def call_read_button_state(request,pk):
    arduino_port = 'COM7'
    baud_rate = 9600

    try:
        serial_port = serial.Serial(arduino_port, baud_rate, timeout=1)
        print(f"Connected to {arduino_port} at {baud_rate} baud.")
        read_button_state(serial_port,request,pk)
        return JsonResponse({'status': 'success'})
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)})



def timer_video_feed(request, pk):
    cap = cv2.VideoCapture(cv2.CAP_ANY)  # Use 0 for the default webcam
    if not cap.isOpened():
        raise IOError("Webcam cannot be opened.")
    
    skip_frames = 2
    frame_count = 0

    def generate_frames():
        nonlocal frame_count
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames
            frame_count += 1

            if frame_count % skip_frames != 0:
                continue

            # Retrieve values
            frame = cv2.resize(frame, (500, 500)) 
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')