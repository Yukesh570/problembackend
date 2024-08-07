import cv2
import numpy as np
from django.http import StreamingHttpResponse
from .views import start_timer, stop_timer,Check_timer
import time
from django.http import JsonResponse
from rest_framework.decorators import api_view
import threading
from django.http import HttpResponse
from .models import *
from django.shortcuts import render ,get_object_or_404
from django.contrib.sessions.models import Session
from celery import shared_task
from django.core.cache import cache
import subprocess
from collections import deque

from django.http import HttpResponseServerError
from decimal import Decimal

# Define the lower and upper bounds for red color in HSV space




# Connect the signal handler
def detectRedObjects(frame):
    lower_red1 = np.array([160, 40, 145])
    upper_red1 = np.array([180, 255, 255])
    lower_red2 = np.array([160, 40, 145])
    upper_red2 = np.array([180, 255, 255])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    return red_mask

def detectObjects(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 170])
    upper_white = np.array([180,40, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    return mask



def contour_touching(contour1,contour2,threshold_distance):
    if contour2 is None:
        return False
    for contour1 in contour1:
        for contour2 in contour2:
            # Convert contours to numpy arrays

            contour1 = np.squeeze(contour1)
            contour2 = np.squeeze(contour2)

            # Calculate distances between points on the two contours
            distances = np.linalg.norm(contour1[:, None] - contour2, axis=-1)

            # Check if minimum distance is less than threshold
            min_distance = np.min(distances)
            # print(min_distance)

            if min_distance < threshold_distance:
                return True
            else:
                return False
            


# def gen_frames():  # generate frame by frame from camera
#     global current_frame,frames  # Declare frame as global
#     cap = cv2.VideoCapture(cv2.CAP_ANY)  # Use 0 for the default webcam
#     if not cap.isOpened():
#         return

#     try:
#         while True:
#             success, frame = cap.read()
#             if not success:
#                 break

#             ret, jpeg = cv2.imencode('.jpg', frame)
#             if not ret:
#                 break

#             cache.set('current_frame', frame, timeout=5)      
            
#             # frame = jpeg.tobytes()
#             # yield (b'--frame\r\n'
#             #        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
        
#     finally:
#         cap.release()

# def video_feedmain(request):
#     gen_frames()
#     return HttpResponse('hello')
    # return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')


# # @shared_task(bind=True)
# def test(request):
#     global current_frame, gameStarted,frames
#     current_frame = cache.get('current_frame')  # Retrieve frame from cach
#     while True:
#         print(current_frame)
#     time.sleep(1)  # Sleep for a second before fetching the next frame to avoid high CPU usage

#     return HttpResponse("done")


# def test(request):
#     test_func.delay()
#     return HttpResponse("done")
current_frame_position = 0
frames=None
worker_processes = {}

gameStarted = False
current_frame = {}  # Initialize frame as a global variable
nowframe=0


@shared_task(bind=True)
def background_video_processing(self,pk,pk1):
    print(f"Task started with pk: {pk}, pk1: {pk1}")

    global current_frame, current_frame_position, gameStarted

    # Retrieve session variables
    person=get_object_or_404(Person,id=pk1)
    table=get_object_or_404(Table,tableno=pk)
    cap = cv2.VideoCapture("rtsp://admin:android18)@192.168.20.39:554/cam/realmonitor?channel=1&subtype=0")  # Use 0 for the default webcam
    if not cap.isOpened():
        raise IOError("Webcam cannot be opened.")
       
    # print('=======================',current_frame,'===')    
    skip_frames = 2
    frame_count = current_frame_position
    gameStarted=False
    count = 0 
    area=0
    approxed=0
    skip_until=0 
    while cap.isOpened():
        current_time=time.time()      #to get the current time
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames
        # frame_count += 1
        # current_frame_position = frame_count  # Update the global frame position

        # if frame_count % skip_frames != 0:
        #     continue

        #retreive values
        frame=cv2.resize(frame,(500,500))
        
        l=20
        u=145
            
        red_mask=detectRedObjects(frame)
        mask=detectObjects(frame)
        median_blur= cv2.medianBlur(red_mask,23)
        median_blur_white= cv2.medianBlur(mask,21)

        canny=cv2.Canny(median_blur,l,u)
        canny2=cv2.Canny(median_blur_white,l,u)
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, hierarchy = cv2.findContours(canny2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # frame_copy=frame.copy()
        
        def detection(area,approxed,gameStarted,count):  
            for contour in contours:
                epsilon = 0.03 * cv2.arcLength(contour, True)
                approxed = cv2.approxPolyDP(contour, epsilon, True)
                approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

                M = cv2.moments(approx)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    # Define position and size of the text area
                    x = cX - 60  # Adjust as needed
                    y = cY + 40  # Adjust as needed
                    w = 100      # Width of the text area

                # Calculate the area of the contour

                area = cv2.contourArea(contour)
                if 13000 > area > 10000:
                    # Put the area text on the frame, positioned near the bottom right of the contour's bounding rectangle
                    
                    cv2.putText(frame, "Area: " + str(int(area)), (x + w - 60, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
                if len(approxed)==3:     
                    cv2.putText(frame, "Points: " + str(len(approxed)), (x + w - 60, y + 60), cv2.FONT_HERSHEY_COMPLEX, 0.5,(0, 255, 0), 1)
                

            return area , approxed
        
        area, approxed = detection(area,approxed,gameStarted,count)
            # Loop through each contour in the second set of contours
        for contour2 in contours2:
            approx = cv2.approxPolyDP(contour2, 0.01 * cv2.arcLength(contour2, True), True)

            M = cv2.moments(approx)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Define position and size of the text area
                x = cX - 50  # Adjust as needed
                y = cY + 30  # Adjust as needed
                w = 100      # Width of the text area
            # Calculate the area of the contour
            area2 = cv2.contourArea(contour2)
            
            # Put the area text on the frame, positioned near the bottom right of the contour's bounding rectangle
            cv2.putText(frame, "Area: " + str(int(area2)), (x + w - 60, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

        for contour in contours:
            epsilon = 0.03 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(frame, [approx], -1, (255, 0, 0), 2)
        cv2.drawContours(frame,contours2,-1,(255,0,0),10)
        threshold_distance = 10
        if frame is not None:
    
            if contour_touching(contours, contours2,threshold_distance):
                cv2.putText(frame, "PLAY", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)                
                
                
            if not gameStarted:     #enters if condition if it is false because not gameStarted is true(! of false = true)
                print('hello-----------------')
                start_timer(pk)
                print('timer ==========================')
                frame
                gameStarted=True
                count += 1
                
                if count>person.frame:
                    print(person.frame)
                    print("Played more than their limited frame")
                
                table.price=Decimal('0.0')
                table.price=table.price+table.per_frame
                table.save(update_fields=['price'])
                print('-------',table.price)
                        
                print('count',count)

                skip_until=current_time + 1    # time.sleep(1)  # Delay for 1 second
            if start_timer:                                
                Check_timer(pk)
            # ret, jpeg = cv2.imencode('.jpg', frame)
            # frame = jpeg.tobytes()
            # yield (b'--frame\r\n'
            #         b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        # time.sleep(1)  # Delay for 1 second

        # if gameStarted and current_time > skip_until:
        #         print(area,'-----------------------')
        #         if 13000 > area > 10000 and len(approxed)==3:
        #             cv2.putText(frame, "end", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)                
        #             print('*******game end*******')
        #             gameStarted = False
        #             stop_timer(request,pk)
        #             print('boolean:', gameStarted)
        current_frame[pk]=frame
        cache.set('current_frame', current_frame, timeout=2)
        

        # Save frame to database or file system accessible to Django

       
       
# def video_feed(request):
#     return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
# def video_feed(request):
#     cap = cv2.VideoCapture(1, cv2.CAP_MSMF)  # Try CAP_MSMF backend

#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return HttpResponse("Failed to open webcam.")

#     def generate():
#         while True:
#             success, frame = cap.read()
#             if not success:
#                 break
#             else:
#                 # Convert frame to JPEG format
#                 ret, buffer = cv2.imencode('.jpg', frame)
#                 frame = buffer.tobytes()
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

#     # Return StreamingHttpResponse with multipart content type
#     return StreamingHttpResponse(generate(), content_type='multipart/x-mixed-replace; boundary=frame')
def video_feed(request,pk):
    frame_buffer = deque(maxlen=40)  # Buffer for 10 frames to smooth streaming

    def frame_generator(pk):
        while True:
                current_frame = cache.get('current_frame', {})  # Retrieve frame from cache

                if pk in current_frame and isinstance(current_frame[pk], np.ndarray):
                    ret, jpeg = cv2.imencode('.jpg', current_frame[pk])
                    if ret:
                        frame = jpeg.tobytes()
                        frame_buffer.append(frame)  # Add frame to buffer
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
                        


                else:
                    # Wait for a short period before retrying
                    time.sleep(0.03)  # Adjust sleep time as needed
            #     else:
            #         # Yield a frame from buffer if current_frame is not valid
            #         if frame_buffer:
            #             yield frame_buffer[-1]  # Yield the latest frame from buffer

            #         # Yield a placeholder image or a blank frame
            #         blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            #         ret, jpeg = cv2.imencode('.jpg', blank_frame)
            #         if ret:
            #             frame = jpeg.tobytes()
            #             yield (b'--frame\r\n'
            #                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

            # except Exception as e:
            #     print(f"Error in frame_generator: {e}")

    return StreamingHttpResponse(frame_generator(pk), content_type='multipart/x-mixed-replace; boundary=frame')
def index(request):
    return render(request, 'index.html')



def index(request,pk):
    return render(request, 'index.html',{'pk':pk})


#//////////////////////////////////////////////////////TESTING


def background_run(request,pk,pk1):
    global worker_processes
    nodename = f"worker_{pk}@{pk}"
    queue_name = f"queue_{pk}"
    if pk in worker_processes and worker_processes[pk].poll() is None:
        worker_process= worker_processes[pk]
    else:
        worker_process = subprocess.Popen([
            'celery', '-A', 'backend.celery', 'worker',
            '--pool=solo',
            '-Q', queue_name,
            '-n', nodename,
            '-l', 'info'
        ])
        worker_processes[pk] = worker_process

    task = background_video_processing.apply_async((pk, pk1), queue=queue_name)
    response_data={
        "message":"hello",
        "pid":worker_process.pid,
        "task_id": task.id
    }
    return JsonResponse(response_data)
    # return StreamingHttpResponse(background_video_processing(pk,pk1) , content_type='multipart/x-mixed-replace; boundary=frame')





# #//////////////////////////////////////////////////////WEBCAM
# def background_run(request,pk):
#          camera = cv2.VideoCapture(0)  # Use 0 for default webcam

#          return StreamingHttpResponse(background_video_processing(camera,request, pk),content_type='multipart/x-mixed-replace; boundary=frame')

# def video_feed(request,pk):
#     def frame_generator():
#         global current_frame
#         while True:
#             current_frame = cache.get('current_frame')  # Retrieve frame from cach

#             print(';;;',current_frame)
#             if  current_frame is not None :
#                 ret, jpeg = cv2.imencode('.jpg', current_frame)
#                 if ret:
#                     frame = jpeg.tobytes()
#                     yield (b'--frame\r\n'
#                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
#             else:
#                 # Yield a placeholder image or a blank frame if current_frame is not valid
#                 blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
#                 ret, jpeg = cv2.imencode('.jpg', blank_frame)
#                 if ret:
#                     frame = jpeg.tobytes()
#                     yield (b'--frame\r\n'
#                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

#     return StreamingHttpResponse(frame_generator(), content_type='multipart/x-mixed-replace; boundary=frame')

#//////////////////////////////////////////////////////ORIGINAL
# def video_stream(request,pk):
#         return StreamingHttpResponse(video_feed(request, pk),content_type='multipart/x-mixed-replace; boundary=frame')
    



# import cv2
# import numpy as np
# from django.http import StreamingHttpResponse
# from .views import start_timer, stop_timer,Check_timer
# import time
# from django.http import JsonResponse
# from rest_framework.decorators import api_view
# import threading
# from django.http import HttpResponse
# from .models import *
# from django.shortcuts import render ,get_object_or_404
# from django.contrib.sessions.models import Session
# from celery import shared_task
# from django.core.cache import cache

# from django.http import HttpResponseServerError
# from decimal import Decimal

# # Define the lower and upper bounds for red color in HSV space




# # Connect the signal handler
# def detectRedObjects(frame):
#     lower_red1 = np.array([160, 40, 145])
#     upper_red1 = np.array([180, 255, 255])
#     lower_red2 = np.array([160, 40, 145])
#     upper_red2 = np.array([180, 255, 255])

#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
#     mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
#     red_mask = cv2.bitwise_or(mask1, mask2)
#     return red_mask

# def detectObjects(frame):
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     lower_white = np.array([0, 0, 170])
#     upper_white = np.array([180,40, 255])
#     mask = cv2.inRange(hsv, lower_white, upper_white)
#     return mask



# def contour_touching(contour1,contour2,threshold_distance):
#     if contour2 is None:
#         return False
#     for contour1 in contour1:
#         for contour2 in contour2:
#             # Convert contours to numpy arrays

#             contour1 = np.squeeze(contour1)
#             contour2 = np.squeeze(contour2)

#             # Calculate distances between points on the two contours
#             distances = np.linalg.norm(contour1[:, None] - contour2, axis=-1)

#             # Check if minimum distance is less than threshold
#             min_distance = np.min(distances)
#             # print(min_distance)

#             if min_distance < threshold_distance:
#                 return True
#             else:
#                 return False
            

# current_frame_position = 0
# frames=None
# gameStarted = False
# current_frame = None  # Initialize frame as a global variable
# def gen_frames():  # generate frame by frame from camera
#     global current_frame,frames  # Declare frame as global
#     cap = cv2.VideoCapture(cv2.CAP_ANY)  # Use 0 for the default webcam
#     if not cap.isOpened():
#         return

#     try:
#         while True:
#             success, frame = cap.read()
#             if not success:
#                 break

#             ret, jpeg = cv2.imencode('.jpg', frame)
#             if not ret:
#                 break

#             cache.set('current_frame', frame, timeout=None)      
            
#             # frame = jpeg.tobytes()
#             # yield (b'--frame\r\n'
#             #        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
        
#     finally:
#         cap.release()

# def video_feedmain(request):
#     gen_frames()
#     return HttpResponse('hello')
#     # return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')


# # # @shared_task(bind=True)
# # def test(request):
# #     global current_frame, gameStarted,frames
# #     current_frame = cache.get('current_frame')  # Retrieve frame from cach
# #     while True:
# #         print(current_frame)
# #     time.sleep(1)  # Sleep for a second before fetching the next frame to avoid high CPU usage

# #     return HttpResponse("done")


# # def test(request):
# #     test_func.delay()
# #     return HttpResponse("done")

# # @shared_task(bind=True)
# def background_video_processing(pk,pk1):
#     global current_frame, gameStarted

#     # Retrieve session variables
#     person=get_object_or_404(Person,id=pk1)
#     table=get_object_or_404(Table,tableno=pk)
  
#     gameStarted = False  # Ensure gameStarted is initialized
    
#     # print('=======================',current_frame,'===')    
#     skip_frames = 2
#     # frame_count = current_frame_position
#     gameStarted=False
#     count = 0 
#     area=0
#     approxed=0
#     skip_until=0 
#     while True:
#         current_frame = cache.get('current_frame')  # Retrieve frame from cach

#         if current_frame is not None:
#             frame=current_frame
#             # print('=======================',current_frame,'===')    

        
#             current_time=time.time()      #to get the current time
        
#             l=20
#             u=145
            
#             red_mask=detectRedObjects(frame)
#             mask=detectObjects(frame)
#             median_blur= cv2.medianBlur(red_mask,23)
#             median_blur_white= cv2.medianBlur(mask,21)

#             canny=cv2.Canny(median_blur,l,u)
#             canny2=cv2.Canny(median_blur_white,l,u)
#             contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             contours2, hierarchy = cv2.findContours(canny2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             # frame_copy=frame.copy()
            
#             def detection(area,approxed,gameStarted,count):  
#                 for contour in contours:
#                     epsilon = 0.03 * cv2.arcLength(contour, True)
#                     approxed = cv2.approxPolyDP(contour, epsilon, True)
#                     approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

#                     M = cv2.moments(approx)
#                     if M["m00"] != 0:
#                         cX = int(M["m10"] / M["m00"])
#                         cY = int(M["m01"] / M["m00"])

#                         # Define position and size of the text area
#                         x = cX - 60  # Adjust as needed
#                         y = cY + 40  # Adjust as needed
#                         w = 100      # Width of the text area

#                     # Calculate the area of the contour

#                     area = cv2.contourArea(contour)
#                     if 13000 > area > 10000:
#                         # Put the area text on the frame, positioned near the bottom right of the contour's bounding rectangle
                        
#                         cv2.putText(frame, "Area: " + str(int(area)), (x + w - 60, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
#                     if len(approxed)==3:     
#                         cv2.putText(frame, "Points: " + str(len(approxed)), (x + w - 60, y + 60), cv2.FONT_HERSHEY_COMPLEX, 0.5,(0, 255, 0), 1)
                    

#                 return area , approxed
            
#             area, approxed = detection(area,approxed,gameStarted,count)
#                 # Loop through each contour in the second set of contours
#             for contour2 in contours2:
#                 approx = cv2.approxPolyDP(contour2, 0.01 * cv2.arcLength(contour2, True), True)

#                 M = cv2.moments(approx)
#                 if M["m00"] != 0:
#                     cX = int(M["m10"] / M["m00"])
#                     cY = int(M["m01"] / M["m00"])

#                     # Define position and size of the text area
#                     x = cX - 50  # Adjust as needed
#                     y = cY + 30  # Adjust as needed
#                     w = 100      # Width of the text area
#                 # Calculate the area of the contour
#                 area2 = cv2.contourArea(contour2)
                
#                 # Put the area text on the frame, positioned near the bottom right of the contour's bounding rectangle
#                 cv2.putText(frame, "Area: " + str(int(area2)), (x + w - 60, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

#             for contour in contours:
#                 epsilon = 0.03 * cv2.arcLength(contour, True)
#                 approx = cv2.approxPolyDP(contour, epsilon, True)
#                 cv2.drawContours(frame, [approx], -1, (255, 0, 0), 2)
#             cv2.drawContours(frame,contours2,-1,(255,0,0),10)
#             threshold_distance = 10
#             if frame is not None:
        
#                 if contour_touching(contours, contours2,threshold_distance):
#                     cv2.putText(frame, "PLAY", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)                
                    
                    
#                 if not gameStarted:     #enters if condition if it is false because not gameStarted is true(! of false = true)
#                     print('hello-----------------')
#                     start_timer(pk)
#                     print('timer ==========================')
#                     frame
#                     gameStarted=True
#                     count += 1
                    
#                     if count>person.frame:
#                         print(person.frame)
#                         print("Played more than their limited frame")
                    
#                     table.price=Decimal('0.0')
#                     table.price=table.price+table.per_frame
#                     table.save(update_fields=['price'])
#                     print('-------',table.price)
                            
#                     print('count',count)

#                     skip_until=current_time + 1    # time.sleep(1)  # Delay for 1 second
#                 if start_timer:                                
#                     Check_timer(pk)
#                 ret, jpeg = cv2.imencode('.jpg', frame)
#                 frame = jpeg.tobytes()
#                 yield (b'--frame\r\n'
#                         b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
#         # time.sleep(1)  # Delay for 1 second

#         # if gameStarted and current_time > skip_until:
#         #         print(area,'-----------------------')
#         #         if 13000 > area > 10000 and len(approxed)==3:
#         #             cv2.putText(frame, "end", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)                
#         #             print('*******game end*******')
#         #             gameStarted = False
#         #             stop_timer(request,pk)
#         #             print('boolean:', gameStarted)

#         # Save frame to database or file system accessible to Django
        
       
       
# # def video_feed(request):
# #     return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
# # def video_feed(request):
# #     cap = cv2.VideoCapture(1, cv2.CAP_MSMF)  # Try CAP_MSMF backend

# #     if not cap.isOpened():
# #         print("Error: Could not open webcam.")
# #         return HttpResponse("Failed to open webcam.")

# #     def generate():
# #         while True:
# #             success, frame = cap.read()
# #             if not success:
# #                 break
# #             else:
# #                 # Convert frame to JPEG format
# #                 ret, buffer = cv2.imencode('.jpg', frame)
# #                 frame = buffer.tobytes()
# #                 yield (b'--frame\r\n'
# #                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# #     # Return StreamingHttpResponse with multipart content type
# #     return StreamingHttpResponse(generate(), content_type='multipart/x-mixed-replace; boundary=frame')
# def video_feed(request):
#     def frame_generator():
#         global current_frame
#         while True:
#             current_frame = cache.get('current_frame')  # Retrieve frame from cach

#             print(';;;',current_frame)
#             if  current_frame is not None :
#                 ret, jpeg = cv2.imencode('.jpg', current_frame)
#                 if ret:
#                     frame = jpeg.tobytes()
#                     yield (b'--frame\r\n'
#                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
#             else:
#                 # Yield a placeholder image or a blank frame if current_frame is not valid
#                 blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
#                 ret, jpeg = cv2.imencode('.jpg', blank_frame)
#                 if ret:
#                     frame = jpeg.tobytes()
#                     yield (b'--frame\r\n'
#                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

#     return StreamingHttpResponse(frame_generator(), content_type='multipart/x-mixed-replace; boundary=frame')
# def index(request):
#     return render(request, 'index.html')



# def index(request,pk):
#     return render(request, 'index.html',{'pk':pk})


