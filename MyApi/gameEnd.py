#////////////////////////////////////////////////////////////////// ORIGINAL

# def video_feed(request, pk):
#     global current_frame_position

#     video_path = r'C:\Users\Yukesh\Downloads\snookervideo\comined2.mp4'  # Replace with your video file path

   
#     print('==========================',pk)
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#             raise IOError("Video file cannot be opened.")
        
#     skip_frames = 2
#     frame_count = current_frame_position
#     gameStarted=False
#     count = 0 
#     area=0
#     approxed=0
#     skip_until=0 
     
#     while cap.isOpened():
#         current_time=time.time()      #to get the current time
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Skip frames
#         frame_count += 1
#         current_frame_position = frame_count  # Update the global frame position
#         if frame_count % skip_frames != 0:
#             continue

#         #retreive values
#         frame=cv2.resize(frame,(500,500))
#         # l = cv2.getTrackbarPos('Lower','Trackbars')
#         # u = cv2.getTrackbarPos('Upper', 'Trackbars')
#         l=20
#         u=145
#         # print(l,u)
#         # imgBlur = cv2.GaussianBlur(frame, (25, 25), 1)

#         # dst = cv2.fastNlMeansDenoisingColored(frame,None,15,15,3,10)
#         red_mask=detectRedObjects(frame)
#         mask=detectObjects(frame)
#         median_blur= cv2.medianBlur(red_mask,23)
#         median_blur_white= cv2.medianBlur(mask,21)
#         # median_blur = median_blur.astype(np.uint8)
#         # median_blur_white = median_blur_white.astype(np.uint8)

#         # imgGray = cv2.cvtColor(median_blur, cv2.COLOR_BGR2GRAY)
#         canny=cv2.Canny(median_blur,l,u)
#         canny2=cv2.Canny(median_blur_white,l,u)
#         contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         contours2, hierarchy = cv2.findContours(canny2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         frame_copy=frame.copy()
       
#         def detection(area,approxed,gameStarted,count):  
#             for contour in contours:
#                 epsilon = 0.03 * cv2.arcLength(contour, True)
#                 approxed = cv2.approxPolyDP(contour, epsilon, True)
#                 approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

#                 M = cv2.moments(approx)
#                 if M["m00"] != 0:
#                     cX = int(M["m10"] / M["m00"])
#                     cY = int(M["m01"] / M["m00"])

#                     # Define position and size of the text area
#                     x = cX - 60  # Adjust as needed
#                     y = cY + 40  # Adjust as needed
#                     w = 100      # Width of the text area

#                 # Calculate the area of the contour

#                 area = cv2.contourArea(contour)
#                 if 13000 > area > 10000:
#                     # Put the area text on the frame, positioned near the bottom right of the contour's bounding rectangle
                    
#                     cv2.putText(frame, "Area: " + str(int(area)), (x + w - 60, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
#                 if len(approxed)==3:     
#                     cv2.putText(frame, "Points: " + str(len(approxed)), (x + w - 60, y + 60), cv2.FONT_HERSHEY_COMPLEX, 0.5,(0, 255, 0), 1)
                

#                 # print('-----boolean----:', gameStarted)

#                 # if gameStarted :
#                 #     print(area,'-----------------------',len(approxed))
#                 #     if 13000 > area > 10000 and len(approxed)==3:


#                 #         cv2.putText(frame, "end", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)                
#                 #         print('*******game end*******')
#                 #         count += 1 
#                 #         print('count',count)

#                 #         gameStarted = False
#                 #         print('boolean:', gameStarted)
#             return area , approxed
        
#         area, approxed = detection(area,approxed,gameStarted,count)
# # Loop through each contour in the second set of contours
#         for contour2 in contours2:
#             approx = cv2.approxPolyDP(contour2, 0.01 * cv2.arcLength(contour2, True), True)

#             M = cv2.moments(approx)
#             if M["m00"] != 0:
#                 cX = int(M["m10"] / M["m00"])
#                 cY = int(M["m01"] / M["m00"])

#                 # Define position and size of the text area
#                 x = cX - 50  # Adjust as needed
#                 y = cY + 30  # Adjust as needed
#                 w = 100      # Width of the text area
#             # Calculate the area of the contour
#             area2 = cv2.contourArea(contour2)
            
#             # Put the area text on the frame, positioned near the bottom right of the contour's bounding rectangle
#             cv2.putText(frame, "Area: " + str(int(area2)), (x + w - 60, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

#         for contour in contours:
#             epsilon = 0.03 * cv2.arcLength(contour, True)
#             approx = cv2.approxPolyDP(contour, epsilon, True)
#             cv2.drawContours(frame, [approx], -1, (255, 0, 0), 2)
#         cv2.drawContours(frame,contours2,-1,(255,0,0),10)
      

    
        
#         # cv2.drawContours(frame_copy,contours2,-1,(255,0,0),10)
#         # combined_frame = np.hstack((frame,frame_copy))
        
#         threshold_distance = 10
#     # Adjust this value according to your requirement

#         # Check if the frame is not empty
#         if frame is not None:
#             # cv2.imshow('binary video',dst)
#             # cv2.imshow('median video',median_blur_white)
#             if contour_touching(contours, contours2,threshold_distance):
#                 cv2.putText(frame, "PLAY", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)                
                
                
#                 if not gameStarted:     #enters if condition if it is true so not gameStarted is true(! of false = true)
#                     start_timer(request, pk)
#                     print('timer started==========================')
#                     frame
#                     gameStarted=True
#                     count += 1
#                     print('count',count)
#                     # time.sleep(1)  # Delay for 1 second

#                 skip_until=current_time + 1   

#             if gameStarted and current_time > skip_until:
#                     print(area,'-----------------------')
#                     if 13000 > area > 10000 and len(approxed)==3:
#                         cv2.putText(frame, "end", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)                
#                         print('*******game end*******')
#                         gameStarted = False
#                         stop_timer(request,pk)
#                         print('boolean:', gameStarted)

                    
#             # if masks_overlap(red_mask, mask):
#             #     print("Play")
#             # else:
#             #     print("Game Over")
#                 # cv2.putText(frame, "PLAY", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             if request.path == f'/api/video_stream/{pk}/':
#                 ret, jpeg = cv2.imencode('.jpg', frame)
#                 frame = jpeg.tobytes()
#                 yield (b'--frame\r\n'
#                     b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
         
#             # cv2.imshow('blur', imgGray)
#             # cv2.imshow('binary video',canny2)

#             # cv2.imshow('Input', frame_copy)
#             # cv2.imshow('red', median_blur)

#             # cv2.imshow('white', median_blur_white)
        
                    

#         # Press 'q' to exit the loop
        

#     # Release the video capture object and close all windows
#     cap.release()
    

#////////////////////////////////////////////////////////////////// TESTING



# def background_video_processing(request, pk):
#     global current_frame, current_frame_position, gameStarted

#     video_path = r'C:\Users\Yukesh\Downloads\snookervideo\comined2.mp4'  # Replace with your video file path

   
#     print('==========================',pk)
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#             raise IOError("Video file cannot be opened.")
        
#     skip_frames = 2
#     frame_count = current_frame_position
#     gameStarted=False
#     count = 0 
#     area=0
#     approxed=0
#     skip_until=0 
     
#     while cap.isOpened():
#         current_time=time.time()      #to get the current time
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Skip frames
#         frame_count += 1
#         current_frame_position = frame_count  # Update the global frame position
#         if frame_count % skip_frames != 0:
#             continue

#         #retreive values
#         frame=cv2.resize(frame,(500,500))
#         # l = cv2.getTrackbarPos('Lower','Trackbars')
#         # u = cv2.getTrackbarPos('Upper', 'Trackbars')
#         l=20
#         u=145
#         # print(l,u)
#         # imgBlur = cv2.GaussianBlur(frame, (25, 25), 1)

#         # dst = cv2.fastNlMeansDenoisingColored(frame,None,15,15,3,10)
#         red_mask=detectRedObjects(frame)
#         mask=detectObjects(frame)
#         median_blur= cv2.medianBlur(red_mask,23)
#         median_blur_white= cv2.medianBlur(mask,21)
#         # median_blur = median_blur.astype(np.uint8)
#         # median_blur_white = median_blur_white.astype(np.uint8)

#         # imgGray = cv2.cvtColor(median_blur, cv2.COLOR_BGR2GRAY)
#         canny=cv2.Canny(median_blur,l,u)
#         canny2=cv2.Canny(median_blur_white,l,u)
#         contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         contours2, hierarchy = cv2.findContours(canny2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         frame_copy=frame.copy()
       
#         def detection(area,approxed,gameStarted,count):  
#             for contour in contours:
#                 epsilon = 0.03 * cv2.arcLength(contour, True)
#                 approxed = cv2.approxPolyDP(contour, epsilon, True)
#                 approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

#                 M = cv2.moments(approx)
#                 if M["m00"] != 0:
#                     cX = int(M["m10"] / M["m00"])
#                     cY = int(M["m01"] / M["m00"])

#                     # Define position and size of the text area
#                     x = cX - 60  # Adjust as needed
#                     y = cY + 40  # Adjust as needed
#                     w = 100      # Width of the text area

#                 # Calculate the area of the contour

#                 area = cv2.contourArea(contour)
#                 if 13000 > area > 10000:
#                     # Put the area text on the frame, positioned near the bottom right of the contour's bounding rectangle
                    
#                     cv2.putText(frame, "Area: " + str(int(area)), (x + w - 60, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
#                 if len(approxed)==3:     
#                     cv2.putText(frame, "Points: " + str(len(approxed)), (x + w - 60, y + 60), cv2.FONT_HERSHEY_COMPLEX, 0.5,(0, 255, 0), 1)
                

#                 # print('-----boolean----:', gameStarted)

#                 # if gameStarted :
#                 #     print(area,'-----------------------',len(approxed))
#                 #     if 13000 > area > 10000 and len(approxed)==3:


#                 #         cv2.putText(frame, "end", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)                
#                 #         print('*******game end*******')
#                 #         count += 1 
#                 #         print('count',count)

#                 #         gameStarted = False
#                 #         print('boolean:', gameStarted)
#             return area , approxed
        
#         area, approxed = detection(area,approxed,gameStarted,count)
# # Loop through each contour in the second set of contours
#         for contour2 in contours2:
#             approx = cv2.approxPolyDP(contour2, 0.01 * cv2.arcLength(contour2, True), True)

#             M = cv2.moments(approx)
#             if M["m00"] != 0:
#                 cX = int(M["m10"] / M["m00"])
#                 cY = int(M["m01"] / M["m00"])

#                 # Define position and size of the text area
#                 x = cX - 50  # Adjust as needed
#                 y = cY + 30  # Adjust as needed
#                 w = 100      # Width of the text area
#             # Calculate the area of the contour
#             area2 = cv2.contourArea(contour2)
            
#             # Put the area text on the frame, positioned near the bottom right of the contour's bounding rectangle
#             cv2.putText(frame, "Area: " + str(int(area2)), (x + w - 60, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

#         for contour in contours:
#             epsilon = 0.03 * cv2.arcLength(contour, True)
#             approx = cv2.approxPolyDP(contour, epsilon, True)
#             cv2.drawContours(frame, [approx], -1, (255, 0, 0), 2)
#         cv2.drawContours(frame,contours2,-1,(255,0,0),10)
      

    
        
#         # cv2.drawContours(frame_copy,contours2,-1,(255,0,0),10)
#         # combined_frame = np.hstack((frame,frame_copy))
        
#         threshold_distance = 10
#     # Adjust this value according to your requirement

#         # Check if the frame is not empty
#         if frame is not None:
#             # cv2.imshow('binary video',dst)
#             # cv2.imshow('median video',median_blur_white)
#             if contour_touching(contours, contours2,threshold_distance):
#                 cv2.putText(frame, "PLAY", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)                
                
                
#                 if not gameStarted:     #enters if condition if it is true so not gameStarted is true(! of false = true)
#                     start_timer(request, pk)
#                     print('timer started==========================')
#                     frame
#                     gameStarted=True
#                     count += 1
#                     print('count',count)
#                     # time.sleep(1)  # Delay for 1 second

#                 skip_until=current_time + 1   

#             if gameStarted and current_time > skip_until:
#                     print(area,'-----------------------')
#                     if 13000 > area > 10000 and len(approxed)==3:
#                         cv2.putText(frame, "end", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)                
#                         print('*******game end*******')
#                         gameStarted = False
#                         stop_timer(request,pk)
#                         print('boolean:', gameStarted)

                    
#             # if masks_overlap(red_mask, mask):
#             #     print("Play")
#             # else:
#             #     print("Game Over")
#                 # cv2.putText(frame, "PLAY", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             # if request.path == f'/api/video_stream/{pk}/':
#             #     ret, jpeg = cv2.imencode('.jpg', frame)
#             #     frame = jpeg.tobytes()
#             #     yield (b'--frame\r\n'
#             #         b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
         
#             # cv2.imshow('blur', imgGray)
#             # cv2.imshow('binary video',canny2)

#             # cv2.imshow('Input', frame_copy)
#             # cv2.imshow('red', median_blur)

#             # cv2.imshow('white', median_blur_white)
        
                    

#         # Press 'q' to exit the loop
        
#         current_frame = frame  # Update the global current frame

#     # Release the video capture object and close all windows
#     cap.release()
# def video_feed(request, pk):

#     def frame_generator():
#         global current_frame
#         while True:
#             if current_frame is not None:
#                 ret, jpeg = cv2.imencode('.jpg', current_frame)
#                 frame = jpeg.tobytes()
#                 yield (b'--frame\r\n'
#                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

#     return StreamingHttpResponse(frame_generator(), content_type='multipart/x-mixed-replace; boundary=frame')

