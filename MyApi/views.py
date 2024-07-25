from django.shortcuts import render ,get_object_or_404
from .models import *
from django.core.mail import send_mail 
from django.conf import settings
from .serializers import *
from django.utils.timezone import now
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.db import IntegrityError
from django.http import JsonResponse
import time
from decimal import Decimal
import datetime
from django.http import StreamingHttpResponse
import cv2
from datetime import timedelta
from django.core.validators import validate_email
from django.http import HttpResponse
from django.utils import timezone

# from .python import background_video_processing



def parse_duration(duration_str):
    try:
        parts = duration_str.split(':')
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
            return datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
        return None
    except:
        return None
    
def gen(camera):
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_feed(request):
    return StreamingHttpResponse(gen(cv2.VideoCapture(0)),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

def index(request):
    return render(request, 'index.html')


    
# def index(request):
#     if request.method=='POST':
#         email=request.POST['email']
#         send_mail('SnookerGame',
#                   'You just requested to play the game',
#                   'settings.EMAIL_HOST_USER',
#                   [email],
#                   fail_silently=False,
#                   )
        
#     return render(request,'index.html')


@api_view(['POST'])
def registerUser(request):
    data=request.data

    try:
        user= Person.objects.create(
            Name=data['name'],
            Address=data['address'],
            Phonenumber=data['phonenumber'],
            email=data['email'],
            frame=data['frame'],
            played_table=data['tableno'],
        )
        print(data['tableno'])
        print(Table.objects.filter(tableno=int(data['tableno'])))
        Table.objects.filter(tableno=int(data['tableno'])).update(person=user)
        serializer=PersonaldataSerializer(user,many=False)
        return Response(serializer.data)
    
    except:
        message={'detail':'User with this email already exists'}
        return Response(message,status=status.HTTP_400_BAD_REQUEST)
    

    
@api_view(['POST'])
def registerTable(request):
    data=request.data  
    try:
        frame_limit_duration = parse_duration(data['frame_limit'])

        table_detail= Table.objects.create(
        tableno=data['tableno'],
            # persondetail=person_detail,
        per_frame=data['per_frame'],
        rate=data['rate'],
        # frame_time_limit=data['frame_time_limit'],
        frame_limit=frame_limit_duration,
        ac=data['ac'],
     

        )
        serializers=TableSerializer(table_detail,many=False)
       
        return Response(serializers.data, status=status.HTTP_201_CREATED)
  
    except : 

        return Response({'detail': 'Table cannot be booked'}, status=status.HTTP_400_BAD_REQUEST)

@api_view(['PUT'])
def updatetable(request,pk):
    data=request.data
    try:
        table_instance = Table.objects.get(tableno=pk)
        table_instance.is_running = data.get('is_running', table_instance.is_running)

        table_instance.save()
        serializer = TableSerializer(table_instance, many=False)
        return Response(serializer.data, status=status.HTTP_200_OK)
    
    except Table.DoesNotExist:
        return Response({'detail': 'Table not found'}, status=status.HTTP_404_NOT_FOUND)
    except Person.DoesNotExist:
        return Response({'detail': 'Associated Person not found'}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({'detail': str(e)}, status=status.HTTP_400_BAD_REQUEST)

    
@api_view(['PUT'])
def chooseGame(request,pk):
    data=request.data
    try:
        table_instance = Table.objects.get(tableno=pk)
        table_instance.frame_based = data.get('frame_based', table_instance.frame_based)
        table_instance.time_based = data.get('time_based', table_instance.time_based)

        table_instance.save()
        serializer = TableSerializer(table_instance, many=False)
        return Response(serializer.data, status=status.HTTP_200_OK)
    
    except Table.DoesNotExist:
        return Response({'detail': 'Table not found'}, status=status.HTTP_404_NOT_FOUND)
    except Person.DoesNotExist:
        return Response({'detail': 'Associated Person not found'}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({'detail': str(e)}, status=status.HTTP_400_BAD_REQUEST)


@api_view(['PUT'])
def updatewholetable(request,pk):
    data=request.data
    try:
        frame_limit_duration = parse_duration(data['frame_limit'])

        table_instance = Table.objects.get(tableno=pk)
        table_instance.rate = data.get('rate', table_instance.rate)
        table_instance.frame_limit = frame_limit_duration
        table_instance.ac = data.get('ac', table_instance.ac)
        table_instance.per_frame = data.get('per_frame', table_instance.per_frame)

        table_instance.save()
        serializer = TableSerializer(table_instance, many=False)
        return Response(serializer.data, status=status.HTTP_200_OK)
    
    except Table.DoesNotExist:
        return Response({'detail': 'Table not found'}, status=status.HTTP_404_NOT_FOUND)
    except Person.DoesNotExist:
        return Response({'detail': 'Associated Person not found'}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({'detail': str(e)}, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
def gettable(request,pk):
    table=Table.objects.get(tableno=pk)
    serializers=TableSerializer(table,many=False)
    return Response(serializers.data)


        
@api_view(['GET'])
def getalltable(request):
    table=Table.objects.all()
    serializers=TableSerializer(table,many=True)
    return Response(serializers.data)



# def index(request):
#     return render(request,'index.html')
@api_view(['PUT'])
def deletetable(request,pk):
    try:
        table = Table.objects.get(pk=pk)
        table.person = None
        table.price = None
        table.start_time = None
        table.elapsed_time = None
        table.end_time = None
        table.button = False
        table.inactive = False
        table.is_running = False
        table.played_time = None
        table.frame_based = False
        table.time_based = False

        table.save()
        return Response({"msg":"Table data have been deleted"}, status=status.HTTP_204_NO_CONTENT)
    except Table.DoesNotExist:
        return Response({'error': 'Table not found'}, status=status.HTTP_404_NOT_FOUND)


def start_timer(pk):
    table =get_object_or_404(Table,tableno=pk)
    if table.start_time is not None:
       return JsonResponse({'status': 'Timer already started'})
    # request.session['start_time'] = time.time()
    # table.start_time= time.time()
    current_time = timezone.now()    
    table.start_time = current_time
    # table.start_time= timezone.now()
    # table.is_running =True
    table.save()
    print('=====================================================stared',table.start_time)


# def stop_timer(request,pk):
    
#     if 'start_time' in request.session:
#         start_time = request.session.pop('start_time')

#         elapsed_time = time.time() - start_time
#         table = get_object_or_404(Table, id=pk)  # Replace with the correct logic to identify the Table instance
#         table.is_running = False
#         table.time = Decimal(elapsed_time)
#         table.end_time= now()
#         table.price=(table.rate * table.time)/Decimal(60)
#         table.save()
#         return JsonResponse({'status': 'Timer stopped', 'elapsed_time': elapsed_time})
#     else:
#         return JsonResponse({'status': 'No timer started'})
    
def stop_timer(request,pk):
        table = get_object_or_404(Table, tableno=pk)  # Replace with the correct logic to identify the Table instance
        # current_time = time.time()  # Get current time in seconds since the epoch
        # elapsed_time = current_time - table.start_time.timestamp() 
        # table.is_running = False
        # table.elapsed_time = Decimal(elapsed_time)
        table.end_time= now()
        # table.price=(table.rate * table.elapsed_time)/Decimal(60)
        # total_seconds = int(table.elapsed_time)
        # hours, remainder = divmod(total_seconds, 3600)
        # minutes, seconds = divmod(remainder, 60)
        # table.played_time = f"{hours:02}:{minutes:02}:{seconds:02}"
        table.save()
        print('working')
        return JsonResponse({'status': 'Timer stopped' })

# def stop_running(request, pk):
#     table = get_object_or_404(Table, tableno=pk)  # Get the table instance by its tableno
    
#     if table.is_running:  # Check if the table is actually running
#         table.is_running = False  # Stop the table running state
#         table.save()  # Save the updated state to the database
#         return JsonResponse({'status': 'Timer stopped'})
#     else:
#         return JsonResponse({'status': 'Timer was not running'})

def Check_timer(pk):
    table=get_object_or_404(Table,tableno=pk)
    if table.start_time is None:
        return JsonResponse({'error': 'Start time is not set'}, status=400)
    current_time = time.time()  # Get current time in seconds since the epoch
    elapsed_time = current_time - table.start_time.timestamp() 
    table.elapsed_time = Decimal(elapsed_time) 
    total_seconds = int(table.elapsed_time)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    table.played_time = f"{hours:02}:{minutes:02}:{seconds:02}"
    played_time_str=table.played_time
    #convert played_time to timedelta
    hours,minutes,seconds= map(int,played_time_str.split(':'))
    played_time_td=timedelta(hours=hours,minutes=minutes,seconds=seconds)
    if played_time_td == table.frame_limit:
        print('stop')
    table.save()


    
def watch_timer(pk):
    table=get_object_or_404(Table,tableno=pk)
    if table.start_time is None:
        return JsonResponse({'error': 'Start time is not set'}, status=400)
    current_time = time.time()  # Get current time in seconds since the epoch
    elapsed_time = current_time - table.start_time.timestamp() 
    table.elapsed_time = Decimal(elapsed_time) 
    table.price=(table.rate * table.elapsed_time)/Decimal(60)
    print('---------',table.price)

    total_seconds = int(table.elapsed_time)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    table.played_time = f"{hours:02}:{minutes:02}:{seconds:02}"
    played_time_str=table.played_time
    #convert played_time to timedelta
    hours,minutes,seconds= map(int,played_time_str.split(':'))
    
    table.save()

  
