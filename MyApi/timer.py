from .models import *
import time
import datetime
from decimal import Decimal
from .views import start_timer, stop_timer,Check_timer
from django.dispatch import Signal
from .arduino import button_state_changed, read_button_state
import serial
import threading
from django.shortcuts import render ,get_object_or_404
from django.http import JsonResponse
from decimal import Decimal

from django.http import HttpResponseServerError

# global_button_state = ""
def timer(request,pk):
    table=get_object_or_404(Table,tableno=pk)



    arduino_port = 'COM7'  # Replace with your Arduino serial port
    baud_rate = 9600
    try:
        print("========")

        serial_port = serial.Serial(arduino_port, baud_rate, timeout=1)
        print(f"Connected to {arduino_port} at {baud_rate} baud.")
        print("========")

    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        return HttpResponseServerError("Error opening serial port")

    # Start reading button state in a separate thread
    threading.Thread(target=read_button_state, args=(serial_port,), daemon=True).start()
    global_button_state = request.session.get('global_button_state', '')
    print("========")

    if global_button_state == "0":
            print("Button pressed")
            start_timer(request, pk)
            # Handle button pressed action here
            # Example: Start a timer, update game state, etc.
    elif global_button_state == "1":
        print("Button released")
        stop_timer(request,pk)
    global_button_state = ""
    print("========")
    if start_timer is not None:
        Check_timer(request,pk) 
        # table.elapsed_time = Decimal(elapsed_time)
        if table.rate is None:
            table.rate = Decimal('0.0') 
        if table.elapsed_time:
            table.price = (table.rate * table.elapsed_time) / Decimal(60)
        else:
            table.price = Decimal('0.0')
    return JsonResponse({'status': 'Timer' })
