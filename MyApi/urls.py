from django.urls import path
from . import  views , python ,timer, arduino
from rest_framework.routers import DefaultRouter



urlpatterns = [
    path('userregister/', views.registerUser, name='registerUser'),
    path('registerTable/', views.registerTable, name='registerTable '),

    path('updatetable/<str:pk>/', views.updatetable, name='updatetable'),
    path('updatewholetable/<str:pk>/', views.updatewholetable, name='updatewholetable'),

    path('chooseGame/<str:pk>/', views.chooseGame, name='chooseGame'),
    path('video_feed/<int:pk>/', python.video_feed, name='video_feed'),
    # path('video_feedmain/', python.video_feedmain, name='video_feedmain'),

    path('timer_video_feed/<int:pk>/', arduino.timer_video_feed, name='timer_video_feed'),

    path('gettable/<str:pk>/', views.gettable, name='gettable'),
    path('getalltable/', views.getalltable, name='getalltable'),
    path('timer/<str:pk>/', timer.timer, name='timer'),
    # path('txt/', python.test, name='test_func'),

    path('index/', views.index, name='index'),
    path('arduino/<str:pk>/', arduino.call_read_button_state, name='call_read_button_state'),
    path('start_timer/<str:pk>/', views.start_timer, name='start_timer'),

    path('stop_timer/<str:pk>/', views.stop_timer, name='stop_timer'),
    path('background_run/<int:pk>/<int:pk1>/', python.background_run, name='video_stream'),
    #//
    # path('video_feed/<int:pk>/', python.video_feed, name='video_feed'),
    #//
    path('game_start/<int:pk>/', python.background_run, name='video_stream'),

    path('deletetable/<int:pk>/', views.deletetable, name='deletetable'),

]
   