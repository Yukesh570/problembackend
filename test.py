
import os
import django

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')  # Replace 'myproject.settings' with your actual settings module
django.setup()


from MyApi.models import Person
from django.shortcuts import  get_object_or_404


person=get_object_or_404(Person,id=3)
# print(person.objects.all())
# print('hi')
def say_hi():
    print(Person.objects.all())

    # print("hiasdfasdf")  # This will print "hi" to the console
    # return HttpResponse("hi")
say_hi()
