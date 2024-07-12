from django.contrib import admin

# Register your models here.

from .models import *



class PersonAdmin(admin.ModelAdmin):
    list_display = ['email','Name','played_table']
class TableAdmin(admin.ModelAdmin):
    list_display=['tableno','is_running','person']
admin.site.register(Person,PersonAdmin) 
admin.site.register(Table,TableAdmin)
