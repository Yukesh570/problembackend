# Generated by Django 5.0.6 on 2024-06-24 10:58

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('MyApi', '0036_alter_table_frame_time_limit'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='table',
            name='frame_time_limit',
        ),
    ]