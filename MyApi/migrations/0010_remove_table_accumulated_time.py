# Generated by Django 5.0.4 on 2024-06-10 11:56

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('MyApi', '0009_table_end_time'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='table',
            name='accumulated_time',
        ),
    ]