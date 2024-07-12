# Generated by Django 5.0.6 on 2024-07-10 07:17

import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('MyApi', '0066_remove_table_id_alter_table_tableno'),
    ]

    operations = [
        migrations.AlterField(
            model_name='table',
            name='end_time',
            field=models.DateTimeField(blank=True, default=django.utils.timezone.now, null=True),
        ),
        migrations.AlterField(
            model_name='table',
            name='start_time',
            field=models.DateTimeField(blank=True, default=django.utils.timezone.now, null=True),
        ),
    ]