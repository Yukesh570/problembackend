# Generated by Django 5.0.6 on 2024-06-20 06:19

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('MyApi', '0025_alter_table_table_type'),
    ]

    operations = [
        migrations.AlterField(
            model_name='table',
            name='time',
            field=models.TimeField(blank=True, null=True),
        ),
    ]
