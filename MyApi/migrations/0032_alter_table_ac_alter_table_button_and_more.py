# Generated by Django 5.0.6 on 2024-06-24 10:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('MyApi', '0031_alter_table_button_alter_table_inactive'),
    ]

    operations = [
        migrations.AlterField(
            model_name='table',
            name='ac',
            field=models.BooleanField(default=False),
        ),
        migrations.AlterField(
            model_name='table',
            name='button',
            field=models.BooleanField(default=False),
        ),
        migrations.AlterField(
            model_name='table',
            name='inactive',
            field=models.BooleanField(default=False),
        ),
    ]