# Generated by Django 5.0.6 on 2024-07-02 06:52

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('MyApi', '0061_rename_per_frame_table_per_frame'),
    ]

    operations = [
        migrations.AddField(
            model_name='table',
            name='frame_limit',
            field=models.DurationField(blank=True, null=True),
        ),
    ]