# Generated by Django 5.0.6 on 2024-06-26 10:56

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('MyApi', '0046_remove_table_frame_remove_table_is_running_and_more'),
    ]

    operations = [
        migrations.RenameField(
            model_name='person',
            old_name='tabletype',
            new_name='tabletype_id',
        ),
    ]