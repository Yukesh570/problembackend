# Generated by Django 5.0.4 on 2024-06-11 06:03

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('MyApi', '0014_alter_person_tabletype'),
    ]

    operations = [
        migrations.AlterField(
            model_name='person',
            name='tabletype',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, to='MyApi.table'),
        ),
    ]
