# Generated by Django 5.0.6 on 2024-07-01 07:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('MyApi', '0054_alter_person_email'),
    ]

    operations = [
        migrations.AddField(
            model_name='table',
            name='Perframe',
            field=models.DecimalField(blank=True, decimal_places=2, max_digits=7, null=True),
        ),
    ]