from rest_framework import serializers
from .models import Person ,Table


class PersonaldataSerializer(serializers.ModelSerializer):
    class Meta:
        model=Person
        fields='__all__'




class TableSerializer(serializers.ModelSerializer):
    person = PersonaldataSerializer()

    class Meta:
        model=Table
        fields='__all__'
    # def update(self, instance, validated_data):
    #         # Extract nested data for persondetail
    #         person_data = validated_data.pop('persondetail', None)
            
    #         # Update Table instance
    #         for attr, value in validated_data.items():
    #             setattr(instance, attr, value)
    #         instance.save()

    #         # Update or create persondetail if it exists in the data
    #         if person_data:
    #             person_instance, created = Person.objects.update_or_create(
    #                 pk=instance.persondetail.pk,
    #                 defaults=person_data
    #             )

    #         return instance


