from rest_framework import serializers
from .models import Products


class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = Products
        fields = [
            "id",
            "name",
            "price",
            "gender",
            "category",
            "color",
            "length",
            "fit",
            "activity",
            "fabric",
            "description",
            "image1_url",
            "image2_url",
            "image3_url",
        ]
