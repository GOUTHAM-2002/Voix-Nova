# Generated by Django 5.1.3 on 2024-11-25 17:47

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0002_remove_products_size_remove_products_size_type'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='products',
            name='collection',
        ),
    ]
