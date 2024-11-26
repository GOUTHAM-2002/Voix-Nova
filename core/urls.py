# your_app/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r"products", views.ProductViewSet, basename="product")
router.register(r"vector-search", views.VectorSearchViewSet, basename="vector-search")

urlpatterns = [
    path("api/", include(router.urls)),
]
