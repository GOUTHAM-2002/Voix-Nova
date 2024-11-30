from django.urls import path
from .views import (
    home,
    group_search_view,
    particular_search_view,
    add_to_cart,
    finalize_cart,
    get_all_products,
    get_cart_products,
    home_page_conversationalist,
)

urlpatterns = [
    path(
        "api/home_page_conversationalist/",
        home_page_conversationalist,
        name="home_page_conversationalist",
    ),
    path("api/get_all_products/", get_all_products, name="get_all_products"),
    path("api/group_search/", group_search_view, name="group_search"),
    path("api/particular_search/", particular_search_view, name="particular_search"),
    path("api/add_to_cart/", add_to_cart, name="add_to_cart"),
    path("api/finalize_cart/", finalize_cart, name="finalize_cart"),
    path("api/get_cart_products/", get_cart_products, name="get_cart_products"),
    path("home/", home, name="home"),
]
