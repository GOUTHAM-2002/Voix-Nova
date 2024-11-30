from django.db.models import Q
from django.shortcuts import render
from django.http import JsonResponse
from .models import Products, Cart, PreviousOrders
from .ai_model import GeminiClient
from .serializers import ProductSerializer
from .tfidf import tfidf_search
from rest_framework.decorators import api_view
import json
from django.views.decorators.csrf import csrf_exempt

from .utils import find_similar_images

ai = GeminiClient()


def home(request):
    similar_products = find_similar_images('core/product_images','core/product_images/product_1_image1.jpg')
    print(len(similar_products))
    print(similar_products)
    return render(request, "search.html")


def home_page_conversationalist(request):
    if request.method == "GET":
        transcript = ai.home_page()
        return JsonResponse({"message": transcript})
    return JsonResponse({"message": "Invalid request"})


def get_all_products(request):
    print("here")
    products = Products.objects.all()
    serializer = ProductSerializer(products, many=True)
    return JsonResponse(serializer.data, safe=False)


def group_search_view(request):
    query = request.GET.get("q", "")
    print(f"Search Query: {query}")
    results = tfidf_search(query)
    relevant_data = str(results["results"])
    print(len(results["results"]))
    return JsonResponse(
        {
            "query": query,
            "results": results.get("results", []),
        },
        safe=False,
    )


def particular_search_view(request):
    query = request.GET.get("q", "")
    print(f"Search Query: {query}")
    results = tfidf_search(query)
    temp = results["results"]
    results["results"] = temp[0]
    return JsonResponse(
        {
            "query": query,
            "results": results.get("results", []),
        },
        safe=False,
    )


def add_to_cart(request):
    if request.method == "GET":
        query = request.GET.get("q", "")
        print(f"Search Query bsias: {query}")

        # Perform the search
        results = tfidf_search(query)["results"]

        if not results:
            return JsonResponse(
                {"message": "No products found to add to cart"}, status=404
            )

        add_to_cart_id = results[0]["id"]
        product = Products.objects.get(id=add_to_cart_id)
        cart_item = Cart(product=product)
        cart_item.save()

        # Serialize the product data to return
        product_data = ProductSerializer(product).data
        return JsonResponse(
            {"message": "Added to cart", "product": product_data}, status=201
        )
    else:
        return JsonResponse({"message": "Invalid request method"}, status=405)


def finalize_cart(request):
    # Ensure the request method is GET
    if request.method == "GET":
        # Retrieve all cart items
        cart_items = Cart.objects.all()

        if not cart_items.exists():
            return JsonResponse(
                {"total_cost": 0.0, "message": "Cart is empty."}, status=200
            )

        # Initialize total cost
        total_cost = 0.0

        # Move products from cart to previous orders and calculate total cost
        for cart_item in cart_items:
            total_cost += float(
                cart_item.product.price
            )  # Assuming price is a Decimal or float

            # Create a PreviousOrders entry for each product
            PreviousOrders.objects.create(product=cart_item.product)

        # Clear the cart by deleting all cart items
        cart_items.delete()

        # Return the total cost as a JSON response
        return JsonResponse({"total_cost": str(total_cost)}, status=200)

    # If the request method is not GET, return an error response
    return JsonResponse({"error": "Invalid request method"}, status=400)


def get_cart_products(request):
    if request.method == "GET":
        # Retrieve all cart items
        cart_items = Cart.objects.all()

        if not cart_items.exists():
            return JsonResponse({"cart": [], "message": "Cart is empty."}, status=200)

        # Serialize the products in the cart using ProductSerializer
        products = [cart_item.product for cart_item in cart_items]
        serialized_products = ProductSerializer(products, many=True).data

        return JsonResponse({"cart": serialized_products}, status=200)

    # If the request method is not GET, return an error response
    return JsonResponse({"error": "Invalid request method"}, status=400)
