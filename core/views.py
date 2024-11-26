from .models import Products
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .serializers import ProductSerializer
from .vector_db import vector_search


class ProductViewSet(viewsets.ModelViewSet):
    queryset = Products.objects.all()
    serializer_class = ProductSerializer


class VectorSearchViewSet(viewsets.ViewSet):
    @action(detail=False, methods=["POST"])
    def search(self, request):
        try:
            # Get the search query from request data
            search_query = request.data.get("query", "")

            if not search_query:
                return Response(
                    {"error": "Search query is required"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Use the vector_search function from vector_db.py
            results = vector_search(search_query)

            if not results:
                return Response(
                    {
                        "results": [],
                        "query": search_query,
                        "message": "No matching products found",
                    }
                )

            return Response(
                {"results": results, "query": search_query, "count": len(results)}
            )

        except Exception as e:
            return Response(
                {"error": "An error occurred during search"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
