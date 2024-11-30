from django.db.models import Q

from .models import Products
from collections import Counter
import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime


import os
import numpy as np
import cv2
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def find_similar_images(image_folder, query_image_path, top_n=5):
    def get_products_for_paths(image_paths):
        products = Products.objects.filter(
            Q(image1_url__in=image_paths) |
            Q(image2_url__in=image_paths) |
            Q(image3_url__in=image_paths)
        )
        return products

    def remove_background(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.bitwise_and(image, image, mask=mask)

    def extract_features(image_path, bins=(16, 16, 16)):
        print(f"Reading image from path: {image_path}")  # Debug statement
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error reading image: {image_path}")  # Error statement
            return None
        image = remove_background(image)
        image = cv2.resize(image, (256, 256))
        hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
        return cv2.normalize(hist, hist).flatten()

    def build_feature_database(image_folder):
        features = []
        image_paths = []
        for filename in os.listdir(image_folder):
            if filename.endswith(".jpg"):
                path = os.path.join(image_folder, filename)
                feature = extract_features(path)
                if feature is not None:
                    image_paths.append(path)
                    features.append(feature)
        return np.array(features), image_paths

    def search_similar_images(query_image_path, features, image_paths, top_n):
        query_features = extract_features(query_image_path)
        if query_features is None:
            print("Error processing query image.")
            return []
        distances = cdist([query_features], features, metric='euclidean')[0]
        sorted_indices = np.argsort(distances)
        return [(image_paths[idx], distances[idx]) for idx in sorted_indices][:top_n]

    def display_results(query_image_path, results):
        query_image = cv2.imread(query_image_path)
        if query_image is None:
            print(f"Failed to read query image: {query_image_path}")
            return  # Exit if the image cannot be read

        query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(15, 5))
        plt.subplot(1, len(results) + 1, 1)
        plt.imshow(query_image)
        plt.title("Query Image")
        plt.axis("off")

        for i, (path, distance) in enumerate(results):
            result_image = cv2.imread(path)
            if result_image is None:
                print(f"Failed to read result image: {path}")
                continue  # Skip if the image cannot be read

            result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            plt.subplot(1, len(results) + 1, i + 2)
            plt.imshow(result_image)
            plt.title(f"Dist: {distance:.2f}")
            plt.axis("off")

        plt.show()

    print("Building feature database...")
    features, image_paths = build_feature_database(image_folder)

    print("Searching for similar images...")
    results = search_similar_images(query_image_path, features, image_paths, top_n)

    result_paths = [path for path, _ in results]
    products = get_products_for_paths(result_paths)

    print("Displaying results...")
    display_results(query_image_path, results)
    return products


def get_combined_descriptions():
    # Retrieve all product descriptions and concatenate them into a single string
    products = Products.objects.all()
    combined_description = " ".join(product.description for product in products)
    print(combined_description)
    return combined_description




from django.db.models import Q

from decimal import Decimal

def recommend_similar_products(previous_orders):
    """
    This function takes a queryset of `PreviousOrders` and recommends similar products.

    Args:
        previous_orders: A queryset of `PreviousOrders` objects representing the user's previous purchases.

    Returns:
        A queryset of `Products` objects containing recommended products.
    """

    # Get a list of product IDs from previous orders
    previous_order_product_ids = [order.product.id for order in previous_orders]

    # Exclude previously purchased products from recommendations
    exclude_products = Q(pk__in=previous_order_product_ids)

    # Define similarity filters based on product attributes
    filters = Q()
    for order in previous_orders:
        product = order.product

        # Add filters based on available attributes
        filters |= Q(
            Q(gender=product.gender) | Q(category=product.category),
            Q(length=product.length),
            Q(fit=product.fit),
            Q(color=product.color),
            Q(activity=product.activity),
            Q(fabric=product.fabric),
            # Adjust price range based on your preference, ensuring decimal precision
            Q(price__gte=product.price * Decimal('0.8')),
            Q(price__lte=product.price * Decimal('1.2')),
        )

    # Combine filters and exclude previously purchased items
    recommended_products = Products.objects.filter(filters).exclude(exclude_products)

    # Order recommendations (optional)
    # recommended_products = recommended_products.order_by('-price')  # Order by descending price

    return recommended_products



from collections import Counter
import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime


def recommend_filters(previous_orders):
    """
    Generate intelligent recommendations for product filters based on previous orders.
    :param previous_orders: Queryset of PreviousOrders objects.
    :return: A dictionary of recommended filters.
    """
    if not previous_orders.exists():
        return {
            "message": "No previous orders found. Recommendations unavailable."
        }

    # Extract attribute data from previous orders
    attributes = {
        "gender": [],
        "color": [],
        "fabric": [],
        "activity": [],
        "fit": [],
        "length": [],
        "category": [],
        "price": [],
    }

    for order in previous_orders:
        product = order.product
        attributes["gender"].append(product.gender)
        attributes["color"].append(product.color)
        attributes["fabric"].append(product.fabric)
        attributes["activity"].append(product.activity)
        attributes["fit"].append(product.fit)
        attributes["length"].append(product.length)
        attributes["category"].append(product.category)
        attributes["price"].append(product.price)

    # 1. Frequency Analysis
    recommendations = {}
    for key, values in attributes.items():
        if key == "price":
            # Compute price statistics
            prices = np.array(values)
            recommendations["price_range"] = {
                "min": round(np.min(prices), 2),
                "max": round(np.max(prices), 2),
                "avg": round(np.mean(prices), 2),
            }
        else:
            counter = Counter(values)
            top_choices = counter.most_common(3)  # Top 3 most common filters
            recommendations[key] = [choice[0] for choice in top_choices if choice[0]]

    # 2. Clustering for Diversity
    try:
        # Create feature vectors for clustering (skip price)
        feature_vectors = []
        for i in range(len(attributes["gender"])):
            vector = [
                hash(attributes["gender"][i]) % 100,
                hash(attributes["color"][i]) % 100,
                hash(attributes["fabric"][i]) % 100,
                hash(attributes["activity"][i]) % 100,
                hash(attributes["fit"][i]) % 100,
                hash(attributes["length"][i]) % 100,
                hash(attributes["category"][i]) % 100,
            ]
            feature_vectors.append(vector)

        feature_vectors = np.array(feature_vectors)

        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=min(3, len(feature_vectors)), random_state=0)
        clusters = kmeans.fit_predict(feature_vectors)

        # Find a diverse recommendation from each cluster
        cluster_recommendations = []
        for cluster_idx in range(max(clusters) + 1):
            indices = np.where(clusters == cluster_idx)[0]
            cluster_recommendations.append(attributes["category"][indices[0]])

        recommendations["diverse_categories"] = list(set(cluster_recommendations))

    except Exception as e:
        recommendations["clustering_error"] = str(e)

    # 3. Seasonal/Time-Based Suggestions
    current_month = datetime.now().month
    if current_month in [12, 1, 2]:
        recommendations["seasonal"] = ["Sweaters", "Long Sleeve Shirts", "Wool"]
    elif current_month in [6, 7, 8]:
        recommendations["seasonal"] = ["Tank Tops", "Shorts", "Cotton"]
    else:
        recommendations["seasonal"] = ["T-Shirts", "Pima Cotton", "Casual"]

    return recommendations

# Usage Example
# previous_orders = PreviousOrders.objects.all()
# filters = recommend_filters(previous_orders)
# print(filters)



