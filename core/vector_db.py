import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
import logging
import json
from .models import Products

logger = logging.getLogger(__name__)


class ChromaDBSingleton:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChromaDBSingleton, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            try:
                logger.setLevel(logging.DEBUG)

                persist_directory = os.path.join("vector_db")
                os.makedirs(persist_directory, exist_ok=True)

                logger.debug(
                    f"Initializing ChromaDB with directory: {persist_directory}"
                )

                self.client = chromadb.PersistentClient(path=persist_directory)
                self.encoder = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2"
                )

                self.collection = self.client.get_or_create_collection(
                    name="products", metadata={"hnsw:space": "cosine"}
                )

                products = list(Products.objects.all())

                if products and isinstance(products, list):
                    self.add_products_to_vectordb(products)
                else:
                    logger.warning("No products found in database")

                self._initialized = True
                logger.info("ChromaDB initialization completed")

            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON: {e}")
            except Exception as e:
                logger.error(f"Error during ChromaDB initialization: {str(e)}")
                raise

    def add_products_to_vectordb(self, products=None):
        """
        Add multiple products to vector database with Django ORM integration
        """
        try:
            if products is None:
                products = Products.objects.all()

            existing_collection = self.collection.get()
            existing_ids = set(existing_collection.get("ids", []))
            logger.debug(f"Found {len(existing_ids)} existing products in vector DB")

            ids = []
            embeddings = []
            documents = []
            metadatas = []

            for product in products:
                try:
                    product_id = str(product.id)

                    if product_id in existing_ids:
                        logger.debug(
                            f"Product {product_id} already exists in vector DB"
                        )
                        continue

                    # Create a rich description combining all relevant fields
                    rich_description = (
                        f"{product.name} - {product.description or ''} "
                        f"This {product.gender}'s {product.category} is perfect for {product.activity or 'various activities'}. "
                        f"Made with {product.fabric or 'quality material'}, it features a {product.fit or 'comfortable fit'} "
                        f"and {product.length or 'standard'} length. Available in {product.color}."
                    )
                    print(rich_description)

                    # Generate embedding from rich description
                    embedding = self.encoder.encode(rich_description).tolist()

                    ids.append(product_id)
                    embeddings.append(embedding)
                    documents.append(rich_description)

                    # Enhanced metadata with all relevant fields
                    metadata = {
                        "name": product.name,
                        "price": str(product.price),
                        "gender": product.gender,
                        "category": product.category,
                        "color": product.color,
                        "length": product.length or "",
                        "fit": product.fit or "",
                        "activity": product.activity or "",
                        "fabric": product.fabric or "",
                        "image1_url": product.image1_url or "",
                        "image2_url": product.image2_url or "",
                        "image3_url": product.image3_url or "",
                    }
                    metadatas.append(metadata)

                except Exception as e:
                    logger.error(f"Error processing product {product.id}: {str(e)}")

            # Batch upsert products
            if ids:
                self.collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
                logger.info(f"Successfully added {len(ids)} new products to vector DB")
            else:
                logger.info("No new products to add to vector DB")

        except Exception as e:
            logger.error(f"Error in batch upload to vector DB: {str(e)}")
            raise

    def search_similar(self, query_text, n_results=5, similarity_threshold=0.75):
        """Search for similar products"""
        try:
            query_embedding = self.encoder.encode(query_text).tolist()

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["distances", "metadatas", "documents"],
            )
            return results
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return {"ids": [], "distances": [], "documents": [], "metadatas": []}


def vector_search(query, filters=None):
    try:
        db = ChromaDBSingleton()
        logger.info(f"Received query: {query}")

        formatted_results = []
        if query:
            results = db.search_similar(query, 10, 0.50)
            ids = results.get("ids", [[]])[0]
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for id_, doc, metadata, distance in zip(
                ids, documents, metadatas, distances
            ):
                if distance < 0.50:
                    formatted_results.append(
                        {
                            "ID": id_,
                            "NAME": metadata.get("name", "Unknown"),
                            "PRICE": metadata.get("price", "0.00"),
                            "GENDER": metadata.get("gender", ""),
                            "CATEGORY": metadata.get("category", ""),
                            "COLOR": metadata.get("color", ""),
                            "LENGTH": metadata.get("length", ""),
                            "FIT": metadata.get("fit", ""),
                            "ACTIVITY": metadata.get("activity", ""),
                            "FABRIC": metadata.get("fabric", ""),
                            "DESCRIPTION": doc,
                            "DISTANCE": distance,
                            "IMAGE1_URL": metadata.get("image1_url", ""),
                            "IMAGE2_URL": metadata.get("image2_url", ""),
                            "IMAGE3_URL": metadata.get("image3_url", ""),
                        }
                    )
            sorted_formatted_list = sorted(
                formatted_results, key=lambda k: k["DISTANCE"]
            )

            return sorted_formatted_list
    except Exception as e:
        logger.error(f"Error in vector search: {str(e)}")
        return []
