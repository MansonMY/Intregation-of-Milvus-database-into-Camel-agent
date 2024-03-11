from pymilvus import MilvusClient, Collection
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import uuid

from camel.storages.vectordb_storages import (
    BaseVectorStorage,
    VectorDBQuery,
    VectorDBQueryResult,
    VectorDBStatus,
    VectorRecord,
)

class MilvusStorage(BaseVectorStorage):
    
    def __init__(self, vector_dim: int, url_and_api: Tuple[str, str], collection_name: Optional[str] = None, **kwargs: Any) -> None:
        self._client: MilvusClient
        self.vector_dim = vector_dim
        self.create_client(url_and_api, **kwargs)   
        self.collection_name = self.create_collection()

    def create_client(self, url_and_api_key: Tuple[str, str], **kwargs: Any) -> None:
        self._client = MilvusClient(uri=url_and_api_key[0], token=url_and_api_key[1], **kwargs)

    def create_collection(self, **kwargs: Any,) -> str:
        try:    
            collection_name  = self.generate_collection_name()
            self._client.create_collection(collection_name = collection_name, dimension = self.vector_dim, **kwargs,)
            return collection_name
        except Exception as e:
            print(f"Failed to create collection: {e}")
            raise  

    def delete_collection(self, collection_name: str) -> None:
        self._client.drop_collection(collection_name=collection_name)

    def generate_collection_name(self) -> str:
        return str(uuid.uuid4())
    
    def add(self, records: List[VectorRecord], **kwargs) -> None:
        try:
            vectors = [record.vector for record in records]
            ids = [record.id for record in records] if hasattr(records[0], 'id') else None
            self._client.insert(collection_name=self.collection_name, entities=vectors, ids=ids, **kwargs)
            self._client.flush()
        except Exception as e:
            print(f"Failed to add records: {e}")
            raise

    def delete(self, ids: List[str],**kwargs: Any) -> None:
        results = self._client.delete(collection_name=self.collection_name, pks=ids, **kwargs)

    def status(self) -> VectorDBStatus:
        collection = Collection(self.collection_name)
        return VectorDBStatus(
            vector_dim = collection.schema.fields[-1].params['dim'], 
            vector_count = len(collection.indexes))
    
    def query(self, query: VectorDBQuery, **kwargs: Any) -> List[VectorDBQueryResult]:
        search_results = self._client.search(collection_name=self.collection_name, data=query.query_vector, limit=query.top_k, **kwargs)
        query_results = [VectorDBQueryResult(similarity=(1 - point.distance), id=str(point.id)) for point in search_results]
        return query_results

    def clear(self) -> None:
        self.delete_collection(self.collection_name)
        self.collection_name  = self.create_collection()

    @property
    def client(self) -> Any:
        return self._client