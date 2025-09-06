# scripts/smoke.py
import sys, os, time
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from bs4 import BeautifulSoup
import wikipedia

# 1) Embeddings load + encode
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
dim = model.get_sentence_embedding_dimension()
vecs = model.encode(["hello world", "linear algebra is about vector spaces"], normalize_embeddings=True)

print(f"[OK] sentence-transformers loaded, dim={dim}, sample vec shape={vecs.shape}")

# 2) Wikipedia fetch + HTML parse (needs internet)
page = wikipedia.page(title="Linear algebra", auto_suggest=False, redirect=True)
html = page.html()
text = BeautifulSoup(html, "lxml").get_text(" ", strip=True)
print(f"[OK] wikipedia fetch: {page.title}, text chars={len(text)}")

# 3) Milvus round-trip: create → insert → search
connections.connect(alias="default", host="localhost", port="19530")
name = "whiteboard_smoke"
if utility.has_collection(name):
    utility.drop_collection(name)

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
]
schema = CollectionSchema(fields, description="smoke test collection")
col = Collection(name=name, schema=schema)

col.create_index(field_name="vector", index_params={"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 1024}})
col.load()

entities = [vecs.tolist(), ["hello world", "linear algebra is about vector spaces"]]
# auto_id=True → omit ids
col.insert(entities)

# search
time.sleep(0.5)  # small wait for insert visibility
q = model.encode(["what is linear algebra?"], normalize_embeddings=True)
res = col.search(data=q.tolist(), anns_field="vector", param={"nprobe": 16}, limit=2, output_fields=["text"])
print(f"[OK] milvus search top hits: {[hit.entity.get('text') for hit in res[0]]}")

print("\nALL GOOD ✅")
