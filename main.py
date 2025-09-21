from sentence_transformers import SentenceTransformer
from datetime import datetime, timezone

dt_before = datetime.now(timezone.utc)
print(str(dt_before))
model = SentenceTransformer("multi-qa-mpnet-base-cos-v1")
dt_after = datetime.now(timezone.utc)
print(str(dt_after), str(dt_after - dt_before))

print("=" * 20)

dt_before = datetime.now(timezone.utc)
print(str(dt_before))
model.encode(["殺して解して並べて揃えて晒してやんよ"], convert_to_tensor=False)
dt_after = datetime.now(timezone.utc)
print(str(dt_after), str(dt_after - dt_before))
