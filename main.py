from sentence_transformers import SentenceTransformer
import tiktoken
from datetime import datetime, timezone

dt_before = datetime.now(timezone.utc)
print(str(dt_before))
model = SentenceTransformer("multi-qa-mpnet-base-cos-v1")
dt_after = datetime.now(timezone.utc)
print(str(dt_after), str(dt_after - dt_before))

print("=" * 20)

dt_before = datetime.now(timezone.utc)
print(str(dt_before))
resp = tiktoken.encoding_for_model("gpt-4")
dt_after = datetime.now(timezone.utc)
print(str(dt_after), str(dt_after - dt_before))
print(resp)

print("=" * 20)

dt_before = datetime.now(timezone.utc)
print(str(dt_before))
resp = model.encode(["殺して解して並べて揃えて晒してやんよ"], convert_to_tensor=False)
dt_after = datetime.now(timezone.utc)
print(str(dt_after), str(dt_after - dt_before))
print(resp)

print("=" * 20)

