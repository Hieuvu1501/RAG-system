import torch
from PIL import Image
import sys, types
sys.modules["decord"] = types.ModuleType("decord")  # dummy stub
from transformers import AutoConfig, AutoModel, AutoProcessor

model_name = "5CD-AI/Vintern-Embedding-1B"
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load config first so we can force-disable flash attention
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
for attr in ("use_flash_attn",):
    if hasattr(config, attr):
        setattr(config, attr, False)
    if hasattr(config, "llm_config") and hasattr(config.llm_config, attr):
        setattr(config.llm_config, attr, False)

model = AutoModel.from_pretrained(
    model_name,
    config=config,
    torch_dtype=torch.bfloat16,   # if this errors on your torch version, try torch.float32
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).eval().to(device)

processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

# ---- text-only example ----
queries = ["Sử dụng ma tuý bị phạt gì?"]
documents = [
    "Ma túy, thuốc gây nghiện, thuốc hướng thần và tiền chất ma túy...",
    "Cảng Hải Phòng là một cụm cảng biển tổng hợp cấp quốc gia...",
]

batch_queries = processor.process_queries(queries)
batch_docs = processor.process_docs(documents)

for batch in (batch_queries, batch_docs):
    batch["input_ids"] = batch["input_ids"].to(device)
    batch["attention_mask"] = batch["attention_mask"].to(device).bfloat16()

with torch.no_grad():
    query_embeddings = model(**batch_queries)
    doc_embeddings = model(**batch_docs)

scores = processor.score_multi_vector(query_embeddings, doc_embeddings)
print(documents[scores.argmax(dim=1)[0]])