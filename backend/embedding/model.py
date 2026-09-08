from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer

from backend.config import DEVICE
from backend.schemas import IndexManifest


def count_tokens(text: str, model: SentenceTransformer) -> int:
    token_ids = model.tokenizer.encode(
        text,
        add_special_tokens=True,
        truncation=False
    )
    return len(token_ids)

def load_model_for_ingestion(
    model_name: str
) -> tuple[SentenceTransformer, str]:
    revision = HfApi().model_info(model_name).sha

    if not revision:
        raise ValueError("Could not resolve the model revision")

    model = SentenceTransformer(
        model_name,
        revision=revision,
        device=DEVICE
    )

    return model, revision

def load_model_for_query(
    manifest: IndexManifest
) -> SentenceTransformer:
    return SentenceTransformer(
        manifest.embedding_model,
        revision=manifest.embedding_revision,
        device=DEVICE
    )
