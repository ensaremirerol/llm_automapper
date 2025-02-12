import numpy as np
from kink import di
from sentence_transformers import SentenceTransformer
from torch import Tensor

from utils.embedding_generators.embedding_generator_i import (
    EmbeddingGeneratorI,
)
from utils.text_extractor.extractors.llm_concept_text_extractor import (
    LLMConceptTextExtractor,
)


class LLMConceptEmbeddingGenerator(
    EmbeddingGeneratorI[list[dict]]
):
    def generate(self, data) -> tuple[list[str], Tensor]:
        sentence_transformer = di[SentenceTransformer]
        embeddings = []
        keys = []
        concept_text_extractor = LLMConceptTextExtractor()

        for i, concept in enumerate(data):
            text = concept_text_extractor.extract(concept)
            if text:
                embedding = sentence_transformer.encode(
                    text
                )
                embeddings.append(embedding)
                keys.append(
                    concept.get(
                        "concept_label", f"concept_{i}"
                    )
                )

        return keys, Tensor(np.array(embeddings))
