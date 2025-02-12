import numpy as np
from kink import di
from sentence_transformers import SentenceTransformer
from torch import Tensor

from utils.embedding_generators.embedding_generator_i import (
    EmbeddingGeneratorI,
)
from utils.text_extractor.extractors.rdf_craft_class_text_extractor import (
    RDFCraftClassTextExtractor,
)


class RDFCraftOntologyEmbeddingGenerator(
    EmbeddingGeneratorI[dict]
):
    def generate(self, data) -> tuple[list[str], Tensor]:
        sentence_transformer = di[SentenceTransformer]
        embeddings = []
        keys = []
        class_text_extractor = RDFCraftClassTextExtractor()
        classes = data.get("classes", [])
        for _class in classes:
            uri = _class.get("full_uri", None)
            text = class_text_extractor.extract(_class)
            if uri and text:
                embedding = sentence_transformer.encode(
                    text
                )
                embeddings.append(embedding)
                keys.append(uri)

        properties = data.get("properties", [])
        for prop in properties:
            uri = prop.get("full_uri", None)
            text = class_text_extractor.extract(prop)
            if uri and text:
                embedding = sentence_transformer.encode(
                    text
                )
                embeddings.append(embedding)
                keys.append(uri)

        return keys, Tensor(np.array(embeddings))
