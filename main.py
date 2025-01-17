import json

import numpy as np
from sentence_transformers import SentenceTransformer


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def load_json(json_file_path):
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def gather_entity_text(entity):
    texts = []

    if "label" in entity:
        for label_obj in entity["label"]:
            label_value = label_obj.get("value", "")
            texts.append(label_value)

    if "description" in entity:
        for desc_obj in entity["description"]:
            desc_value = desc_obj.get("value", "")
            texts.append(desc_value)

    combined_text = ". ".join(texts)
    return combined_text.strip()


def get_embeddings_for_entities(json_data, model):
    embeddings_dict = {}

    classes = json_data.get("classes", [])
    for cls in classes:
        uri = cls.get("full_uri", None)
        text = gather_entity_text(cls)
        if uri and text:
            embedding = model.encode(text)
            embeddings_dict[uri] = embedding

    properties = json_data.get("properties", [])
    for prop in properties:
        uri = prop.get("full_uri", None)
        text = gather_entity_text(prop)
        if uri and text:
            embedding = model.encode(text)
            embeddings_dict[uri] = embedding

    # TODO: handle individuals

    return embeddings_dict


def main(json_file_path):
    data = load_json(json_file_path)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = get_embeddings_for_entities(data, model)

    for uri, emb in embeddings.items():
        print(f"URI: {uri}")
        print(f"Embedding shape: {emb.shape}")
        print("-" * 50)

    with open("embeddings.json", "w") as f:
        json.dump(embeddings, f, cls=NumpyEncoder)


if __name__ == "__main__":
    json_file_path = "ontology.json"
    main(json_file_path)
