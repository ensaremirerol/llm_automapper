import json
import os

import numpy as np
from dotenv import load_dotenv
from kink import di
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.fastembed_common import QueryResponse
from sentence_transformers import SentenceTransformer
from torch import cosine_similarity

from utils.embedding_generators.llm_concept_embedding_generator import (
    LLMConceptEmbeddingGenerator,
)
from utils.embedding_generators.rdf_craft_ontology_embedding_generator import (
    RDFCraftOntologyEmbeddingGenerator,
)

load_dotenv()


LLM_API_KEY = os.getenv("LLM_API_KEY") or "ollama"

LLM_MODEL: str = os.getenv("LLM_MODEL") or "phi-4"

di["llm_api_key"] = LLM_API_KEY
di["llm_model"] = LLM_MODEL

di[OpenAI] = OpenAI(
    api_key=di["llm_api_key"],
)

di[SentenceTransformer] = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def load_json(json_file_path):
    with open(json_file_path, "r") as f:
        data = json.load(f)

    return data


messages = []


def get_concepts(data, messages: list = messages):
    ABSTRACT_VIEW_SYSTEM_PROMPT = """

        You are helpful RDF mapping assistant that helps to map table and json data to RDF.

        Your task is to make guesses on the possible RDF Classes and Properties that might be used to represent the data.

        Return with following format:

        [
            "Person: A human being"
        ]

        You can return more than one concept in the array.

        JUST RETURN JSON
        """

    openai = di[OpenAI]

    messages.extend(
        [
            {
                "role": "system",
                "content": ABSTRACT_VIEW_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "data": data,
                    },
                    cls=NumpyEncoder,
                ),
            },
        ]
    )

    prompt = openai.chat.completions.create(
        model=di["llm_model"],
        messages=messages,
        temperature=0.7,
        max_tokens=-1,
        stream=False,
    )

    messages.append(
        {
            "role": "assistant",
            "content": prompt.choices[0].message.content
            or "",
        }
    )

    return prompt.choices[0].message.content or ""


def get_relevant_properties(
    data, messages: list = messages
):
    ABSTRACT_VIEW_SYSTEM_PROMPT = """

        Now you will receive list of properties that are relevant to the concepts you provided.
        Which of these properties do you need to map the data to RDF?

        While mapping if you think you don't have enough information to map a entity fully but enough to generate URI, then set return_more to false.
        Else, return_more to true.

        Now Provide in following format:

        [
            {
                "full_uri": "http://example.com/property1",
                "return_more": true
            }
        ]

        JUST RETURN JSON ARRAY OF OBJECTS

        """

    openai = di[OpenAI]

    messages.extend(
        [
            {
                "role": "system",
                "content": ABSTRACT_VIEW_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": "\n".join(data),
            },
        ]
    )

    prompt = openai.chat.completions.create(
        model=di["llm_model"],
        messages=messages,
        temperature=0.7,
        max_tokens=-1,
        stream=False,
    )

    messages.append(
        {
            "role": "assistant",
            "content": prompt.choices[0].message.content
            or "",
        }
    )

    return prompt.choices[0].message.content or ""


MAX_RETRIES = 3


def main(json_file_path):
    qdrant = QdrantClient(":memory:")
    data = load_json(json_file_path)

    classes = data.get("classes", [])
    properties = data.get("properties", [])
    individual = data.get("individuals", [])

    classes_metadata = [
        {
            "label": _cls["label"][0].get("value", "")
            if "label" in _cls and len(_cls["label"]) > 0
            else "",
            "description": _cls["description"][0].get(
                "value", ""
            )
            if "description" in _cls
            and len(_cls["description"]) > 0
            else "",
            "full_uri": _cls["full_uri"],
        }
        for _cls in classes
    ]

    properties_metadata = [
        {
            "label": prop["label"][0].get("value", "")
            if "label" in prop and len(prop["label"]) > 0
            else "",
            "description": prop["description"][0].get(
                "value", ""
            )
            if "description" in prop
            and len(prop["description"]) > 0
            else "",
            "full_uri": prop["full_uri"],
        }
        for prop in properties
    ]

    qdrant.add(
        "classes",
        [str(_cls) for _cls in classes],
        classes_metadata,
    )

    qdrant.add(
        "properties",
        [str(prop) for prop in properties],
        properties_metadata,
    )

    example_csv_metadata = load_json(
        "example_csv_metadata.json"
    )

    openai = di[OpenAI]

    concepts = None

    for _ in range(MAX_RETRIES):
        concepts = get_concepts(example_csv_metadata)

        print(concepts)

        if concepts and concepts != "":
            try:
                json.loads(concepts)
            except Exception:
                continue
            break

    else:
        raise Exception("Failed to get concepts")

    concepts_arr: list[str] = json.loads(concepts)

    relevant_concepts: list[QueryResponse] = []

    for concept_str in concepts_arr:
        search_results: list[QueryResponse] = qdrant.query(
            "classes",
            concept_str,
        )

        if search_results:
            relevant_concepts.append(search_results[0])

    relevant_concepts_full_uris = [
        concept.metadata["full_uri"]
        for concept in relevant_concepts
    ]

    relevant_classes = [
        _cls
        for _cls in classes
        if _cls["full_uri"] in relevant_concepts_full_uris
    ]

    relevant_properties = [
        _prop
        for _prop in properties
        if any(
            _cls["full_uri"] in _prop["domain"]
            for _cls in relevant_classes
        )
        or any(
            _cls["full_uri"] in _prop["range"]
            for _cls in relevant_classes
        )
    ]

    possible_props_for_relevant_classes = [
        prop["full_uri"]
        for prop in relevant_properties
        if any(
            _cls["full_uri"] in prop["domain"]
            for _cls in relevant_classes
        )
    ]

    properties_arr = None

    for _ in range(MAX_RETRIES):
        properties_arr = get_relevant_properties(
            possible_props_for_relevant_classes
        )

        print(properties_arr)

        if properties_arr and properties_arr != "":
            try:
                json.loads(properties_arr)
            except Exception:
                continue
            break

    else:
        raise Exception("Failed to get properties")

    with open("relevant_classes.json", "w") as f:
        json.dump(relevant_classes, f, indent=4)

    with open("relevant_properties.json", "w") as f:
        json.dump(relevant_properties, f, indent=4)


if __name__ == "__main__":
    json_file_path = "ontology.json"
    main(json_file_path)
