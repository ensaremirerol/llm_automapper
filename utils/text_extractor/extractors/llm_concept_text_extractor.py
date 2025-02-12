from utils.text_extractor.text_extractor_i import (
    TextExtractorI,
)


class LLMConceptTextExtractor(TextExtractorI[dict]):
    def extract(self, data):
        texts = []

        texts.append(data.get("concept_label", ""))
        texts.append(data.get("concept_description", ""))

        combined_text = ". ".join(texts)
        return combined_text.strip()
