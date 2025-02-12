from utils.text_extractor.text_extractor_i import (
    TextExtractorI,
)


class RDFCraftClassTextExtractor(TextExtractorI[dict]):
    def extract(self, data):
        texts = []

        if "label" in data:
            for label_obj in data["label"]:
                label_value = label_obj.get("value", "")
                texts.append(label_value)

        if "description" in data:
            for desc_obj in data["description"]:
                desc_value = desc_obj.get("value", "")
                texts.append(desc_value)

        combined_text = ". ".join(texts)
        return combined_text.strip()
