
from typing import Dict, Literal, Any
from .semantic_document_splitting_layerwise_edits import parse_doc

def text_to_ocr_format(text: str, filename: str = "input_text") -> Dict:
    """
    Wraps a raw string into the expected OCR dictionary format with a single dummy cluster.
    """
    return {
        filename: [
            {
                "pdf_page_num": 1,
                "OCR_text_clusters": [
                    {
                        "text": text,
                        "bb_x_min": 0, "bb_y_min": 0, "bb_x_max": 1000, "bb_y_max": 1000,
                        "cluster_number": 0
                    }
                ],
                "non_text_objects": []
            }
        ]
    }

def parse_doc_text(text: str, doc_id: str = "text_doc", parsing_mode: Literal["snippet", "delimiter"] = "snippet", max_depth: int = 10):
    """
    Convenience function to parse a raw text string.
    """
    raw_doc = text_to_ocr_format(text)
    return parse_doc(doc_id, raw_doc, parsing_mode=parsing_mode, max_depth=max_depth)
