
import pytest
from src.text_processing_utils import parse_doc_text
from src.semantic_document_splitting_layerwise_edits import HydratedTextPointer, resolve_delimiter_pointer

def test_resolve_delimiter_simple():
    # Setup source map
    source_text = "Section 1\nThis is content of section 1.\nSection 2\nThis is content of section 2."
    source_map = {
        "p1_c0": {
            "text": source_text,
            "id": "p1_c0"
        }
    }
    
    # Pointer with unique delimiters
    pointer = HydratedTextPointer(
        source_cluster_id="p1_c0",
        start_char=0, # Placeholder
        end_char=0,   # Placeholder
        start_delimiter="Section 1",
        end_delimiter="Section 2"
    )
    
    # Resolve
    resolved = resolve_delimiter_pointer(pointer, source_map)
    assert resolved is not None
    assert resolved.verbatim_text.startswith("Section 1")
    assert resolved.verbatim_text.endswith("Section 2")
    # Actually my logic includes end delimiter.
    # Text: "Section 1\nThis is content of section 1.\nSection 2"
    
    # Test Uniqueness Error
    source_text_dupe = "Section 1 ... Section 1 ... Section 2"
    source_map_dupe = {"p1_c0": {"text": source_text_dupe, "id": "p1_c0"}}
    pointer_dupe = HydratedTextPointer(
        source_cluster_id="p1_c0",
        start_char=0, end_char=0,
        start_delimiter="Section 1",
        end_delimiter="Section 2"
    )
    
    with pytest.raises(ValueError, match="is not unique"):
        resolve_delimiter_pointer(pointer_dupe, source_map_dupe)

if __name__ == "__main__":
    test_resolve_delimiter_simple()
    print("Test passed!")
