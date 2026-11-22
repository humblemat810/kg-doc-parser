use serde_json::{Map, Value};

fn first_text(record: &Map<String, Value>, keys: &[&str]) -> Option<String> {
    for key in keys {
        if let Some(Value::String(value)) = record.get(*key) {
            if !value.trim().is_empty() {
                return Some(value.clone());
            }
        }
    }
    None
}

pub fn source_map_entry_text(entry: &Value) -> String {
    let Some(record) = entry.as_object() else {
        return String::new();
    };
    if let Some(direct_text) = first_text(
        record,
        &["text", "content", "verbatim_text", "snippet", "summary"],
    ) {
        return direct_text;
    }

    let mut parts: Vec<String> = Vec::new();
    if let Some(Value::Array(raw_clusters)) = record.get("OCR_text_clusters") {
        for raw_cluster in raw_clusters {
            if let Some(cluster) = raw_cluster.as_object() {
                if let Some(text) =
                    first_text(cluster, &["text", "content", "verbatim_text", "snippet"])
                {
                    parts.push(text);
                }
            }
        }
        if !parts.is_empty() {
            return parts.join("\n");
        }
    }

    if let Some(Value::Array(raw_objects)) = record.get("non_text_objects") {
        for raw_object in raw_objects {
            if let Some(item) = raw_object.as_object() {
                if let Some(text) = first_text(item, &["description", "text"]) {
                    parts.push(text);
                }
            }
        }
        if !parts.is_empty() {
            return parts.join("\n");
        }
    }

    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn reads_only_real_text_fields() {
        assert_eq!(
            source_map_entry_text(&json!({"text": "primary", "content": "fallback"})),
            "primary"
        );
        assert_eq!(
            source_map_entry_text(&json!({"text": " ", "content": "fallback"})),
            "fallback"
        );
        assert_eq!(source_map_entry_text(&json!(7)), "");
    }

    #[test]
    fn rejects_non_string_nested_evidence() {
        let source = json!({
            "OCR_text_clusters": [
                {"text": 17, "content": "actual cluster"},
                {"text": "second cluster"},
                "not a cluster"
            ],
            "non_text_objects": [
                {"description": false},
                {"text": "Figure caption"}
            ]
        });
        assert_eq!(
            source_map_entry_text(&source),
            "actual cluster\nsecond cluster"
        );
        assert_eq!(
            source_map_entry_text(
                &json!({"non_text_objects": [{"description": false}, {"text": "Figure caption"}]})
            ),
            "Figure caption"
        );
    }
}
