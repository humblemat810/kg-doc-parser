use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

fn source_text(record: Option<&Value>) -> String {
    record
        .and_then(Value::as_object)
        .and_then(|record| record.get("text"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

pub fn source_map_fingerprint(parser_source_map: &Value) -> String {
    let empty = Map::new();
    let records = parser_source_map
        .as_object()
        .unwrap_or(&empty)
        .iter()
        .map(|(source_cluster_id, record)| {
            serde_json::json!({
                "source_cluster_id": source_cluster_id,
                "text": source_text(Some(record)),
            })
        })
        .collect::<Vec<_>>();
    let canonical = crate::semantic_indexing::json_compact(&serde_json::to_value(records).unwrap());
    let mut digest = Sha256::new();
    digest.update(canonical.as_bytes());
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn fingerprints_sorted_source_map() {
        json!({
            "z": {"text": "second"},
            "a": {"text": "first"},
        });
        let fingerprint = source_map_fingerprint(&json!({
            "a": {"text": "first"},
            "z": {"text": "second"},
        }));
        assert_eq!(
            fingerprint,
            source_map_fingerprint(&json!({
                "a": {"text": "first"},
                "z": {"text": "second"},
            }))
        );
    }
}
