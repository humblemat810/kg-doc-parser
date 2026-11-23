use anyhow::{bail, Result};
use serde_json::{json, Map, Value};
use sha1::{Digest as Sha1Digest, Sha1};

use crate::source_text::source_map_entry_text;

#[derive(Debug, Clone)]
pub struct PreparedLayerwiseDocument {
    pub document: Value,
    pub source_map: Map<String, Value>,
}

fn as_object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("{label} must be an object"))
}

fn as_array<'a>(value: &'a Value, label: &str) -> Result<&'a Vec<Value>> {
    value
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("{label} must be a list"))
}

fn as_object_mut<'a>(value: &'a mut Value, label: &str) -> Result<&'a mut Map<String, Value>> {
    value
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("{label} must be an object"))
}

pub fn prepare_document_for_llm(doc_dict: &Value) -> Result<PreparedLayerwiseDocument> {
    let document_object = as_object(doc_dict, "raw OCR document")?;
    if document_object.is_empty() {
        bail!("raw OCR document must contain a filename entry");
    }
    let (filename, pages_value) = document_object.iter().next().unwrap();
    let raw_pages = as_array(pages_value, "raw OCR document pages")?;
    let mut pages = Vec::with_capacity(raw_pages.len());
    let mut source_map = Map::new();
    for (page_index, page_value) in raw_pages.iter().enumerate() {
        let page = as_object(page_value, &format!("raw OCR page {page_index}"))?;
        let mut page = Value::Object(page.clone());
        let page_object = as_object_mut(&mut page, &format!("raw OCR page {page_index}"))?;
        if !(page_object.contains_key("pdf_page_num")
            && page_object.contains_key("OCR_text_clusters"))
        {
            pages.push(page);
            continue;
        }
        let page_number = page_object["pdf_page_num"].clone();
        for key in ["OCR_text_clusters", "non_text_objects"] {
            if !page_object.contains_key(key) {
                continue;
            }
            let label = format!("raw OCR page {page_index} field {key:?}");
            let raw_clusters = as_array(&page_object[key], &label)?;
            let mut clusters = Vec::with_capacity(raw_clusters.len());
            for (cluster_index, cluster_value) in raw_clusters.iter().enumerate() {
                let cluster = as_object(cluster_value, &format!("{label} item {cluster_index}"))?;
                let cluster_id = format!("p{page_number}_c{cluster_index}");
                let mut enriched = Value::Object(cluster.clone());
                enriched["id"] = json!(cluster_id);
                source_map.insert(cluster_id, enriched.clone());
                clusters.push(enriched);
            }
            page_object[key] = Value::Array(clusters);
        }
        pages.push(page);
    }
    Ok(PreparedLayerwiseDocument {
        document: json!({
            "document_filename": filename,
            "pages": pages,
        }),
        source_map,
    })
}

pub fn stable_id(kind: &str, parts: &[&str]) -> String {
    let mut identity_parts = vec![kind.to_string()];
    identity_parts.extend(parts.iter().map(|part| (*part).to_string()));
    let identity = serde_json::to_string(&identity_parts).unwrap();
    let mut namespace_bytes = [0u8; 16];
    namespace_bytes.copy_from_slice(&[
        0xf9, 0xa5, 0x59, 0x40, 0x43, 0x30, 0x51, 0x84, 0x9d, 0xee, 0xa6, 0x01, 0xc0, 0xed, 0x8a,
        0xbd,
    ]);
    let mut digest = Sha1::new();
    digest.update(namespace_bytes);
    digest.update(identity.as_bytes());
    let hash = digest.finalize();
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&hash[..16]);
    bytes[6] = (bytes[6] & 0x0f) | 0x50;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
        bytes[7], bytes[8], bytes[9], bytes[10], bytes[11], bytes[12],
        bytes[13], bytes[14], bytes[15]
    )
}

pub fn layerwise_root_node(title: &str, source_map: &Map<String, Value>) -> Value {
    let cluster_ids = source_map.keys().cloned().collect::<Vec<_>>();
    let node_id = stable_id("legacy.root_node", &[title, &cluster_ids.join("|")]);
    let mut pointers = Vec::new();
    for cluster_id in cluster_ids {
        pointers.push(json!({
            "source_cluster_id": cluster_id,
            "start_char": 0,
            "end_char": -1,
            "verbatim_text": source_map_entry_text(&source_map[&cluster_id]),
            "start_delimiter": Value::Null,
            "end_delimiter": Value::Null,
            "validation_method": Value::Null,
        }));
    }
    json!({
        "node_id": node_id,
        "title": title,
        "node_type": "DOCUMENT_ROOT",
        "total_content_pointers": pointers,
        "child_nodes": [],
        "level_from_root": 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn prepares_source_map_with_deterministic_cluster_ids() {
        let raw = json!({
            "input.pdf": [
                {
                    "pdf_page_num": 1,
                    "OCR_text_clusters": [{"text": "first"}, {"text": "second"}]
                }
            ]
        });
        let prepared = prepare_document_for_llm(&raw).unwrap();
        assert_eq!(prepared.document["document_filename"], "input.pdf");
        assert_eq!(prepared.source_map["p1_c0"]["text"], "first");
        assert_eq!(prepared.source_map["p1_c0"]["id"], "p1_c0");
        assert_eq!(prepared.source_map["p1_c1"]["text"], "second");
    }

    #[test]
    fn rejects_invalid_document_shapes() {
        assert!(prepare_document_for_llm(&json!({})).is_err());
        assert!(prepare_document_for_llm(&json!({"file.pdf": {}})).is_err());
        assert!(prepare_document_for_llm(&json!({"file.pdf": [{}]})).is_ok());
        assert!(prepare_document_for_llm(&json!({
            "file.pdf": [{"pdf_page_num": 1, "OCR_text_clusters": {}}]
        }))
        .is_err());
        assert!(prepare_document_for_llm(&json!({
            "file.pdf": [{"pdf_page_num": 1, "OCR_text_clusters": ["not object"]}]
        }))
        .is_err());
    }

    #[test]
    fn legacy_root_node_matches_python_stable_id() {
        let raw = json!({
            "input.pdf": [
                {
                    "pdf_page_num": 1,
                    "OCR_text_clusters": [{"text": "first"}, {"text": "second"}]
                }
            ]
        });
        let prepared = prepare_document_for_llm(&raw).unwrap();
        let root = layerwise_root_node("Document", &prepared.source_map);
        assert_eq!(root["node_id"], "dfd8171e-781d-5b90-840f-1170a69aaa81");
        assert_eq!(root["total_content_pointers"][0]["end_char"], -1);
        assert_eq!(root["total_content_pointers"][1]["verbatim_text"], "second");
    }
}
