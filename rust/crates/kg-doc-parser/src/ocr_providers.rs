use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OCRProviderName {
    Anthropic,
    Codex,
    Custom,
    KServe,
    Docling,
    Rapidocr,
    Easyocr,
    Fake,
    Gemini,
    Ollama,
    Openai,
    Bonsai,
    Azure,
    Vertex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CoordinatePrecision {
    Page,
    Region,
    Token,
    TableCell,
    #[default]
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OCRProviderCapabilities {
    pub provider: OCRProviderName,
    #[serde(default)]
    pub engine: Option<String>,
    pub vision: bool,
    pub local: bool,
    pub coordinates: bool,
    pub structured_output: bool,
    #[serde(default)]
    pub optional_dependency: Option<String>,
    #[serde(default)]
    pub platforms: Vec<String>,
    #[serde(default)]
    pub coordinate_precision: CoordinatePrecision,
}

fn docling_capabilities() -> Vec<OCRProviderCapabilities> {
    [
        ("rapidocr", "docling"),
        ("easyocr", "docling[easyocr]"),
        ("tesseract", "docling + tesseract"),
        ("tesserocr", "docling[tesserocr]"),
        ("ocrmac", "docling[ocrmac]"),
        ("nemotron-ocr", "docling[nemotron]"),
    ]
    .iter()
    .map(|(engine, dependency)| OCRProviderCapabilities {
        provider: OCRProviderName::Docling,
        engine: Some(engine.to_string()),
        vision: true,
        local: true,
        coordinates: true,
        structured_output: false,
        optional_dependency: Some(dependency.to_string()),
        platforms: if *engine == "ocrmac" {
            vec!["macos".to_string()]
        } else if *engine == "nemotron-ocr" {
            vec!["linux".to_string()]
        } else {
            vec![
                "linux".to_string(),
                "windows".to_string(),
                "macos".to_string(),
            ]
        },
        coordinate_precision: CoordinatePrecision::Region,
    })
    .collect()
}

fn direct_capabilities() -> Vec<OCRProviderCapabilities> {
    [
        (OCRProviderName::Rapidocr, "rapidocr-onnxruntime"),
        (OCRProviderName::Easyocr, "easyocr"),
    ]
    .iter()
    .map(|(provider, dependency)| OCRProviderCapabilities {
        provider: *provider,
        engine: None,
        vision: true,
        local: true,
        coordinates: true,
        structured_output: false,
        optional_dependency: Some(dependency.to_string()),
        platforms: vec![
            "linux".to_string(),
            "windows".to_string(),
            "macos".to_string(),
        ],
        coordinate_precision: CoordinatePrecision::Region,
    })
    .collect()
}

fn chat_capabilities() -> Vec<OCRProviderCapabilities> {
    [
        (OCRProviderName::Bonsai, Some("langchain-openai")),
        (OCRProviderName::Gemini, Some("langchain-google-genai")),
        (OCRProviderName::Ollama, Some("langchain-ollama")),
        (OCRProviderName::Openai, Some("langchain-openai")),
        (OCRProviderName::Azure, Some("langchain-openai")),
        (OCRProviderName::Vertex, Some("langchain-google-vertexai")),
        (OCRProviderName::Anthropic, Some("langchain-anthropic")),
        (OCRProviderName::Codex, None),
        (OCRProviderName::Fake, None),
        (OCRProviderName::Custom, None),
        (OCRProviderName::KServe, Some("urllib")),
    ]
    .iter()
    .map(|(provider, dependency)| OCRProviderCapabilities {
        provider: *provider,
        engine: None,
        vision: *provider != OCRProviderName::Fake,
        local: matches!(provider, OCRProviderName::Ollama | OCRProviderName::Fake),
        coordinates: false,
        structured_output: true,
        optional_dependency: dependency.map(|value| value.to_string()),
        platforms: vec![
            "linux".to_string(),
            "windows".to_string(),
            "macos".to_string(),
        ],
        coordinate_precision: CoordinatePrecision::Unknown,
    })
    .collect()
}

pub fn list_ocr_provider_capabilities() -> Vec<OCRProviderCapabilities> {
    let mut capabilities = docling_capabilities();
    capabilities.extend(direct_capabilities());
    capabilities.extend(chat_capabilities());
    capabilities
}

pub fn normalize_ocr_provider(
    value: &str,
    engine: Option<&str>,
) -> Result<(OCRProviderName, Option<String>)> {
    let normalized_value = value.trim().to_lowercase();
    let mut parts = normalized_value.splitn(2, ':');
    let mut provider = parts.next().unwrap_or_default().to_string();
    let inline_engine = parts.next().map(|engine| engine.to_string());
    if provider == "bonsai2" {
        provider = "bonsai".to_string();
    }
    let valid = [
        "anthropic",
        "codex",
        "custom",
        "kserve",
        "docling",
        "rapidocr",
        "easyocr",
        "fake",
        "gemini",
        "ollama",
        "openai",
        "bonsai",
        "azure",
        "vertex",
    ];
    if !valid.contains(&provider.as_str()) {
        bail!("unsupported OCR provider: {value}");
    }
    if provider == "docling" {
        let selected = engine
            .map(|value| value.to_string())
            .or(inline_engine)
            .unwrap_or_else(|| "rapidocr".to_string())
            .trim()
            .to_lowercase();
        if !docling_capabilities()
            .iter()
            .any(|capability| capability.engine.as_deref() == Some(selected.as_str()))
        {
            bail!("unsupported Docling OCR engine: {selected}");
        }
        return Ok((OCRProviderName::Docling, Some(selected)));
    }
    if provider == "rapidocr" || provider == "easyocr" {
        if engine.is_some() || inline_engine.is_some() {
            bail!("direct OCR provider does not accept an engine: {provider}");
        }
        return Ok((
            if provider == "rapidocr" {
                OCRProviderName::Rapidocr
            } else {
                OCRProviderName::Easyocr
            },
            None,
        ));
    }
    Ok((
        serde_json::from_value(serde_json::Value::String(provider)).unwrap(),
        None,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inventory_contains_expected_provider_families() {
        let capabilities = list_ocr_provider_capabilities();
        assert_eq!(capabilities.len(), 19);
        assert_eq!(capabilities[0].provider, OCRProviderName::Docling);
        assert_eq!(capabilities[6].provider, OCRProviderName::Rapidocr);
        assert_eq!(capabilities[8].provider, OCRProviderName::Bonsai);
    }

    #[test]
    fn normalizes_docling_engine_aliases() {
        let (provider, engine) = normalize_ocr_provider("docling:easyocr", None).unwrap();
        assert_eq!(provider, OCRProviderName::Docling);
        assert_eq!(engine.as_deref(), Some("easyocr"));
        let (provider, engine) = normalize_ocr_provider("docling", Some("EASYOCR")).unwrap();
        assert_eq!(provider, OCRProviderName::Docling);
        assert_eq!(engine.as_deref(), Some("easyocr"));
    }

    #[test]
    fn rejects_unsupported_provider_and_engine() {
        assert!(normalize_ocr_provider("unknown", None).is_err());
        assert!(normalize_ocr_provider("docling:unknown", None).is_err());
        assert!(normalize_ocr_provider("rapidocr", Some("easyocr")).is_err());
    }
}
