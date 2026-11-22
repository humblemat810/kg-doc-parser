use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProposalMode {
    Children,
    Boundaries,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParseStrategy {
    Auto,
    LayerExcerpt,
    LayerBoundary,
    PageIndex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PdfTextMode {
    Disabled,
    PreferNative,
    NativeOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderEndpointConfig {
    #[serde(default = "default_provider")]
    pub provider: String,
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default)]
    pub api_version: Option<String>,
    #[serde(default)]
    pub project: Option<String>,
    #[serde(default)]
    pub ocr_engine: Option<String>,
    #[serde(default = "default_pdf_text_mode")]
    pub pdf_text_mode: PdfTextMode,
    #[serde(default = "default_ocr_language")]
    pub ocr_language: String,
    #[serde(default)]
    pub location: Option<String>,
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    #[serde(default)]
    pub max_output_tokens: Option<u64>,
    #[serde(default)]
    pub ollama_num_ctx: Option<u64>,
    #[serde(default = "default_timeout_seconds")]
    pub timeout_seconds: f64,
    #[serde(default = "default_retry_backoff_seconds")]
    pub retry_backoff_seconds: f64,
    #[serde(default = "default_retry_backoff_max_seconds")]
    pub retry_backoff_max_seconds: f64,
    #[serde(default = "default_max_in_flight_calls")]
    pub max_in_flight_calls: u32,
}

fn default_provider() -> String {
    "gemini".to_string()
}

fn default_model() -> String {
    "gemini-2.5-flash".to_string()
}

fn default_temperature() -> f64 {
    0.1
}

fn default_pdf_text_mode() -> PdfTextMode {
    PdfTextMode::Disabled
}

fn default_ocr_language() -> String {
    "en".to_string()
}

fn default_max_retries() -> u32 {
    2
}

fn default_timeout_seconds() -> f64 {
    120.0
}

fn default_retry_backoff_seconds() -> f64 {
    0.25
}

fn default_retry_backoff_max_seconds() -> f64 {
    5.0
}

fn default_max_in_flight_calls() -> u32 {
    1
}

impl Default for ProviderEndpointConfig {
    fn default() -> Self {
        serde_json::from_value(serde_json::json!({})).unwrap()
    }
}

impl ProviderEndpointConfig {
    pub fn validate(&self) -> Result<()> {
        if !(0.0..=2.0).contains(&self.temperature) {
            bail!("provider temperature must be between 0 and 2");
        }
        if self.max_output_tokens.is_some_and(|value| value == 0) {
            bail!("provider max_output_tokens must be positive");
        }
        if self.ollama_num_ctx.is_some_and(|value| value == 0) {
            bail!("provider ollama_num_ctx must be positive");
        }
        if self.timeout_seconds <= 0.0 {
            bail!("provider timeout_seconds must be positive");
        }
        if self.retry_backoff_seconds < 0.0 || self.retry_backoff_seconds > 30.0 {
            bail!("provider retry_backoff_seconds must be between 0 and 30");
        }
        if self.retry_backoff_max_seconds < 0.0 || self.retry_backoff_max_seconds > 120.0 {
            bail!("provider retry_backoff_max_seconds must be between 0 and 120");
        }
        if self.max_in_flight_calls < 1 || self.max_in_flight_calls > 32 {
            bail!("provider max_in_flight_calls must be between 1 and 32");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowProviderSettings {
    #[serde(default = "default_proposal_mode")]
    pub proposal_mode: ProposalMode,
    #[serde(default = "default_parse_strategy")]
    pub parse_strategy: ParseStrategy,
    #[serde(default = "default_parse_strategy_order")]
    pub parse_strategy_order: Vec<String>,
    #[serde(default = "default_true")]
    pub triage_enabled: bool,
    #[serde(default = "default_true")]
    pub page_index_summary_enabled: bool,
    #[serde(default)]
    pub page_index_hierarchical_summary_enabled: bool,
    #[serde(default = "default_layer_frontier_batch_size")]
    pub layer_frontier_batch_size: usize,
    #[serde(default = "default_boundary_max_points")]
    pub boundary_max_points: usize,
    #[serde(default = "default_boundary_max_repair_shift_chars")]
    pub boundary_max_repair_shift_chars: usize,
    #[serde(default)]
    pub proposal_timeout_seconds: Option<f64>,
    #[serde(default)]
    pub review_timeout_seconds: Option<f64>,
    #[serde(default)]
    pub triage_timeout_seconds: Option<f64>,
    #[serde(default)]
    pub ocr: ProviderEndpointConfig,
    #[serde(default)]
    pub parser: ProviderEndpointConfig,
}

fn default_true() -> bool {
    true
}

fn default_proposal_mode() -> ProposalMode {
    ProposalMode::Children
}

fn default_parse_strategy() -> ParseStrategy {
    ParseStrategy::Auto
}

fn default_parse_strategy_order() -> Vec<String> {
    [
        "layer_excerpt".to_string(),
        "layer_boundary".to_string(),
        "page_index".to_string(),
    ]
    .to_vec()
}

fn default_layer_frontier_batch_size() -> usize {
    1
}

fn default_boundary_max_points() -> usize {
    128
}

fn default_boundary_max_repair_shift_chars() -> usize {
    8
}

impl Default for WorkflowProviderSettings {
    fn default() -> Self {
        serde_json::from_value(serde_json::json!({})).unwrap()
    }
}

impl WorkflowProviderSettings {
    pub fn validate(&self) -> Result<()> {
        let expected: Vec<&str> = vec!["layer_excerpt", "layer_boundary", "page_index"];
        if self.parse_strategy_order.len() != expected.len()
            || self
                .parse_strategy_order
                .iter()
                .any(|strategy| !expected.contains(&strategy.as_str()))
            || self
                .parse_strategy_order
                .iter()
                .collect::<std::collections::HashSet<_>>()
                .len()
                != expected.len()
        {
            bail!("parse_strategy_order must contain layer_excerpt, layer_boundary, and page_index exactly once");
        }
        self.ocr.validate()?;
        self.parser.validate()?;
        if self.layer_frontier_batch_size < 1 {
            bail!("layer_frontier_batch_size must be positive");
        }
        if self.boundary_max_points < 8 || self.boundary_max_points > 512 {
            bail!("boundary_max_points must be between 8 and 512");
        }
        if self.boundary_max_repair_shift_chars > 4096 {
            bail!("boundary_max_repair_shift_chars must be at most 4096");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn default_settings_validate() {
        let settings = WorkflowProviderSettings::default();
        settings.validate().unwrap();
        assert_eq!(settings.proposal_mode, ProposalMode::Children);
        assert_eq!(settings.parse_strategy, ParseStrategy::Auto);
        assert_eq!(settings.ocr.provider, "gemini");
        assert_eq!(settings.parser.provider, "gemini");
    }

    #[test]
    fn rejects_invalid_strategy_order() {
        let settings: WorkflowProviderSettings = serde_json::from_value(json!({
            "parse_strategy_order": ["layer_excerpt", "layer_excerpt", "page_index"]
        }))
        .unwrap();
        assert!(settings.validate().is_err());
    }

    #[test]
    fn rejects_invalid_provider_limits() {
        let settings: WorkflowProviderSettings = serde_json::from_value(json!({
            "parser": {
                "max_in_flight_calls": 64
            }
        }))
        .unwrap();
        assert!(settings.validate().is_err());
    }
}
