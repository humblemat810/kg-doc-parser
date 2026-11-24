use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderHttpRequest {
    pub url: String,
    pub headers: BTreeMap<String, String>,
    pub body: Value,
    pub timeout_seconds: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderHttpResponse {
    pub status: u16,
    pub body: String,
}

pub trait ProviderTransport {
    fn send(&self, request: &ProviderHttpRequest) -> Result<ProviderHttpResponse>;
}

impl<F> ProviderTransport for F
where
    F: Fn(&ProviderHttpRequest) -> Result<ProviderHttpResponse>,
{
    fn send(&self, request: &ProviderHttpRequest) -> Result<ProviderHttpResponse> {
        self(request)
    }
}

pub struct UreqProviderTransport;

impl ProviderTransport for UreqProviderTransport {
    fn send(&self, request: &ProviderHttpRequest) -> Result<ProviderHttpResponse> {
        let timeout = std::time::Duration::from_secs_f64(request.timeout_seconds);
        let agent = ureq::Agent::new_with_config(
            ureq::Agent::config_builder()
                .timeout_global(Some(timeout))
                .http_status_as_error(false)
                .build(),
        );
        let mut builder = agent.post(request.url.as_str());
        for (name, value) in &request.headers {
            builder = builder.header(name.as_str(), value.as_str());
        }
        let mut response = builder.send(request.body.to_string().as_bytes())?;
        Ok(ProviderHttpResponse {
            status: response.status().as_u16(),
            body: response.body_mut().read_to_string()?,
        })
    }
}

pub struct ProviderChatMessage {
    pub role: String,
    pub content: String,
}

fn parse_provider_http_response(response: &ProviderHttpResponse) -> Result<Value> {
    let payload: Value = serde_json::from_str(&response.body)
        .map_err(|error| anyhow::anyhow!("provider returned malformed JSON: {error}"))?;
    if response.status >= 400 {
        bail!(
            "provider HTTP {}: {}",
            response.status,
            serde_json::to_string(&payload)?
        );
    }
    Ok(payload)
}

pub fn build_openai_compatible_chat_request(
    config: &ProviderEndpointConfig,
    messages: &[ProviderChatMessage],
    response_format: Option<Value>,
) -> Result<ProviderHttpRequest> {
    if config.provider != "openai" && config.provider != "custom" && config.provider != "bonsai" {
        bail!("OpenAI-compatible provider requires openai, custom, or bonsai");
    }
    let Some(base_url) = config
        .base_url
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        bail!("OpenAI-compatible provider requires base_url");
    };
    let api_key = match config.api_key_env.as_deref() {
        Some(name) => std::env::var(name).map_err(|_| {
            anyhow::anyhow!("provider API key environment {name:?} is not configured")
        })?,
        None => bail!("OpenAI-compatible provider requires api_key_env"),
    };
    let mut body = json!({
        "model": config.model,
        "temperature": config.temperature,
        "messages": messages.iter().map(|message| json!({
            "role": message.role,
            "content": message.content,
        })).collect::<Vec<_>>(),
    });
    if let Some(max_output_tokens) = config.max_output_tokens {
        body["max_tokens"] = json!(max_output_tokens);
    }
    if let Some(response_format) = response_format {
        body["response_format"] = response_format;
    }
    let mut headers = BTreeMap::new();
    headers.insert("Content-Type".to_string(), "application/json".to_string());
    headers.insert("Accept".to_string(), "application/json".to_string());
    headers.insert("Authorization".to_string(), format!("Bearer {api_key}"));
    Ok(ProviderHttpRequest {
        url: format!("{}/chat/completions", base_url.trim_end_matches('/')),
        headers,
        body,
        timeout_seconds: config.timeout_seconds,
    })
}

pub fn invoke_openai_compatible_chat(
    config: &ProviderEndpointConfig,
    messages: &[ProviderChatMessage],
    response_format: Option<Value>,
    transport: &dyn ProviderTransport,
) -> Result<String> {
    let request = build_openai_compatible_chat_request(config, messages, response_format)?;
    let attempts = config.max_retries.saturating_add(1).max(1);
    let mut backoff = config.retry_backoff_seconds.max(0.0);
    for attempt in 1..=attempts {
        let response = match transport.send(&request) {
            Ok(response) => response,
            Err(error) => {
                if attempt == attempts {
                    return Err(error);
                }
                std::thread::sleep(std::time::Duration::from_secs_f64(backoff));
                backoff = (backoff * 2.0).min(config.retry_backoff_max_seconds.max(0.0));
                continue;
            }
        };
        if (response.status == 429 || response.status >= 500) && attempt < attempts {
            std::thread::sleep(std::time::Duration::from_secs_f64(backoff));
            backoff = (backoff * 2.0).min(config.retry_backoff_max_seconds.max(0.0));
            continue;
        }
        let payload = parse_provider_http_response(&response)?;
        let content = payload
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                anyhow::anyhow!("provider response is missing choices[0].message.content")
            })?;
        return Ok(content.to_string());
    }
    unreachable!("provider retry loop must return or continue")
}

pub fn build_anthropic_chat_request(
    config: &ProviderEndpointConfig,
    messages: &[ProviderChatMessage],
) -> Result<ProviderHttpRequest> {
    if config.provider != "anthropic" {
        bail!("Anthropic provider requires provider=anthropic");
    }
    let base_url = config
        .base_url
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("https://api.anthropic.com/v1");
    let api_key = match config.api_key_env.as_deref() {
        Some(name) => std::env::var(name).map_err(|_| {
            anyhow::anyhow!("provider API key environment {name:?} is not configured")
        })?,
        None => bail!("Anthropic provider requires api_key_env"),
    };
    let (system_messages, chat_messages): (Vec<&ProviderChatMessage>, Vec<&ProviderChatMessage>) =
        messages
            .iter()
            .partition(|message| message.role.eq_ignore_ascii_case("system"));
    let mut body = json!({
        "model": config.model,
        "max_tokens": config.max_output_tokens.unwrap_or(1024),
        "messages": chat_messages.iter().map(|message| json!({
            "role": message.role,
            "content": message.content,
        })).collect::<Vec<_>>(),
    });
    if !system_messages.is_empty() {
        body["system"] = json!(system_messages
            .iter()
            .map(|message| message.content.as_str())
            .collect::<Vec<_>>()
            .join("\n\n"));
    }
    let mut headers = BTreeMap::new();
    headers.insert("Content-Type".to_string(), "application/json".to_string());
    headers.insert("Accept".to_string(), "application/json".to_string());
    headers.insert("x-api-key".to_string(), api_key);
    headers.insert("anthropic-version".to_string(), "2023-06-01".to_string());
    Ok(ProviderHttpRequest {
        url: format!("{}/messages", base_url.trim_end_matches('/')),
        headers,
        body,
        timeout_seconds: config.timeout_seconds,
    })
}

fn anthropic_content(payload: &Value) -> Result<String> {
    let content = payload
        .get("content")
        .ok_or_else(|| anyhow::anyhow!("Anthropic response is missing content"))?;
    if let Some(text) = content.as_str() {
        return Ok(text.to_string());
    }
    let content = content.as_array().ok_or_else(|| {
        anyhow::anyhow!("Anthropic response content must be a string or block list")
    })?;
    let mut text = String::new();
    for block in content {
        let block_text = block
            .get("text")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("Anthropic content block is missing text"))?;
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str(block_text);
    }
    Ok(text)
}

fn retry_provider_request(
    config: &ProviderEndpointConfig,
    request: &ProviderHttpRequest,
    transport: &dyn ProviderTransport,
    extract_content: impl Fn(&Value) -> Result<String>,
) -> Result<String> {
    let attempts = config.max_retries.saturating_add(1).max(1);
    let mut backoff = config.retry_backoff_seconds.max(0.0);
    for attempt in 1..=attempts {
        let response = match transport.send(request) {
            Ok(response) => response,
            Err(error) => {
                if attempt == attempts {
                    return Err(error);
                }
                std::thread::sleep(std::time::Duration::from_secs_f64(backoff));
                backoff = (backoff * 2.0).min(config.retry_backoff_max_seconds.max(0.0));
                continue;
            }
        };
        if (response.status == 429 || response.status >= 500) && attempt < attempts {
            std::thread::sleep(std::time::Duration::from_secs_f64(backoff));
            backoff = (backoff * 2.0).min(config.retry_backoff_max_seconds.max(0.0));
            continue;
        }
        let payload = parse_provider_http_response(&response)?;
        return extract_content(&payload);
    }
    unreachable!("provider retry loop must return or continue")
}

pub fn invoke_anthropic_chat(
    config: &ProviderEndpointConfig,
    messages: &[ProviderChatMessage],
    transport: &dyn ProviderTransport,
) -> Result<String> {
    let request = build_anthropic_chat_request(config, messages)?;
    retry_provider_request(config, &request, transport, anthropic_content)
}

pub fn build_ollama_chat_request(
    config: &ProviderEndpointConfig,
    messages: &[ProviderChatMessage],
) -> Result<ProviderHttpRequest> {
    if config.provider != "ollama" {
        bail!("Ollama provider requires provider=ollama");
    }
    let base_url = config
        .base_url
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("http://127.0.0.1:11434");
    let mut options = json!({"temperature": config.temperature});
    if let Some(max_output_tokens) = config.max_output_tokens {
        options["num_predict"] = json!(max_output_tokens);
    }
    if let Some(num_ctx) = config.ollama_num_ctx {
        options["num_ctx"] = json!(num_ctx);
    }
    let mut headers = BTreeMap::new();
    headers.insert("Content-Type".to_string(), "application/json".to_string());
    headers.insert("Accept".to_string(), "application/json".to_string());
    Ok(ProviderHttpRequest {
        url: format!("{}/api/chat", base_url.trim_end_matches('/')),
        headers,
        body: json!({
            "model": config.model,
            "messages": messages.iter().map(|message| json!({
                "role": message.role,
                "content": message.content,
            })).collect::<Vec<_>>(),
            "stream": false,
            "options": options,
        }),
        timeout_seconds: config.timeout_seconds,
    })
}

fn ollama_content(payload: &Value) -> Result<String> {
    payload
        .get("message")
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .ok_or_else(|| anyhow::anyhow!("Ollama response is missing message.content"))
}

pub fn invoke_ollama_chat(
    config: &ProviderEndpointConfig,
    messages: &[ProviderChatMessage],
    transport: &dyn ProviderTransport,
) -> Result<String> {
    let request = build_ollama_chat_request(config, messages)?;
    retry_provider_request(config, &request, transport, ollama_content)
}

pub fn invoke_provider_chat(
    config: &ProviderEndpointConfig,
    messages: &[ProviderChatMessage],
    response_format: Option<Value>,
    transport: &dyn ProviderTransport,
) -> Result<String> {
    if config.provider == "anthropic" {
        invoke_anthropic_chat(config, messages, transport)
    } else if config.provider == "ollama" {
        invoke_ollama_chat(config, messages, transport)
    } else {
        invoke_openai_compatible_chat(config, messages, response_format, transport)
    }
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

    fn config() -> ProviderEndpointConfig {
        serde_json::from_value(json!({
            "provider": "openai",
            "model": "test-model",
            "temperature": 0.2,
            "base_url": "http://localhost/v1",
            "api_key_env": "KG_DOC_PARSER_TEST_API_KEY",
            "max_retries": 1,
            "retry_backoff_seconds": 0.0,
            "retry_backoff_max_seconds": 0.0
        }))
        .unwrap()
    }

    fn messages() -> Vec<ProviderChatMessage> {
        vec![ProviderChatMessage {
            role: "user".to_string(),
            content: "parse this".to_string(),
        }]
    }

    #[test]
    fn builds_openai_compatible_chat_request() {
        std::env::set_var("KG_DOC_PARSER_TEST_API_KEY", "secret");
        let request = build_openai_compatible_chat_request(
            &config(),
            &messages(),
            Some(json!({"type": "json_object"})),
        )
        .unwrap();
        assert_eq!(request.url, "http://localhost/v1/chat/completions");
        assert_eq!(request.headers["Authorization"], "Bearer secret");
        assert_eq!(request.body["model"], "test-model");
        assert_eq!(request.body["messages"][0]["content"], "parse this");
        assert_eq!(request.body["response_format"]["type"], "json_object");
    }

    #[test]
    fn rejects_unsupported_openai_compatible_provider() {
        let mut invalid = config();
        invalid.provider = "fake".to_string();
        assert!(build_openai_compatible_chat_request(&invalid, &messages(), None).is_err());
    }

    #[test]
    fn retries_server_errors_and_extracts_chat_content() {
        std::env::set_var("KG_DOC_PARSER_TEST_API_KEY", "secret");
        let calls = std::sync::atomic::AtomicUsize::new(0);
        let transport = |_request: &ProviderHttpRequest| -> Result<ProviderHttpResponse> {
            if calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                return Ok(ProviderHttpResponse {
                    status: 500,
                    body: json!({"error": "temporary"}).to_string(),
                });
            }
            Ok(ProviderHttpResponse {
                status: 200,
                body: json!({"choices":[{"message":{"content":"structured result"}}]}).to_string(),
            })
        };
        let content =
            invoke_openai_compatible_chat(&config(), &messages(), None, &transport).unwrap();
        assert_eq!(content, "structured result");
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    fn anthropic_config() -> ProviderEndpointConfig {
        serde_json::from_value(json!({
            "provider": "anthropic",
            "model": "claude-test",
            "temperature": 0.2,
            "base_url": "http://localhost",
            "api_key_env": "KG_DOC_PARSER_TEST_ANTHROPIC_KEY",
            "max_output_tokens": 64,
            "max_retries": 1,
            "retry_backoff_seconds": 0.0,
            "retry_backoff_max_seconds": 0.0
        }))
        .unwrap()
    }

    #[test]
    fn builds_anthropic_chat_request_with_system_header_and_max_tokens() {
        std::env::set_var("KG_DOC_PARSER_TEST_ANTHROPIC_KEY", "anthropic-secret");
        let request = build_anthropic_chat_request(
            &anthropic_config(),
            &[
                ProviderChatMessage {
                    role: "system".to_string(),
                    content: "be precise".to_string(),
                },
                ProviderChatMessage {
                    role: "user".to_string(),
                    content: "parse this".to_string(),
                },
            ],
        )
        .unwrap();
        assert_eq!(request.url, "http://localhost/messages");
        assert_eq!(request.headers["x-api-key"], "anthropic-secret");
        assert_eq!(request.headers["anthropic-version"], "2023-06-01");
        assert_eq!(request.body["system"], "be precise");
        assert_eq!(request.body["max_tokens"], 64);
        assert_eq!(request.body["messages"][0]["role"], "user");
    }

    #[test]
    fn invokes_anthropic_blocks_and_retries_server_errors() {
        std::env::set_var("KG_DOC_PARSER_TEST_ANTHROPIC_KEY", "anthropic-secret");
        let calls = std::sync::atomic::AtomicUsize::new(0);
        let transport = |_request: &ProviderHttpRequest| -> Result<ProviderHttpResponse> {
            if calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                return Ok(ProviderHttpResponse {
                    status: 503,
                    body: json!({"error": "temporary"}).to_string(),
                });
            }
            Ok(ProviderHttpResponse {
                status: 200,
                body: json!({"content":[{"type":"text","text":"part one"},{"type":"text","text":"part two"}]}).to_string(),
            })
        };
        let content =
            invoke_provider_chat(&anthropic_config(), &messages(), None, &transport).unwrap();
        assert_eq!(content, "part one\npart two");
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    fn ollama_config() -> ProviderEndpointConfig {
        serde_json::from_value(json!({
            "provider": "ollama",
            "model": "llama-test",
            "temperature": 0.2,
            "base_url": "http://localhost:11434",
            "max_output_tokens": 64,
            "ollama_num_ctx": 2048,
            "max_retries": 1,
            "retry_backoff_seconds": 0.0,
            "retry_backoff_max_seconds": 0.0
        }))
        .unwrap()
    }

    #[test]
    fn builds_ollama_chat_request_with_options() {
        let request = build_ollama_chat_request(&ollama_config(), &messages()).unwrap();
        assert_eq!(request.url, "http://localhost:11434/api/chat");
        assert_eq!(request.body["model"], "llama-test");
        assert_eq!(request.body["stream"], false);
        assert_eq!(request.body["options"]["num_predict"], 64);
        assert_eq!(request.body["options"]["num_ctx"], 2048);
        assert_eq!(request.body["options"]["temperature"], 0.2);
    }

    #[test]
    fn invokes_ollama_chat_and_retries_server_errors() {
        let calls = std::sync::atomic::AtomicUsize::new(0);
        let transport = |_request: &ProviderHttpRequest| -> Result<ProviderHttpResponse> {
            if calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0 {
                return Ok(ProviderHttpResponse {
                    status: 503,
                    body: json!({"error": "temporary"}).to_string(),
                });
            }
            Ok(ProviderHttpResponse {
                status: 200,
                body: json!({"message": {"role": "assistant", "content": "ollama result"}})
                    .to_string(),
            })
        };
        let content =
            invoke_provider_chat(&ollama_config(), &messages(), None, &transport).unwrap();
        assert_eq!(content, "ollama result");
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

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
