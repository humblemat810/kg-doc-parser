use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub type StrategyTriageFn<'a> = dyn Fn(&Value) -> Result<ParseStrategyTriage> + 'a;

pub const HARD_CODED_STRATEGY_PRIORITY: [ParseStrategy; 3] = [
    ParseStrategy::LayerExcerpt,
    ParseStrategy::LayerBoundary,
    ParseStrategy::PageIndex,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParseStrategy {
    LayerExcerpt,
    LayerBoundary,
    PageIndex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParseStrategyRequest {
    Auto,
    LayerExcerpt,
    LayerBoundary,
    PageIndex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParseStrategySource {
    Config,
    HardcodedFallback,
    LlmTriage,
    LlmTriageFallback,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseStrategyAssessment {
    pub strategy: ParseStrategy,
    #[serde(default)]
    pub advantages: Vec<String>,
    #[serde(default)]
    pub limitations: Vec<String>,
}

impl ParseStrategyAssessment {
    pub fn validate(&self) -> Result<()> {
        if self.advantages.len() > 4 || self.limitations.len() > 4 {
            bail!("strategy assessment arrays must contain at most 4 items");
        }
        if self.advantages.iter().all(|value| value.trim().is_empty())
            || self.limitations.iter().all(|value| value.trim().is_empty())
        {
            bail!("strategy assessments require advantages and limitations");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseStrategyTriage {
    pub selected_strategy: ParseStrategy,
    pub confidence: f64,
    #[serde(default)]
    pub rationale: String,
    #[serde(default)]
    pub assessments: Vec<ParseStrategyAssessment>,
}

impl ParseStrategyTriage {
    pub fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.confidence) {
            bail!("strategy triage confidence must be between 0 and 1");
        }
        if self.rationale.chars().count() > 800 {
            bail!("strategy triage rationale must contain at most 800 characters");
        }
        if self.assessments.len() > 3 {
            bail!("strategy triage must assess at most 3 strategies");
        }
        let seen: std::collections::HashSet<ParseStrategy> =
            self.assessments.iter().map(|item| item.strategy).collect();
        if seen.len() != HARD_CODED_STRATEGY_PRIORITY.len()
            || HARD_CODED_STRATEGY_PRIORITY
                .iter()
                .any(|strategy| !seen.contains(strategy))
        {
            bail!("strategy triage must assess all allowed strategies exactly once");
        }
        for assessment in &self.assessments {
            assessment.validate()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseStrategyDecision {
    pub selected_strategy: ParseStrategy,
    pub source: ParseStrategySource,
    pub confidence: f64,
    #[serde(default)]
    pub rationale: String,
    #[serde(default)]
    pub assessments: Vec<ParseStrategyAssessment>,
    #[serde(default = "hard_coded_strategy_priority")]
    pub fallback_order: Vec<ParseStrategy>,
}

fn hard_coded_strategy_priority() -> Vec<ParseStrategy> {
    HARD_CODED_STRATEGY_PRIORITY.to_vec()
}

pub fn hardcoded_strategy(
    requested: ParseStrategyRequest,
    strategy_order: &[ParseStrategy],
    disabled_strategies: &[ParseStrategy],
    reason: &str,
) -> Result<ParseStrategyDecision> {
    let expected: std::collections::HashSet<ParseStrategy> =
        HARD_CODED_STRATEGY_PRIORITY.iter().copied().collect();
    let provided: std::collections::HashSet<ParseStrategy> =
        strategy_order.iter().copied().collect();
    if strategy_order.len() != HARD_CODED_STRATEGY_PRIORITY.len() || provided != expected {
        bail!("strategy_order must contain each parser strategy exactly once");
    }
    let disabled: std::collections::HashSet<ParseStrategy> =
        disabled_strategies.iter().copied().collect();
    let mut order: Vec<ParseStrategy> = match requested {
        ParseStrategyRequest::Auto => strategy_order.to_vec(),
        requested => {
            let explicit = match requested {
                ParseStrategyRequest::LayerExcerpt => ParseStrategy::LayerExcerpt,
                ParseStrategyRequest::LayerBoundary => ParseStrategy::LayerBoundary,
                ParseStrategyRequest::PageIndex => ParseStrategy::PageIndex,
                ParseStrategyRequest::Auto => unreachable!(),
            };
            let mut order = vec![explicit];
            order.extend(
                strategy_order
                    .iter()
                    .copied()
                    .filter(|strategy| *strategy != explicit),
            );
            order
        }
    };
    order.retain(|strategy| !disabled.contains(strategy));
    if order.is_empty() {
        bail!("all parser strategies are disabled");
    }
    Ok(ParseStrategyDecision {
        selected_strategy: order[0],
        source: ParseStrategySource::Config,
        confidence: 1.0,
        rationale: reason.to_string(),
        assessments: Vec::new(),
        fallback_order: order,
    })
}

pub fn select_parse_strategy(
    requested: ParseStrategyRequest,
    context: &Value,
    triage_enabled: bool,
    triage_fn: Option<&StrategyTriageFn<'_>>,
    strategy_order: &[ParseStrategy],
    disabled_strategies: &[ParseStrategy],
    minimum_confidence: f64,
) -> Result<ParseStrategyDecision> {
    let configured = hardcoded_strategy(
        requested,
        strategy_order,
        disabled_strategies,
        "deterministic priority policy",
    )?;
    if requested != ParseStrategyRequest::Auto || !triage_enabled {
        return Ok(configured);
    }
    let Some(triage_fn) = triage_fn else {
        return Ok(configured);
    };
    match triage_fn(context) {
        Ok(triage) => {
            if triage.confidence < minimum_confidence
                || disabled_strategies.contains(&triage.selected_strategy)
            {
                Ok(ParseStrategyDecision {
                    source: ParseStrategySource::LlmTriageFallback,
                    rationale: if triage.confidence < minimum_confidence {
                        format!(
                            "triage confidence {:.2} below {:.2}",
                            triage.confidence, minimum_confidence
                        )
                    } else {
                        format!(
                            "triage selected disabled strategy {}",
                            serde_json::to_string(&triage.selected_strategy).unwrap_or_default()
                        )
                    },
                    ..configured
                })
            } else {
                Ok(ParseStrategyDecision {
                    selected_strategy: triage.selected_strategy,
                    source: ParseStrategySource::LlmTriage,
                    confidence: triage.confidence,
                    rationale: triage.rationale,
                    assessments: triage.assessments,
                    fallback_order: strategy_order.to_vec(),
                })
            }
        }
        Err(error) => Ok(ParseStrategyDecision {
            source: ParseStrategySource::LlmTriageFallback,
            rationale: format!("triage failed: {error}"),
            ..configured
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn triage() -> ParseStrategyTriage {
        ParseStrategyTriage {
            selected_strategy: ParseStrategy::PageIndex,
            confidence: 0.8,
            rationale: "structured page-like source".to_string(),
            assessments: HARD_CODED_STRATEGY_PRIORITY
                .iter()
                .map(|strategy| ParseStrategyAssessment {
                    strategy: *strategy,
                    advantages: vec!["cheap".to_string()],
                    limitations: vec!["approximate".to_string()],
                })
                .collect(),
        }
    }

    #[test]
    fn hardcoded_auto_uses_configured_priority() {
        let decision = hardcoded_strategy(
            ParseStrategyRequest::Auto,
            &HARD_CODED_STRATEGY_PRIORITY,
            &[],
            "deterministic priority policy",
        )
        .unwrap();
        assert_eq!(decision.selected_strategy, ParseStrategy::LayerExcerpt);
        assert_eq!(decision.source, ParseStrategySource::Config);
    }

    #[test]
    fn hardcoded_request_promotes_explicit_strategy() {
        let decision = hardcoded_strategy(
            ParseStrategyRequest::PageIndex,
            &HARD_CODED_STRATEGY_PRIORITY,
            &[],
            "deterministic priority policy",
        )
        .unwrap();
        assert_eq!(decision.selected_strategy, ParseStrategy::PageIndex);
        assert_eq!(decision.fallback_order[0], ParseStrategy::PageIndex);
    }

    #[test]
    fn triage_replaces_auto_decision() {
        let decision = select_parse_strategy(
            ParseStrategyRequest::Auto,
            &json!({"source": "page-index"}),
            true,
            Some(&|_| Ok(triage())),
            &HARD_CODED_STRATEGY_PRIORITY,
            &[],
            0.55,
        )
        .unwrap();
        assert_eq!(decision.selected_strategy, ParseStrategy::PageIndex);
        assert_eq!(decision.source, ParseStrategySource::LlmTriage);
    }

    #[test]
    fn triage_failure_falls_back_to_hardcoded() {
        let decision = select_parse_strategy(
            ParseStrategyRequest::Auto,
            &json!({}),
            true,
            Some(&|_| bail!("provider unavailable")),
            &HARD_CODED_STRATEGY_PRIORITY,
            &[],
            0.55,
        )
        .unwrap();
        assert_eq!(decision.source, ParseStrategySource::LlmTriageFallback);
        assert!(decision.rationale.starts_with("triage failed"));
    }
}
