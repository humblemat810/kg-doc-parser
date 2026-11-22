use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PageIndexSourceFormat {
    Text,
    Markdown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum PageIndexNodeType {
    Heading,
    Paragraph,
    Term,
    Table,
    TableRowGroup,
    TableRow,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockSpan {
    pub start_char: usize,
    pub end_char: usize,
    pub text: String,
    pub node_type: PageIndexNodeType,
    pub title: String,
    pub line_start: usize,
    pub line_end: usize,
    pub indent: usize,
    pub kind_hint: String,
    pub confidence: f64,
    pub heading_level: Option<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateBlock {
    pub block_id: String,
    pub page_number: u32,
    pub order: u32,
    pub start_char: usize,
    pub end_char: usize,
    pub line_start: usize,
    pub line_end: usize,
    pub indent: usize,
    pub kind_hint: String,
    pub confidence: f64,
    pub text: String,
    pub node_type_hint: PageIndexNodeType,
    pub title_hint: String,
    pub heading_level: Option<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockAssignment {
    pub block_id: String,
    pub parent_id: Option<String>,
    pub node_type: PageIndexNodeType,
    pub title: String,
    pub summary: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PageIndexSourceRole {
    Heading,
    Content,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageIndexBlockSpec {
    pub title: String,
    pub node_type: PageIndexNodeType,
    pub excerpt: String,
    pub display_excerpt: Option<String>,
    pub summary: String,
    pub source_role: PageIndexSourceRole,
    pub child_nodes: Vec<PageIndexBlockSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageIndexValidationResult {
    pub valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub fallback_reason: Option<String>,
}
