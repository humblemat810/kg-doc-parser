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
