use crate::page_index_contracts::{BlockSpan, PageIndexNodeType, PageIndexSourceFormat};
use regex::Regex;
use std::sync::LazyLock;

static SETEXT_UNDERLINE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^=+|-+$").unwrap());
static NUMERIC_GROUP: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b\d+\b(?:\s+\b\d+\b){2,}").unwrap());
static MARKDOWN_TABLE_SEPARATOR_CELL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\s*:?-{3,}:?\s*$").unwrap());
static MARKDOWN_HEADING: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(#{1,6})\s+(.*)$").unwrap());
static PLAIN_HEADING: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(?i:Section|Clause|Article|Definitions)\b[:\s].*").unwrap());
static NUMBERED_HEADING: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\d+(?:\.\d+){0,4}(?:[.)])?\s+\S+").unwrap());
static LEGAL_CLAUSE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\s*\((?:\d+|[a-z]|[ivx]+)\)\s+\S+").unwrap());
static BULLET_ITEM: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\s*(?:[-*+]|\d{1,3}[.)])\s+\S+").unwrap());
static ITEM_PREFIX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\s*(?:[-*+]|\d{1,3}[.)]|\((?:\d+|[a-z]|[ivx]+)\))\s+").unwrap());

pub fn is_setext_underline(line: &str) -> bool {
    let stripped = line.trim();
    !stripped.is_empty() && SETEXT_UNDERLINE.is_match(stripped)
}

pub fn is_numeric_heavy_or_table_like(text: &str) -> bool {
    let compact = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.is_empty() {
        return false;
    }
    let digit_count = compact.chars().filter(char::is_ascii_digit).count();
    let alpha_count = compact
        .chars()
        .filter(|character| character.is_alphabetic())
        .count();
    let pipe_count = compact.matches('|').count();
    let tab_count = compact.matches('\t').count();
    if pipe_count >= 2 || tab_count >= 2 {
        return true;
    }
    if alpha_count == 0 && digit_count >= 3 {
        return true;
    }
    if digit_count >= 6 && digit_count > alpha_count {
        return true;
    }
    NUMERIC_GROUP.is_match(&compact)
}

pub fn is_markdown_table_row(text: &str) -> bool {
    let stripped = text.trim();
    if stripped.starts_with("```") || stripped.starts_with("~~~") {
        return false;
    }
    let cells = stripped
        .trim_matches('|')
        .split('|')
        .map(str::trim)
        .collect::<Vec<_>>();
    cells.len() >= 2 && stripped.contains('|')
}

pub fn is_markdown_table_block(text: &str, source_format: PageIndexSourceFormat) -> bool {
    if source_format != PageIndexSourceFormat::Markdown {
        return false;
    }
    let lines = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    if lines.len() < 2 || !lines.iter().all(|line| is_markdown_table_row(line)) {
        return false;
    }
    let separator = lines[1]
        .trim_matches('|')
        .split('|')
        .map(str::trim)
        .collect::<Vec<_>>();
    separator.len() >= 2
        && separator
            .iter()
            .all(|cell| MARKDOWN_TABLE_SEPARATOR_CELL.is_match(cell))
}

pub struct ClassifyBlockOptions {
    pub source_format: PageIndexSourceFormat,
    pub is_first: bool,
    pub has_blank_before: bool,
    pub has_blank_after: bool,
    pub line_start: usize,
    pub line_end: usize,
    pub indent: usize,
    pub heading_level: Option<u8>,
}

impl Default for ClassifyBlockOptions {
    fn default() -> Self {
        Self {
            source_format: PageIndexSourceFormat::Text,
            is_first: false,
            has_blank_before: false,
            has_blank_after: false,
            line_start: 1,
            line_end: 1,
            indent: 0,
            heading_level: None,
        }
    }
}

pub fn classify_block(
    text: &str,
    start_char: usize,
    end_char: usize,
    options: ClassifyBlockOptions,
) -> BlockSpan {
    let stripped = text.trim();
    let first_line = stripped.lines().next().unwrap_or_default().trim();
    let word_count = first_line.split_whitespace().count();
    let is_sentence_like = first_line.ends_with(['.', '!', '?']);
    let surrounded_by_blank_lines = options.has_blank_before && options.has_blank_after;
    let numeric_heavy = is_numeric_heavy_or_table_like(first_line);

    if is_markdown_table_block(stripped, options.source_format) {
        return BlockSpan {
            start_char,
            end_char,
            text: stripped.to_string(),
            node_type: PageIndexNodeType::Table,
            title: "Table".to_string(),
            line_start: options.line_start,
            line_end: options.line_end,
            indent: options.indent,
            kind_hint: "markdown_table".to_string(),
            confidence: 0.98,
            heading_level: None,
        };
    }

    if let Some(heading) = MARKDOWN_HEADING.captures(first_line) {
        if options.source_format == PageIndexSourceFormat::Markdown {
            let level = heading[1].chars().count() as u8;
            let title = heading[2].trim();
            let title = if title.is_empty() {
                first_line.trim_start_matches('#').trim()
            } else {
                title
            };
            return BlockSpan {
                start_char,
                end_char,
                text: stripped.to_string(),
                node_type: PageIndexNodeType::Heading,
                title: title.to_string(),
                line_start: options.line_start,
                line_end: options.line_end,
                indent: options.indent,
                kind_hint: "heading_markdown".to_string(),
                confidence: 0.98,
                heading_level: Some(level),
            };
        }
    }

    if let Some(level) = options.heading_level {
        return BlockSpan {
            start_char,
            end_char,
            text: stripped.to_string(),
            node_type: PageIndexNodeType::Heading,
            title: first_line.trim_end_matches(':').to_string(),
            line_start: options.line_start,
            line_end: options.line_end,
            indent: options.indent,
            kind_hint: "heading_setext".to_string(),
            confidence: 0.97,
            heading_level: Some(level),
        };
    }

    let plain_heading = PLAIN_HEADING.is_match(first_line);
    let numbered_heading = !numeric_heavy && NUMBERED_HEADING.is_match(first_line);
    let legal_clause = LEGAL_CLAUSE.is_match(first_line);
    let bullet_item = BULLET_ITEM.is_match(first_line);
    let all_caps_heading = word_count <= 8
        && first_line.chars().any(char::is_alphabetic)
        && first_line.to_uppercase() == first_line
        && !is_sentence_like
        && !numeric_heavy
        && (options.is_first || surrounded_by_blank_lines);
    let title_like_first_line =
        options.is_first && word_count <= 10 && !is_sentence_like && !numeric_heavy;
    let short_title_case = word_count <= 12
        && !is_sentence_like
        && first_line.chars().next().is_some_and(char::is_uppercase)
        && first_line.chars().any(char::is_alphabetic)
        && !numeric_heavy
        && (options.is_first || surrounded_by_blank_lines);

    if plain_heading
        || numbered_heading
        || all_caps_heading
        || title_like_first_line
        || short_title_case
    {
        let level: u8 = if numbered_heading {
            (first_line.matches('.').count() as u8 + 2).max(2)
        } else if title_like_first_line || (all_caps_heading && options.is_first) {
            1
        } else {
            2
        };
        let mut confidence: f64 = if plain_heading || numbered_heading {
            0.95
        } else {
            0.76
        };
        if surrounded_by_blank_lines {
            confidence = (confidence + 0.06).min(0.98);
        }
        if numeric_heavy {
            confidence = (confidence - 0.2).max(0.4);
        }
        return BlockSpan {
            start_char,
            end_char,
            text: stripped.to_string(),
            node_type: PageIndexNodeType::Heading,
            title: first_line.trim_end_matches(':').to_string(),
            line_start: options.line_start,
            line_end: options.line_end,
            indent: options.indent,
            kind_hint: "heading_plain".to_string(),
            confidence,
            heading_level: Some(level),
        };
    }

    if legal_clause || bullet_item {
        let title = ITEM_PREFIX.replace(first_line, "").trim().to_string();
        return BlockSpan {
            start_char,
            end_char,
            text: stripped.to_string(),
            node_type: PageIndexNodeType::Term,
            title: if title.is_empty() {
                first_line.to_string()
            } else {
                title
            },
            line_start: options.line_start,
            line_end: options.line_end,
            indent: options.indent,
            kind_hint: "term_list_item".to_string(),
            confidence: 0.74,
            heading_level: None,
        };
    }

    let title = first_line
        .chars()
        .take(120)
        .collect::<String>()
        .trim_end()
        .to_string();
    BlockSpan {
        start_char,
        end_char,
        text: stripped.to_string(),
        node_type: PageIndexNodeType::Paragraph,
        title,
        line_start: options.line_start,
        line_end: options.line_end,
        indent: options.indent,
        kind_hint: if numeric_heavy {
            "paragraph_table_like".to_string()
        } else {
            "paragraph".to_string()
        },
        confidence: if numeric_heavy {
            0.62
        } else if word_count > 1 {
            0.8
        } else {
            0.65
        },
        heading_level: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_setext_underlines_and_table_like_text() {
        assert!(is_setext_underline("==="));
        assert!(is_setext_underline("=="));
        assert!(is_numeric_heavy_or_table_like("12 34 56"));
        assert!(!is_numeric_heavy_or_table_like("alpha beta"));
    }

    #[test]
    fn detects_markdown_tables_and_headings() {
        let table = "| a | b |\n| --- | --- |";
        assert!(is_markdown_table_block(
            table,
            PageIndexSourceFormat::Markdown
        ));
        let heading = classify_block(
            "# Section One",
            0,
            12,
            ClassifyBlockOptions {
                source_format: PageIndexSourceFormat::Markdown,
                is_first: true,
                ..Default::default()
            },
        );
        assert_eq!(heading.node_type, PageIndexNodeType::Heading);
        assert_eq!(heading.title, "Section One");
        assert_eq!(heading.heading_level, Some(1));
    }
}
