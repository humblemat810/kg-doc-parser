use crate::page_index_classification::{classify_block, is_setext_underline, ClassifyBlockOptions};
use crate::page_index_contracts::{
    BlockAssignment, CandidateBlock, PageIndexNodeType, PageIndexSourceFormat,
};

struct SplitPageBlocksState {
    blocks: Vec<crate::page_index_contracts::BlockSpan>,
    paragraph_start: Option<usize>,
    paragraph_start_line: usize,
    paragraph_end: usize,
    paragraph_end_line: usize,
    paragraph_indent: usize,
    saw_nonblank: bool,
}

impl SplitPageBlocksState {
    fn new() -> Self {
        Self {
            blocks: Vec::new(),
            paragraph_start: None,
            paragraph_start_line: 0,
            paragraph_end: 0,
            paragraph_end_line: 0,
            paragraph_indent: 0,
            saw_nonblank: false,
        }
    }

    fn flush_paragraph(&mut self, page_text: &str, source_format: PageIndexSourceFormat) {
        let Some(paragraph_start) = self.paragraph_start else {
            return;
        };
        let raw = &page_text[paragraph_start..self.paragraph_end];
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            let relative_start = raw.find(trimmed).unwrap_or(0);
            let start_char = paragraph_start + relative_start;
            let end_char = start_char + trimmed.chars().count() - 1;
            self.blocks.push(classify_block(
                trimmed,
                start_char,
                end_char,
                ClassifyBlockOptions {
                    source_format,
                    is_first: false,
                    has_blank_before: true,
                    has_blank_after: true,
                    line_start: if self.paragraph_start_line == 0 {
                        1
                    } else {
                        self.paragraph_start_line
                    },
                    line_end: if self.paragraph_end_line == 0 {
                        self.paragraph_start_line.max(1)
                    } else {
                        self.paragraph_end_line
                    },
                    indent: self.paragraph_indent,
                    heading_level: None,
                },
            ));
        }
        self.paragraph_start = None;
        self.paragraph_start_line = 0;
        self.paragraph_end = 0;
        self.paragraph_end_line = 0;
        self.paragraph_indent = 0;
    }
}

pub fn split_page_blocks(
    page_text: &str,
    source_format: PageIndexSourceFormat,
) -> Vec<crate::page_index_contracts::BlockSpan> {
    let mut state = SplitPageBlocksState::new();
    let lines = page_text.split_inclusive('\n').collect::<Vec<_>>();
    let mut cursor = 0usize;
    let mut index = 0usize;
    while index < lines.len() {
        let line = lines[index];
        let line_start = cursor;
        let line_end = cursor + line.chars().count();
        let line_body = line.trim_end_matches(['\r', '\n']);
        let stripped = line_body.trim();
        let indent = line_body.len() - line_body.trim_start_matches([' ', '\t']).len();
        let has_blank_before = index == 0 || lines[index - 1].trim().is_empty();
        let has_blank_after = index + 1 >= lines.len() || lines[index + 1].trim().is_empty();
        if stripped.is_empty() {
            state.flush_paragraph(page_text, source_format);
            cursor = line_end;
            index += 1;
            continue;
        }
        let next_line_body = if index + 1 < lines.len() {
            lines[index + 1].trim_end_matches(['\r', '\n'])
        } else {
            ""
        };
        if !next_line_body.is_empty() && is_setext_underline(next_line_body) {
            let underline_level: u8 = if next_line_body.trim_start().starts_with('=') {
                1
            } else {
                2
            };
            state.flush_paragraph(page_text, source_format);
            let start_char = line_start + line_body.find(stripped).unwrap_or(0);
            let end_char = start_char + stripped.chars().count() - 1;
            let line_block = classify_block(
                stripped,
                start_char,
                end_char,
                ClassifyBlockOptions {
                    source_format,
                    is_first: !state.saw_nonblank,
                    has_blank_before,
                    has_blank_after,
                    line_start: index + 1,
                    line_end: index + 1,
                    indent,
                    heading_level: Some(underline_level),
                },
            );
            state.saw_nonblank = true;
            state.blocks.push(line_block);
            cursor = line_end + lines[index + 1].chars().count();
            index += 2;
            continue;
        }
        let line_block = classify_block(
            stripped,
            line_start + line_body.find(stripped).unwrap_or(0),
            line_end.saturating_sub(1),
            ClassifyBlockOptions {
                source_format,
                is_first: !state.saw_nonblank,
                has_blank_before,
                has_blank_after,
                line_start: index + 1,
                line_end: index + 1,
                indent,
                heading_level: None,
            },
        );
        state.saw_nonblank = true;
        if line_block.node_type != PageIndexNodeType::Paragraph {
            state.flush_paragraph(page_text, source_format);
            state.blocks.push(line_block);
        } else {
            if state.paragraph_start.is_none() {
                state.paragraph_start = Some(line_start);
                state.paragraph_start_line = index + 1;
                state.paragraph_indent = indent;
            }
            state.paragraph_end = line_end;
            state.paragraph_end_line = index + 1;
        }
        cursor = line_end;
        index += 1;
    }
    state.flush_paragraph(page_text, source_format);
    state.blocks
}

pub fn extract_candidate_blocks(
    page_text: &str,
    page_number: u32,
    source_format: PageIndexSourceFormat,
) -> Vec<CandidateBlock> {
    split_page_blocks(page_text, source_format)
        .into_iter()
        .enumerate()
        .map(|(index, block)| {
            let order = index as u32 + 1;
            CandidateBlock {
                block_id: format!("p{page_number:04}-b{order:03}"),
                page_number,
                order,
                start_char: block.start_char,
                end_char: block.end_char,
                line_start: block.line_start,
                line_end: block.line_end,
                indent: block.indent,
                kind_hint: block.kind_hint,
                confidence: block.confidence,
                text: block.text,
                node_type_hint: block.node_type,
                title_hint: block.title,
                heading_level: block.heading_level,
            }
        })
        .collect()
}

pub fn candidate_heading_evidence(candidate: &CandidateBlock) -> bool {
    candidate.heading_level.is_some()
        || candidate.node_type_hint == PageIndexNodeType::Heading
        || candidate.kind_hint.starts_with("heading")
}

pub fn deterministic_block_assignments(candidates: &[CandidateBlock]) -> Vec<BlockAssignment> {
    let mut assignments = Vec::new();
    let mut stack: Vec<(u8, String)> = Vec::new();
    for candidate in candidates {
        let mut parent_id = stack.last().map(|(_, block_id)| block_id.clone());
        if candidate_heading_evidence(candidate) {
            let level = candidate.heading_level.unwrap_or(1);
            while stack
                .last()
                .is_some_and(|(stack_level, _)| *stack_level >= level)
            {
                stack.pop();
            }
            parent_id = stack.last().map(|(_, block_id)| block_id.clone());
            assignments.push(BlockAssignment {
                block_id: candidate.block_id.clone(),
                parent_id,
                node_type: PageIndexNodeType::Heading,
                title: candidate.title_hint.clone(),
                summary: String::new(),
            });
            stack.push((level, candidate.block_id.clone()));
            continue;
        }
        assignments.push(BlockAssignment {
            block_id: candidate.block_id.clone(),
            parent_id,
            node_type: candidate.node_type_hint,
            title: if candidate.title_hint.is_empty() {
                candidate.text.chars().take(120).collect()
            } else {
                candidate.title_hint.clone()
            },
            summary: String::new(),
        });
    }
    assignments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_markdown_heading_and_paragraph_candidates() {
        let page_text = "# Section One\n\nParagraph one.\n\nParagraph two.";
        let candidates = extract_candidate_blocks(page_text, 1, PageIndexSourceFormat::Markdown);
        assert_eq!(candidates.len(), 3);
        assert_eq!(candidates[0].node_type_hint, PageIndexNodeType::Heading);
        assert_eq!(candidates[0].title_hint, "Section One");
        assert_eq!(candidates[1].node_type_hint, PageIndexNodeType::Paragraph);
    }

    #[test]
    fn deterministic_assignments_nest_under_headings() {
        let page_text = "# Section One\n\nParagraph one.";
        let candidates = extract_candidate_blocks(page_text, 1, PageIndexSourceFormat::Markdown);
        let assignments = deterministic_block_assignments(&candidates);
        assert_eq!(assignments.len(), 2);
        assert_eq!(assignments[1].parent_id.as_deref(), Some("p0001-b001"));
    }
}
