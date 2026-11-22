use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "name")]
pub enum MatchDiagnostic {
    #[serde(rename = "exact")]
    Exact { collapsed: bool },
    #[serde(rename = "LCSseq.normalized_similarity")]
    Lcsseq { threshold: f64, collapsed: bool },
    #[serde(rename = "difflib.SequenceMatcher")]
    Difflib { threshold: f64, collapsed: bool },
}

pub fn safe_slice(text: &str, start: isize, end_inclusive: isize) -> String {
    let chars: Vec<char> = text.chars().collect();
    let end_exclusive = if end_inclusive == -1 {
        chars.len() as isize
    } else {
        end_inclusive + 1
    };
    let start = start.max(0) as usize;
    let end_exclusive = end_exclusive.max(0) as usize;
    if start >= end_exclusive || start >= chars.len() {
        return String::new();
    }
    chars[start..end_exclusive.min(chars.len())]
        .iter()
        .collect()
}

pub fn all_exact_occurrences(haystack: &str, needle: &str) -> Vec<(usize, usize)> {
    if needle.is_empty() {
        return Vec::new();
    }
    let mut occurrences = Vec::new();
    let mut byte_index = 0;
    while let Some(found) = haystack[byte_index..].find(needle) {
        let absolute_byte_index = byte_index + found;
        let char_index = haystack[..absolute_byte_index].chars().count();
        let end_inclusive = char_index + needle.chars().count() - 1;
        occurrences.push((char_index, end_inclusive));
        byte_index = absolute_byte_index + needle.len();
    }
    occurrences
}

pub fn best_occurrence_by_proximity(
    occurrences: &[(usize, usize)],
    proposed_start: usize,
) -> Option<(usize, usize)> {
    occurrences
        .iter()
        .copied()
        .min_by_key(|(start, _)| start.abs_diff(proposed_start))
}

pub fn whitespace_collapse(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn soft_exact_positions(
    source_text: &str,
    verbatim: &str,
    fuzzy_threshold: Option<f64>,
    fuzzy_len_stretch: f64,
    fuzzy_stride_frac: f64,
) -> (Vec<(usize, usize)>, Option<MatchDiagnostic>) {
    let exact_occurrences = all_exact_occurrences(source_text, verbatim);
    if !exact_occurrences.is_empty() {
        return (
            exact_occurrences,
            Some(MatchDiagnostic::Exact { collapsed: false }),
        );
    }

    let normalized_query = whitespace_collapse(verbatim);
    if normalized_query.is_empty() {
        return (Vec::new(), None);
    }

    let mut source_spans: Vec<(usize, usize)> = Vec::new();
    let mut collapsed_chars: Vec<char> = Vec::new();
    let mut characters = source_text.char_indices().peekable();
    while let Some((byte_index, character)) = characters.next() {
        let char_index = source_text[..byte_index].chars().count();
        if character.is_whitespace() {
            let mut last_char_index = char_index;
            while characters
                .peek()
                .is_some_and(|(_, next_character)| next_character.is_whitespace())
            {
                let (next_byte_index, _) = characters.next().unwrap();
                last_char_index = source_text[..next_byte_index].chars().count();
            }
            collapsed_chars.push(' ');
            source_spans.push((char_index, last_char_index));
        } else {
            collapsed_chars.push(character);
            source_spans.push((char_index, char_index));
        }
    }

    let collapsed_text: String = collapsed_chars.iter().collect();
    let collapsed_occurrences = all_exact_occurrences(&collapsed_text, &normalized_query);
    if collapsed_occurrences.is_empty() && fuzzy_threshold.is_none() {
        return (Vec::new(), None);
    }

    if !collapsed_occurrences.is_empty() {
        let mapped = collapsed_occurrences
            .iter()
            .map(|(start, end)| (source_spans[*start].0, source_spans[*end].1))
            .collect();
        return (mapped, Some(MatchDiagnostic::Exact { collapsed: true }));
    }

    let Some(fuzzy_threshold) = fuzzy_threshold else {
        return (Vec::new(), None);
    };
    let map_back = |start: usize, end: usize| (source_spans[start].0, source_spans[end].1);
    let query_length = normalized_query.chars().count();
    let collapsed_length = collapsed_text.chars().count();
    if query_length == 0 || collapsed_length == 0 {
        return (Vec::new(), None);
    }

    let mut threshold = fuzzy_threshold;
    if threshold <= 1.0 {
        threshold *= 100.0;
    }
    let min_length = ((query_length as f64 * (1.0 - fuzzy_len_stretch)).floor() as usize).max(1);
    let max_length =
        ((query_length as f64 * (1.0 + fuzzy_len_stretch)).ceil() as usize).max(min_length);
    let stride = ((query_length as f64 * fuzzy_stride_frac).round().max(1.0) as usize).max(1);

    let query_chars: Vec<char> = normalized_query.chars().collect();
    let collapsed_chars: Vec<char> = collapsed_text.chars().collect();
    let mut best_score = -1.0;
    let mut best_span: Option<(usize, usize)> = None;
    for window_length in [min_length, query_length, max_length] {
        if window_length > collapsed_length {
            continue;
        }
        for start in (0..collapsed_length - window_length + 1).step_by(stride) {
            let window = &collapsed_chars[start..start + window_length];
            let score = lcs_similarity(&query_chars, window) * 100.0;
            if score > best_score {
                best_score = score;
                best_span = Some((start, start + window_length - 1));
            }
        }
    }

    if let Some(span) = best_span {
        if best_score >= threshold {
            let (start, end) = map_back(span.0, span.1);
            return (
                vec![(start, end)],
                Some(MatchDiagnostic::Lcsseq {
                    threshold,
                    collapsed: true,
                }),
            );
        }
    }

    if let Some(span) = longest_common_substring(&normalized_query, &collapsed_text) {
        let window: String = collapsed_chars[span.0..span.1 + 1].iter().collect();
        best_score = ratio_similarity(&normalized_query, &window) * 100.0;
        if best_score >= threshold {
            let (start, end) = map_back(span.0, span.1);
            return (
                vec![(start, end)],
                Some(MatchDiagnostic::Difflib {
                    threshold,
                    collapsed: true,
                }),
            );
        }
    }

    if let Some(span) = longest_common_substring(verbatim, source_text) {
        let window_start = span.1 + 1 - span.0;
        let window: String = if window_start <= collapsed_length {
            collapsed_chars[span.0..window_start].iter().collect()
        } else {
            String::new()
        };
        best_score = ratio_similarity(verbatim, &window) * 100.0;
        if best_score >= threshold {
            return (
                vec![span],
                Some(MatchDiagnostic::Difflib {
                    threshold,
                    collapsed: false,
                }),
            );
        }
    }

    (Vec::new(), None)
}

fn lcs_similarity(query: &[char], window: &[char]) -> f64 {
    if query.is_empty() || window.is_empty() {
        return 0.0;
    }
    let mut previous = vec![0u64; window.len() + 1];
    let mut current = vec![0u64; window.len() + 1];
    for &query_character in query {
        for (window_index, &window_character) in window.iter().enumerate() {
            current[window_index + 1] = if query_character == window_character {
                previous[window_index] + 1
            } else {
                previous[window_index + 1].max(current[window_index])
            };
        }
        std::mem::swap(&mut previous, &mut current);
        current.fill(0);
    }
    previous[window.len()] as f64 / query.len().max(window.len()) as f64
}

fn longest_common_substring(query: &str, text: &str) -> Option<(usize, usize)> {
    let query_chars: Vec<char> = query
        .chars()
        .map(|character| character.to_lowercase().next().unwrap_or(character))
        .collect();
    let text_chars: Vec<char> = text
        .chars()
        .map(|character| character.to_lowercase().next().unwrap_or(character))
        .collect();
    let mut previous = vec![0u64; text_chars.len() + 1];
    let mut current = vec![0u64; text_chars.len() + 1];
    let mut best_length: usize = 0;
    let mut best_end = 0;
    for &query_character in &query_chars {
        for (text_index, &text_character) in text_chars.iter().enumerate() {
            current[text_index + 1] = if query_character == text_character {
                previous[text_index] + 1
            } else {
                0
            };
            if current[text_index + 1] > best_length as u64 {
                best_length = current[text_index + 1] as usize;
                best_end = text_index + 1;
            }
        }
        std::mem::swap(&mut previous, &mut current);
        current.fill(0);
    }
    if best_length == 0 {
        return None;
    }
    Some((best_end - best_length, best_end - 1))
}

fn ratio_similarity(query: &str, window: &str) -> f64 {
    let query_chars: Vec<char> = query.chars().collect();
    let window_chars: Vec<char> = window.chars().collect();
    if query_chars.is_empty() || window_chars.is_empty() {
        return 0.0;
    }
    let mut previous = vec![0u64; window_chars.len() + 1];
    let mut current = vec![0u64; window_chars.len() + 1];
    for &query_character in &query_chars {
        for (window_index, &window_character) in window_chars.iter().enumerate() {
            current[window_index + 1] = if query_character == window_character {
                previous[window_index] + 1
            } else {
                previous[window_index + 1].max(current[window_index])
            };
        }
        std::mem::swap(&mut previous, &mut current);
        current.fill(0);
    }
    2.0 * previous[window_chars.len()] as f64 / (query_chars.len() + window_chars.len()) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_inclusive_offsets() {
        assert_eq!(safe_slice("alpha beta", 6, 9), "beta");
        assert_eq!(safe_slice("alpha beta", 0, -1), "alpha beta");
        assert_eq!(
            all_exact_occurrences("alpha alpha", "alpha"),
            vec![(0, 4), (6, 10)]
        );
    }

    #[test]
    fn reports_whitespace_normalization() {
        let (matches, diagnostic) =
            soft_exact_positions("alpha\n\tbeta", "alpha beta", None, 0.25, 0.10);
        assert_eq!(matches, vec![(0, 10)]);
        assert_eq!(diagnostic, Some(MatchDiagnostic::Exact { collapsed: true }));
    }

    #[test]
    fn keeps_repair_opt_in() {
        assert_eq!(
            soft_exact_positions("alpha beta", "alpha zeta", None, 0.25, 0.10),
            (Vec::new(), None)
        );
    }

    #[test]
    fn fuzzy_repair_is_opt_in() {
        let (matches, diagnostic) =
            soft_exact_positions("alpha beta", "alpha zeta", Some(0.8), 0.25, 0.10);
        assert_eq!(matches, vec![(0, 9)]);
        assert!(matches!(
            diagnostic,
            Some(MatchDiagnostic::Lcsseq {
                collapsed: true,
                ..
            })
        ));
    }

    #[test]
    fn fuzzy_repair_honors_threshold() {
        let (matches, diagnostic) =
            soft_exact_positions("alpha beta", "alpha zeta", Some(0.99), 0.25, 0.10);
        assert!(matches.is_empty());
        assert!(diagnostic.is_none());
    }
}
