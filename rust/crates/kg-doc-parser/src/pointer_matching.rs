#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchDiagnostic {
    Exact { collapsed: bool },
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
    if collapsed_occurrences.is_empty() {
        return (Vec::new(), None);
    }
    let mapped = collapsed_occurrences
        .iter()
        .map(|(start, end)| (source_spans[*start].0, source_spans[*end].1))
        .collect();
    (mapped, Some(MatchDiagnostic::Exact { collapsed: true }))
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
        let (matches, diagnostic) = soft_exact_positions("alpha\n\tbeta", "alpha beta");
        assert_eq!(matches, vec![(0, 10)]);
        assert_eq!(diagnostic, Some(MatchDiagnostic::Exact { collapsed: true }));
    }

    #[test]
    fn keeps_repair_opt_in() {
        assert_eq!(
            soft_exact_positions("alpha beta", "alpha zeta"),
            (Vec::new(), None)
        );
    }
}
