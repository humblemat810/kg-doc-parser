use regex::Regex;
use std::sync::LazyLock;

static PAGE_BREAK: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)\f|^\s*--- PAGE BREAK ---\s*$").unwrap());

pub fn split_pages(raw_text: &str) -> Vec<String> {
    PAGE_BREAK
        .split(raw_text)
        .map(|page| page.trim_matches('\n').to_string())
        .filter(|page| !page.trim().is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_explicit_page_breaks() {
        let raw_text = "one\n\n--- PAGE BREAK ---\n\ntwo";
        assert_eq!(split_pages(raw_text), vec!["one", "two"]);
    }
}
