use anyhow::{bail, Result};
use kg_doc_parser::page_index_contracts::PageIndexSourceFormat;
use kg_doc_parser::page_index_pipeline::parse_page_index_document_heuristic;
use kg_doc_parser::parser::{parse_file, ParsingMode};
use serde_json::json;
use std::path::PathBuf;

fn usage() -> ! {
    eprintln!("usage: kg-doc-parser <parse|page-index> --input <path> [--mode snippet|delimiter|markdown|text] [--max-depth <depth>] [--output <path>]");
    std::process::exit(2);
}

fn main() -> Result<()> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let command = arguments.first().map(String::as_str).unwrap_or_default();
    if command != "parse" && command != "page-index" {
        usage();
    }
    let mut input = None;
    let mut output = "-";
    let mut mode = if command == "page-index" {
        "markdown".to_string()
    } else {
        "snippet".to_string()
    };
    let mut max_depth = 10;
    let mut index = 1;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--input" => {
                input = Some(PathBuf::from(
                    arguments
                        .get(index + 1)
                        .ok_or_else(|| anyhow::anyhow!("--input requires a path"))?,
                ));
                index += 2;
            }
            "--output" => {
                output = arguments
                    .get(index + 1)
                    .map(String::as_str)
                    .ok_or_else(|| anyhow::anyhow!("--output requires a path"))?;
                index += 2;
            }
            "--mode" => {
                mode = arguments
                    .get(index + 1)
                    .ok_or_else(|| anyhow::anyhow!("--mode requires a value"))?
                    .to_string();
                index += 2;
            }
            "--max-depth" => {
                max_depth = arguments
                    .get(index + 1)
                    .ok_or_else(|| anyhow::anyhow!("--max-depth requires a value"))?
                    .parse()?;
                index += 2;
            }
            _ => bail!("unknown argument: {}", arguments[index]),
        }
    }
    let Some(input) = input else {
        usage();
    };
    let payload = if command == "page-index" {
        let source_format = match mode.as_str() {
            "markdown" => PageIndexSourceFormat::Markdown,
            "text" => PageIndexSourceFormat::Text,
            _ => bail!("page-index mode must be markdown or text"),
        };
        let text = std::fs::read_to_string(&input)?;
        let document_id = input
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("document");
        let result =
            parse_page_index_document_heuristic(document_id, "Document", &text, source_format)?;
        json!({
            "semantic_tree": result.semantic_tree,
            "diagnostics": result.diagnostics,
        })
    } else {
        let parsing_mode = ParsingMode::parse(&mode)?;
        let result = parse_file(&input, parsing_mode, max_depth)?;
        json!({
            "document_tree": result.document_tree,
            "source_map": result.source_map,
        })
    };
    let encoded = serde_json::to_string_pretty(&payload)?;
    if output == "-" {
        println!("{encoded}");
    } else {
        std::fs::write(output, encoded)?;
    }
    Ok(())
}
