use anyhow::{bail, Result};
use kg_doc_parser::parser::{parse_file, ParsingMode};
use serde_json::json;
use std::path::PathBuf;

fn usage() -> ! {
    eprintln!("usage: kg-doc-parser parse --input <path> [--mode snippet|delimiter] [--max-depth <depth>] [--output <path>]");
    std::process::exit(2);
}

fn main() -> Result<()> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    if arguments.first().map(String::as_str) != Some("parse") {
        usage();
    }
    let mut input = None;
    let mut output = "-";
    let mut mode = ParsingMode::Snippet;
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
                mode = ParsingMode::parse(
                    arguments
                        .get(index + 1)
                        .ok_or_else(|| anyhow::anyhow!("--mode requires a value"))?,
                )?;
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
    let result = parse_file(&input, mode, max_depth)?;
    let payload = json!({
        "document_tree": result.document_tree,
        "source_map": result.source_map,
    });
    let encoded = serde_json::to_string_pretty(&payload)?;
    if output == "-" {
        println!("{encoded}");
    } else {
        std::fs::write(output, encoded)?;
    }
    Ok(())
}
