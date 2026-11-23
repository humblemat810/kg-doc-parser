use anyhow::{bail, Result};
use kg_doc_parser::demo_harness::{
    prepare_demo_artifacts, write_demo_summary, DemoHarnessConfig, DemoParserMode,
};
use kg_doc_parser::page_index_contracts::PageIndexSourceFormat;
use kg_doc_parser::page_index_pipeline::parse_page_index_document_heuristic;
use kg_doc_parser::parser::{parse_file, ParsingMode};
use kg_doc_parser::runners::infer_page_index_source_format;
use kg_doc_parser::runners::run_page_index_batch_workflow;
use kg_doc_parser::runners::run_page_index_source_workflow;
use serde_json::json;
use std::path::PathBuf;

fn usage() -> ! {
    eprintln!("usage: kg-doc-parser <parse|page-index|page-index-workflow> --input <path> [--batch] [--mode snippet|delimiter|markdown|text] [--max-depth <depth>] [--output <path>] [--output-dir <dir>]");
    std::process::exit(2);
}

fn main() -> Result<()> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let command = arguments.first().map(String::as_str).unwrap_or_default();
    if command != "parse"
        && command != "page-index"
        && command != "page-index-workflow"
        && command != "demo-summary"
    {
        usage();
    }
    let mut input = None;
    let mut output = "-";
    let mut output_dir = None;
    let mut mode = if command == "page-index" {
        "markdown".to_string()
    } else {
        "snippet".to_string()
    };
    let mut max_depth = 10;
    let mut batch = false;
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
            "--output-dir" => {
                output_dir =
                    Some(PathBuf::from(arguments.get(index + 1).ok_or_else(
                        || anyhow::anyhow!("--output-dir requires a path"),
                    )?));
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
            "--batch" => {
                batch = true;
                index += 1;
            }
            _ => bail!("unknown argument: {}", arguments[index]),
        }
    }
    let Some(input) = input else {
        usage();
    };
    if command == "page-index-workflow" {
        let source_format = infer_page_index_source_format(&input, &mode)?;
        let Some(output_dir) = output_dir else {
            bail!("page-index-workflow requires --output-dir");
        };
        if batch {
            let results =
                run_page_index_batch_workflow(&[input], output_dir.as_path(), source_format)?;
            let payload = serde_json::to_value(&results)?;
            println!("{}", serde_json::to_string_pretty(&payload)?);
        } else {
            let result =
                run_page_index_source_workflow(&input, output_dir.as_path(), source_format)?;
            println!("{}", serde_json::to_string_pretty(&result.extra)?);
        }
        return Ok(());
    }
    if command == "demo-summary" {
        let Some(output_dir) = output_dir else {
            bail!("demo-summary requires --output-dir");
        };
        let mut config = DemoHarnessConfig {
            output_dir,
            ..Default::default()
        };
        config.document_id = input
            .file_stem()
            .and_then(std::ffi::OsStr::to_str)
            .unwrap_or("demo-doc")
            .to_string();
        config.parser_mode = match mode.as_str() {
            "fake_layered" => DemoParserMode::FakeLayered,
            "legacy_cached" => DemoParserMode::LegacyCached,
            _ => bail!("demo-summary mode must be fake_layered or legacy_cached"),
        };
        let artifacts = prepare_demo_artifacts(&config)?;
        write_demo_summary(&artifacts, &config, "children")?;
        let summary = serde_json::to_string_pretty(&artifacts.summary_path)?;
        println!("{summary}");
        return Ok(());
    }
    let payload = if command == "page-index" {
        let source_format = match mode.as_str() {
            "auto" => {
                if input
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .map(|extension| extension.to_lowercase())
                    == Some("md".to_string())
                {
                    PageIndexSourceFormat::Markdown
                } else {
                    PageIndexSourceFormat::Text
                }
            }
            "markdown" => PageIndexSourceFormat::Markdown,
            "text" => PageIndexSourceFormat::Text,
            _ => bail!("page-index mode must be auto, markdown, or text"),
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
