use anyhow::{bail, Result};
use kg_doc_parser::demo_harness::{
    prepare_demo_artifacts, write_demo_summary, DemoHarnessConfig, DemoParserMode,
};
use kg_doc_parser::page_index_contracts::PageIndexSourceFormat;
use kg_doc_parser::page_index_pipeline::parse_page_index_document_heuristic;
use kg_doc_parser::parser::{parse_file, ParsingMode};
use kg_doc_parser::runners::infer_page_index_source_format;
use kg_doc_parser::runners::run_layerwise_batch_workflow;
use kg_doc_parser::runners::run_layerwise_source_workflow;
use kg_doc_parser::runners::run_ocr_batch_workflow;
use kg_doc_parser::runners::run_ocr_source_workflow;
use kg_doc_parser::runners::run_page_index_batch_workflow;
use kg_doc_parser::runners::run_page_index_source_workflow;
use serde_json::{json, Value};
use std::path::PathBuf;

struct IdentityOcrRunner;

impl kg_doc_parser::runners::OcrRunner for IdentityOcrRunner {
    fn provider_name(&self) -> &str {
        "fake"
    }

    fn run_page(&self, image_path: &std::path::Path, page_number: u64) -> anyhow::Result<Value> {
        Ok(json!({
            "metadata": {
                "ocr_json_version": "0.1",
                "ocr_model_name": "identity",
                "source_image_path": image_path,
            },
            "pdf_page_num": page_number,
            "OCR_text_clusters": [{"text": ""}],
            "non_text_objects": [],
        }))
    }
}

fn usage() -> ! {
    eprintln!("usage: kg-doc-parser <parse|page-index|page-index-workflow|layerwise-workflow|ocr-workflow> --input <path> [--batch] [--mode snippet|delimiter|markdown|text] [--max-depth <depth>] [--provider fake] [--output <path>] [--output-dir <dir>]");
    std::process::exit(2);
}

fn main() -> Result<()> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let command = arguments.first().map(String::as_str).unwrap_or_default();
    if command != "parse"
        && command != "page-index"
        && command != "page-index-workflow"
        && command != "layerwise-workflow"
        && command != "ocr-workflow"
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
    let mut title: Option<String> = None;
    let mut text: Option<String> = None;
    let mut provider: Option<String> = None;
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
            "--title" => {
                title = Some(
                    arguments
                        .get(index + 1)
                        .ok_or_else(|| anyhow::anyhow!("--title requires a value"))?
                        .to_string(),
                );
                index += 2;
            }
            "--text" => {
                text = Some(
                    arguments
                        .get(index + 1)
                        .ok_or_else(|| anyhow::anyhow!("--text requires a value"))?
                        .to_string(),
                );
                index += 2;
            }
            "--provider" => {
                provider = Some(
                    arguments
                        .get(index + 1)
                        .ok_or_else(|| anyhow::anyhow!("--provider requires a value"))?
                        .to_string(),
                );
                index += 2;
            }
            _ => bail!("unknown argument: {}", arguments[index]),
        }
    }
    let Some(input) = input else {
        usage();
    };
    if command == "ocr-workflow" {
        let Some(output_dir) = output_dir else {
            bail!("ocr-workflow requires --output-dir");
        };
        let runner = match provider.as_deref().unwrap_or("fake") {
            "fake" => IdentityOcrRunner,
            unsupported => bail!("unsupported OCR provider: {unsupported}"),
        };
        if batch {
            let results = run_ocr_batch_workflow(
                std::slice::from_ref(&input),
                output_dir.as_path(),
                &runner,
                None,
            )?;
            let payload = serde_json::to_value(&results)?;
            println!("{}", serde_json::to_string_pretty(&payload)?);
        } else {
            let result = run_ocr_source_workflow(
                &input,
                output_dir.as_path(),
                &runner,
                None,
                &output_dir.join("workflow-events.jsonl"),
            )?;
            println!("{}", serde_json::to_string_pretty(&result.extra)?);
        }
        return Ok(());
    }
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
    if command == "layerwise-workflow" {
        let parsing_mode = ParsingMode::parse(&mode)?;
        let Some(output_dir) = output_dir else {
            bail!("layerwise-workflow requires --output-dir");
        };
        if batch {
            let results = run_layerwise_batch_workflow(
                &[input],
                output_dir.as_path(),
                parsing_mode,
                max_depth,
            )?;
            let payload = serde_json::to_value(&results)?;
            println!("{}", serde_json::to_string_pretty(&payload)?);
        } else {
            let result = run_layerwise_source_workflow(
                &input,
                output_dir.as_path(),
                parsing_mode,
                max_depth,
            )?;
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
        if let Some(title) = title {
            config.title = title;
        }
        if let Some(text) = text {
            config.text = text;
        }
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
