use anyhow::Result;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn draw_page(path: &Path, title: &str, lines: &[&str]) -> Result<()> {
    let width = 1400;
    let height = 1000;
    let mut pixels = vec![255_u8; (width as usize) * (height as usize) * 3];
    let mut fill = |x: u32, y: u32, extent: u32| {
        if x < width && y < height {
            for offset in 0..extent {
                let index =
                    ((y as usize * width as usize) + (x + offset).min(width - 1) as usize) * 3;
                pixels[index] = 0;
                pixels[index + 1] = 0;
                pixels[index + 2] = 0;
            }
        }
    };
    fill(80, 80, (title.len().min(500) as u32).max(100));
    for (index, line) in lines.iter().enumerate() {
        fill(
            80,
            180 + (index as u32 * 90),
            (line.len().min(600) as u32).max(100),
        );
    }

    let file = std::fs::File::create(path)?;
    let mut encoder = png::Encoder::new(file, width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&pixels)?;
    Ok(())
}

fn pdf_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('(', "\\(")
        .replace(')', "\\)")
}

fn build_pdf(path: &Path) -> Result<()> {
    let mut payloads = Vec::<String>::new();
    payloads.push("<< /Type /Catalog /Pages 2 0 R >>".to_string());
    payloads.push("<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>".to_string());
    for content_object in [6, 7] {
        payloads.push(format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 1400 1000] /Resources << /Font << /F1 5 0 R >> >> /Contents {content_object} 0 R >>"
        ));
    }
    payloads.push("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_string());
    for (title, lines) in [
        (
            "OCR Workflow Smoke Page 1",
            [
                "Section 1. Overview",
                "Alpha clause: the workflow should preserve page order.",
                "Beta clause: per-page OCR artifacts should remain inspectable.",
            ],
        ),
        (
            "OCR Workflow Smoke Page 2",
            [
                "Section 2. Review",
                "Gamma clause: downstream source maps should include OCR text.",
                "Delta clause: resume state should survive reruns.",
            ],
        ),
    ] {
        let mut stream = format!(
            "BT /F1 28 Tf 80 920 Td ({}) Tj /F1 14 Tf 0 -80 Td",
            pdf_escape(title)
        );
        for line in lines {
            stream.push_str(&format!(" ({}) Tj 0 -50 Td", pdf_escape(line)));
        }
        stream.push_str(" ET");
        let content = format!(
            "<< /Length {} >>\nstream\n{stream}\nendstream",
            stream.len()
        );
        payloads.push(content);
    }

    let mut document = Vec::new();
    document.extend_from_slice(b"%PDF-1.4\n");
    let mut object_offsets = Vec::new();
    for payload in &payloads {
        object_offsets.push(document.len() as u64);
        document.extend_from_slice(format!("{} 0 obj\n", object_offsets.len()).as_bytes());
        document.extend_from_slice(payload.as_bytes());
        document.extend_from_slice(b"\nendobj\n");
    }
    let xref_offset = document.len();
    document.extend_from_slice(format!("xref\n0 {}\n", payloads.len() + 1).as_bytes());
    document.extend_from_slice(b"0000000000 65535 f \n");
    for offset in object_offsets {
        document.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
    }
    document.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n",
            payloads.len() + 1
        )
        .as_bytes(),
    );
    std::fs::write(path, document)?;
    Ok(())
}

pub fn generate_ocr_smoke_assets(output_dir: &Path) -> Result<BTreeMap<String, String>> {
    std::fs::create_dir_all(output_dir)?;
    let page_1: PathBuf = output_dir.join("ocr_smoke_page_1.png");
    let page_2: PathBuf = output_dir.join("ocr_smoke_page_2.png");
    let pdf_path: PathBuf = output_dir.join("ocr_smoke_document.pdf");
    draw_page(
        &page_1,
        "OCR Workflow Smoke Page 1",
        &[
            "Section 1. Overview",
            "Alpha clause: the workflow should preserve page order.",
            "Beta clause: per-page OCR artifacts should remain inspectable.",
        ],
    )?;
    draw_page(
        &page_2,
        "OCR Workflow Smoke Page 2",
        &[
            "Section 2. Review",
            "Gamma clause: downstream source maps should include OCR text.",
            "Delta clause: resume state should survive reruns.",
        ],
    )?;
    build_pdf(&pdf_path)?;
    Ok(BTreeMap::from([
        (
            "page_1_png".to_string(),
            page_1.to_string_lossy().to_string(),
        ),
        (
            "page_2_png".to_string(),
            page_2.to_string_lossy().to_string(),
        ),
        ("pdf".to_string(), pdf_path.to_string_lossy().to_string()),
    ]))
}
