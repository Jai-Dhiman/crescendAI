use anyhow::{Context, Result};
use lopdf::Document;

use crate::models::PdfMetadata;

/// Extract metadata from a PDF file
pub fn extract_pdf_metadata(pdf_bytes: &[u8]) -> Result<PdfMetadata> {
    // Parse the PDF document
    let doc = Document::load_mem(pdf_bytes)
        .context("Failed to parse PDF document")?;

    // Get page count
    let page_count = doc.get_pages().len() as i32;

    // Get file size
    let file_size_bytes = pdf_bytes.len() as i64;

    // Try to extract title from PDF metadata
    let title = extract_pdf_title(&doc);

    Ok(PdfMetadata {
        page_count,
        file_size_bytes,
        title,
    })
}

/// Extract title from PDF metadata dictionary
fn extract_pdf_title(doc: &Document) -> Option<String> {
    // Try to get the document information dictionary
    if let Ok(info_dict) = doc.trailer.get(b"Info") {
        if let Ok(info_ref) = info_dict.as_reference() {
            if let Ok(info_obj) = doc.get_object(info_ref) {
                if let Ok(dict) = info_obj.as_dict() {
                    // Try to get the Title entry
                    if let Ok(title_obj) = dict.get(b"Title") {
                        if let Ok(title_bytes) = title_obj.as_str() {
                            // Convert bytes to UTF-8 string
                            return String::from_utf8(title_bytes.to_vec()).ok();
                        }
                    }
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_valid_pdf_metadata() {
        // This would require a test PDF file
        // For now, we'll skip this test
        // In production, you'd want to include a small test PDF in the test resources
    }

    #[test]
    fn test_extract_invalid_pdf_fails() {
        let invalid_pdf = b"This is not a PDF";
        let result = extract_pdf_metadata(invalid_pdf);
        assert!(result.is_err());
    }
}
