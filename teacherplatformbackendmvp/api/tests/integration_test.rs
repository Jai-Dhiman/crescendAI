// Integration test for PDF ingestion pipeline
use piano_api::ingestion::{extract_pdf_text, chunk_pages, ChunkConfig};
use std::fs;

#[tokio::test]
async fn test_pdf_extraction() {
    // Test PDF text extraction
    let pdf_path = "../test_data/sample.pdf";
    let pdf_bytes = fs::read(pdf_path).expect("Failed to read PDF file");

    println!("\n=== Testing PDF Text Extraction ===");
    println!("PDF path: {}", pdf_path);
    println!("PDF size: {} bytes", pdf_bytes.len());

    let result = extract_pdf_text(&pdf_bytes);
    match result {
        Ok(pages) => {
            println!("✓ Successfully extracted {} pages", pages.len());
            for page in pages.iter() {
                println!("\n--- Page {} ({} chars) ---",
                    page.page_number,
                    page.text.len()
                );
                let preview = if page.text.len() > 200 {
                    &page.text[..200]
                } else {
                    &page.text
                };
                println!("{}", preview);
                if page.text.len() > 200 {
                    println!("...");
                }
            }

            // Test chunking
            println!("\n=== Testing PDF Chunking ===");
            let config = ChunkConfig {
                chunk_size: 512,
                overlap: 50,
            };

            let chunks = chunk_pages(pages.clone(), &config).expect("Failed to chunk pages");
            println!("✓ Created {} chunks from {} pages", chunks.len(), pages.len());

            for (i, chunk) in chunks.iter().take(3).enumerate() {
                println!("\n--- Chunk {} ---", i + 1);
                println!("Chars: {} - {}", chunk.start_char, chunk.end_char);
                println!("Page: {:?}", chunk.page_number);
                let preview = if chunk.content.len() > 150 {
                    &chunk.content[..150]
                } else {
                    &chunk.content
                };
                println!("{}", preview);
                if chunk.content.len() > 150 {
                    println!("...");
                }
            }

            if chunks.len() > 3 {
                println!("\n... and {} more chunks", chunks.len() - 3);
            }

            println!("\n=== Test Summary ===");
            println!("✓ PDF extraction: SUCCESS");
            println!("✓ PDF chunking: SUCCESS");
            println!("  Pages extracted: {}", pages.len());
            println!("  Chunks created: {}", chunks.len());
            println!("  Avg chunk size: {} chars",
                chunks.iter().map(|c| c.content.len()).sum::<usize>() / chunks.len().max(1)
            );
        }
        Err(e) => {
            panic!("Failed to extract PDF: {}", e);
        }
    }
}

#[tokio::test]
async fn test_chunking_config() {
    println!("\n=== Testing Chunking Configuration ===");

    // Test different chunk sizes
    let test_configs = vec![
        ChunkConfig { chunk_size: 256, overlap: 25 },
        ChunkConfig { chunk_size: 512, overlap: 50 },
        ChunkConfig { chunk_size: 1024, overlap: 100 },
    ];

    let pdf_path = "../test_data/sample.pdf";
    let pdf_bytes = fs::read(pdf_path).expect("Failed to read PDF file");
    let pages = extract_pdf_text(&pdf_bytes).expect("Failed to extract PDF");

    for config in test_configs {
        let chunks = chunk_pages(pages.clone(), &config).expect("Failed to chunk pages");
        println!(
            "  Config(chunk_size={}, overlap={}) -> {} chunks",
            config.chunk_size,
            config.overlap,
            chunks.len()
        );
    }

    println!("✓ Chunking configuration test: SUCCESS");
}
