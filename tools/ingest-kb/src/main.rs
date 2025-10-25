use anyhow::{Context, Result};
use clap::Parser;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Rust KB ingestion preview tool
/// - Walks piano_pedagogy/**/*.txt
/// - Parses header + body
/// - Chunks text and prints preview of would-be upserts
#[derive(Debug, Parser)]
#[command(name = "ingest-kb", version, about = "KB ingestion preview")] 
struct Cli {
    /// KB directory (default: ./piano_pedagogy)
    #[arg(long)]
    kb_dir: Option<PathBuf>,

    /// Preview limit (chunks to print)
    #[arg(long, default_value_t = 10)]
    preview_limit: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Header {
    id: Option<String>,
    title: Option<String>,
    tags: Option<Vec<String>>,  // or comma-separated
    source: Option<String>,
    url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct UpsertItem {
    index: String,
    id: String,
    metadata: Metadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Metadata {
    id: String,
    doc_id: String,
    title: String,
    tags: Vec<String>,
    source: String,
    url: Option<String>,
    text: String,
    chunk_id: usize,
}

fn default_kb_dir() -> PathBuf {
    PathBuf::from("piano_pedagogy")
}

fn parse_header_and_body(raw: &str) -> (Header, String) {
    let mut header = Header { id: None, title: None, tags: None, source: None, url: None };
    let mut lines = raw.lines();
    let kv_re: Regex = Regex::new(r"^(?P<key>[A-Za-z0-9_\-]+):\s*(?P<val>.*)$").unwrap();

    let mut consumed_header = false;
    let mut body_start = 0usize;

    for (idx, line) in raw.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            body_start = idx + 1; // skip the blank line
            consumed_header = true;
            break;
        }
        if let Some(caps) = kv_re.captures(trimmed) {
            let key = caps.name("key").unwrap().as_str();
            let val = caps.name("val").unwrap().as_str().trim();
            match key {
                "id" => header.id = Some(val.to_string()),
                "title" => header.title = Some(val.to_string()),
                "tags" => {
                    let tags = if val.starts_with('[') && val.ends_with(']') {
                        val.trim_matches(&['[', ']'][..])
                            .split(',')
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .collect::<Vec<_>>()
                    } else {
                        val.split(',')
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .collect::<Vec<_>>()
                    };
                    header.tags = Some(tags);
                }
                "source" => header.source = Some(val.to_string()),
                "url" => header.url = Some(val.to_string()),
                _ => {}
            }
        } else {
            // Not a k:v header line; treat as body start
            body_start = idx;
            break;
        }
    }

    let body = if consumed_header {
        raw.lines().skip(body_start).collect::<Vec<_>>().join("\n").trim().to_string()
    } else {
        // No header; entire doc is body
        raw.to_string()
    };

    (header, body)
}

fn estimate_tokens(text: &str) -> usize { (text.len() + 3) / 4 }

fn chunk_text(text: &str, target_tokens: usize, overlap_ratio: f32) -> Vec<String> {
    let tokens = estimate_tokens(text);
    if tokens <= target_tokens { return vec![text.to_string()]; }
    let approx_chars_per_token = 4usize;
    let target_chars = target_tokens * approx_chars_per_token;
    let step_chars = (target_chars as f32 * (1.0 - overlap_ratio)).floor() as usize;

    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < text.len() {
        let end = std::cmp::min(text.len(), start + target_chars);
        let slice = &text[start..end];
        let trimmed = slice.trim();
        if !trimmed.is_empty() {
            chunks.push(trimmed.to_string());
        }
        if end >= text.len() { break; }
        start += step_chars.max(1);
    }
    chunks
}

fn list_txt_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
        let p = entry.path();
        if p.is_file() && p.extension().map(|e| e == "txt").unwrap_or(false) {
            out.push(p.to_path_buf());
        }
    }
    out
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let kb_dir = cli.kb_dir.unwrap_or_else(default_kb_dir);
    let index_name = std::env::var("VECTORIZE_INDEX_NAME").unwrap_or_else(|_| "crescendai-piano-pedagogy".to_string());

    if !kb_dir.exists() {
        anyhow::bail!("KB directory not found: {}", kb_dir.display());
    }

    println!("[ingest] scanning KB directory: {}", kb_dir.display());
    let files = list_txt_files(&kb_dir);
    if files.is_empty() {
        println!("[ingest] no .txt files found under {}", kb_dir.display());
        return Ok(());
    }
    println!("[ingest] found {} file(s)", files.len());

    let mut printed = 0usize;

    for file in files {
        let raw = fs::read_to_string(&file).with_context(|| format!("read file: {}", file.display()))?;
        let (header, body) = parse_header_and_body(&raw);
        let doc_id = header.id.clone().unwrap_or_else(|| file.to_string_lossy().to_string());
        let title = header.title.clone().unwrap_or_else(|| doc_id.clone());
        let tags = header.tags.clone().unwrap_or_default();
        let source = header.source.clone().unwrap_or_else(|| "unknown".to_string());
        let url = header.url.clone();

        let chunks = chunk_text(&body, 600, 0.12);
        for (idx, text) in chunks.into_iter().enumerate() {
            let id = format!("{}::c{}", doc_id, idx);
            let item = UpsertItem {
                index: index_name.clone(),
                id: id.clone(),
                metadata: Metadata {
                    id: id.clone(),
                    doc_id: doc_id.clone(),
                    title: title.clone(),
                    tags: tags.clone(),
                    source: source.clone(),
                    url: url.clone(),
                    text,
                    chunk_id: idx,
                }
            };
            if printed < cli.preview_limit {
                println!("[preview] upsert -> {} (tags={}) title={}", item.id, item.metadata.tags.join("|"), item.metadata.title);
                printed += 1;
            }
        }
    }

    println!("[ingest] preview complete (printed {} items)", printed);
    println!("[ingest] Next: implement CF embeddings + Vectorize upsert (Phase 1 preview only)");
    Ok(())
}
