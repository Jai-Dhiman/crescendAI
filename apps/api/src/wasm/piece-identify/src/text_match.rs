//! Text query matching using Dice similarity on character bigrams.
//! Ported from `apps/api-rust/src/practice/analysis/piece_match.rs`.

use std::collections::HashSet;

use crate::types::{CatalogEntry, TextMatchResult};

const CONFIDENCE_THRESHOLD: f64 = 0.3;

/// Normalize a string: lowercase, replace non-alphanumeric with space, collapse whitespace.
pub fn normalize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() {
                c.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<&str>>()
        .join(" ")
}

/// Extract character bigrams from a string.
pub fn bigrams(s: &str) -> Vec<String> {
    if s.len() < 2 {
        return vec![];
    }
    let chars: Vec<char> = s.chars().collect();
    chars.windows(2).map(|w| w.iter().collect::<String>()).collect()
}

/// Dice coefficient similarity on character bigrams.
pub fn dice_similarity(a: &str, b: &str) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let bigrams_a: HashSet<String> = bigrams(a).into_iter().collect();
    let bigrams_b: HashSet<String> = bigrams(b).into_iter().collect();

    if bigrams_a.is_empty() && bigrams_b.is_empty() {
        return 1.0;
    }
    if bigrams_a.is_empty() || bigrams_b.is_empty() {
        return 0.0;
    }

    let intersection = bigrams_a.intersection(&bigrams_b).count();
    2.0 * intersection as f64 / (bigrams_a.len() + bigrams_b.len()) as f64
}

/// Find a composer name from the catalog that appears in the query (case-insensitive).
/// Prefers longer matches to avoid partial matches on short names.
fn extract_composer<'a>(query: &str, catalog: &'a [CatalogEntry]) -> Option<&'a str> {
    let query_lower = query.to_ascii_lowercase();
    let mut composers: Vec<&str> = catalog.iter().map(|p| p.composer.as_str()).collect();
    let mut seen = HashSet::new();
    composers.retain(|c| seen.insert(*c));
    composers.sort_by_key(|c| std::cmp::Reverse(c.len()));

    composers
        .into_iter()
        .find(|&composer| query_lower.contains(&composer.to_ascii_lowercase()))
}

/// Strip the composer name from the query for isolated title matching.
fn strip_composer(query: &str, composer: &str) -> String {
    let query_lower = query.to_ascii_lowercase();
    let composer_lower = composer.to_ascii_lowercase();
    let stripped = query_lower.replacen(&composer_lower, "", 1);
    stripped.split_whitespace().collect::<Vec<&str>>().join(" ")
}

/// Match a free-text query against the catalog using Dice similarity on bigrams.
/// Returns the best-matching piece if its score exceeds the confidence threshold.
pub fn match_piece_text(query: &str, catalog: &[CatalogEntry]) -> Option<TextMatchResult> {
    let trimmed = query.trim();
    if trimmed.is_empty() || catalog.is_empty() {
        return None;
    }

    let composer = extract_composer(trimmed, catalog);

    let candidates: Vec<&CatalogEntry> = if let Some(c) = composer {
        catalog.iter().filter(|p| p.composer.eq_ignore_ascii_case(c)).collect()
    } else {
        catalog.iter().collect()
    };

    let title_query = if let Some(c) = composer {
        strip_composer(trimmed, c)
    } else {
        normalize(trimmed)
    };

    let norm_title_query = normalize(&title_query);

    let best = candidates
        .iter()
        .map(|piece| {
            let norm_title = normalize(&piece.title);
            let score = dice_similarity(&norm_title_query, &norm_title);
            (*piece, score)
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    if let Some((piece, confidence)) = best {
        if confidence >= CONFIDENCE_THRESHOLD {
            return Some(TextMatchResult { piece_id: piece.piece_id.clone(), confidence });
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_catalog() -> Vec<CatalogEntry> {
        vec![
            CatalogEntry {
                piece_id: "chopin.ballades.1".to_string(),
                composer: "chopin".to_string(),
                title: "Ballade No. 1".to_string(),
            },
            CatalogEntry {
                piece_id: "chopin.ballades.4".to_string(),
                composer: "chopin".to_string(),
                title: "Ballade No. 4".to_string(),
            },
            CatalogEntry {
                piece_id: "bach.prelude.bwv_846".to_string(),
                composer: "bach".to_string(),
                title: "Prelude - BWV 846".to_string(),
            },
            CatalogEntry {
                piece_id: "chopin.etudes_op_10.3".to_string(),
                composer: "chopin".to_string(),
                title: "Etudes Op. 10 No. 3".to_string(),
            },
        ]
    }

    #[test]
    fn matches_chopin_ballade() {
        let catalog = test_catalog();
        let result = match_piece_text("chopin ballade 1", &catalog);
        assert!(result.is_some());
        assert_eq!(result.unwrap().piece_id, "chopin.ballades.1");
    }

    #[test]
    fn matches_bach_prelude() {
        let catalog = test_catalog();
        let result = match_piece_text("bach prelude bwv 846", &catalog);
        assert!(result.is_some());
        assert_eq!(result.unwrap().piece_id, "bach.prelude.bwv_846");
    }

    #[test]
    fn no_match_for_unknown_piece() {
        let catalog = test_catalog();
        let result = match_piece_text("debussy clair de lune", &catalog);
        assert!(result.is_none());
    }

    #[test]
    fn empty_query_returns_none() {
        let catalog = test_catalog();
        assert!(match_piece_text("", &catalog).is_none());
        assert!(match_piece_text("  ", &catalog).is_none());
    }

    #[test]
    fn empty_catalog_returns_none() {
        let result = match_piece_text("chopin ballade 1", &[]);
        assert!(result.is_none());
    }

    #[test]
    fn case_insensitive() {
        let catalog = test_catalog();
        let result = match_piece_text("CHOPIN BALLADE", &catalog);
        assert!(result.is_some());
        assert!(result.unwrap().piece_id.starts_with("chopin.ballades."));
    }

    #[test]
    fn normalize_removes_punctuation() {
        assert_eq!(normalize("No. 14 in C-sharp"), "no 14 in c sharp");
    }

    #[test]
    fn dice_identical_strings() {
        let score = dice_similarity("hello world", "hello world");
        assert!(score > 0.99, "Expected ~1.0 for identical strings, got {}", score);
    }

    #[test]
    fn dice_completely_different() {
        let score = dice_similarity("abcdef", "xyz123");
        assert!(score < 0.1, "Expected < 0.1 for completely different strings, got {}", score);
    }
}
