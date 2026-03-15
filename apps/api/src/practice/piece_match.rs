use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct CatalogPiece {
    pub piece_id: String,
    pub composer: String,
    pub title: String,
}

#[derive(Debug, Clone)]
pub struct MatchResult {
    pub piece_id: String,
    pub composer: String,
    pub title: String,
    pub confidence: f64,
}

const CONFIDENCE_THRESHOLD: f64 = 0.3;

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

pub fn bigrams(s: &str) -> Vec<String> {
    if s.len() < 2 {
        return vec![];
    }
    let chars: Vec<char> = s.chars().collect();
    chars
        .windows(2)
        .map(|w| w.iter().collect::<String>())
        .collect()
}

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

pub fn extract_composer<'a>(query: &str, catalog: &'a [CatalogPiece]) -> Option<&'a str> {
    let query_lower = query.to_ascii_lowercase();
    let mut composers: Vec<&str> = catalog.iter().map(|p| p.composer.as_str()).collect();
    // Deduplicate while preserving order
    let mut seen = HashSet::new();
    composers.retain(|c| seen.insert(*c));

    // Prefer longer matches to avoid partial matches on short names
    composers.sort_by(|a, b| b.len().cmp(&a.len()));

    for composer in composers {
        if query_lower.contains(&composer.to_ascii_lowercase()) {
            return Some(composer);
        }
    }
    None
}

pub fn strip_composer(query: &str, composer: &str) -> String {
    let query_lower = query.to_ascii_lowercase();
    let composer_lower = composer.to_ascii_lowercase();
    let stripped = query_lower.replacen(&composer_lower, "", 1);
    stripped
        .split_whitespace()
        .collect::<Vec<&str>>()
        .join(" ")
}

pub fn match_piece(query: &str, catalog: &[CatalogPiece]) -> Option<MatchResult> {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return None;
    }
    if catalog.is_empty() {
        return None;
    }

    let composer = extract_composer(trimmed, catalog);

    let candidates: Vec<&CatalogPiece> = if let Some(c) = composer {
        catalog
            .iter()
            .filter(|p| p.composer.to_ascii_lowercase() == c.to_ascii_lowercase())
            .collect()
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
            (piece, score)
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    if let Some((piece, confidence)) = best {
        if confidence >= CONFIDENCE_THRESHOLD {
            return Some(MatchResult {
                piece_id: piece.piece_id.clone(),
                composer: piece.composer.clone(),
                title: piece.title.clone(),
                confidence,
            });
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_catalog() -> Vec<CatalogPiece> {
        vec![
            CatalogPiece {
                piece_id: "chopin.ballades.1".to_string(),
                composer: "chopin".to_string(),
                title: "Ballade No. 1".to_string(),
            },
            CatalogPiece {
                piece_id: "chopin.ballades.4".to_string(),
                composer: "chopin".to_string(),
                title: "Ballade No. 4".to_string(),
            },
            CatalogPiece {
                piece_id: "bach.prelude.bwv_846".to_string(),
                composer: "bach".to_string(),
                title: "Prelude - BWV 846".to_string(),
            },
            CatalogPiece {
                piece_id: "chopin.etudes_op_10.3".to_string(),
                composer: "chopin".to_string(),
                title: "Etudes Op. 10 No. 3".to_string(),
            },
            CatalogPiece {
                piece_id: "bach.fugue.bwv_846".to_string(),
                composer: "bach".to_string(),
                title: "Fugue - BWV 846".to_string(),
            },
        ]
    }

    #[test]
    fn matches_chopin_ballade_casual() {
        let catalog = test_catalog();
        let result = match_piece("chopin ballade 1", &catalog);
        assert!(result.is_some(), "Expected a match for 'chopin ballade 1'");
        let r = result.unwrap();
        assert_eq!(r.piece_id, "chopin.ballades.1", "piece_id mismatch: got {}", r.piece_id);
    }

    #[test]
    fn matches_chopin_ballade_number_only() {
        let catalog = test_catalog();
        let result = match_piece("chopin ballade no 1", &catalog);
        assert!(result.is_some(), "Expected a match for 'chopin ballade no 1'");
        let r = result.unwrap();
        assert_eq!(r.piece_id, "chopin.ballades.1", "piece_id mismatch: got {}", r.piece_id);
    }

    #[test]
    fn matches_bach_prelude() {
        let catalog = test_catalog();
        let result = match_piece("bach prelude bwv 846", &catalog);
        assert!(result.is_some(), "Expected a match for 'bach prelude bwv 846'");
        let r = result.unwrap();
        assert_eq!(r.piece_id, "bach.prelude.bwv_846", "piece_id mismatch: got {}", r.piece_id);
    }

    #[test]
    fn no_match_for_unknown_piece() {
        let catalog = test_catalog();
        let result = match_piece("debussy clair de lune", &catalog);
        assert!(result.is_none(), "Expected no match for unknown piece");
    }

    #[test]
    fn empty_query_returns_none() {
        let catalog = test_catalog();
        assert!(match_piece("", &catalog).is_none());
        assert!(match_piece("  ", &catalog).is_none());
    }

    #[test]
    fn empty_catalog_returns_none() {
        let result = match_piece("chopin ballade 1", &[]);
        assert!(result.is_none());
    }

    #[test]
    fn case_insensitive() {
        let catalog = test_catalog();
        let result = match_piece("CHOPIN BALLADE", &catalog);
        assert!(result.is_some(), "Expected a match for 'CHOPIN BALLADE'");
        let r = result.unwrap();
        assert!(
            r.piece_id.starts_with("chopin.ballades."),
            "Expected a chopin ballade, got {}",
            r.piece_id
        );
    }

    #[test]
    fn filters_by_composer() {
        let catalog = test_catalog();
        let result = match_piece("chopin etude", &catalog);
        assert!(result.is_some(), "Expected a match for 'chopin etude'");
        let r = result.unwrap();
        assert_eq!(
            r.piece_id, "chopin.etudes_op_10.3",
            "Expected chopin etude, got {}",
            r.piece_id
        );
    }

    #[test]
    fn normalize_removes_punctuation() {
        let result = normalize("No. 14 in C-sharp");
        assert_eq!(result, "no 14 in c sharp");
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
