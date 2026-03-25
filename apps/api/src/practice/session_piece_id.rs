use worker::*;

use super::piece_identify::DTW_CONFIRM_THRESHOLD;
use super::score_context::ScoreContext;
use super::score_follower::FollowerState;
use super::session::PracticeSession;

impl PracticeSession {
    /// Attempt piece identification from accumulated AMT notes using sliding windows.
    ///
    /// Tries multiple window sizes (smallest first) on the most recent notes.
    /// Distinctive openings match on small windows; ambiguous passages need larger ones.
    /// Runs N-gram recall + rerank (Stage 1+2), then DTW confirmation (Stage 3)
    /// against the top candidate's score data. If confirmed, locks in the piece and
    /// loads the full ScoreContext for subsequent score following.
    pub(crate) async fn try_identify_piece(&self, ws: &WebSocket) {
        /// Window sizes for identification attempts (smallest first).
        const ID_WINDOW_SIZES: &[usize] = &[60, 120, 200];
        /// Sanity cap on total notes -- stops attempts but does NOT lock (manual naming still works).
        const ID_MAX_TOTAL_NOTES: usize = 600;
        /// Minimum new notes between attempts to avoid redundant work.
        const ID_MIN_NEW_NOTES: usize = 30;

        let (piece_locked, total_notes, last_attempt_count) = {
            let s = self.inner.borrow();
            (
                s.piece_locked,
                s.accumulated_notes.len(),
                s.identification_note_count,
            )
        };

        if piece_locked {
            return;
        }

        // Sanity cap: stop automatic attempts but don't lock (set_piece still works)
        if total_notes > ID_MAX_TOTAL_NOTES {
            console_log!(
                "piece_id: {} notes exceeded sanity cap {}, stopping auto-identification",
                total_notes,
                ID_MAX_TOTAL_NOTES
            );
            return;
        }

        // Throttle: need enough new notes since last attempt
        if total_notes.saturating_sub(last_attempt_count as usize) < ID_MIN_NEW_NOTES {
            return;
        }

        // Lazy-load N-gram index from R2 (cached in session state)
        {
            let needs_index = self.inner.borrow().ngram_index.is_none();
            if needs_index {
                match crate::practice::score_context::load_ngram_index(&self.env).await {
                    Ok(index) => {
                        self.inner.borrow_mut().ngram_index = Some(index);
                    }
                    Err(e) => {
                        console_error!("Failed to load N-gram index: {}", e);
                        return;
                    }
                }
            }
        }
        {
            let needs_features = self.inner.borrow().rerank_features.is_none();
            if needs_features {
                match crate::practice::score_context::load_rerank_features(&self.env).await {
                    Ok(features) => {
                        self.inner.borrow_mut().rerank_features = Some(features);
                    }
                    Err(e) => {
                        console_error!("Failed to load rerank features: {}", e);
                        return;
                    }
                }
            }
        }

        // Clone what we need (no borrows across await)
        let (all_notes, ngram_index, rerank_features) = {
            let s = self.inner.borrow();
            (
                s.accumulated_notes.clone(),
                s.ngram_index.clone().unwrap(),
                s.rerank_features.clone().unwrap(),
            )
        };

        // Try each window size, smallest first (distinctive fragments win early)
        for &window_size in ID_WINDOW_SIZES {
            if all_notes.len() < window_size {
                continue;
            }
            let window = &all_notes[all_notes.len() - window_size..];

            // Stage 1+2: N-gram recall + rerank on this window
            let identification = crate::practice::piece_identify::identify_piece(
                window,
                &ngram_index,
                &rerank_features,
            );

            let candidate = match identification {
                Some(id) => id,
                None => continue, // try next (larger) window
            };

            console_log!(
                "piece_id_attempt: window={} total_notes={} top_piece={} top_score={:.3} method={}",
                window_size,
                all_notes.len(),
                candidate.piece_id,
                candidate.confidence,
                candidate.method
            );

            // Stage 3: DTW confirmation -- load the candidate's score and align
            let score_data =
                match crate::practice::score_context::load_score(&self.env, &candidate.piece_id)
                    .await
                {
                    Ok(s) => s,
                    Err(e) => {
                        console_error!(
                            "Failed to load score for DTW confirmation of {}: {}",
                            candidate.piece_id,
                            e
                        );
                        continue;
                    }
                };

            let mut dtw_state = FollowerState::default();
            let bar_map = crate::practice::score_follower::align_chunk(
                0,
                0.0,
                window,
                &score_data,
                &mut dtw_state,
            );

            let dtw_cost = bar_map.as_ref().map(|bm| 1.0 / bm.confidence - 1.0);
            let dtw_confirmed = dtw_cost.map(|c| c < DTW_CONFIRM_THRESHOLD).unwrap_or(false);

            console_log!(
                "piece_id_dtw: piece={} window={} cost={:.3} threshold={:.3} confirmed={}",
                candidate.piece_id,
                window_size,
                dtw_cost.unwrap_or(f64::MAX),
                DTW_CONFIRM_THRESHOLD,
                dtw_confirmed
            );

            if !dtw_confirmed {
                continue; // try next (larger) window
            }

            // DTW confirmed -- lock in piece and load full ScoreContext
            let reference =
                crate::practice::score_context::load_reference(&self.env, &candidate.piece_id)
                    .await;

            let composer = score_data.composer.clone();
            let title = score_data.title.clone();

            {
                let mut s = self.inner.borrow_mut();
                s.piece_identification = Some(candidate.clone());
                s.piece_locked = true;
                s.score_context = Some(ScoreContext {
                    piece_id: candidate.piece_id.clone(),
                    composer: composer.clone(),
                    title: title.clone(),
                    score: score_data,
                    reference,
                    match_confidence: candidate.confidence,
                });
                s.score_context_loaded = true;
                s.follower_state = FollowerState::default();
            }

            console_log!(
                "piece_id_locked: piece={} ({} - {}) confidence={:.3} method={} window={}",
                candidate.piece_id,
                composer,
                title,
                candidate.confidence,
                candidate.method,
                window_size
            );

            // Notify client
            let msg = serde_json::json!({
                "type": "piece_identified",
                "pieceId": candidate.piece_id,
                "composer": composer,
                "title": title,
                "confidence": candidate.confidence,
                "method": candidate.method,
            });
            let _ = ws.send_with_str(&msg.to_string());

            // Log to piece_requests for demand tracking
            let student_id = self.inner.borrow().student_id.clone();
            crate::practice::score_context::log_fingerprint_piece_request(
                &self.env,
                &student_id,
                &candidate.piece_id,
                candidate.confidence,
                &candidate.method,
            )
            .await;

            return; // identified and locked
        }

        // No window matched -- update attempt counter for throttle
        self.inner.borrow_mut().identification_note_count = all_notes.len() as u32;
        console_log!(
            "piece_id_attempt: total_notes={} windows_tried={:?} no_match",
            all_notes.len(),
            ID_WINDOW_SIZES
                .iter()
                .filter(|&&w| all_notes.len() >= w)
                .collect::<Vec<_>>()
        );
    }
}
