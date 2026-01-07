use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Performance {
    pub id: String,
    pub composer: String,
    pub piece_title: String,
    pub performer: String,
    pub thumbnail_url: String,
    pub audio_url: String,
    pub duration_seconds: u32,
    pub year_recorded: Option<u32>,
    pub description: Option<String>,
}
