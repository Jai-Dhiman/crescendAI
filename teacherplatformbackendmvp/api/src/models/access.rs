use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "access_level", rename_all = "lowercase")]
pub enum AccessLevel {
    View,
    Edit,
    Admin,
}
