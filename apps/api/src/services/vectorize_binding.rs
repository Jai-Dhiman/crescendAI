//! Vectorize JavaScript bindings via wasm-bindgen
//!
//! Since workers-rs doesn't have native Vectorize support, we use JS interop
//! to access the Vectorize binding directly. This provides ~30-50ms latency
//! versus ~50-80ms for HTTP API.

use js_sys::{Array, Object, Reflect};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

// Import the Workers runtime global to access env bindings
#[wasm_bindgen]
extern "C" {
    /// Vectorize index type from Cloudflare Workers runtime
    #[wasm_bindgen(extends = Object)]
    pub type VectorizeIndex;

    /// Query the vector index for similar vectors
    #[wasm_bindgen(method, catch)]
    pub async fn query(
        this: &VectorizeIndex,
        vector: Array,
        options: JsValue,
    ) -> Result<JsValue, JsValue>;

    /// Insert or update vectors in the index
    #[wasm_bindgen(method, catch)]
    pub async fn upsert(this: &VectorizeIndex, vectors: JsValue) -> Result<JsValue, JsValue>;

    /// Delete vectors by ID
    #[wasm_bindgen(method, catch)]
    pub async fn deleteByIds(this: &VectorizeIndex, ids: JsValue) -> Result<JsValue, JsValue>;

    /// Get vectors by ID
    #[wasm_bindgen(method, catch)]
    pub async fn getByIds(this: &VectorizeIndex, ids: JsValue) -> Result<JsValue, JsValue>;

    /// Describe the index configuration
    #[wasm_bindgen(method, catch)]
    pub async fn describe(this: &VectorizeIndex) -> Result<JsValue, JsValue>;
}

/// Metadata stored with each vector for filtering
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorMetadata {
    /// Difficulty level: beginner, intermediate, advanced
    pub difficulty: Option<String>,
    /// Topic hierarchy: technique/articulation, interpretation/phrasing, etc.
    pub topic: Option<String>,
    /// Content type: explanation, exercise, example
    pub content_type: Option<String>,
    /// Primary composer if applicable
    pub composer: Option<String>,
    /// Source type for filtering: book, masterclass, letter, journal
    pub source_type: Option<String>,
}

/// A single vector to upsert
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorRecord {
    pub id: String,
    pub values: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<VectorMetadata>,
}

/// A match returned from vector query
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorMatch {
    pub id: String,
    pub score: f32,
    #[serde(default)]
    pub metadata: Option<VectorMetadata>,
}

/// Query result from Vectorize
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorQueryResult {
    pub matches: Vec<VectorMatch>,
    pub count: usize,
}

/// Get the Vectorize index binding from the Worker environment
///
/// Uses Reflect to access the binding from the Env's inner JS object.
pub fn get_vectorize_index(env: &worker::Env, binding: &str) -> Result<VectorizeIndex, JsValue> {
    // The worker::Env wraps a JS object. We access it using the inner() method
    // which returns a reference that we can use with Reflect::get
    let env_js: &JsValue = env.as_ref();

    let binding_val = Reflect::get(env_js, &JsValue::from_str(binding))
        .map_err(|e| JsValue::from_str(&format!("Failed to get binding '{}': {:?}", binding, e)))?;

    if binding_val.is_undefined() || binding_val.is_null() {
        return Err(JsValue::from_str(&format!("Binding '{}' not found in environment", binding)));
    }

    Ok(binding_val.unchecked_into())
}

/// Query the Vectorize index for similar vectors
pub async fn query_vectors(
    index: &VectorizeIndex,
    embedding: &[f32],
    top_k: usize,
    filter: Option<VectorMetadata>,
) -> Result<Vec<VectorMatch>, String> {
    // Convert embedding to JS Array
    let vector_array = Array::new();
    for val in embedding {
        vector_array.push(&JsValue::from_f64(*val as f64));
    }

    // Build options object
    let options = Object::new();
    Reflect::set(&options, &"topK".into(), &JsValue::from_f64(top_k as f64))
        .map_err(|e| format!("Failed to set topK: {:?}", e))?;
    Reflect::set(&options, &"returnMetadata".into(), &"all".into())
        .map_err(|e| format!("Failed to set returnMetadata: {:?}", e))?;

    // Add filter if provided
    if let Some(f) = filter {
        let filter_obj = Object::new();
        if let Some(difficulty) = &f.difficulty {
            Reflect::set(&filter_obj, &"difficulty".into(), &JsValue::from_str(difficulty))
                .map_err(|e| format!("Failed to set difficulty filter: {:?}", e))?;
        }
        if let Some(topic) = &f.topic {
            Reflect::set(&filter_obj, &"topic".into(), &JsValue::from_str(topic))
                .map_err(|e| format!("Failed to set topic filter: {:?}", e))?;
        }
        if let Some(content_type) = &f.content_type {
            Reflect::set(&filter_obj, &"content_type".into(), &JsValue::from_str(content_type))
                .map_err(|e| format!("Failed to set content_type filter: {:?}", e))?;
        }
        if let Some(composer) = &f.composer {
            Reflect::set(&filter_obj, &"composer".into(), &JsValue::from_str(composer))
                .map_err(|e| format!("Failed to set composer filter: {:?}", e))?;
        }
        if let Some(source_type) = &f.source_type {
            Reflect::set(&filter_obj, &"source_type".into(), &JsValue::from_str(source_type))
                .map_err(|e| format!("Failed to set source_type filter: {:?}", e))?;
        }
        Reflect::set(&options, &"filter".into(), &filter_obj)
            .map_err(|e| format!("Failed to set filter: {:?}", e))?;
    }

    // Execute query
    let result = index
        .query(vector_array, options.into())
        .await
        .map_err(|e| format!("Vectorize query failed: {:?}", e))?;

    // Parse result
    let query_result: VectorQueryResult = serde_wasm_bindgen::from_value(result)
        .map_err(|e| format!("Failed to parse query result: {:?}", e))?;

    Ok(query_result.matches)
}

/// Upsert vectors into the Vectorize index
pub async fn upsert_vectors(
    index: &VectorizeIndex,
    vectors: Vec<VectorRecord>,
) -> Result<usize, String> {
    // Convert vectors to JS value
    let vectors_js = serde_wasm_bindgen::to_value(&vectors)
        .map_err(|e| format!("Failed to serialize vectors: {:?}", e))?;

    // Execute upsert
    let result = index
        .upsert(vectors_js)
        .await
        .map_err(|e| format!("Vectorize upsert failed: {:?}", e))?;

    // Parse result to get count
    let result_obj: serde_json::Value = serde_wasm_bindgen::from_value(result)
        .map_err(|e| format!("Failed to parse upsert result: {:?}", e))?;

    let count = result_obj
        .get("mutationId")
        .and_then(|_| Some(vectors.len()))
        .unwrap_or(0);

    Ok(count)
}

/// Delete vectors by their IDs
pub async fn delete_vectors(index: &VectorizeIndex, ids: Vec<String>) -> Result<(), String> {
    let ids_js = serde_wasm_bindgen::to_value(&ids)
        .map_err(|e| format!("Failed to serialize ids: {:?}", e))?;

    index
        .deleteByIds(ids_js)
        .await
        .map_err(|e| format!("Vectorize delete failed: {:?}", e))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_metadata_serialization() {
        let metadata = VectorMetadata {
            difficulty: Some("intermediate".to_string()),
            topic: Some("technique/articulation".to_string()),
            content_type: Some("explanation".to_string()),
            composer: Some("Chopin".to_string()),
            source_type: Some("book".to_string()),
        };

        let json = serde_json::to_string(&metadata).unwrap();
        assert!(json.contains("intermediate"));
        assert!(json.contains("Chopin"));
    }

    #[test]
    fn test_vector_record_serialization() {
        let record = VectorRecord {
            id: "test-001".to_string(),
            values: vec![0.1, 0.2, 0.3],
            metadata: Some(VectorMetadata {
                difficulty: Some("beginner".to_string()),
                topic: None,
                content_type: None,
                composer: None,
                source_type: None,
            }),
        };

        let json = serde_json::to_string(&record).unwrap();
        assert!(json.contains("test-001"));
        assert!(json.contains("0.1"));
    }
}
