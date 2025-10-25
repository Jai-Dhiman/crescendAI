use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Clone)]
pub struct KVClient {
    http_client: Client,
    account_id: String,
    api_token: String,
    embedding_namespace_id: String,
    search_namespace_id: String,
    llm_namespace_id: String,
}

impl KVClient {
    pub fn new(
        account_id: String,
        api_token: String,
        embedding_namespace_id: String,
        search_namespace_id: String,
        llm_namespace_id: String,
    ) -> Self {
        Self {
            http_client: Client::new(),
            account_id,
            api_token,
            embedding_namespace_id,
            search_namespace_id,
            llm_namespace_id,
        }
    }

    pub async fn get(&self, namespace_id: &str, key: &str) -> Result<Option<Vec<u8>>> {
        let url = format!(
            "https://api.cloudflare.com/client/v4/accounts/{}/storage/kv/namespaces/{}/values/{}",
            self.account_id, namespace_id, key
        );

        let response = self
            .http_client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.api_token))
            .send()
            .await
            .context("Failed to fetch from Workers KV")?;

        if response.status().is_success() {
            let bytes = response
                .bytes()
                .await
                .context("Failed to read response bytes")?;
            Ok(Some(bytes.to_vec()))
        } else if response.status() == 404 {
            Ok(None)
        } else {
            anyhow::bail!("KV GET failed with status: {}", response.status())
        }
    }

    pub async fn put(
        &self,
        namespace_id: &str,
        key: &str,
        value: Vec<u8>,
        ttl_seconds: Option<u64>,
    ) -> Result<()> {
        let mut url = format!(
            "https://api.cloudflare.com/client/v4/accounts/{}/storage/kv/namespaces/{}/values/{}",
            self.account_id, namespace_id, key
        );

        if let Some(ttl) = ttl_seconds {
            url.push_str(&format!("?expiration_ttl={}", ttl));
        }

        let response = self
            .http_client
            .put(&url)
            .header("Authorization", format!("Bearer {}", self.api_token))
            .header("Content-Type", "application/octet-stream")
            .body(value)
            .send()
            .await
            .context("Failed to write to Workers KV")?;

        if !response.status().is_success() {
            anyhow::bail!("KV PUT failed with status: {}", response.status())
        }

        Ok(())
    }

    pub async fn delete(&self, namespace_id: &str, key: &str) -> Result<()> {
        let url = format!(
            "https://api.cloudflare.com/client/v4/accounts/{}/storage/kv/namespaces/{}/values/{}",
            self.account_id, namespace_id, key
        );

        let response = self
            .http_client
            .delete(&url)
            .header("Authorization", format!("Bearer {}", self.api_token))
            .send()
            .await
            .context("Failed to delete from Workers KV")?;

        if !response.status().is_success() {
            anyhow::bail!("KV DELETE failed with status: {}", response.status())
        }

        Ok(())
    }

    pub fn embedding_namespace(&self) -> &str {
        &self.embedding_namespace_id
    }

    pub fn search_namespace(&self) -> &str {
        &self.search_namespace_id
    }

    pub fn llm_namespace(&self) -> &str {
        &self.llm_namespace_id
    }
}
