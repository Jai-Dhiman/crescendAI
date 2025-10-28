use anyhow::{Context as AnyhowContext, Result};
use aws_credential_types::Credentials;
use aws_sdk_s3::presigning::PresigningConfig;
use aws_sdk_s3::Client;
use aws_types::region::Region;
use std::time::Duration;
use tracing::{debug, error, info, instrument};

use crate::config::CloudflareConfig;
use crate::errors::AppError;

/// R2 bucket type for organizing storage
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BucketType {
    /// PDF projects uploaded by users
    Pdfs,
    /// Knowledge base documents (PDFs, videos, web content)
    Knowledge,
}

impl BucketType {
    pub fn as_str(&self) -> &'static str {
        match self {
            BucketType::Pdfs => "pdfs",
            BucketType::Knowledge => "knowledge",
        }
    }
}

/// Production-ready R2 client for Cloudflare R2 storage
///
/// Uses AWS SDK for S3-compatible operations:
/// - Presigned URL generation (upload/download)
/// - Direct object deletion
/// - Object metadata retrieval
///
/// Note: Direct streaming operations should use Worker R2 bindings for zero-latency edge access.
#[derive(Clone)]
pub struct R2Client {
    client: Client,
    bucket_pdfs: String,
    bucket_knowledge: String,
    account_id: String,
}

impl R2Client {
    /// Create a new R2 client with production configuration
    ///
    /// # Arguments
    /// * `config` - Cloudflare configuration with R2 credentials
    ///
    /// # Errors
    /// Returns error if credentials are missing or invalid
    #[instrument(skip(config), fields(account_id = %config.account_id.as_ref().unwrap_or(&"missing".to_string())))]
    pub async fn new(config: &CloudflareConfig) -> Result<Self> {
        // Validate required configuration
        let account_id = config
            .account_id
            .as_ref()
            .context("CLOUDFLARE_ACCOUNT_ID is required for R2 operations")?;

        let access_key_id = config
            .r2_access_key_id
            .as_ref()
            .context("CLOUDFLARE_R2_ACCESS_KEY_ID is required for R2 operations")?;

        let secret_access_key = config
            .r2_secret_access_key
            .as_ref()
            .context("CLOUDFLARE_R2_SECRET_ACCESS_KEY is required for R2 operations")?;

        // R2 endpoint format: https://<account_id>.r2.cloudflarestorage.com
        let endpoint_url = format!("https://{}.r2.cloudflarestorage.com", account_id);

        debug!(
            endpoint = %endpoint_url,
            bucket_pdfs = %config.r2_bucket_pdfs,
            bucket_knowledge = %config.r2_bucket_knowledge,
            "Initializing R2 client"
        );

        // Create AWS credentials
        let credentials = Credentials::new(
            access_key_id,
            secret_access_key,
            None, // session token (not needed for R2)
            None, // expiration (static credentials)
            "r2", // provider name
        );

        // Build S3 config for R2
        let s3_config = aws_sdk_s3::Config::builder()
            .endpoint_url(&endpoint_url)
            .region(Region::new("auto")) // R2 uses "auto" region
            .credentials_provider(credentials)
            .force_path_style(false) // R2 uses virtual-hosted-style URLs
            .build();

        let client = Client::from_conf(s3_config);

        info!(
            account_id = %account_id,
            bucket_pdfs = %config.r2_bucket_pdfs,
            bucket_knowledge = %config.r2_bucket_knowledge,
            "R2 client initialized successfully"
        );

        Ok(Self {
            client,
            bucket_pdfs: config.r2_bucket_pdfs.clone(),
            bucket_knowledge: config.r2_bucket_knowledge.clone(),
            account_id: account_id.clone(),
        })
    }

    /// Generate a presigned upload URL for client direct upload
    ///
    /// # Arguments
    /// * `bucket_type` - Which bucket to upload to (Pdfs or Knowledge)
    /// * `key` - Object key (path within bucket)
    /// * `expires_in_secs` - URL validity duration in seconds (max 604800 = 7 days)
    ///
    /// # Returns
    /// Presigned PUT URL that client can use to upload directly to R2
    ///
    /// # Security
    /// - URLs are time-limited and expire after the specified duration
    /// - Recommend 3600s (1 hour) for uploads to minimize exposure window
    /// - Client must use PUT method and include Content-Type header
    #[instrument(skip(self), fields(account_id = %self.account_id, bucket = %bucket_type.as_str()))]
    pub async fn generate_upload_url(
        &self,
        bucket_type: BucketType,
        key: &str,
        expires_in_secs: u64,
    ) -> Result<String, AppError> {
        // Validate expiration (AWS/R2 max is 7 days)
        if expires_in_secs > 604800 {
            return Err(AppError::BadRequest(
                "Presigned URL expiration cannot exceed 7 days (604800 seconds)".to_string(),
            ));
        }

        let bucket = match bucket_type {
            BucketType::Pdfs => &self.bucket_pdfs,
            BucketType::Knowledge => &self.bucket_knowledge,
        };

        debug!(
            bucket = %bucket,
            key = %key,
            expires_in_secs = %expires_in_secs,
            "Generating presigned upload URL"
        );

        let presigning_config = PresigningConfig::expires_in(Duration::from_secs(expires_in_secs))
            .map_err(|e| {
                error!(error = ?e, "Failed to create presigning config");
                AppError::Internal(format!("Failed to create presigning config: {}", e))
            })?;

        let presigned = self
            .client
            .put_object()
            .bucket(bucket)
            .key(key)
            .presigned(presigning_config)
            .await
            .map_err(|e| {
                error!(
                    error = ?e,
                    bucket = %bucket,
                    key = %key,
                    "Failed to generate presigned upload URL"
                );
                AppError::Internal(format!("Failed to generate presigned upload URL: {}", e))
            })?;

        let url = presigned.uri().to_string();

        info!(
            bucket = %bucket,
            key = %key,
            url_length = %url.len(),
            expires_in_secs = %expires_in_secs,
            "Generated presigned upload URL"
        );

        Ok(url)
    }

    /// Generate a presigned download URL for temporary file access
    ///
    /// # Arguments
    /// * `bucket_type` - Which bucket to download from (Pdfs or Knowledge)
    /// * `key` - Object key (path within bucket)
    /// * `expires_in_secs` - URL validity duration in seconds (max 604800 = 7 days)
    ///
    /// # Returns
    /// Presigned GET URL that client can use to download directly from R2
    ///
    /// # Security
    /// - URLs are time-limited and expire after the specified duration
    /// - Recommend 3600s (1 hour) for downloads for security
    /// - No authentication required once URL is generated
    /// - Consider using Worker streaming for frequently accessed files
    #[instrument(skip(self), fields(account_id = %self.account_id, bucket = %bucket_type.as_str()))]
    pub async fn generate_download_url(
        &self,
        bucket_type: BucketType,
        key: &str,
        expires_in_secs: u64,
    ) -> Result<String, AppError> {
        // Validate expiration (AWS/R2 max is 7 days)
        if expires_in_secs > 604800 {
            return Err(AppError::BadRequest(
                "Presigned URL expiration cannot exceed 7 days (604800 seconds)".to_string(),
            ));
        }

        let bucket = match bucket_type {
            BucketType::Pdfs => &self.bucket_pdfs,
            BucketType::Knowledge => &self.bucket_knowledge,
        };

        debug!(
            bucket = %bucket,
            key = %key,
            expires_in_secs = %expires_in_secs,
            "Generating presigned download URL"
        );

        let presigning_config = PresigningConfig::expires_in(Duration::from_secs(expires_in_secs))
            .map_err(|e| {
                error!(error = ?e, "Failed to create presigning config");
                AppError::Internal(format!("Failed to create presigning config: {}", e))
            })?;

        let presigned = self
            .client
            .get_object()
            .bucket(bucket)
            .key(key)
            .presigned(presigning_config)
            .await
            .map_err(|e| {
                error!(
                    error = ?e,
                    bucket = %bucket,
                    key = %key,
                    "Failed to generate presigned download URL"
                );
                AppError::Internal(format!("Failed to generate presigned download URL: {}", e))
            })?;

        let url = presigned.uri().to_string();

        info!(
            bucket = %bucket,
            key = %key,
            url_length = %url.len(),
            expires_in_secs = %expires_in_secs,
            "Generated presigned download URL"
        );

        Ok(url)
    }

    /// Delete an object from R2
    ///
    /// # Arguments
    /// * `bucket_type` - Which bucket to delete from
    /// * `key` - Object key to delete
    ///
    /// # Returns
    /// Ok(()) if deletion succeeded or object didn't exist
    ///
    /// # Note
    /// R2 delete operations are idempotent - deleting a non-existent object succeeds
    #[instrument(skip(self), fields(account_id = %self.account_id, bucket = %bucket_type.as_str()))]
    pub async fn delete_object(
        &self,
        bucket_type: BucketType,
        key: &str,
    ) -> Result<(), AppError> {
        let bucket = match bucket_type {
            BucketType::Pdfs => &self.bucket_pdfs,
            BucketType::Knowledge => &self.bucket_knowledge,
        };

        debug!(
            bucket = %bucket,
            key = %key,
            "Deleting object from R2"
        );

        self.client
            .delete_object()
            .bucket(bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| {
                error!(
                    error = ?e,
                    bucket = %bucket,
                    key = %key,
                    "Failed to delete object from R2"
                );
                AppError::Internal(format!("Failed to delete object from R2: {}", e))
            })?;

        info!(
            bucket = %bucket,
            key = %key,
            "Successfully deleted object from R2"
        );

        Ok(())
    }

    /// Check if an object exists in R2
    ///
    /// # Arguments
    /// * `bucket_type` - Which bucket to check
    /// * `key` - Object key to check
    ///
    /// # Returns
    /// true if object exists, false otherwise
    #[instrument(skip(self), fields(account_id = %self.account_id, bucket = %bucket_type.as_str()))]
    pub async fn object_exists(
        &self,
        bucket_type: BucketType,
        key: &str,
    ) -> Result<bool, AppError> {
        let bucket = match bucket_type {
            BucketType::Pdfs => &self.bucket_pdfs,
            BucketType::Knowledge => &self.bucket_knowledge,
        };

        debug!(
            bucket = %bucket,
            key = %key,
            "Checking if object exists in R2"
        );

        match self
            .client
            .head_object()
            .bucket(bucket)
            .key(key)
            .send()
            .await
        {
            Ok(_) => {
                debug!(bucket = %bucket, key = %key, "Object exists");
                Ok(true)
            }
            Err(e) => {
                // Check if it's a "not found" error
                if e.to_string().contains("404") || e.to_string().contains("NotFound") {
                    debug!(bucket = %bucket, key = %key, "Object does not exist");
                    Ok(false)
                } else {
                    error!(
                        error = ?e,
                        bucket = %bucket,
                        key = %key,
                        "Failed to check object existence"
                    );
                    Err(AppError::Internal(format!(
                        "Failed to check object existence: {}",
                        e
                    )))
                }
            }
        }
    }

    /// Get object metadata without downloading the object
    ///
    /// # Arguments
    /// * `bucket_type` - Which bucket to query
    /// * `key` - Object key
    ///
    /// # Returns
    /// Object metadata including size, content type, etag, last modified
    #[instrument(skip(self), fields(account_id = %self.account_id, bucket = %bucket_type.as_str()))]
    pub async fn get_object_metadata(
        &self,
        bucket_type: BucketType,
        key: &str,
    ) -> Result<ObjectMetadata, AppError> {
        let bucket = match bucket_type {
            BucketType::Pdfs => &self.bucket_pdfs,
            BucketType::Knowledge => &self.bucket_knowledge,
        };

        debug!(
            bucket = %bucket,
            key = %key,
            "Getting object metadata from R2"
        );

        let response = self
            .client
            .head_object()
            .bucket(bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| {
                error!(
                    error = ?e,
                    bucket = %bucket,
                    key = %key,
                    "Failed to get object metadata"
                );
                if e.to_string().contains("404") || e.to_string().contains("NotFound") {
                    AppError::NotFound(format!("Object not found: {}", key))
                } else {
                    AppError::Internal(format!("Failed to get object metadata: {}", e))
                }
            })?;

        let metadata = ObjectMetadata {
            size_bytes: response.content_length().unwrap_or(0),
            content_type: response.content_type().map(|s| s.to_string()),
            etag: response.e_tag().map(|s| s.to_string()),
            last_modified: response.last_modified().map(|dt| dt.to_string()),
        };

        debug!(
            bucket = %bucket,
            key = %key,
            size_bytes = %metadata.size_bytes,
            "Retrieved object metadata"
        );

        Ok(metadata)
    }

    /// Get the bucket name for a given bucket type
    pub fn get_bucket_name(&self, bucket_type: BucketType) -> &str {
        match bucket_type {
            BucketType::Pdfs => &self.bucket_pdfs,
            BucketType::Knowledge => &self.bucket_knowledge,
        }
    }

    /// Download an object from R2
    ///
    /// # Arguments
    /// * `bucket_type` - Which bucket to download from
    /// * `key` - Object key to download
    ///
    /// # Returns
    /// Object bytes as Vec<u8>
    ///
    /// # Note
    /// For frequently accessed files, consider using Worker streaming with R2 bindings
    /// instead of this method for lower latency
    #[instrument(skip(self), fields(account_id = %self.account_id, bucket = %bucket_type.as_str()))]
    pub async fn download_object(
        &self,
        bucket_type: BucketType,
        key: &str,
    ) -> Result<Vec<u8>, AppError> {
        let bucket = match bucket_type {
            BucketType::Pdfs => &self.bucket_pdfs,
            BucketType::Knowledge => &self.bucket_knowledge,
        };

        debug!(
            bucket = %bucket,
            key = %key,
            "Downloading object from R2"
        );

        let response = self
            .client
            .get_object()
            .bucket(bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| {
                error!(
                    error = ?e,
                    bucket = %bucket,
                    key = %key,
                    "Failed to fetch object from R2"
                );
                if e.to_string().contains("404") || e.to_string().contains("NotFound") {
                    AppError::NotFound(format!("Object not found: {}", key))
                } else {
                    AppError::Internal(format!("Failed to fetch object from R2: {}", e))
                }
            })?;

        // Stream the object bytes into memory
        let data = response.body.collect().await.map_err(|e| {
            error!(
                error = ?e,
                bucket = %bucket,
                key = %key,
                "Failed to read object bytes from R2"
            );
            AppError::Internal(format!("Failed to read object bytes: {}", e))
        })?;

        let bytes = data.into_bytes().to_vec();

        info!(
            bucket = %bucket,
            key = %key,
            size_bytes = %bytes.len(),
            "Downloaded object from R2"
        );

        Ok(bytes)
    }
}

/// Object metadata retrieved from R2
#[derive(Debug, Clone)]
pub struct ObjectMetadata {
    /// Object size in bytes
    pub size_bytes: i64,
    /// Content type (MIME type)
    pub content_type: Option<String>,
    /// ETag for versioning/caching
    pub etag: Option<String>,
    /// Last modified timestamp
    pub last_modified: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_type_as_str() {
        assert_eq!(BucketType::Pdfs.as_str(), "pdfs");
        assert_eq!(BucketType::Knowledge.as_str(), "knowledge");
    }

    #[test]
    fn test_bucket_type_equality() {
        assert_eq!(BucketType::Pdfs, BucketType::Pdfs);
        assert_ne!(BucketType::Pdfs, BucketType::Knowledge);
    }
}
