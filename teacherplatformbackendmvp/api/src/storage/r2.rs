use anyhow::{Context, Result};
use aws_config::Region;
use aws_credential_types::Credentials;
use aws_sdk_s3::{
    config::Builder as S3ConfigBuilder,
    presigning::PresigningConfig,
    primitives::ByteStream,
    Client as S3Client,
};
use std::time::Duration;

#[derive(Clone)]
pub struct R2Client {
    client: S3Client,
    pdfs_bucket: String,
    knowledge_bucket: String,
}

impl R2Client {
    pub fn new(
        account_id: &str,
        access_key_id: &str,
        secret_access_key: &str,
        pdfs_bucket: String,
        knowledge_bucket: String,
    ) -> Result<Self> {
        // R2 endpoint format: https://<account_id>.r2.cloudflarestorage.com
        let endpoint = format!("https://{}.r2.cloudflarestorage.com", account_id);

        // Create credentials
        let credentials = Credentials::new(
            access_key_id,
            secret_access_key,
            None,
            None,
            "r2-credentials",
        );

        // Configure S3 client for R2
        let s3_config = S3ConfigBuilder::new()
            .endpoint_url(endpoint)
            .region(Region::new("auto"))
            .credentials_provider(credentials)
            .build();

        let client = S3Client::from_conf(s3_config);

        Ok(Self {
            client,
            pdfs_bucket,
            knowledge_bucket,
        })
    }

    pub async fn upload_file(
        &self,
        bucket: &str,
        key: &str,
        data: ByteStream,
        content_type: Option<&str>,
    ) -> Result<()> {
        let mut request = self
            .client
            .put_object()
            .bucket(bucket)
            .key(key)
            .body(data);

        if let Some(ct) = content_type {
            request = request.content_type(ct);
        }

        request
            .send()
            .await
            .context("Failed to upload file to R2")?;

        tracing::info!("Uploaded file to R2: bucket={}, key={}", bucket, key);
        Ok(())
    }

    pub async fn generate_presigned_upload_url(
        &self,
        bucket: &str,
        key: &str,
        expires_in: Duration,
    ) -> Result<String> {
        let presigning_config = PresigningConfig::expires_in(expires_in)
            .context("Failed to create presigning config")?;

        let presigned_request = self
            .client
            .put_object()
            .bucket(bucket)
            .key(key)
            .presigned(presigning_config)
            .await
            .context("Failed to generate presigned upload URL")?;

        Ok(presigned_request.uri().to_string())
    }

    pub async fn generate_presigned_download_url(
        &self,
        bucket: &str,
        key: &str,
        expires_in: Duration,
    ) -> Result<String> {
        let presigning_config = PresigningConfig::expires_in(expires_in)
            .context("Failed to create presigning config")?;

        let presigned_request = self
            .client
            .get_object()
            .bucket(bucket)
            .key(key)
            .presigned(presigning_config)
            .await
            .context("Failed to generate presigned download URL")?;

        Ok(presigned_request.uri().to_string())
    }

    pub async fn delete_file(&self, bucket: &str, key: &str) -> Result<()> {
        self.client
            .delete_object()
            .bucket(bucket)
            .key(key)
            .send()
            .await
            .context("Failed to delete file from R2")?;

        tracing::info!("Deleted file from R2: bucket={}, key={}", bucket, key);
        Ok(())
    }

    pub async fn get_file_metadata(
        &self,
        bucket: &str,
        key: &str,
    ) -> Result<FileMetadata> {
        let response = self
            .client
            .head_object()
            .bucket(bucket)
            .key(key)
            .send()
            .await
            .context("Failed to get file metadata from R2")?;

        Ok(FileMetadata {
            size: response.content_length().unwrap_or(0) as u64,
            content_type: response.content_type().map(|s| s.to_string()),
            last_modified: response
                .last_modified()
                .map(|dt| dt.to_string())
                .unwrap_or_default(),
        })
    }

    pub fn pdfs_bucket(&self) -> &str {
        &self.pdfs_bucket
    }

    pub fn knowledge_bucket(&self) -> &str {
        &self.knowledge_bucket
    }
}

#[derive(Debug, Clone)]
pub struct FileMetadata {
    pub size: u64,
    pub content_type: Option<String>,
    pub last_modified: String,
}
