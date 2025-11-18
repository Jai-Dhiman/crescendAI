#!/usr/bin/env python3
"""
Download training data directly from cloud storage (bypasses Google Drive FUSE).

Future-proof solutions for large datasets:
1. Google Cloud Storage (GCS) - gsutil/gcsfs
2. AWS S3 - boto3
3. Azure Blob - azure-storage-blob
4. Hugging Face Hub - huggingface_hub
5. DagsHub Storage - dagshub

This script supports multiple backends and downloads directly to local SSD,
completely avoiding Google Drive FUSE mount issues.
"""

import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm


def download_from_gcs(bucket_name, prefix, local_root):
    """
    Download from Google Cloud Storage (recommended for Colab).

    Setup:
    1. Upload data to GCS bucket: gsutil -m cp -r crescendai_data gs://your-bucket/
    2. Make bucket public or authenticate with service account
    3. Download directly to Colab (no Drive needed!)

    Benefits:
    - No file count limits (unlike Drive)
    - Much faster (10-100x vs Drive FUSE)
    - Parallel downloads
    - Resumable
    """
    try:
        from google.cloud import storage
    except ImportError:
        print("Installing google-cloud-storage...")
        import subprocess
        subprocess.run(["pip", "install", "-q", "google-cloud-storage"], check=True)
        from google.cloud import storage

    print(f"Downloading from gs://{bucket_name}/{prefix}...")

    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)

    # List all blobs with prefix
    blobs = list(bucket.list_blobs(prefix=prefix))
    print(f"Found {len(blobs):,} files to download")

    # Download each blob
    for blob in tqdm(blobs, desc="Downloading"):
        # Calculate local path
        rel_path = blob.name[len(prefix):].lstrip('/')
        local_path = Path(local_root) / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if already exists
        if local_path.exists() and local_path.stat().st_size == blob.size:
            continue

        # Download
        blob.download_to_filename(str(local_path))

    print(f"✓ Downloaded to {local_root}")


def download_from_s3(bucket_name, prefix, local_root):
    """Download from AWS S3."""
    try:
        import boto3
    except ImportError:
        print("Installing boto3...")
        import subprocess
        subprocess.run(["pip", "install", "-q", "boto3"], check=True)
        import boto3

    print(f"Downloading from s3://{bucket_name}/{prefix}...")

    s3 = boto3.client('s3', config=boto3.session.Config(signature_version='UNSIGNED'))

    # List objects
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    files = []
    for page in pages:
        files.extend(page.get('Contents', []))

    print(f"Found {len(files):,} files to download")

    # Download
    for obj in tqdm(files, desc="Downloading"):
        key = obj['Key']
        rel_path = key[len(prefix):].lstrip('/')
        local_path = Path(local_root) / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if local_path.exists() and local_path.stat().st_size == obj['Size']:
            continue

        s3.download_file(bucket_name, key, str(local_path))

    print(f"✓ Downloaded to {local_root}")


def download_from_hf_hub(repo_id, subfolder, local_root):
    """
    Download from Hugging Face Hub.

    Setup:
    1. Create HF dataset repo: huggingface-cli repo create your-username/crescendai-data --type dataset
    2. Upload data: huggingface-cli upload your-username/crescendai-data ./crescendai_data
    3. Download in Colab (no Drive needed!)

    Benefits:
    - Free hosting for public datasets
    - Built-in versioning
    - Fast downloads
    - Great for ML datasets
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        import subprocess
        subprocess.run(["pip", "install", "-q", "huggingface_hub"], check=True)
        from huggingface_hub import snapshot_download

    print(f"Downloading from Hugging Face: {repo_id}")

    local_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=f"{subfolder}/*" if subfolder else None,
        local_dir=local_root,
        local_dir_use_symlinks=False,
    )

    print(f"✓ Downloaded to {local_path}")


def download_from_http(base_url, file_list, local_root):
    """
    Download from HTTP/HTTPS (e.g., university server, personal hosting).

    Setup:
    1. Host files on any web server
    2. Create file_list.txt with all file URLs
    3. Download directly

    Good for:
    - One-time datasets
    - Custom hosting
    - No cloud account needed
    """
    import requests

    print(f"Downloading from {base_url}...")

    # Read file list
    with open(file_list) as f:
        files = [line.strip() for line in f if line.strip()]

    print(f"Found {len(files):,} files to download")

    # Download each file
    for rel_path in tqdm(files, desc="Downloading"):
        url = f"{base_url}/{rel_path}"
        local_path = Path(local_root) / rel_path
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip if exists
        if local_path.exists():
            continue

        # Download
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    print(f"✓ Downloaded to {local_root}")


def create_tar_archive_instructions():
    """
    Alternative: Create a single tar.gz archive (best for very large datasets).

    This bypasses the "too many files" issue entirely!
    """
    instructions = """
# RECOMMENDED: Single Archive Solution

Instead of copying individual files, create a tar.gz archive:

## Step 1: Create archive (run locally or in Colab with Drive mounted)
```bash
# From your crescendai_data folder
cd /content/drive/MyDrive/
tar -czf crescendai_data.tar.gz crescendai_data/

# This creates ONE file instead of 65,000+ files!
# Size: ~20-30 GB compressed
```

## Step 2: Upload to cloud (pick ONE)

### Option A: Google Cloud Storage (Recommended for Colab)
```bash
# Upload single file (much faster than 65K files!)
gsutil cp crescendai_data.tar.gz gs://your-bucket/

# Make publicly readable (or use service account)
gsutil acl ch -u AllUsers:R gs://your-bucket/crescendai_data.tar.gz
```

### Option B: Hugging Face Hub (Free for public datasets)
```bash
# Install CLI
pip install huggingface_hub[cli]

# Login
huggingface-cli login

# Upload
huggingface-cli upload your-username/crescendai-data crescendai_data.tar.gz
```

### Option C: AWS S3
```bash
aws s3 cp crescendai_data.tar.gz s3://your-bucket/ --acl public-read
```

## Step 3: Download in Colab (FAST!)

### From GCS:
```python
!wget https://storage.googleapis.com/your-bucket/crescendai_data.tar.gz
!tar -xzf crescendai_data.tar.gz -C /tmp/
```

### From Hugging Face:
```python
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="your-username/crescendai-data",
    filename="crescendai_data.tar.gz",
    repo_type="dataset",
    local_dir="/tmp/"
)
!tar -xzf /tmp/crescendai_data.tar.gz -C /tmp/
```

### From S3:
```python
!wget https://your-bucket.s3.amazonaws.com/crescendai_data.tar.gz
!tar -xzf crescendai_data.tar.gz -C /tmp/
```

## Benefits:
- ✓ Single file = no "too many files" issues
- ✓ 10-100x faster download than Drive
- ✓ Compressed = smaller storage/transfer
- ✓ Resumable downloads
- ✓ No Drive mounting needed
- ✓ Works reliably every time

## Time Comparison:
- Drive FUSE (65K files): 30-60 min + frequent failures
- Single tar.gz: 5-10 min, reliable
"""
    return instructions


def main():
    parser = argparse.ArgumentParser(
        description='Download training data from cloud storage',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=create_tar_archive_instructions()
    )

    parser.add_argument(
        '--backend',
        choices=['gcs', 's3', 'hf', 'http', 'instructions'],
        default='instructions',
        help='Cloud storage backend'
    )
    parser.add_argument('--bucket', help='GCS/S3 bucket name')
    parser.add_argument('--prefix', help='Prefix/folder in bucket')
    parser.add_argument('--repo-id', help='Hugging Face repo ID')
    parser.add_argument('--base-url', help='Base URL for HTTP downloads')
    parser.add_argument('--file-list', help='File list for HTTP downloads')
    parser.add_argument(
        '--local-root',
        default='/tmp/crescendai_data',
        help='Local destination directory'
    )

    args = parser.parse_args()

    if args.backend == 'instructions':
        print("="*70)
        print("CLOUD STORAGE SETUP INSTRUCTIONS")
        print("="*70)
        print(create_tar_archive_instructions())
        return 0

    # Create local directory
    Path(args.local_root).mkdir(parents=True, exist_ok=True)

    # Download based on backend
    try:
        if args.backend == 'gcs':
            if not args.bucket or not args.prefix:
                print("Error: --bucket and --prefix required for GCS")
                return 1
            download_from_gcs(args.bucket, args.prefix, args.local_root)

        elif args.backend == 's3':
            if not args.bucket or not args.prefix:
                print("Error: --bucket and --prefix required for S3")
                return 1
            download_from_s3(args.bucket, args.prefix, args.local_root)

        elif args.backend == 'hf':
            if not args.repo_id:
                print("Error: --repo-id required for Hugging Face")
                return 1
            download_from_hf_hub(args.repo_id, args.prefix or '', args.local_root)

        elif args.backend == 'http':
            if not args.base_url or not args.file_list:
                print("Error: --base-url and --file-list required for HTTP")
                return 1
            download_from_http(args.base_url, args.file_list, args.local_root)

        print("\n✓ Download complete!")
        return 0

    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
