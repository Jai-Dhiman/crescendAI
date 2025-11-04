# CrescendAI Dedalus Worker

Python Cloudflare Worker that wraps the Dedalus SDK for AI-powered piano tutoring.

## Overview

This worker provides a simple HTTP interface for the main Rust worker to make chat completion requests using the Dedalus SDK. It's connected via Cloudflare Workers service binding for low-latency communication.

## Architecture

```
Web App → Rust Worker (main) → Python Worker (Dedalus SDK) → Dedalus API
                ↓                        ↓
            D1 + Vectorize          Service Binding
```

- [Cloudflare Python Workers](https://developers.cloudflare.com/workers/languages/python/)
- [Cloudflare Service Bindings](https://developers.cloudflare.com/workers/runtime-apis/bindings/service-bindings/)
- [Dedalus Documentation](https://docs.dedaluslabs.ai/)
