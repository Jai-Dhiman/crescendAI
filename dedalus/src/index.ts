/**
 * CrescendAI Dedalus Worker
 *
 * TypeScript Cloudflare Worker that wraps the Dedalus API using the official SDK.
 * Connects to the main Rust worker via service binding.
 */

import { Hono } from 'hono';
import { cors } from 'hono/cors';
import { Dedalus } from 'dedalus-labs';

type Bindings = {
  DEDALUS_API_KEY: string;
};

const app = new Hono<{ Bindings: Bindings }>();

// Enable CORS
app.use('*', cors());

// Health check endpoint
app.get('/health', (c) => {
  return c.json({ status: 'ok', service: 'dedalus-worker' });
});

// Chat completion endpoint
app.post('/chat', async (c) => {
  try {
    const apiKey = c.env.DEDALUS_API_KEY;

    if (!apiKey) {
      return c.json({ error: 'DEDALUS_API_KEY not configured' }, 500);
    }

    // Parse request body
    const body = await c.req.json();

    // Initialize Dedalus client
    const client = new Dedalus({
      apiKey: apiKey,
    });

    // Make request using Dedalus SDK
    const completion = await client.chat.create({
      model: body.model || 'openai/gpt-4o-mini',
      input: body.messages || body.input || [],
      tools: body.tools,
      mcp_servers: body.mcp_servers,
      temperature: body.temperature || 0.7,
      max_tokens: body.max_tokens || 2000,
      stream: false,
    });

    return c.json(completion);

  } catch (error) {
    console.error('Error in dedalus worker:', error);

    // Handle Dedalus SDK errors
    if (error && typeof error === 'object' && 'status' in error) {
      return c.json(
        { error: error.message || 'Dedalus API error' },
        (error as any).status || 500
      );
    }

    return c.json(
      { error: `Internal error: ${error instanceof Error ? error.message : String(error)}` },
      500
    );
  }
});

// 404 handler
app.notFound((c) => {
  return c.json({ error: 'Not found' }, 404);
});

export default app;
