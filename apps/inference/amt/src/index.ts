// CF Containers is in beta -- the API surface (Container base class, containerFetch,
// startAndWaitForPorts) may evolve. Pin @cloudflare/containers and test after upgrades.

import { Container } from "@cloudflare/containers";

export interface Env {
  AMT_CONTAINER: DurableObjectNamespace;
  POOL_SIZE: string;
  INSTANCE_TYPE: string;
  SLEEP_AFTER_S: string;
}

interface InstanceState {
  busy: boolean;
  lastUsed: number;
  inferenceCount: number;
}

// AmtContainer is both a Durable Object and a Container.
// Each named instance (amt-0, amt-1, ...) runs one container process.
// DO storage persists busy/lastUsed/inferenceCount across requests within an instance lifetime.
export class AmtContainer extends Container {
  env: Env;

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    this.env = env;
  }

  // defaultPort is the port the container process listens on.
  override defaultPort = 8080;

  // sleepAfter returns idle timeout in milliseconds.
  // Container will be suspended after this many ms of inactivity.
  get sleepAfter(): number {
    const seconds = parseInt(this.env.SLEEP_AFTER_S ?? "300", 10);
    return seconds * 1000;
  }

  // instanceType declares the hardware tier for this container.
  get instanceType(): string {
    return this.env.INSTANCE_TYPE ?? "standard-4";
  }

  private async getInstanceState(): Promise<InstanceState> {
    const stored = await this.ctx.storage.get<InstanceState>("state");
    if (stored !== undefined) {
      return stored;
    }
    const initial: InstanceState = {
      busy: false,
      lastUsed: Date.now(),
      inferenceCount: 0,
    };
    await this.ctx.storage.put("state", initial);
    return initial;
  }

  private async markBusy(): Promise<void> {
    const state = await this.getInstanceState();
    state.busy = true;
    state.lastUsed = Date.now();
    await this.ctx.storage.put("state", state);
  }

  private async markIdle(): Promise<void> {
    const state = await this.getInstanceState();
    state.busy = false;
    state.lastUsed = Date.now();
    state.inferenceCount += 1;
    await this.ctx.storage.put("state", state);
  }

  override async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url);

    if (request.method === "GET" && url.pathname === "/is-busy") {
      const state = await this.getInstanceState();
      return Response.json({ busy: state.busy });
    }

    if (request.method === "GET" && url.pathname === "/health") {
      // Forward health check to the container process.
      return this.containerFetch(new Request(request.url, { method: "GET" }));
    }

    if (request.method === "POST" && url.pathname === "/transcribe") {
      await this.markBusy();
      try {
        // containerFetch sends the request to the container process.
        // startAndWaitForPorts() is called implicitly on first containerFetch.
        return await this.containerFetch(request);
      } finally {
        await this.markIdle();
      }
    }

    return new Response(JSON.stringify({ error: { code: "NOT_FOUND", message: "Route not found" } }), {
      status: 404,
      headers: { "Content-Type": "application/json" },
    });
  }
}

// Default export: the routing Worker that distributes requests across the container pool.
export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    if (request.method === "GET" && url.pathname === "/health") {
      return Response.json({ status: "routing_worker_ok" });
    }

    if (request.method === "POST" && url.pathname === "/transcribe") {
      const poolSize = parseInt(env.POOL_SIZE ?? "2", 10);

      // Iterate named instances amt-0..amt-{poolSize-1} to find an idle one.
      for (let i = 0; i < poolSize; i++) {
        const instanceName = `amt-${i}`;
        const id = env.AMT_CONTAINER.idFromName(instanceName);
        const stub = env.AMT_CONTAINER.get(id);

        const busyResponse = await stub.fetch(
          new Request(`${url.origin}/is-busy`, { method: "GET" })
        );
        const { busy } = (await busyResponse.json()) as { busy: boolean };

        if (!busy) {
          // Forward the original request body to this instance.
          return stub.fetch(
            new Request(`${url.origin}/transcribe`, {
              method: "POST",
              headers: request.headers,
              body: request.body,
            })
          );
        }
      }

      // All instances are busy.
      return new Response(
        JSON.stringify({
          error: {
            code: "POOL_EXHAUSTED",
            message: `All ${poolSize} AMT container instances are currently busy. Retry after a moment.`,
          },
        }),
        {
          status: 503,
          headers: { "Content-Type": "application/json" },
        }
      );
    }

    return new Response(
      JSON.stringify({ error: { code: "NOT_FOUND", message: "Route not found" } }),
      {
        status: 404,
        headers: { "Content-Type": "application/json" },
      }
    );
  },
};
