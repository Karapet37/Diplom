const DEFAULT_MODE = "integration";
const DEFAULT_HOST = "generic";
const DEFAULT_APP_ID = "external_app";

function asObject(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return {};
  }
  return value;
}

function asToken(value, fallback) {
  const raw = String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^\w.-]+/g, "_")
    .slice(0, 64);
  return raw || fallback;
}

function asMode(value) {
  const token = String(value || "").trim().toLowerCase();
  if (token === "standalone" || token === "integration") {
    return token;
  }
  return DEFAULT_MODE;
}

async function defaultHttpRequester(method, url, payload = null, options = {}) {
  const fetchImpl = typeof options.fetchImpl === "function" ? options.fetchImpl : globalThis.fetch;
  if (typeof fetchImpl !== "function") {
    throw new Error("fetch is not available; provide fetchImpl in client config");
  }
  const headers = {
    Accept: "application/json",
    ...(asObject(options.headers) || {}),
  };
  const requestInit = {
    method: String(method || "GET").trim().toUpperCase() || "GET",
    headers,
  };
  if (requestInit.method !== "GET") {
    headers["Content-Type"] = headers["Content-Type"] || "application/json";
    requestInit.body = JSON.stringify(asObject(payload));
  }

  const response = await fetchImpl(url, requestInit);
  const rawText = await response.text();
  let parsed = {};
  if (String(rawText || "").trim()) {
    try {
      parsed = JSON.parse(rawText);
    } catch (_error) {
      parsed = { raw: rawText };
    }
  }
  if (!response.ok) {
    const detail =
      parsed && typeof parsed === "object" && !Array.isArray(parsed) && parsed.detail
        ? parsed.detail
        : `${response.status} ${response.statusText}`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    return { data: parsed };
  }
  return parsed;
}

export function createIntegrationLayerClient(config = {}) {
  const root = asObject(config);
  const mode = asMode(root.mode);
  const defaultHost = asToken(root.host, DEFAULT_HOST);
  const defaultAppId = asToken(root.appId || root.app_id, DEFAULT_APP_ID);
  const baseUrl = String(root.baseUrl || root.base_url || "").trim().replace(/\/+$/, "");
  const standaloneManifest = typeof root.standaloneManifest === "function" ? root.standaloneManifest : null;
  const standaloneInvoke = typeof root.standaloneInvoke === "function" ? root.standaloneInvoke : null;
  const httpRequester =
    typeof root.httpRequester === "function" ? root.httpRequester : defaultHttpRequester;

  async function manifest(params = {}) {
    const payload = asObject(params);
    const requestPayload = {
      host: asToken(payload.host, defaultHost),
      app_id: asToken(payload.app_id || payload.appId, defaultAppId),
    };
    if (mode === "standalone") {
      if (!standaloneManifest) {
        throw new Error("standalone manifest handler is not configured");
      }
      return standaloneManifest(requestPayload);
    }
    if (!baseUrl) {
      throw new Error("baseUrl is required in integration mode");
    }
    const query = new URLSearchParams(requestPayload).toString();
    return httpRequester("GET", `${baseUrl}/api/integration/layer/manifest?${query}`, null, root);
  }

  async function invoke(payload = {}) {
    const rootPayload = asObject(payload);
    const requestPayload = {
      ...rootPayload,
      host: asToken(rootPayload.host, defaultHost),
      app_id: asToken(rootPayload.app_id || rootPayload.appId, defaultAppId),
    };
    if (mode === "standalone") {
      if (!standaloneInvoke) {
        throw new Error("standalone invoke handler is not configured");
      }
      return standaloneInvoke(requestPayload);
    }
    if (!baseUrl) {
      throw new Error("baseUrl is required in integration mode");
    }
    return httpRequester("POST", `${baseUrl}/api/integration/layer/invoke`, requestPayload, root);
  }

  async function invokeAction(action, params = {}) {
    const rootParams = asObject(params);
    return invoke({
      ...rootParams,
      action: String(action || "").trim(),
      input: asObject(rootParams.input),
      options: asObject(rootParams.options),
    });
  }

  async function respond(message, params = {}) {
    const rootParams = asObject(params);
    return invokeAction("wrapper.respond", {
      ...rootParams,
      input: {
        ...asObject(rootParams.input),
        message: String(message || "").trim(),
      },
    });
  }

  async function archiveChat(message, params = {}) {
    const rootParams = asObject(params);
    return invokeAction("archive.chat", {
      ...rootParams,
      input: {
        ...asObject(rootParams.input),
        message: String(message || "").trim(),
      },
    });
  }

  async function updateUserGraph(text, params = {}) {
    const rootParams = asObject(params);
    return invokeAction("user_graph.update", {
      ...rootParams,
      input: {
        ...asObject(rootParams.input),
        message: String(text || "").trim(),
      },
    });
  }

  async function ingestPersonalTree(text, params = {}) {
    const rootParams = asObject(params);
    return invokeAction("personal_tree.ingest", {
      ...rootParams,
      input: {
        ...asObject(rootParams.input),
        message: String(text || "").trim(),
      },
    });
  }

  return Object.freeze({
    mode,
    defaults: Object.freeze({
      host: defaultHost,
      app_id: defaultAppId,
      base_url: baseUrl,
    }),
    manifest,
    invoke,
    invokeAction,
    respond,
    archiveChat,
    updateUserGraph,
    ingestPersonalTree,
  });
}

export function createHttpIntegrationLayerClient(config = {}) {
  return createIntegrationLayerClient({
    ...asObject(config),
    mode: "integration",
  });
}

export function createStandaloneIntegrationLayerClient(config = {}) {
  return createIntegrationLayerClient({
    ...asObject(config),
    mode: "standalone",
  });
}

export default createIntegrationLayerClient;
