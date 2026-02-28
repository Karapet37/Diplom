import { getProjectIntegrationLayerManifest, invokeProjectIntegrationLayer } from "./api";

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

export function createIntegrationLayerClient(config = {}) {
  const root = asObject(config);
  const mode = asMode(root.mode);
  const defaultHost = asToken(root.host, DEFAULT_HOST);
  const defaultAppId = asToken(root.appId || root.app_id, DEFAULT_APP_ID);
  const manifestRequest =
    typeof root.manifestRequest === "function" ? root.manifestRequest : getProjectIntegrationLayerManifest;
  const invokeRequest =
    typeof root.invokeRequest === "function" ? root.invokeRequest : invokeProjectIntegrationLayer;
  const standaloneManifest = typeof root.standaloneManifest === "function" ? root.standaloneManifest : null;
  const standaloneInvoke = typeof root.standaloneInvoke === "function" ? root.standaloneInvoke : null;

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
    return manifestRequest(requestPayload);
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
    return invokeRequest(requestPayload);
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
        ...(asObject(rootParams.input) || {}),
        message: String(message || "").trim(),
      },
    });
  }

  async function archiveChat(message, params = {}) {
    const rootParams = asObject(params);
    return invokeAction("archive.chat", {
      ...rootParams,
      input: {
        ...(asObject(rootParams.input) || {}),
        message: String(message || "").trim(),
      },
    });
  }

  async function updateUserGraph(text, params = {}) {
    const rootParams = asObject(params);
    return invokeAction("user_graph.update", {
      ...rootParams,
      input: {
        ...(asObject(rootParams.input) || {}),
        message: String(text || "").trim(),
      },
    });
  }

  async function ingestPersonalTree(text, params = {}) {
    const rootParams = asObject(params);
    return invokeAction("personal_tree.ingest", {
      ...rootParams,
      input: {
        ...(asObject(rootParams.input) || {}),
        message: String(text || "").trim(),
      },
    });
  }

  return Object.freeze({
    mode,
    defaults: Object.freeze({
      host: defaultHost,
      app_id: defaultAppId,
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

export default createIntegrationLayerClient;
