import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles.css";

const rootElement = document.getElementById("root");

if (!rootElement) {
  throw new Error("Frontend mount error: #root element not found");
}

try {
  createRoot(rootElement).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
} catch (error) {
  rootElement.innerHTML = `
    <div style="padding:16px;font-family:monospace;color:#7a1d1d;">
      <h3>Frontend bootstrap error</h3>
      <pre style="white-space:pre-wrap;">${String(error?.stack || error?.message || error)}</pre>
    </div>
  `;
}
