import React from "react";
import { createRoot } from "react-dom/client";

function ScoreHostApp() {
  return <div id="scorehost-container" style={{ width: "100%", height: "100%" }} />;
}

const rootEl = document.getElementById("score-root");
if (!rootEl) {
  throw new Error("score-root element not found");
}
createRoot(rootEl).render(<ScoreHostApp />);
