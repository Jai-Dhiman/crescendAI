import React, { useEffect } from "react";
import { createRoot } from "react-dom/client";
import "./score-host";

function ScoreHostApp() {
  useEffect(() => {
    void (
      window as unknown as { ScoreHost: { ready(): Promise<void> } }
    ).ScoreHost.ready();
  }, []);

  return (
    <div
      id="scorehost-container"
      style={{ width: "100%", height: "100%", position: "relative" }}
    />
  );
}

const rootEl = document.getElementById("score-root");
if (!rootEl) {
  throw new Error("score-root element not found");
}
createRoot(rootEl).render(<ScoreHostApp />);
