import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Set CWD to project root so that paths like "docs/harness/skills/..." resolve correctly.
// This file is at apps/api/scripts/, so project root is three levels up.
export async function setup() {
	process.chdir(resolve(__dirname, "../../.."));
}
