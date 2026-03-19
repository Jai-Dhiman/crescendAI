export function handsLabel(hands: "left" | "right" | "both"): string {
	if (hands === "left") return "LH";
	if (hands === "right") return "RH";
	return "Both";
}
