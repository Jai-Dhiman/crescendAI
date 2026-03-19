import { createContext, useContext } from "react";

export const ArtifactScrollContext = createContext<React.RefObject<HTMLDivElement | null> | null>(null);

export function useArtifactScrollContext() {
	return useContext(ArtifactScrollContext);
}
