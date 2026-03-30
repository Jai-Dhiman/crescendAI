import { afterEach, describe, expect, it } from "vitest";
import { useArtifactStore, getExpandedArtifact } from "./artifact";
import type { InlineComponent } from "../lib/types";

const sampleComponent: InlineComponent = {
	type: "exercise_set",
	config: {
		sourcePassage: "mm. 1-4",
		targetSkill: "legato",
		exercises: [
			{
				title: "Slow legato",
				instruction: "Play mm. 1-4 slowly with legato touch",
				focusDimension: "articulation",
				exerciseId: "ex-1",
			},
		],
	},
};

const anotherComponent: InlineComponent = {
	type: "score_highlight",
	config: { measures: [1, 2, 3] },
};

afterEach(() => {
	// Unregister all entries to reset state between tests
	const store = useArtifactStore.getState();
	const ids = Object.keys(store.states);
	for (const id of ids) {
		store.unregister(id);
	}
});

describe("register", () => {
	it("sets initial state to inline with the component", () => {
		const store = useArtifactStore.getState();
		store.register("a1", sampleComponent);

		const entry = useArtifactStore.getState().states["a1"];
		expect(entry).toBeDefined();
		expect(entry.state).toBe("inline");
		expect(entry.component).toEqual(sampleComponent);
	});
});

describe("collapse", () => {
	it("transitions from inline to collapsed", () => {
		const store = useArtifactStore.getState();
		store.register("a1", sampleComponent);
		useArtifactStore.getState().collapse("a1");

		expect(useArtifactStore.getState().states["a1"].state).toBe("collapsed");
	});

	it("is no-op from expanded", () => {
		const store = useArtifactStore.getState();
		store.register("a1", sampleComponent);
		useArtifactStore.getState().expand("a1");
		useArtifactStore.getState().collapse("a1");

		expect(useArtifactStore.getState().states["a1"].state).toBe("expanded");
	});

	it("is no-op if already collapsed", () => {
		const store = useArtifactStore.getState();
		store.register("a1", sampleComponent);
		useArtifactStore.getState().collapse("a1");
		useArtifactStore.getState().collapse("a1");

		expect(useArtifactStore.getState().states["a1"].state).toBe("collapsed");
	});
});

describe("expand", () => {
	it("transitions from collapsed to expanded", () => {
		const store = useArtifactStore.getState();
		store.register("a1", sampleComponent);
		useArtifactStore.getState().collapse("a1");
		useArtifactStore.getState().expand("a1");

		expect(useArtifactStore.getState().states["a1"].state).toBe("expanded");
	});

	it("transitions from inline to expanded", () => {
		const store = useArtifactStore.getState();
		store.register("a1", sampleComponent);
		useArtifactStore.getState().expand("a1");

		expect(useArtifactStore.getState().states["a1"].state).toBe("expanded");
	});

	it("sets previous expanded artifact back to inline", () => {
		const store = useArtifactStore.getState();
		store.register("a1", sampleComponent);
		store.register("a2", anotherComponent);

		useArtifactStore.getState().expand("a1");
		expect(useArtifactStore.getState().states["a1"].state).toBe("expanded");

		useArtifactStore.getState().expand("a2");
		expect(useArtifactStore.getState().states["a2"].state).toBe("expanded");
		expect(useArtifactStore.getState().states["a1"].state).toBe("inline");
	});
});

describe("restore", () => {
	it("transitions from collapsed to inline", () => {
		const store = useArtifactStore.getState();
		store.register("a1", sampleComponent);
		useArtifactStore.getState().collapse("a1");
		useArtifactStore.getState().restore("a1");

		expect(useArtifactStore.getState().states["a1"].state).toBe("inline");
	});

	it("is no-op from inline", () => {
		const store = useArtifactStore.getState();
		store.register("a1", sampleComponent);
		useArtifactStore.getState().restore("a1");

		expect(useArtifactStore.getState().states["a1"].state).toBe("inline");
	});
});

describe("closeOverlay", () => {
	it("transitions from expanded to inline", () => {
		const store = useArtifactStore.getState();
		store.register("a1", sampleComponent);
		useArtifactStore.getState().expand("a1");
		useArtifactStore.getState().closeOverlay("a1");

		expect(useArtifactStore.getState().states["a1"].state).toBe("inline");
	});

	it("is no-op if not expanded", () => {
		const store = useArtifactStore.getState();
		store.register("a1", sampleComponent);
		useArtifactStore.getState().closeOverlay("a1");

		expect(useArtifactStore.getState().states["a1"].state).toBe("inline");
	});

	it("is no-op from collapsed", () => {
		const store = useArtifactStore.getState();
		store.register("a1", sampleComponent);
		useArtifactStore.getState().collapse("a1");
		useArtifactStore.getState().closeOverlay("a1");

		expect(useArtifactStore.getState().states["a1"].state).toBe("collapsed");
	});
});

describe("unregister", () => {
	it("removes entry", () => {
		const store = useArtifactStore.getState();
		store.register("a1", sampleComponent);
		useArtifactStore.getState().unregister("a1");

		expect(useArtifactStore.getState().states["a1"]).toBeUndefined();
	});
});

describe("getExpandedArtifact", () => {
	it("returns null when nothing is expanded", () => {
		const store = useArtifactStore.getState();
		store.register("a1", sampleComponent);

		const result = getExpandedArtifact(useArtifactStore.getState());
		expect(result).toBeNull();
	});

	it("returns { id, entry } when something is expanded", () => {
		const store = useArtifactStore.getState();
		store.register("a1", sampleComponent);
		useArtifactStore.getState().expand("a1");

		const result = getExpandedArtifact(useArtifactStore.getState());
		expect(result).not.toBeNull();
		expect(result!.id).toBe("a1");
		expect(result!.entry.state).toBe("expanded");
		expect(result!.entry.component).toEqual(sampleComponent);
	});
});

describe("setExerciseStatus", () => {
	it("tracks exercise state with status", () => {
		const store = useArtifactStore.getState();
		store.register("a1", sampleComponent);
		useArtifactStore.getState().setExerciseStatus("a1", "ex-1", "loading");

		const exerciseStates = useArtifactStore.getState().states["a1"].exerciseStates;
		expect(exerciseStates).toBeDefined();
		expect(exerciseStates!["ex-1"]).toEqual({ status: "loading" });
	});

	it("tracks exercise state with status and studentExerciseId", () => {
		const store = useArtifactStore.getState();
		store.register("a1", sampleComponent);
		useArtifactStore.getState().setExerciseStatus("a1", "ex-1", "assigned", "student-ex-42");

		const exerciseStates = useArtifactStore.getState().states["a1"].exerciseStates;
		expect(exerciseStates!["ex-1"]).toEqual({ status: "assigned", studentExerciseId: "student-ex-42" });
	});

	it("updates existing exercise state", () => {
		const store = useArtifactStore.getState();
		store.register("a1", sampleComponent);
		useArtifactStore.getState().setExerciseStatus("a1", "ex-1", "loading");
		useArtifactStore.getState().setExerciseStatus("a1", "ex-1", "completed", "student-ex-42");

		const exerciseStates = useArtifactStore.getState().states["a1"].exerciseStates;
		expect(exerciseStates!["ex-1"]).toEqual({ status: "completed", studentExerciseId: "student-ex-42" });
	});
});
