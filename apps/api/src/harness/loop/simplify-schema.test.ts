import { describe, expect, it } from "vitest";
import { simplifyConstrainedSchema } from "./simplify-schema";

describe("simplifyConstrainedSchema — const -> enum", () => {
	it("rewrites const to a single-value enum", () => {
		const out = simplifyConstrainedSchema({
			type: "object",
			properties: { kind: { type: "string", const: "loop" } },
		});
		expect((out.properties as Record<string, unknown>).kind).toEqual({
			type: "string",
			enum: ["loop"],
		});
	});
});

describe("simplifyConstrainedSchema — tuple items -> uniform", () => {
	it("collapses tuple-form items (array of schemas) to a single item schema", () => {
		const out = simplifyConstrainedSchema({
			type: "array",
			minItems: 2,
			maxItems: 2,
			items: [
				{ type: "integer", exclusiveMinimum: 0 },
				{ type: "integer", exclusiveMinimum: 0 },
			],
		});
		expect(out.items).toEqual({ type: "integer", exclusiveMinimum: 0 });
	});
});

describe("simplifyConstrainedSchema — object union -> single object", () => {
	it("merges an anyOf of object branches into one object with merged discriminator enum and intersected required", () => {
		const out = simplifyConstrainedSchema({
			anyOf: [
				{
					type: "object",
					properties: {
						kind: { type: "string", const: "a" },
						shared: { type: "string" },
						onlyA: { type: "number" },
					},
					required: ["kind", "shared", "onlyA"],
				},
				{
					type: "object",
					properties: {
						kind: { type: "string", const: "b" },
						shared: { type: "string" },
						onlyB: { type: "string" },
					},
					required: ["kind", "shared"],
				},
			],
		});
		expect(out.type).toBe("object");
		const props = out.properties as Record<string, Record<string, unknown>>;
		// discriminator enum is the union of both branch consts
		expect(props.kind).toEqual({ type: "string", enum: ["a", "b"] });
		// union of all properties present
		expect(Object.keys(props).sort()).toEqual([
			"kind",
			"onlyA",
			"onlyB",
			"shared",
		]);
		// required is the intersection (onlyA dropped since not required in branch b)
		expect(out.required).toEqual(["kind", "shared"]);
		expect(out.additionalProperties).toBe(false);
	});

	it("makes the merged object nullable when the union has a null branch (incl. nested anyOf)", () => {
		const out = simplifyConstrainedSchema({
			anyOf: [
				{
					anyOf: [
						{
							type: "object",
							properties: { kind: { type: "string", const: "a" } },
							required: ["kind"],
						},
						{
							type: "object",
							properties: { kind: { type: "string", const: "b" } },
							required: ["kind"],
						},
					],
				},
				{ type: "null" },
			],
		});
		expect(out.type).toEqual(["object", "null"]);
		const props = out.properties as Record<string, Record<string, unknown>>;
		expect(props.kind).toEqual({ type: "string", enum: ["a", "b"] });
	});
});

describe("simplifyConstrainedSchema — pass-through (do not over-transform)", () => {
	it("leaves a scalar nullable union (anyOf string|null) untouched", () => {
		const input = {
			anyOf: [{ type: "string", minLength: 1 }, { type: "null" }],
		};
		const out = simplifyConstrainedSchema(input);
		expect(out).toEqual(input);
	});

	it("leaves a plain object schema structurally intact", () => {
		const input = {
			type: "object",
			properties: { headline: { type: "string" }, n: { type: "integer" } },
			required: ["headline"],
			additionalProperties: false,
		};
		expect(simplifyConstrainedSchema(input)).toEqual(input);
	});

	it("recurses into nested properties (object union nested inside a property)", () => {
		const out = simplifyConstrainedSchema({
			type: "object",
			properties: {
				ex: {
					anyOf: [
						{
							type: "object",
							properties: { kind: { type: "string", const: "a" } },
							required: ["kind"],
						},
						{ type: "null" },
					],
				},
			},
		});
		const ex = (out.properties as Record<string, Record<string, unknown>>).ex;
		expect(ex.type).toEqual(["object", "null"]);
	});
});
