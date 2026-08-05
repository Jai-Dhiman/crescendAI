// verovio ships no type declarations, and its subpath exports ("verovio/esm",
// "verovio/wasm") are the only entrypoints this app imports. Without these
// declarations every import site is an implicit-any error (TS7016), including
// the ones that immediately cast to a precise shape.
//
// These describe the module boundary only. The toolkit surface itself stays
// untyped -- see VerovioTk in src/lib/score-worker.ts.

declare module "verovio/wasm" {
	const createVerovioModule: () => Promise<unknown>;
	export default createVerovioModule;
}

declare module "verovio/esm" {
	export const VerovioToolkit: new (module: unknown) => unknown;
}
