import adapter from '@sveltejs/adapter-static';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [
		sveltekit({
			compilerOptions: {
				// Force runes mode for the project, except for libraries. Can be removed in svelte 6.
				runes: ({ filename }) =>
					filename.split(/[/\\]/).includes('node_modules') ? undefined : true
			},
			adapter: adapter()
		})
	],
	// The wasm package and sample images live outside web/.
	server: { fs: { allow: ['..'] } },
	// Vite must not pre-bundle the wasm-bindgen glue, or its `new URL(.wasm, import.meta.url)` breaks.
	optimizeDeps: { exclude: ['fx-wasm'] }
});
