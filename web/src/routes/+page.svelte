<script lang="ts">
	import { onMount } from 'svelte';
	import sample from '../../../assets/grumpy-cat.jpg';
	import { createRenderer, decodeImage, defaults, type EffectInfo, type ParamDesc, type Renderer } from '#lib/fx.js';

	let canvas: HTMLCanvasElement;
	let renderer = $state.raw<Renderer | null>(null);
	let effects = $state.raw<EffectInfo[]>([]);
	let effectKey = $state('identity');
	let values = $state<Record<string, Record<string, number>>>({});
	let imageVersion = $state(0);
	let comparing = $state(false);
	let dragging = $state(false);
	let error = $state('');
	let renderMs = $state(0);

	const current = $derived(effects.find((e) => e.key === effectKey));

	onMount(() => {
		(async () => {
			try {
				const r = await createRenderer(canvas);
				effects = r.effects;
				values = Object.fromEntries(effects.map((e) => [e.key, defaults(e)]));
				effectKey = effects.some((e) => e.key === 'kuwahara') ? 'kuwahara' : 'identity';
				renderer = r.renderer;
				await loadImage(await (await fetch(sample)).blob());
			} catch (e) {
				error = `${e}`;
			}
		})();
		return () => renderer?.free();
	});

	async function loadImage(file: Blob) {
		if (!renderer) return;
		try {
			const img = await decodeImage(file);
			renderer.setImage(img.rgba, img.width, img.height);
			imageVersion++;
			error = '';
		} catch (e) {
			error = `${e}`;
		}
	}

	// Re-render (at most once per frame) whenever the effect, a parameter, or the image changes.
	let frame = 0;
	$effect(() => {
		const key = comparing ? 'identity' : effectKey;
		const params = JSON.stringify(values[effectKey] ?? {});
		if (!renderer || imageVersion === 0) return;
		cancelAnimationFrame(frame);
		frame = requestAnimationFrame(() => {
			try {
				const start = performance.now();
				renderer?.render(key, params);
				renderMs = performance.now() - start;
			} catch (e) {
				error = `${e}`;
			}
		});
	});

	function download() {
		if (!renderer) return;
		// A WebGPU canvas is only readable in the task that rendered it, so render right before toBlob.
		renderer.render(effectKey, JSON.stringify(values[effectKey] ?? {}));
		canvas.toBlob((blob) => {
			if (!blob) return;
			const a = document.createElement('a');
			a.href = URL.createObjectURL(blob);
			a.download = `${effectKey}.png`;
			a.click();
			URL.revokeObjectURL(a.href);
		}, 'image/png');
	}

	// Parameters that only take 0 or 1 are on/off switches.
	const isToggle = (p: ParamDesc) => p.min === 0 && p.max === 1 && p.step === 1;

	function onDrop(e: DragEvent) {
		e.preventDefault();
		dragging = false;
		const file = e.dataTransfer?.files[0];
		if (file) loadImage(file);
	}

	function onKey(e: KeyboardEvent, down: boolean) {
		if (e.code !== 'Space' || e.target instanceof HTMLInputElement) return;
		e.preventDefault();
		comparing = down;
	}
</script>

<svelte:window
	onkeydown={(e) => onKey(e, true)}
	onkeyup={(e) => onKey(e, false)}
	ondragover={(e) => {
		e.preventDefault();
		dragging = true;
	}}
	ondragleave={() => (dragging = false)}
	ondrop={onDrop}
/>

<svelte:head><title>Brushwork</title></svelte:head>

<div class={['app', { dragging }]}>
	<aside>
		<header>
			<h1>Brushwork</h1>
			<p>Painterly image styles, rendered on your GPU</p>
		</header>

		<label class="field">
			<span>Style</span>
			<select bind:value={effectKey} disabled={!renderer}>
				{#each effects as e (e.key)}
					<option value={e.key}>{e.label}</option>
				{/each}
			</select>
		</label>

		{#if current}
			<p class="description">{current.description}</p>
		{/if}

		{#if current && values[current.key]}
			{#each current.params as p (p.key)}
				{#if isToggle(p)}
					<label class="toggle">
						{p.label}
						<input
							type="checkbox"
							role="switch"
							checked={values[current.key][p.key] === p.max}
							onchange={(e) => (values[current.key][p.key] = e.currentTarget.checked ? p.max : p.min)}
						/>
					</label>
				{:else}
					<label class="field">
						<span>{p.label} <output>{values[current.key][p.key]}</output></span>
						<input type="range" min={p.min} max={p.max} step={p.step} bind:value={values[current.key][p.key]} />
					</label>
				{/if}
			{/each}
			{#if current.params.length}
				<button onclick={() => (values[current.key] = defaults(current))}>Reset</button>
			{/if}
		{/if}

		<label class="file">
			Open image…
			<input type="file" accept="image/*" onchange={(e) => e.currentTarget.files?.[0] && loadImage(e.currentTarget.files[0])} />
		</label>
		<button onclick={download} disabled={!renderer}>Download PNG</button>
		<button
			onpointerdown={() => (comparing = true)}
			onpointerup={() => (comparing = false)}
			onpointerleave={() => (comparing = false)}
			disabled={!renderer}>Hold to compare (Space)</button
		>

		<p class="status">
			{#if error}<span class="error">{error}</span>{:else if renderer}submit {renderMs.toFixed(1)} ms (CPU){:else}Starting WebGPU…{/if}
		</p>
	</aside>

	<main>
		<canvas bind:this={canvas}></canvas>
		<p class="hint">Drop an image anywhere</p>
	</main>
</div>

<style>
	:global(body) {
		margin: 0;
		background: #16171a;
		color: #d8d8dc;
		font: 14px/1.4 system-ui, sans-serif;
	}
	.app {
		display: grid;
		grid-template-columns: 260px 1fr;
		height: 100vh;
	}
	.app.dragging main {
		outline: 2px dashed #8ab4f8;
		outline-offset: -8px;
	}
	aside {
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
		padding: 1rem;
		background: #1e1f23;
		overflow-y: auto;
	}
	header p,
	.description {
		margin: 0;
		color: #9a9aa2;
		font-size: 13px;
	}
	h1 {
		margin: 0;
		font-size: 1.2rem;
	}
	.field {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}
	.field span {
		display: flex;
		justify-content: space-between;
	}
	.toggle {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 0.5rem;
		cursor: pointer;
	}
	.toggle input {
		appearance: none;
		position: relative;
		flex: none;
		width: 2.25rem;
		height: 1.25rem;
		margin: 0;
		border-radius: 999px;
		background: #3a3b42;
		cursor: pointer;
		transition: background 0.15s;
	}
	.toggle input::before {
		content: '';
		position: absolute;
		top: 0.125rem;
		left: 0.125rem;
		width: 1rem;
		height: 1rem;
		border-radius: 50%;
		background: #d8d8dc;
		transition: transform 0.15s;
	}
	.toggle input:checked {
		background: #3b82f6;
	}
	.toggle input:checked::before {
		transform: translateX(1rem);
	}
	.toggle input:focus-visible {
		outline: 2px solid #8ab4f8;
		outline-offset: 2px;
	}
	output {
		font-variant-numeric: tabular-nums;
		color: #9a9aa2;
	}
	select,
	button,
	.file {
		padding: 0.4rem 0.6rem;
		background: #2a2b30;
		color: inherit;
		border: 1px solid #3a3b42;
		border-radius: 4px;
		font: inherit;
		cursor: pointer;
		text-align: center;
	}
	.file input {
		display: none;
	}
	button:disabled {
		opacity: 0.5;
		cursor: default;
	}
	.status {
		margin-top: auto;
		color: #9a9aa2;
		font-variant-numeric: tabular-nums;
	}
	.error {
		color: #f28b82;
		white-space: pre-wrap;
		word-break: break-word;
	}
	main {
		display: grid;
		place-items: center;
		min-width: 0;
		padding: 1rem;
		position: relative;
	}
	canvas {
		max-width: 100%;
		max-height: calc(100vh - 2rem);
	}
	.hint {
		position: absolute;
		bottom: 0.5rem;
		margin: 0;
		color: #5a5a62;
		font-size: 12px;
	}
	@media (max-width: 700px) {
		.app {
			grid-template-columns: 1fr;
			grid-template-rows: auto 1fr;
			height: auto;
		}
	}
</style>
