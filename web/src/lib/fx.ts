import init, { Renderer, effects } from 'fx-wasm';

export type { Renderer };

export interface ParamDesc {
	key: string;
	label: string;
	min: number;
	max: number;
	step: number;
	default: number;
}

export interface EffectInfo {
	key: string;
	label: string;
	description: string;
	params: ParamDesc[];
}

export async function createRenderer(canvas: HTMLCanvasElement) {
	await init();
	return {
		renderer: await Renderer.create(canvas),
		effects: JSON.parse(effects()) as EffectInfo[]
	};
}

/** Decodes an image file into tightly packed sRGB RGBA bytes. */
export async function decodeImage(file: Blob) {
	const bitmap = await createImageBitmap(file);
	const { width, height } = bitmap;
	const ctx = new OffscreenCanvas(width, height).getContext('2d');
	if (!ctx) throw new Error('2D canvas unavailable');
	ctx.drawImage(bitmap, 0, 0);
	bitmap.close();
	const data = ctx.getImageData(0, 0, width, height).data;
	return { rgba: new Uint8Array(data.buffer), width, height };
}

export function defaults(effect: EffectInfo): Record<string, number> {
	return Object.fromEntries(effect.params.map((p) => [p.key, p.default]));
}
