import type { ImageMetadata } from 'astro';

/** Every figure a post can import lives under src/assets/blog/ (keys look like "/src/assets/blog/<folder>/<file>.png"). */
const images = import.meta.glob<{ default: ImageMetadata }>('/src/assets/blog/**/*.{png,jpg,jpeg,webp}', { eager: true });

/**
 * The first image a post imports (`import name from '../../../assets/blog/.../file.png'`), or undefined when it has none.
 * Posts without a figure get no thumbnail: the index never shows a placeholder.
 */
export function firstPostImage(body: string | undefined): ImageMetadata | undefined {
  const m = body?.match(/^import\s+\w+\s+from\s+['"][./]*assets\/(blog\/[^'"]+\.(?:png|jpe?g|webp))['"]/m);
  return m ? images[`/src/assets/${m[1]}`]?.default : undefined;
}
