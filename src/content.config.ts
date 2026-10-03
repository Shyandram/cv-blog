import { defineCollection } from 'astro:content';
import { z } from 'astro/zod';
import { glob } from 'astro/loaders';

import { TRACKS, LEVELS } from './lib/taxonomy';

// Entry ids look like "en/image-processing/convolution"; the first segment is the language.
const tutorials = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/tutorials' }),
  schema: ({ image }) =>
    z.object({
      title: z.string(),
      description: z.string(),
      track: z.enum(TRACKS),
      order: z.number().int(),
      level: z.enum(LEVELS),
      prereqs: z.array(z.string()).default([]), // "<track>/<slug>", language-independent
      tags: z.array(z.string()).default([]),
      notebook: z.string().optional(), // repo-relative path, e.g. notebooks/image-processing/convolution.ipynb
      cover: image().optional(),
      published: z.coerce.date(),
      updated: z.coerce.date().optional(),
      draft: z.boolean().default(false),
    }),
});

const posts = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/posts' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    tags: z.array(z.string()).default([]),
    published: z.coerce.date(),
    updated: z.coerce.date().optional(),
    draft: z.boolean().default(false),
    original: z.string().url().optional(), // where the post first appeared, e.g. Medium
  }),
});

export const collections = { tutorials, posts };
