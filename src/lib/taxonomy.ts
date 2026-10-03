export const TRACKS = ['foundations', 'image-processing', 'classical-cv', 'ml', 'deep-learning'] as const;
export const LEVELS = ['beginner', 'intermediate', 'advanced'] as const;
export type Track = (typeof TRACKS)[number];
