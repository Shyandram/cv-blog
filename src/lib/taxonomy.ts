export const TRACKS = ['foundations', 'image-processing', 'pattern-recognition', 'ml', 'deep-learning', 'nlp'] as const;
export const LEVELS = ['beginner', 'intermediate', 'advanced'] as const;
export type Track = (typeof TRACKS)[number];
