export const TRACKS = ['foundations', 'image-processing', 'nlp', 'pattern-recognition', 'ml', 'deep-learning', 'optimization', 'information-theory'] as const;
export const LEVELS = ['beginner', 'intermediate', 'advanced'] as const;
export type Track = (typeof TRACKS)[number];
