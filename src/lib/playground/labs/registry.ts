// Static metadata for the server (page cards, coverage checks). The browser loads each lab lazily through loaders.ts.
import type { LabDef } from './types.ts';
import { lab as frequencyFilter } from './frequency-filter.ts';
import { lab as segmentationThreshold } from './segmentation-threshold.ts';
import { lab as lowlightPhotons } from './lowlight-photons.ts';

export const labs: LabDef[] = [frequencyFilter, segmentationThreshold, lowlightPhotons];
