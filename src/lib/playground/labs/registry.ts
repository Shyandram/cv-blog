// Static metadata for the server (page cards, coverage checks). The browser loads each lab lazily through loaders.ts.
import type { LabDef } from './types.ts';
import { lab as frequencyFilter } from './frequency-filter.ts';
import { lab as segmentationThreshold } from './segmentation-threshold.ts';
import { lab as lowlightPhotons } from './lowlight-photons.ts';

import { lab as transformCoding } from './transform-coding.ts';
import { lab as inverseRegularization } from './inverse-regularization.ts';
import { lab as morphologyShapes } from './morphology-shapes.ts';
import { lab as arrayDtype } from './array-dtype.ts';
import { lab as pixelSampling } from './pixel-sampling.ts';
import { lab as colorClipping } from './color-clipping.ts';
import { lab as cornerTensor } from './corner-tensor.ts';
import { lab as receptiveField } from './receptive-field.ts';
import { lab as neighborGeometry } from './neighbor-geometry.ts';
import { lab as diffusionTargets } from './diffusion-targets.ts';
import { lab as baggingTrees } from './bagging-trees.ts';
import { lab as linearTransform } from './linear-transform.ts';
import { lab as contourEnergy } from './contour-energy.ts';
import { lab as mlpTraining } from './mlp.ts';
import { lab as trackingAssociation } from './tracking-association.ts';
import { lab as imageFusion } from './image-fusion.ts';
import { lab as lutTrilinear } from './lut-trilinear.ts';
import { lab as segmentationMetrics } from './segmentation-metrics.ts';
import { lab as stereoMatching } from './stereo-matching.ts';
import { lab as adversarialBudget } from './adversarial-budget.ts';
import { lab as ctcDecoding } from './ctc-decoding.ts';
export const labs: LabDef[] = [frequencyFilter, segmentationThreshold, lowlightPhotons, transformCoding, inverseRegularization, morphologyShapes, arrayDtype, pixelSampling, colorClipping, cornerTensor, receptiveField, neighborGeometry, diffusionTargets, baggingTrees, linearTransform, contourEnergy, mlpTraining, trackingAssociation, imageFusion, lutTrilinear, segmentationMetrics, stereoMatching, adversarialBudget, ctcDecoding];
