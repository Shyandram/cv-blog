import { TRACKS, type Track } from '../taxonomy';
import { visionExperiments } from './vision';
import { languageExperiments } from './language';
import { mathExperiments } from './math';
import { chapterFoundationsExperiments } from './chapters-foundations';
import { chapterOptimizationExperiments } from './chapters-optimization';
import { chapterInformationExperiments } from './chapters-information';
import { chapterRecognitionExperiments } from './chapters-recognition';
import { chapterVisionExperiments } from './chapters-vision';
import { chapterLanguageExperiments } from './chapters-language';
export const experiments = [...visionExperiments, ...languageExperiments, ...mathExperiments, ...chapterFoundationsExperiments, ...chapterOptimizationExperiments, ...chapterInformationExperiments, ...chapterRecognitionExperiments, ...chapterVisionExperiments, ...chapterLanguageExperiments]
    .sort((a, b) => TRACKS.indexOf(a.track) - TRACKS.indexOf(b.track));
export interface Source {
    title: string;
    url: string;
}
const source = (title: string, url: string): Source => ({ title, url });
export const sources: Record<Track, Source[]> = {
    foundations: [source('NumPy contributors, Data types and overflow, current documentation', 'https://numpy.org/doc/stable/user/basics.types.html'), source('Gilbert Strang, Linear Algebra, MIT OpenCourseWare, 2010', 'https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/'), source('Antonio Torralba, Phillip Isola & William T. Freeman, Foundations of Computer Vision, MIT Press, 2024', 'https://visionbook.mit.edu/')],
    'image-processing': [source('Rafael C. Gonzalez & Richard E. Woods, Digital Image Processing, 4th ed., Pearson, 2018', 'https://www.imageprocessingplace.com/DIP-4E/dip4e_main_page.htm'), source('Antonio Torralba, Phillip Isola & William T. Freeman, Foundations of Computer Vision, MIT Press, 2024', 'https://visionbook.mit.edu/')],
    nlp: [source('Daniel Jurafsky & James H. Martin, Speech and Language Processing, 3rd ed. draft, 2026', 'https://web.stanford.edu/~jurafsky/slp3/'), source('Tomas Mikolov et al., Efficient Estimation of Word Representations in Vector Space, ICLR Workshop, 2013', 'https://arxiv.org/abs/1301.3781')],
    'pattern-recognition': [source('Sergios Theodoridis & Konstantinos Koutroumbas, Pattern Recognition, 4th ed., Academic Press, 2009', 'https://shop.elsevier.com/books/pattern-recognition/koutroumbas/978-1-59749-272-0'), source('Christopher M. Bishop, Pattern Recognition and Machine Learning, Springer, 2006', 'https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/')],
    ml: [source('Ian Goodfellow, Yoshua Bengio & Aaron Courville, Deep Learning, MIT Press, 2016', 'https://www.deeplearningbook.org/'), source('Ashish Vaswani et al., Attention Is All You Need, NeurIPS, 2017', 'https://arxiv.org/abs/1706.03762'), source('Jonathan Ho, Ajay Jain & Pieter Abbeel, Denoising Diffusion Probabilistic Models, NeurIPS, 2020', 'https://arxiv.org/abs/2006.11239'), source('Tim Salimans & Jonathan Ho, Progressive Distillation for Fast Sampling of Diffusion Models, ICLR, 2022', 'https://arxiv.org/abs/2202.00512')],
    'deep-learning': [source('Torchvision contributors, Non-maximum suppression, current documentation', 'https://docs.pytorch.org/vision/stable/generated/torchvision.ops.nms.html')],
    optimization: [source('Stephen Boyd & Lieven Vandenberghe, Convex Optimization, Cambridge University Press, 2004', 'https://web.stanford.edu/~boyd/cvxbook/')],
    'information-theory': [source('Thomas M. Cover & Joy A. Thomas, Elements of Information Theory, 2nd ed., Wiley, 2006', 'https://onlinelibrary.wiley.com/doi/book/10.1002/047174882X')],
};
/** Specific papers supplement the track textbooks where the teaching model needs them. */
const calibrationSource = source('Chuan Guo, Geoff Pleiss, Yu Sun & Kilian Q. Weinberger, On Calibration of Modern Neural Networks, ICML, 2017', 'https://proceedings.mlr.press/v70/guo17a.html');
export const experimentSources: Record<string, Source[]> = {
    'subword-merges': [source('Rico Sennrich, Barry Haddow & Alexandra Birch, Neural Machine Translation of Rare Words with Subword Units, ACL, 2016', 'https://aclanthology.org/P16-1162/')],
    'masked-objectives': [source('Jacob Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, NAACL-HLT, 2019', 'https://aclanthology.org/N19-1423/')],
    'contrastive-alignment': [source('Ting Chen, Simon Kornblith, Mohammad Norouzi & Geoffrey Hinton, A Simple Framework for Contrastive Learning of Visual Representations, ICML, 2020', 'https://proceedings.mlr.press/v119/chen20j.html'), source('Alec Radford et al., Learning Transferable Visual Models From Natural Language Supervision, ICML, 2021', 'https://proceedings.mlr.press/v139/radford21a.html')],
    'selective-memory': [source('Albert Gu & Tri Dao, Mamba: Linear-Time Sequence Modeling with Selective State Spaces, arXiv, 2023', 'https://arxiv.org/abs/2312.00752'), sources.ml[0]],
    'transport-steps': [source('Yaron Lipman et al., Flow Matching for Generative Modeling, ICLR, 2023', 'https://arxiv.org/abs/2210.02747')],
    'classification-threshold': [sources['pattern-recognition'][1]],
    'temperature-calibration': [calibrationSource],
    'restoration-ambiguity': [source('Yochai Blau & Tomer Michaeli, The Perception-Distortion Tradeoff, CVPR, 2018', 'https://arxiv.org/abs/1711.06077')],
    'world-model-rollout': [source('David Ha & Jürgen Schmidhuber, World Models, arXiv, 2018', 'https://arxiv.org/abs/1803.10122')],
};
