---
title: "Introducing Pixels → Perception"
description: "A bilingual place to study image processing, machine learning, and the ideas behind them through explanations, code, and interactive experiments."
tags: [meta, learning, computer-vision]
published: 2026-10-04
---

An image begins as an array of numbers. Understanding what those numbers represent leads to questions about signals, patterns, learning, and uncertainty. How can a small operation change the appearance of an image? How does a model learn to recognize something? What should we check before trusting its output?

I’m introducing [Pixels → Perception](/en/), a tutorial website in English and Traditional Chinese that brings these questions into a connected learning path. It covers image processing, pattern recognition, machine learning, deep learning, computer vision, natural language processing, optimization, and information theory. The aim is to help readers move between an intuitive explanation, the mathematics behind it, and an experiment they can inspect for themselves.

There is a particular gap this site tries to address: the distance between recognizing an equation and understanding what happens when you use it. A definition may be clear on the page, yet its assumptions can become much easier to see when a parameter changes, a plot moves, or an example fails. The tutorials are organized to make those connections explicit.

The [learning section](/en/learn/) has eight tracks:

- **Foundations** introduces the site, suggested learning paths, and basic ways to represent and work with images.
- **Image Processing** develops the tools for treating images as signals, from intensity and spatial operations to restoration, compression, and segmentation.
- **Natural Language Processing** explores how to represent language and build systems that work with text, connecting linguistic ideas with learning methods.
- **Pattern Recognition** focuses on features, statistical decisions, classification, and the uncertainty involved in recognizing patterns.
- **Machine Learning** covers mathematical foundations, neural networks, learned representations, generative models, and evaluation.
- **Deep Learning for Computer Vision** follows those ideas into tasks such as image classification, object detection, restoration, enhancement, and tracking.
- **Optimization** examines how to formulate a problem, express its constraints, and choose a way to solve it.
- **Information Theory** studies uncertainty, compression, and communication, providing another perspective on ideas that appear throughout the other tracks.

These subjects share a great deal of common ground. The mathematical tools used to reason about a visual model also help explain a language model. A question about image quality can lead back to the assumptions of an imaging process. A training problem can become clearer when you examine the objective being optimized. Keeping the tracks together makes it easier to follow those connections without losing the structure of an individual subject.

The tutorials follow a consistent rhythm. They begin with learning goals and a plain-language picture of the topic, then develop definitions, equations, and examples. Code and figures give readers something concrete to examine. Later sections connect the fundamentals with research literature, discuss limitations, and end with takeaways and exercises. Prerequisite links help identify where to go when an explanation assumes something unfamiliar.

For example, the [interactive playground](/en/playground/) includes an image-filtering experiment. You can edit a small grid of weights and see the output change. That creates a useful learning loop: make a prediction, adjust the weights, inspect the result, and return to the explanation. Other experiments explore ideas from the broader curriculum. The controls run in the browser, so you can start experimenting without setting up a programming environment.

The code examples serve a different purpose: they let you inspect the operations themselves. Readers who want to go further can change an example, compare outputs, or work through an exercise. Some tutorials also link to a runnable notebook. The small examples are intended to explain mechanisms; they do not reproduce the scale or performance of a full research system.

Both language versions are intended to support the same learning experience. The Traditional Chinese pages use terminology familiar to readers in Taiwan, and the language switcher connects corresponding pages. This can be useful when you understand an idea more comfortably in one language but need to recognize its terminology in the other.

The site also has a [blog](/en/blog/) alongside the structured tutorials. The curriculum provides an ordered route through a subject; the blog provides room for introductions, notes, and discussions that sit outside that sequence. Search is available when you want to return to a specific idea without following a track from the beginning.

I also want to be clear about how the site was made. Much of its draft content and code was created with AI assistance. Human direction shapes the topics and course structure, and human review remains essential. **The content is still under review.** An explanation can be readable and still contain a mistake, and a working example can leave an important assumption unstated. Readers should check the cited sources when a detail matters and treat small demonstrations as a starting point for investigation.

If you are new to the subjects, start with the [getting-started guide](/en/learn/foundations/getting-started/) and [Images as Arrays](/en/learn/foundations/images-as-arrays/). If you already have a particular question, browse the tracks or open the playground and follow an experiment back to its related lesson. You can read in sequence, revisit a mathematical prerequisite, or spend a few minutes exploring one idea.

The invitation is simple: choose a question, make a prediction, and try an example. I hope Pixels → Perception helps turn something that looks abstract on a page into something you can explain, test, and use.

Visit [Pixels → Perception in English](/en/) or [繁體中文](/zh-tw/).
