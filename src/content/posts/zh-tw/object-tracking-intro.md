---
title: "物件追蹤Object Tracking 簡介"
description: "物件追蹤是什麼？介紹 Re-ID、單物件追蹤（VOT）與多物件追蹤（MOT）的差別、挑戰與代表模型。"
tags: [computer-vision, object-tracking]
published: 2023-11-28
updated: 2026-10-04
original: https://medium.com/@shyangen104/%E7%89%A9%E4%BB%B6%E8%BF%BD%E8%B9%A4object-tracking-%E7%B0%A1%E4%BB%8B-f52d3af122
---

> **2026 年 10 月更新：** 本文寫於 2023 年 11 月，這次補上 2024–2026 年的發展：SAM 2／SAM 3 等影片分割模型直接拿來追蹤、MOT 的新 baseline 與端到端模型、HOTA 指標，以及 DukeMTMC 已下架。原文保留，新增處皆有標示。

物件跟蹤／追蹤是什麼？

## 前言

電腦視覺（Computer Vision）在現今已經是非常重要的專業領域，無論是在軍事、無人駕駛、日常生活中都有所連繫。目前科技的進展在AI和深度學習上已經達到前所未有的強大能力，…

物件分類 -> 物件偵測 -> ？

### 物件偵測（Object Detection）

> 是模仿人類視覺對畫面中目標的判斷能力

## 追蹤(Tracking)在做什麼？

![Figure 1](./images/object-tracking-intro/01.png)

> 模仿人類視覺對運動物體的感知能力衍伸出的專業領域，對攝影機畫面的圖像序列進行解析。

## 追蹤的分類

![Figure 2](./images/object-tracking-intro/02.png)

- 依照作法可分為兩個類別：Re-ID、Object Tracking
- 依照目標可分為兩個類別：Visual Object Tracking (VOT, 又稱SOT)、Multiple Object Tracking (MOT)
- 終極目標：MTMOT （多目標多物件追蹤）

> **更正：** 原文提到終極目標是「MTMOT（多目標多物件追蹤）」，更精確的說法是 MTMCT（Multi-Target Multi-Camera Tracking，多目標跨攝影機追蹤），重點在跨攝影機維持同一目標的身分。

> 大多以**行人的移動**作為檢測目標

## Re-ID

重識別、跨鏡頭追蹤

根據目標的表觀特徵、姿態動作追蹤該目標

挑戰：數據因素、環境因素

應用領域：監視系統-天網系統

代表模型：CTL Model (ResNet50, 256x128)

資料庫: DukeMTMC、Market-1501

![DukeMTMC](./images/object-tracking-intro/03.png)

*DukeMTMC*

> **更新（2026）：** DukeMTMC 因隱私與倫理爭議已於 2019 年被作者撤下，不再提供下載（見 [Peng et al., 2021](https://arxiv.org/abs/2108.02922)）。

## VOT

![SiamFC](./images/object-tracking-intro/04.png)

*SiamFC*

- 又稱Single Object Tracking (SOT)
- 單純對單一畫面的同一目標追蹤
- 挑戰：複雜的外觀、移動模式、環境變化（光照、遮擋等）
- 代表模型：SiamFC、SiamRPN
- 資料庫：VOT Challenge

> **更新（2026）：** 2022 年後主流已從 Siamese 轉為 Transformer，如 [OSTrack](https://arxiv.org/abs/2203.11991)（2022）、[SeqTrack](https://arxiv.org/abs/2304.14394)（2023）。

### 用影片分割模型做追蹤（2026 補充）

Meta 的 [SAM 2](https://arxiv.org/abs/2408.00714)（2024）只要在一幀點選物體，就能在整段影片中持續分割它，等於一個輸出遮罩的 SOT；[SAMURAI](https://arxiv.org/abs/2411.11922)（2024）加上運動模型，不需訓練即可 zero-shot 追蹤。[SAM 3](https://arxiv.org/abs/2511.16719)（2025）可用名詞片語當提示，追蹤所有符合的物件並給予各自 ID，SOT 與 MOT 的界線因此更模糊。

## Multi-Object Tracking

對單一畫格中的複數目標進行追蹤

挑戰：頻繁遮擋、軌跡初始及結束、相似外觀、目標相互影響

資料庫：MOT Challenge、AI CITY CHALLENGE

代表模型：Deep SORT、Tracktor、FairMOT

> **更正：** 原文提到「對單一畫格中的複數目標進行追蹤」，更精確的說法是在連續影格中同時追蹤多個目標，並為每個目標維持一致的 ID；單一畫格只能做偵測。

> **更新（2026）：** 常用 baseline 已換成 [ByteTrack](https://arxiv.org/abs/2110.06864)、[OC-SORT](https://arxiv.org/abs/2203.14360)、[BoT-SORT](https://arxiv.org/abs/2206.14651)；端到端路線有 [MOTR](https://arxiv.org/abs/2105.03247)、[MOTRv2](https://arxiv.org/abs/2211.09791) 與 [MOTIP](https://arxiv.org/abs/2403.16848)（CVPR 2025）。評估上 MOTA 偏重偵測、IDF1 偏重關聯，[HOTA](https://arxiv.org/abs/2009.07736) 兼顧兩者，已成主要指標之一。

## MTMOT

終極目標

> **更新（2026）：** [第 9 屆 AI City Challenge](https://arxiv.org/abs/2508.13564)（2025）Track 1 已是倉儲場景的多類別 3D 多攝影機追蹤（行人、機器人、堆高機）。

## Reference

- [云从科技资深算法研究员：详解跨镜追踪(ReID)技术实现及难点](https://zhuanlan.zhihu.com/p/38603624)
- [基于深度学习的视觉目标跟踪方法介绍](https://zhuanlan.zhihu.com/p/84183000)
- [SOT入门之Siam系列（一）：SiamFC](https://zhuanlan.zhihu.com/p/107428605)
- [mot数据集\_谈谈ReID与MOT的关系-CSDN博客](https://blog.csdn.net/weixin_42452580/article/details/112162074)

## 延伸閱讀（2024–2026）

- [SAMURAI 官方程式碼](https://github.com/yangchris11/samurai)
- 點追蹤：[TAPIR](https://arxiv.org/abs/2306.08637)、[CoTracker](https://arxiv.org/abs/2307.07635)、[CoTracker3](https://arxiv.org/abs/2410.11831)（2024）
- Re-ID：[CLIP-ReID](https://arxiv.org/abs/2211.13977)（AAAI 2023）
- MOT 資料集：[DanceTrack](https://arxiv.org/abs/2111.14690)、[SportsMOT](https://arxiv.org/abs/2304.05170)、[BDD100K](https://github.com/bdd100k/bdd100k)
- 開放詞彙 MOT：[OVTrack](https://arxiv.org/abs/2304.08408)（CVPR 2023）
