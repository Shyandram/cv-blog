# Brief: translate a new English tutorial into Traditional Chinese (Taiwan)

Create ONLY the zh-tw twin file named in your assignment; edit nothing else; do not commit; no git stash/reset/checkout/restore/
clean. Keep your scratch files in your own subfolder of
/tmp/claude-0/-home-user-cv-blog/46b65cc4-04fc-53dc-9c17-ea7809aadfc9/scratchpad/ (several agents share the scratchpad).

Read first: /home/user/cv-blog/AGENTS.md (sections 2-5; especially section 4 Language: natural Traditional Chinese with Taiwanese
terminology — 影像、像素、資料、資料集、網路、模型、訓練、演算法、參數、最佳化、損失函數、卷積、濾波器、頻域、直方圖、雜訊、
解析度、取樣、量化、內插、影片; no Simplified characters or mainland terms; keep model/paper/dataset names in English; give the
English term in parentheses at first use). Use the finished twin
/home/user/cv-blog/src/content/tutorials/zh-tw/deep-learning/dl-04a-super-resolution.mdx as the model for conventions: title
format "DLCV NNX・標題" with the full-width middle dot; section names 你將學到, 先看全貌, 現代觀點, 重點整理, 練習, 補充資料,
論文導讀, 參考文獻; `<Callout type="tip" title="用最簡單的話說">`; `<details>` with `<summary>` on its own line; pointer lines
`*選讀：[標題](#slug)（見「補充資料」頁）。…*` and `（見「論文導讀」頁）`; paper-guide labels 要解決的問題／核心想法／它呈現了什麼／
閱讀時要檢查什麼.

Rules: same structure, sections and order; same figures (same imports and PNGs; Chinese alt text and captions "圖 n — …"); code
blocks and their printed outputs copied VERBATIM (comments stay English); same display and inline math; the same reference list
copied verbatim (authors/titles untranslated) with the same numbers and the same citation counts. Internal links: /en/ → /zh-tw/,
and recompute every #anchor from the zh-tw target page's headings with github-slugger
(`import GithubSlugger from '/home/user/cv-blog/node_modules/github-slugger/index.js'`; one slugger per page, headings in order);
anchors to items in another page's Supplementary notes or Paper guides use that page's …/notes/ or …/papers/ path. In-page anchors
must match the Chinese headings you write. Frontmatter: translate title and description; keep track, order, level, prereqs, tags,
published identical. Translate the meaning naturally, not word by word; keep it easy to understand quickly (the owner's top
priority).

Validate: `export PATH=$HOME/.nvm/versions/node/v22.23.3/bin:$PATH; node scripts/check-mdx.mjs <zh-tw file>`; every #anchor
(in-page and cross-page) resolves; grep for Simplified characters / mainland terms (数据 图像 网络 视频 信息 默认 质量 采样 优化
计算机 模块 文本 博客 软件 硬件 程序 内存 等); compare counts with the English file (## and ### sections, imports, Figure,
code blocks, display math, inline math, reference entries, citations, details). Report: path, the counts table, terminology choices
worth reviewing, anything uncertain.
