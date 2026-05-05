# Билет 10

## Q1. Quantization

**Квантизация** — это понижение точности представления весов и/или активаций нейросети (с FP32/FP16 до INT8 / INT4 / FP8 и т. д.) ради экономии памяти и ускорения инференса. Для LLM это критическая техника: модель на 70B параметров в FP16 занимает ~140 GB памяти, в INT4 — около 35 GB, что позволяет вместить её в одну GPU.

### Зачем квантовать

1. **Память**: размер модели сокращается в 2× (INT8) – 4× (INT4) – 8× (INT2) раз;
2. **Скорость**: INT8/INT4-операции быстрее на современных GPU/CPU (TensorCores с поддержкой FP8/INT4, Apple Neural Engine);
3. **Энергопотребление** и latency на edge-устройствах;
4. **Размер KV-cache** — отдельно квантуют и его (чувствительная для длинных контекстов вещь).

### Базовая идея — линейное отображение

Чтобы из FP-числа $x$ получить целое $q$ (n-битное):

$$q = \mathrm{round}\!\left(\frac{x}{s}\right) + z, \quad \hat{x} = s \cdot (q - z),$$

где:
- $s$ (scale) — шаг квантования (positive float);
- $z$ (zero-point) — целочисленный сдвиг.

**Symmetric** quantization: $z = 0$, диапазон симметричен относительно нуля. **Asymmetric**: $z \ne 0$, удобно для активаций после ReLU/GELU (не симметричных).

Параметры $s, z$ выбираются по статистике диапазона тензора (min/max, percentile) или через оптимизацию.

### Гранулярность

- **Per-tensor** — один $(s, z)$ на весь тензор: грубо, плохо для outlier'ов;
- **Per-channel / per-row** — свои $(s, z)$ на каждый выходной канал/строку матрицы: стандарт для весов;
- **Per-group / per-block** — на блок размера 32/64/128 элементов: лучший компромисс, использует GPTQ / AWQ;
- **Per-token** — на каждый токен (для активаций, SmoothQuant).

### Что квантуется

1. **Только веса (Weight-Only Quantization)** — самое популярное для LLM-инференса. Веса в INT4/INT8, активации в FP16. Распаковка происходит «на лету» при умножении. Это GPTQ, AWQ, GGUF (`Q4_K_M`, `Q5_K_M`, `Q8_0` и т. д.).
2. **Веса + активации (W8A8 / W8A16)** — обе стороны умножения квантованы. Сложнее из-за outlier'ов в активациях LLM (особенно после LayerNorm). SmoothQuant, LLM.int8(), FP8.
3. **KV-кэш** — отдельно квантуют до INT8/FP8/INT4 для длинных контекстов.

### Подходы по моменту квантизации

#### PTQ (Post-Training Quantization)

Квантуем уже обученную модель **без переобучения**, иногда с использованием небольшого калибровочного датасета.

- **GPTQ** (Frantar et al., 2022) — слой за слоем оптимизирует квантование по min reconstruction error через приближённое решение задачи $\min \|WX - \hat W X\|^2$ с использованием Hessian. Качество близко к FP16 даже при 4 битах. Стандарт для LLM.
- **AWQ** (Lin et al., 2023) — Activation-aware Weight Quantization. Защищает «важные» каналы (с большой активацией) масштабированием перед квантизацией. Лучше работает на инструкционных моделях.
- **SmoothQuant** — переносит сложность с активаций на веса: `Y = (X / s) (s W)`, чтобы и активации, и веса оказались в более удобном диапазоне.
- **LLM.int8()** (Dettmers et al., 2022) — выделяет outlier-каналы активаций в FP16, остальные в INT8.
- **GGUF / GGML quants** — popular формат для llama.cpp, набор группированных схем разной точности.

#### QAT (Quantization-Aware Training)

Квантование внедряется в граф **во время обучения** через fake-quant узлы (forward квантуется, gradient — Straight-Through Estimator). Модель компенсирует ошибку квантизации. Качество выше PTQ при низких битах, но требует доступа к данным и compute. Для LLM применяется реже из-за стоимости.

#### Quant-aware fine-tuning

Гибрид: PTQ + лёгкое доучивание. Например, **QLoRA** (Dettmers et al., 2023): базовые веса в **NF4** (NormalFloat-4) + LoRA-адаптеры в FP16, LoRA дообучается. Позволяет fine-tune 65B-модели на одной 48 GB GPU.

### Влияние на качество

- **INT8 weight-only**: практически без потерь (perplexity ↑ 0.1–0.3%);
- **INT4 (GPTQ/AWQ)**: близко к FP16, чуть хуже на сложных reasoning-задачах (потеря ~1% метрик);
- **INT3 / INT2**: уже заметная просадка, но активно исследуются;
- **W8A8**: outlier-проблемы; SmoothQuant и LLM.int8() справляются.

В LLM ключевая трудность — **outliers в активациях**: несколько каналов имеют очень большие значения после LayerNorm, и их грубое квантование портит всё. Все современные методы PTQ — это в первую очередь обработка outlier'ов.

### Современные форматы

- **FP8 (E4M3, E5M2)** на H100 / B200 — half-precision-like, но в 8 бит. Используется для тренировки и инференса.
- **NF4** (NormalFloat) — нелинейная схема под нормально-распределённые веса (QLoRA).
- **MXFP4 / MX-формат** — block-floating-point с групповым экспонентой.

---

## Q2. RAG. Rewriting and reranking. Agentic RAG.

### RAG (Retrieval-Augmented Generation)

**RAG** (Lewis et al., NeurIPS 2020) — парадигма, в которой генеративная LLM дополняется внешней непараметрической памятью (non-parametric memory). Мотивация — три проблемы чисто параметрических LLM: устаревание знаний (staleness), галлюцинации (hallucination) и дорогое обновление весов. Идея: модель учит язык и рассуждение, а факты хранятся в обновляемом внешнем хранилище.

Pipeline: **User Query → Retriever → Top-k Passages → Generator → Answer**.

В оригинале ретривер — **Dense Passage Retriever (DPR)** (Karpukhin et al., EMNLP 2020): два BERT-энкодера (query и passage), обучаемые контрастивной потерей на парах (query, gold passage). Поиск по индексу ~21M пассажей Википедии через FAISS с inner-product-similarity:

$$s(q, p) = E_Q(q)^\top E_P(p).$$

Альтернатива dense — **sparse retrieval** (BM25), TF-IDF-веса с нормализацией по длине документа. На практике их часто комбинируют (hybrid).

Эмбеддинги пассажей хранятся в **vector store** (Chroma, FAISS, Pinecone, pgvector); документы режутся text splitter'ом на чанки фиксированного размера с перекрытием.

Генератор в оригинальном RAG — **BART-large**; на вход — конкатенация query и top-k passages. Два режима:
- **RAG-Sequence**: $p(y \mid x) = \sum_{z} p_\eta(z \mid x)\, p_\theta(y \mid x, z)$ — один документ на ответ;
- **RAG-Token**: на каждом шаге генерации можно «смотреть» на разные документы, $p(y_t \mid x, y_{1:t-1}) = \sum_z p_\eta(z \mid x) p_\theta(y_t \mid x, z, y_{1:t-1})$.

**Преимущества**: (1) актуальность — обновили векторную базу, перетренировка не нужна; (2) groundedness и цитируемость — ответ опирается на конкретные пассажи; (3) экономия — небольшая модель + большая база ≈ гигантская параметрическая LLM.

### Rewriting and Reranking (Advanced RAG)

**Naive RAG** часто проваливается: запрос лексически не совпадает с документами, top-k содержит шум. **Advanced RAG** добавляет два шага: **query rewriting** (до retrieval) и **post-retrieval reranking** → pipeline `Rewrite → Search → Rerank → Generate`.

#### Query rewriting / expansion

- **HyDE (Hypothetical Document Embeddings)** — модель пишет гипотетический «идеальный» ответ, эмбеддинг этого псевдо-документа используется для retrieval вместо эмбеддинга запроса. Снимает vocabulary mismatch между question и passage.
- **Multi-query / query expansion** — LLM генерирует несколько перефразировок запроса, retrieval делается по каждой, результаты сливаются через **Reciprocal Rank Fusion**:
$$\mathrm{RRF}(d) = \sum_i \frac{1}{k + r_i(d)}.$$
- **Step-back prompting** — обобщение запроса, чтобы поднять более общие документы;
- **Decomposition** — разбиение multi-hop вопроса на подзапросы.

#### Reranking (two-stage retrieval)

1. Быстрый **bi-encoder** (DPR-стиль) достаёт сотни кандидатов через ANN-поиск: эмбеддинги пассажей предвычислены, близость — один dot product;
2. Точный **cross-encoder** (MS-MARCO MiniLM, BGE-reranker, Cohere Rerank) попарно скорит каждую пару (query, passage), принимая их совместно через self-attention:
$$s(q, p) = \mathrm{CrossEnc}([q; p]).$$

Cross-encoder качественнее (видит токенное взаимодействие), но непригоден для поиска по миллионам документов в реальном времени. Top-N от bi-encoder'а пересортируется cross-encoder'ом, и в LLM попадает действительно лучший top-k.

#### Diversity через MMR (Maximal Marginal Relevance)

$$\mathrm{MMR} = \arg\max_{d_i \in R \setminus S}\Big[\lambda\, \mathrm{sim}(d_i, q) - (1-\lambda) \max_{d_j \in S} \mathrm{sim}(d_i, d_j)\Big].$$

$\lambda$ балансирует релевантность и непохожесть на уже выбранные.

### Agentic RAG

**Agentic RAG** (2024–2026) — переход от пассивных pipeline «retrieve-and-generate» к **автономным агентам**, которые планируют, итеративно ищут, самокорректируются и используют инструменты. Принципиальная разница: в naive RAG retrieval вызывается ровно один раз и всегда; в agentic — LLM сама решает, нужен ли retrieval вообще, в каком источнике искать, как переформулировать запрос, и достаточно ли уже собранных доказательств, чтобы ответить.

Архитектура — stateful loop, например в **LangGraph**:

```
Plan → Retrieve → Grade → Generate
   ↑___________________|
   (Rewrite query if Grade fails)
```

- **Plan**: LLM-планировщик решает, какие подзапросы нужны (декомпозиция multi-hop вопроса);
- **Retrieve**: агент выбирает источник — внутренний vector store, веб-поиск, SQL-база, API; **Tool RAG** — retrieval применяется не только к документам, но и к выбору инструмента;
- **Grade**: grader-агент или LLM-судья оценивает релевантность найденного и faithfulness потенциального ответа. Если оценка низкая — query переписывается, цикл повторяется;
- **Generate**: финальный ответ.

#### Конкретные методы

- **Self-RAG** (Asai et al., ICLR 2024) — модель использует специальные reflection-токены `[Retrieve]`, `[Relevant]`, `[Supported]`, `[Useful]`, чтобы решать, когда искать и оценивать собственный вывод;
- **CRAG (Corrective RAG)** — частный случай: при провале grader'а триггерится fallback на web-search или query rewrite;
- **GraphRAG** (Microsoft, 2024) — строит граф сущностей и отношений по корпусу, отвечает на global/thematic вопросы; даёт +77% MRR на сложных задачах;
- **Multi-Agent RAG** — несколько специализированных агентов (Orchestrator, Retriever, Analyst, Critic, Writer) работают параллельно;
- **Multi-Modal RAG** — эмбеддинги изображений, таблиц, сканов через CLIP / ColPali и retrieval по визуальному контенту.

#### Когда применять

Agentic RAG нужен для сложных, многошаговых задач с несколькими источниками; платой служат latency (несколько LLM-вызовов на один user query) и стоимость.
