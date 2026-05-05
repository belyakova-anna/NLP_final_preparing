# Билет 3

## Q1. Rotary Positional Embeddings (RoPE)

**RoPE** (Rotary Position Embedding; Su et al., 2021, статья «RoFormer») — способ инжектирования информации о позиции токена напрямую в векторы Query и Key через **поворот в комплексной плоскости**. Используется в LLaMA-1/2/3, PaLM, GPT-NeoX, Mistral, Qwen, DeepSeek и большинстве современных decoder-only LLM.

### Проблема, которую решает RoPE

Обычные positional embeddings — обучаемые (BERT) или синусоидальные (Vaswani'17) — добавляются к token embedding на входе. Минусы:
- Абсолютная позиция, плохая экстраполяция за пределы тренировочной длины;
- Position и content смешаны — attention видит «позицию + содержание», а хочется чистого относительного смещения;
- Для длинных контекстов нужна другая схема.

RoPE даёт способ закодировать **относительную** позицию, не меняя skoring-формулу attention.

### Идея

Вместо суммирования эмбеддинга позиции с токеном RoPE **поворачивает** Q и K в каждой паре каналов на угол, пропорциональный позиции. Скалярное произведение двух повёрнутых векторов зависит **только от разности позиций**, что и означает относительное кодирование.

Для двумерного случая (одной пары каналов) поворот на угол $\theta$:

$$R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}.$$

Обозначим вектор $x = (x_1, x_2)$. Тогда $\langle R(\theta_m) x, R(\theta_n) y \rangle = \langle x, R(\theta_n - \theta_m) y \rangle$ — зависит только от $n - m$.

### Формула RoPE

Размерность головы $d_h$ разбивается на $d_h / 2$ пар каналов. Для пары $j$ и позиции $m$ задаётся частота:

$$\theta_j = 10000^{-2j/d_h}, \quad j = 0, 1, \ldots, d_h/2 - 1.$$

Вектор $\mathbf{q}_m$ (позиция $m$) поворачивается блочно:

$$\mathrm{RoPE}(\mathbf{q}&#95;m, m)&#95;{[2j, 2j+1]} = \begin{pmatrix} \cos(m\theta&#95;j) & -\sin(m\theta&#95;j) \\\\ \sin(m\theta&#95;j) & \cos(m\theta&#95;j) \end{pmatrix} \begin{pmatrix} q&#95;{m, 2j} \\\\ q&#95;{m, 2j+1} \end{pmatrix}.$$

То же для $\mathbf{k}&#95;n$ с углом $n\theta&#95;j$. Тогда:

$$\langle \mathrm{RoPE}(\mathbf{q}&#95;m, m), \mathrm{RoPE}(\mathbf{k}&#95;n, n) \rangle = \mathrm{Re}\bigl[\sum&#95;j (q&#95;{m,2j} + i q&#95;{m,2j+1})(k&#95;{n,2j} - i k&#95;{n,2j+1}) e^{i(m-n)\theta&#95;j}\bigr],$$

где зависимость от позиций входит **только через разность** $m - n$.

### Свойства

1. **Относительность позиции** — attention зависит только от расстояния между токенами;
2. **Не меняет архитектуру attention** — применяется к Q и K *перед* скалярным произведением;
3. **Не вносит дополнительных параметров** (фиксированные синусы);
4. **Хорошо параллелизуется** — поэлементно;
5. **Расширение контекста** — позволяет эвристики типа **NTK-aware scaling**, **YaRN**, **Position Interpolation** для extrapolation на длины >> тренировочной;
6. **Длинные затухания**: высокочастотные компоненты ($\theta_j$ малый) — для соседних токенов, низкочастотные — для дальних.

### Расширение контекста

- **Position Interpolation (PI)** — масштабируем позиции $m \to m / s$, чтобы новый длинный контекст «помещался» в обученные углы;
- **NTK-aware** / **YaRN** — динамическая корректировка $\theta_j$ для разных частот;
- **Long-RoPE** — отдельная корректировка по dimension-группам.

### Где используется

LLaMA (с самого начала), Mistral, Qwen, DeepSeek, GPT-NeoX, Falcon, Gemma. Можно сказать, RoPE стал де-факто стандартом для positional encoding в современных decoder-only LLM.

---

## Q2. GPT and GPT-like language models

**GPT** (Generative Pre-trained Transformer; Radford et al., OpenAI, 2018) — семейство **decoder-only** авторегрессивных трансформеров, обученных на задаче **next-token prediction**. Стандартная парадигма: huge-scale pretraining → SFT → RLHF/DPO для alignment.

### Архитектура

- Только **decoder-стек** трансформера, **causal mask** в self-attention (токен видит только себя и левых соседей):
  $$\mathrm{mask}_{ij} = \begin{cases} 0, & j \le i \\ -\infty, & j > i \end{cases}.$$
- Каждый блок: Self-Attention → FFN, обёрнутые pre-LN или post-LN, residual.
- Token embeddings + positional encoding.
- Финальная LayerNorm, проекция в словарь, softmax.

### Тренировочная цель

$$\mathcal{L}&#95;{LM}(\theta) = -\sum&#95;{t=1}^{T} \log P&#95;\theta(x_t \mid x&#95;{1:t-1}).$$

Один лосс, никакого NSP, никаких masked-токенов. На gigantic-корпусах (Common Crawl, books, code) это даёт сильнейший сигнал.

### Эволюция GPT

- **GPT-1 (2018)**: 117M параметров, 12 слоёв, BookCorpus. Идея: pretrain → fine-tune. Бенчмарк GLUE.
- **GPT-2 (2019)**: 1.5B, 48 слоёв, 40 GB WebText. Демонстрация **zero-shot** способности — без fine-tuning отвечает на задачи через text prompts. Изначально не выложили из соображений безопасности.
- **GPT-3 (2020, Brown et al.)**: 175B параметров, 96 слоёв, dimension 12288, 96 голов, 300B токенов тренировки. Появление **few-shot in-context learning** — модель решает задачу по нескольким примерам в промпте без апдейта весов. **Эмерджентность** способностей при scale.
- **InstructGPT (2022)**: 1.3B / 6B / 175B + SFT + RLHF. Показало, что aligned 1.3B модель может превзойти base 175B.
- **GPT-3.5 / ChatGPT (2022)**: расширенная RLHF, чат-формат, conversational alignment.
- **GPT-4 (2023)**: предположительно MoE, мультимодальность (vision), большой контекст. Точные параметры не раскрыты.
- **GPT-4o / o1 / o3 / GPT-5**: добавление reasoning через скрытый CoT, расширение модальностей, инференс-time compute scaling.

### GPT-like модели (семейство)

- **GPT-Neo / GPT-J / GPT-NeoX** (EleutherAI) — открытые реплики GPT-3 разного размера;
- **OPT** (Meta, 2022) — открытая GPT-3-копия, 175B;
- **BLOOM** (BigScience, 2022) — 176B, мультиязычная;
- **LLaMA / LLaMA-2 / LLaMA-3 / LLaMA-4** (Meta) — открытая серия decoder-only с современными оптимизациями (RMSNorm, SwiGLU, RoPE, GQA);
- **Mistral / Mixtral / Mistral Large** — Mistral.AI, в т. ч. MoE;
- **Falcon, Qwen, DeepSeek, Yi, Gemma** — другие сильные семьи;
- **Claude** (Anthropic), **Gemini** (Google) — закрытые, но архитектурно decoder-only LLM той же парадигмы.

### Особенности современных GPT-like LLM

1. **Decoder-only** + causal LM-цель;
2. **Современные оптимизации**: RoPE, RMSNorm, SwiGLU/GeGLU FFN, GQA, FlashAttention, KV-cache;
3. **Большой контекст** (8K → 32K → 128K → 1M+), часто с YaRN/NTK-scaling;
4. **Tokenizer**: BPE / Tiktoken / SentencePiece;
5. **Инструмент-вызовы (function calling)**, JSON-mode, structured output;
6. **Alignment**: SFT → RLHF/DPO/GRPO → red-teaming;
7. **Mixture-of-Experts** во многих современных (Mixtral, DeepSeek-V3, GPT-4).

### Сильные и слабые стороны

**+** Один универсальный pretraining объект → масштабируется. **+** Авторегрессивная генерация — естественная для текста. **+** In-context learning, instruction-following, reasoning.

**−** Видит только левый контекст (хуже для NLU-encoding-задач, чем BERT). **−** Параметрические знания устаревают, hallucinations. **−** Дорогой инференс с длинным контекстом (квадратичность attention, размер KV-кэша).
