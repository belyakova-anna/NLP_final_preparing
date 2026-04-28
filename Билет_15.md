# Билет 15

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

Свойство: $\langle R(\theta_m) x, R(\theta_n) y \rangle = \langle x, R(\theta_n - \theta_m) y \rangle$ — зависит только от $n - m$.

### Формула RoPE

Размерность головы $d_h$ разбивается на $d_h / 2$ пар каналов. Для пары $j$ и позиции $m$ задаётся частота:

$$\theta_j = 10000^{-2j/d_h}, \quad j = 0, 1, \ldots, d_h/2 - 1.$$

Вектор $\mathbf{q}_m$ (позиция $m$) поворачивается блочно:

$$\mathrm{RoPE}(\mathbf{q}_m, m)_{[2j, 2j+1]} = \begin{pmatrix} \cos(m\theta_j) & -\sin(m\theta_j) \\ \sin(m\theta_j) & \cos(m\theta_j) \end{pmatrix} \begin{pmatrix} q_{m, 2j} \\ q_{m, 2j+1} \end{pmatrix}.$$

То же для $\mathbf{k}_n$ с углом $n\theta_j$. Скалярное произведение $\langle \mathrm{RoPE}(\mathbf{q}_m, m), \mathrm{RoPE}(\mathbf{k}_n, n) \rangle$ зависит от позиций **только через разность** $m - n$.

### Свойства

1. **Относительность позиции** — attention зависит только от расстояния между токенами;
2. **Не меняет архитектуру attention** — применяется к Q и K *перед* скалярным произведением;
3. **Не вносит дополнительных параметров** (фиксированные синусы);
4. **Хорошо параллелизуется** — поэлементно;
5. **Расширение контекста** — позволяет эвристики **NTK-aware**, **YaRN**, **Position Interpolation** для extrapolation на длины >> тренировочной;
6. **Long-distance decay**: высокочастотные компоненты — для соседних токенов, низкочастотные — для дальних.

### Расширение контекста

- **Position Interpolation (PI)** — масштабируем позиции $m \to m / s$, чтобы новый длинный контекст «помещался» в обученные углы;
- **NTK-aware** / **YaRN** — динамическая корректировка $\theta_j$ для разных частот;
- **Long-RoPE** — отдельная корректировка по dimension-группам.

### Где используется

LLaMA (с самого начала), Mistral, Qwen, DeepSeek, GPT-NeoX, Falcon, Gemma. RoPE стал де-факто стандартом для positional encoding в современных decoder-only LLM.

---

## Q2. Scaling Laws of LLMs

**Scaling laws** — это степенные закономерности, описывающие, как качество модели (loss, perplexity) зависит от трёх ресурсов: **числа параметров $N$**, **размера датасета $D$** и **compute $C$**. Эти законы — основа planning'а тренировки больших моделей.

### Kaplan et al. 2020 (OpenAI)

Первая масштабная работа — Kaplan et al., «Scaling Laws for Neural Language Models». Эмпирически показано, что test loss $L$ зависит от $N$ (parameters), $D$ (dataset tokens) и $C$ (compute) по **power laws**:

$$L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad L(C) \approx \left(\frac{C_c}{C}\right)^{\alpha_C}.$$

Конкретные показатели по Kaplan'у: $\alpha_N \approx 0.076$, $\alpha_D \approx 0.095$, $\alpha_C \approx 0.050$. То есть удвоение $N$ снижает loss на ~5%, удвоение $D$ — на ~6.5%, удвоение compute — на ~3.5%.

#### Главный вывод Kaplan 2020

При фиксированном compute **больше параметров полезнее, чем больше данных**. Это привело к гонке «больше параметров за любые данные»: GPT-3 175B обучен на ~300B токенов — то есть ~1.7 токена на параметр.

### Chinchilla (Hoffmann et al., 2022, DeepMind)

Hoffmann et al., «Training Compute-Optimal Large Language Models». Авторы заявили, что Kaplan'овский анализ был неоптимальным: если compute фиксирован, **оптимально обучать модель меньше, но на гораздо большем числе токенов**. Они тренировали 400+ моделей и выявили compute-optimal закон:

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta},$$

с $\alpha \approx 0.34$, $\beta \approx 0.28$, $E \approx 1.69$. При оптимизации по compute $C \approx 6 N D$ они получили **soft rule of thumb**: на оптимуме $N$ и $D$ должны масштабироваться **примерно одинаково**, и оптимально иметь **~20 токенов на параметр**.

#### Доказательство

Авторы обучили **Chinchilla 70B на 1.4T токенов** (т. е. 20 токенов / параметр). Эта модель **превзошла Gopher 280B**, обученный на 300B токенов (1.07 ток/парам). Это перевернуло индустрию: вышла LLaMA-1, в которой 7B/13B/65B обучены на 1–1.4T токенов — следуя Chinchilla recipe.

### Современный взгляд (2023–2026)

После Chinchilla стало понятно, что **«inference-optimal»** часто отличается от «training-optimal». Если модель будет инференситься триллионы раз, имеет смысл обучить **меньшую** модель **дольше** — она дешевле в эксплуатации.

Поэтому LLaMA-3 8B обучен на **15T токенов** (1875 ток/парам — гораздо больше Chinchilla-optimal), потому что у Meta огромные inference-нагрузки и хочется маленькую быструю модель.

Также появились scaling laws для:
- **Sparse / MoE моделей** (Clark et al. 2022, DeepMind);
- **Multimodal моделей** (Aghajanyan et al.);
- **Mixed-precision и квантизации**;
- **RLHF и preference data**;
- **Inference-time compute** — масштабирование test-time через CoT/Self-Consistency/o1-style reasoning.

### Compute как основная переменная

$$C \approx 6 N D \quad \text{(FLOPs тренировки трансформера)}.$$

Множитель 6: 2 FLOPs на forward + 4 на backward на каждый параметр на каждый токен. Для GPT-3 175B на 300B токенов это $\sim 3.14 \times 10^{23}$ FLOPs. Frontier-модели 2024–2026 тратят $10^{25}$–$10^{26}$ FLOPs.

### Practical implications

1. **Знание scaling laws** позволяет предсказать loss большой модели по экспериментам с малыми. Это реальная практика: тренируют $N$ маленьких моделей, фиттят power law, экстраполируют на frontier-размер.
2. **Chinchilla-optimal vs inference-optimal trade-off**: для исследовательских моделей — Chinchilla; для продакшен-моделей — over-train на гораздо большем числе токенов.
3. **«Дата как боттлнек»**: высококачественных уникальных текстовых токенов в интернете — порядка 10–20 триллионов. Это упирает frontier-модели в потолок данных, отсюда интерес к synthetic data, multi-modal, code, reasoning-traces.

### Ограничения scaling laws

- Power laws — **эмпирические** на данном архитектурном семействе и данных; смена архитектуры (Transformer → SSM/Mamba, MoE) меняет константы;
- Не предсказывают **emergent abilities** — точечные скачки качества на конкретных бенчмарках при определённом scale (математика, multi-step reasoning, in-context learning);
- Не учитывают **alignment / RLHF** — это пост-обучение с другими scaling-кривыми;
- **Задача-специфичный loss**: на одних бенчмарках scale играет огромную роль, на других — нет.

### Эмерджентность

Wei et al. 2022 показали, что многие способности (in-context learning, CoT, multi-step arithmetic) появляются **скачкообразно** при определённом scale, не предсказываясь гладкой scaling law. Дискуссия об «эмерджентности vs sharp metric» (Schaeffer et al., 2023) показала, что часть эмерджентности — артефакт дискретных метрик, но часть остаётся реальной.

### Главные тезисы

1. Loss падает по power law с увеличением $N, D, C$;
2. Compute-optimal: $N$ и $D$ масштабировать примерно одинаково (~20 ток/парам);
3. Inference-optimal: over-train маленькую модель;
4. Compute $\approx 6 N D$;
5. Scaling laws — лучший инструмент планирования frontier-models, но не предсказывают emergent abilities.
