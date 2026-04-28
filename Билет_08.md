# Билет 8

## Q1. Proximal Policy Optimization (PPO)

### Постановка задачи как RL

Файнтюнинг LLM рассматривается как token-level Markov Decision Process (MDP):
- **state $s_t$** — это промпт $x$ плюс уже сгенерированные токены $y_{<t}$;
- **action $a_t$** — следующий токен из словаря $V$ (~50K токенов);
- **policy $\pi_\theta(a_t \mid s_t)$** — сама LLM;
- **reward $R(x, y)$** — скаляр от reward-model в конце последовательности (sparse terminal reward);
- эпизод длится 128–1024 токена.

Цель: $\max_\theta \mathbb{E}[R(x, y)]$.

### Почему не vanilla policy gradient

Базовый policy gradient

$$\nabla_\theta J(\theta) = \mathbb{E}\bigl[\nabla_\theta \log \pi_\theta(a_t \mid s_t) \, A^\pi(s_t, a_t)\bigr]$$

имеет высокую дисперсию, on-policy (выкидывает старые данные), не контролирует размер шага и плохо масштабируется на огромное action-пространство.

**TRPO** (Trust Region Policy Optimization) добавляет жёсткое ограничение через KL-дивергенцию между новой и старой полиси, но требует second-order оптимизации (Hessian) — нереалистично для миллиардов параметров.

**PPO** (Schulman et al., 2017) — дешёвая аппроксимация TRPO через **clipping** probability ratio. Не требует Hessian, использует обычный SGD.

### Probability ratio и clipped surrogate objective

Определяется отношение вероятностей новой и старой полиси:

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}.$$

**Главный лосс PPO**:

$$L^{CLIP}(\theta) = \mathbb{E}_t \Bigl[\min\bigl(r_t(\theta) \, \hat{A}_t,\ \mathrm{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \, \hat{A}_t\bigr)\Bigr],$$

где $\hat{A}_t$ — оценка advantage (часто GAE — Generalized Advantage Estimation), $\varepsilon = 0.2$ — типичное значение в InstructGPT.

**Логика min**:
- Если $\hat A_t > 0$ (хорошее действие), хотим увеличить $\pi_\theta$, но clip ограничивает $r_t$ сверху $1+\varepsilon$ — нельзя стать слишком уверенным за один шаг;
- Если $\hat A_t < 0$ (плохое действие), нельзя занулить вероятность ниже $1-\varepsilon$ — это предохранитель от катастрофического «забывания».

### Полный лосс в RLHF-PPO

К $L^{CLIP}$ добавляется:
- **value loss** $L^{VF} = (V_\theta(s_t) - V_t^\text{target})^2$ (критик-сеть оценивает $V$);
- **entropy bonus** $H[\pi_\theta]$ — поощряет exploration:

$$L^\text{PPO} = L^{CLIP} - c_1 L^{VF} + c_2 H[\pi_\theta].$$

Дополнительно reward модифицируется per-token через **KL-штраф к референсной модели $\pi_\text{SFT}$**:

$$r(x, y) = r_\text{RM}(x, y) - \beta \bigl(\log \pi_\text{RL}(y \mid x) - \log \pi_\text{SFT}(y \mid x)\bigr).$$

Это критично для борьбы с reward hacking (см. билет 2).

### GAE — Generalized Advantage Estimation

Advantage оценивается через комбинацию TD-residuals:

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t),$$
$$\hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \, \delta_{t+l}.$$

Параметр $\lambda$ регулирует bias-variance trade-off ($\lambda = 0$ → одношаговый TD; $\lambda = 1$ → Monte-Carlo).

### Цикл PPO в RLHF (4 шага, повторяется ~1000 итераций при batch=512)

1. **Sample prompts**: $x \sim D_\text{prompt}$;
2. **Generate**: $y \sim \pi_\theta(\cdot \mid x)$ авторегрессивно, фиксируем $\pi_\text{old} = \pi_\theta$;
3. **Score**: $r(x, y) = r_\phi(x, y) - \beta \log[\pi_\theta / \pi_\text{SFT}]$;
4. **PPO update**: $K = 4$ mini-epochs по rollout-батчу с lr=1e-5, $\varepsilon = 0.2$, $\beta = 0.02$, batch = 512.

### Преимущества PPO

1. **Стабильность** через clipping;
2. **Простота имплементации** — first-order, no Hessian;
3. **Sample-efficient** благодаря K-epochs reuse;
4. **De facto стандарт** в RLHF (InstructGPT, ChatGPT, GPT-4, Claude).

### Недостатки

1. **Сложный пайплайн** — надо одновременно держать в памяти 4 модели (actor, critic, RM, ref);
2. **Чувствительность к гиперпараметрам** ($\varepsilon, \beta, \gamma, \lambda, $ lr);
3. **Reward hacking** при недостаточном KL-якоре;
4. **Дорогой rollout** — генерация on-policy.

Альтернативы: **DPO** (нет RM и RL), **GRPO** (нет critic), **REINFORCE с baseline**.

---

## Q2. Attention mechanism. Self-attention, Cross-attention, Multi-head attention

### Базовая идея attention

Attention — механизм взвешенной агрегации, в котором каждая позиция выходной последовательности **«смотрит»** на все позиции входа с **обучаемыми весами внимания**, зависящими от содержимого.

Формула scaled dot-product attention:

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V,$$

где $Q \in \mathbb{R}^{n_q \times d_k}$, $K \in \mathbb{R}^{n_k \times d_k}$, $V \in \mathbb{R}^{n_k \times d_v}$. Q (queries), K (keys), V (values) получаются линейными проекциями входа.

### Self-attention

В **self-attention** Q, K, V происходят **из одной и той же** последовательности $X$:

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V.$$

Это позволяет каждому токену взвешенно агрегировать информацию от всех токенов **в той же** последовательности. Используется в encoder-блоке Transformer (BERT, T5-encoder) и в decoder-блоке (с causal mask).

**С causal mask** (decoder-only, GPT):

$$\mathrm{mask}_{ij} = \begin{cases} 0, & j \le i \\ -\infty, & j > i \end{cases},$$

это запрещает токену смотреть в будущее, что нужно для авторегрессивного LM.

**Без маски** (encoder, BERT) — каждый токен видит весь контекст с обеих сторон → bidirectional representations.

**Сложность**: $O(n^2 \cdot d)$ по compute, $O(n^2)$ по памяти (матрица attention scores).

### Cross-attention

В **cross-attention** Q приходит из **одной** последовательности (target), а K, V — из **другой** (source):

$$Q = Y W_Q, \quad K = X W_K, \quad V = X W_V,$$

где $X$ — выход encoder'а, $Y$ — состояние decoder'а.

Используется в:
- **Encoder-Decoder Transformer** (T5, BART, оригинальный Vaswani-2017) — между encoder и decoder;
- **RETRO** — chunked cross-attention к retrieved chunks (см. билет 4);
- **Multimodal**: текст-decoder смотрит на vision-encoder (Flamingo, BLIP, LLaVA);
- **Diffusion models** (Stable Diffusion) — UNet с cross-attention к text embeddings.

**Это и есть тот механизм**, который Bahdanau и Luong применяли в RNN-seq2seq, но в Transformer-форме (см. билет 7).

### Multi-Head Attention (MHA)

**Идея**: вместо одной attention-операции с полной размерностью $d$ запустить $H$ **параллельных** «голов» с урезанной размерностью $d_h = d / H$, и потом сконкатенировать:

$$Q^{(i)} = X W_Q^{(i)}, \quad K^{(i)} = X W_K^{(i)}, \quad V^{(i)} = X W_V^{(i)}, \quad i = 1, \ldots, H,$$

$$\mathrm{head}_i = \mathrm{Attention}(Q^{(i)}, K^{(i)}, V^{(i)}),$$

$$\mathrm{MHA}(X) = \mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_H) W_O.$$

Каждая голова имеет свои матрицы $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)} \in \mathbb{R}^{d \times d_h}$ и общую финальную проекцию $W_O \in \mathbb{R}^{d \times d}$.

**Зачем нужны несколько голов**:
1. Каждая голова может специализироваться на разных типах зависимостей (синтаксис, кореференция, дальние/ближние связи);
2. Это похоже на ансамблирование внутри одного слоя;
3. Empirically работает лучше, чем одна большая голова с тем же compute.

### Современные оптимизации

- **MQA (Multi-Query Attention)** — одна общая пара K, V на все Q-головы → меньше KV-кэш;
- **GQA (Grouped-Query Attention)** — компромисс: $G$ групп голов с общими K, V (см. билет 4);
- **MLA (Multi-head Latent Attention)** в DeepSeek — low-rank сжатие KV;
- **FlashAttention** — IO-aware вычисление (см. билет 5);
- **Sliding window attention** (Mistral) — каждый токен видит окно $W$ предыдущих;
- **Sparse attention** (Longformer, BigBird) — паттерны вместо плотной матрицы.

### Сводная таблица

| Тип | Q | K, V | Mask | Используется в |
|---|---|---|---|---|
| Encoder self-attn | X | X | нет | BERT, T5-encoder |
| Decoder masked self-attn | Y | Y | causal | GPT, decoder |
| Cross-attn | Y (decoder) | X (encoder) | нет | T5, BART, RETRO, MM-LLMs |
| Multi-head | любая | любая | любая | везде |

### Связь с темами курса

Attention — фундамент Transformer'а. Все темы билетов 1 (KV-cache), 3 (RoPE), 4 (GQA), 5 (FlashAttention), 7 (Bahdanau/Luong) — это либо предки, либо оптимизации этой формулы.
