# Билет 7

## Q1. In-context learning: Few-shot prompting, Chain-of-Thought (CoT)

**In-context learning (ICL)** — способность LLM решать новую задачу, не обновляя веса, а только из примеров в промпте. Открытие GPT-3 (Brown et al., NeurIPS 2020): достаточно дать модели описание задачи и несколько демонстраций, и она обобщает.

Различают:
- **Zero-shot** — только инструкция, без примеров;
- **One-shot** — один пример;
- **Few-shot** — несколько примеров (обычно 2–32).

### Few-shot prompting

Структура промпта:

```
[Описание задачи]
Q: <input_1>
A: <output_1>

Q: <input_2>
A: <output_2>

...

Q: <input_n>
A: <output_n>

Q: <тестовый вопрос>
A:
```

Модель «выучивает» паттерн из демонстраций и продолжает в том же стиле. На GPT-3 175B это даёт SOTA-уровень во многих задачах без какого-либо fine-tuning.

**Ключевые свойства:**
1. **Никаких обновлений весов** — обучение происходит «in context» (только в forward pass);
2. **Эмерджентность** — у маленьких моделей few-shot не работает или слабый эффект; на ~10B+ резко появляется;
3. **Зависимость от формата** — порядок примеров, формулировка инструкции, разделители существенно влияют;
4. **Чувствительность к recency** — последние примеры влияют сильнее.

**Гипотезы о механизме:**
- Имплицитный градиентный спуск — attention реализует «псевдо-обновление» (von Oswald et al., 2023);
- Pattern matching — модель находит ближайший шаблон в памяти;
- Bayesian inference поверх «латентных задач», полученных при претрейне.

### Chain-of-Thought (CoT)

**CoT prompting** (Wei et al., NeurIPS 2022, «Chain-of-Thought Prompting Elicits Reasoning in Large Language Models») — техника, в которой модель просят генерировать **промежуточные шаги рассуждения** перед финальным ответом, а не сразу ответ. Это драматически улучшает качество на задачах, требующих многошагового reasoning (математика, логика, common sense).

**Manual CoT (Wei et al.)**: в few-shot промпте каждый пример содержит явное пошаговое рассуждение:

```
Q: Roger has 5 tennis balls. He buys 2 cans of 3 balls each. How many?
A: Roger started with 5 balls. 2 cans × 3 = 6. 5 + 6 = 11. The answer is 11.

Q: <new question>
A:
```

Модель учится продолжать в том же стиле и генерирует свою цепочку рассуждений. На GSM8K (математика) CoT поднял точность PaLM 540B с ~18% до ~57%.

**Zero-shot CoT (Kojima et al., 2022)**: показано, что простая фраза **«Let's think step by step»** в конце промпта без всяких примеров уже включает reasoning-режим у больших моделей. На GSM8K дало прирост с ~18% до ~40% для GPT-3 InstructGPT.

**Auto-CoT** — автоматическая генерация CoT-демонстраций без ручной разметки.

### Почему CoT работает

1. **Декомпозиция** — сложная задача разбивается на простые подшаги, на каждом из которых модель ошибается реже;
2. **Использование контекста** — модель «вынимает» промежуточные результаты в видимый контекст и потом их использует;
3. **Compute scaling** — больше токенов = больше «time to think»;
4. **Эмерджентность** — у моделей <100B effect мал, у больших — резкий скачок.

### Развитие идеи

- **Self-Consistency** (Wang et al., 2022) — семплируем N CoT, голосуем большинством за ответ (см. билет 14);
- **Tree of Thoughts** (Yao et al., 2023) — дерево вариантов рассуждения с поиском;
- **Graph of Thoughts**, **Skeleton-of-Thoughts** — другие топологии;
- **Inner monologue / scratchpad / hidden reasoning** — современные reasoning-модели (o1, o3, R1) обучены делать длинный внутренний CoT перед ответом, иногда скрытый от пользователя.

### Когда CoT не помогает

- На простых, прямых задачах CoT может **ухудшить** результат (модель «передумывает»);
- Маленькие модели; CoT эмерджентен;
- При плохой инструкции / шумных демонстрациях.

---

## Q2. Scaled dot-product attention, Bahdanau attention, Luong attention

Attention-механизм исторически прошёл три ключевые формы. Все три считают **взвешенную сумму значений (V) с весами, зависящими от соответствия запроса (Q) и ключей (K)**, но по-разному задают score-функцию.

### Bahdanau attention (2014, additive)

**Bahdanau et al., ICLR 2015, «Neural Machine Translation by Jointly Learning to Align and Translate»** — первая статья, где attention внедрён в seq2seq для машинного перевода. До этого encoder сжимал всё предложение в один вектор → бутылочное горлышко на длинных входах.

Идея: декодер на каждом шаге $t$ может «смотреть» на все hidden-states энкодера и брать их взвешенную сумму:

$$c_t = \sum_{i=1}^{T_x} \alpha_{t,i} \, h_i,$$

где $\alpha_{t,i}$ — веса внимания, $h_i$ — энкодерные states.

**Score (additive / MLP)**:

$$e_{t,i} = v_a^\top \tanh(W_a s_{t-1} + U_a h_i),$$

где $s_{t-1}$ — скрытое состояние декодера на предыдущем шаге, $h_i$ — i-й state энкодера, $v_a, W_a, U_a$ — обучаемые параметры. После softmax:

$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j} \exp(e_{t,j})}.$$

Контекст-вектор $c_t$ конкатенируется с входом декодера. Вся attention bidirectional (использует $h_i$ от bidirectional GRU).

**Особенности:**
- **Additive** — score через MLP с tanh;
- Параметры $W_a, U_a, v_a$ обучаются;
- Использует $s_{t-1}$ (state до шага), что усложняет вычисление.

### Luong attention (2015, multiplicative)

**Luong et al., EMNLP 2015, «Effective Approaches to Attention-based Neural Machine Translation»** — упрощение Bahdanau с акцентом на эффективность.

Главные отличия:
1. Использует $s_t$ (а не $s_{t-1}$) — current state декодера;
2. **Score-функции — multiplicative** (нет MLP), три варианта:
   - **dot**: $e_{t,i} = s_t^\top h_i$;
   - **general**: $e_{t,i} = s_t^\top W_a h_i$;
   - **concat**: $e_{t,i} = v_a^\top \tanh(W_a [s_t; h_i])$ (близко к Bahdanau).

После attention выход смешивается с $s_t$:

$$\tilde{s}_t = \tanh(W_c [c_t; s_t]).$$

**Global vs Local attention**:
- **Global** — внимание ко всем энкодер-states (как у Bahdanau);
- **Local** — окно вокруг предсказанной позиции $p_t$, дешевле для длинных последовательностей.

### Scaled Dot-Product Attention (2017, Vaswani)

**Vaswani et al., NeurIPS 2017, «Attention Is All You Need»** — формализация, ставшая основой Transformer.

Все три тензора $Q, K, V$ — линейные проекции одного входа (или Q из декодера, K, V из энкодера для cross-attention):

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V.$$

Score-функция — масштабированное скалярное произведение:

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V.$$

**Зачем масштабирование $1/\sqrt{d_k}$**: при больших $d_k$ скалярное произведение даёт большие магнитуды, softmax «насыщается» (один argmax, остальные ≈ 0), градиенты исчезают. Деление на $\sqrt{d_k}$ нормализует дисперсию.

**Свойства:**
1. **Параллельность** — все позиции считаются одновременно (нет рекуррентности), отлично подходит для GPU;
2. **Multiplicative** (как Luong-dot), но scaled;
3. **Multi-head**: разделить $d_k$ на $H$ голов, $H$ параллельных attention'ов, конкатенировать;
4. **Self-attention vs cross-attention**: $Q, K, V$ из одного источника или из разных.

### Сравнение

| | Bahdanau | Luong | Scaled Dot-Product |
|---|---|---|---|
| Год | 2014 | 2015 | 2017 |
| Score | additive (MLP) | multiplicative (dot/general/concat) | multiplicative (dot, scaled) |
| Параметры score | $W_a, U_a, v_a$ | $W_a$ или нет | $W_Q, W_K$ (через проекции $Q, K$) |
| Использует state | $s_{t-1}$ | $s_t$ | произвольные $Q, K, V$ |
| Архитектура | внутри RNN seq2seq | внутри RNN seq2seq | основа Transformer |
| Параллельность | требует RNN | требует RNN | полная |
| Сложность | $O(T_x \cdot d)$ на шаг | $O(T_x \cdot d)$ | $O(n^2 \cdot d)$ |
| Масштабирование | нет | нет | $1/\sqrt{d_k}$ |

### Историческое значение

Bahdanau ввёл сам принцип alignment-based attention в нейронных моделях; Luong упростил и ускорил; Vaswani обобщил, убрал RNN целиком («Attention Is All You Need») и открыл эру трансформеров.
