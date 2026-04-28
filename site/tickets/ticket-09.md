# Билет 9

## Q1. Language models as world models

**Идея.** Гипотеза «LLM как world model» утверждает: чтобы хорошо предсказывать следующий токен на огромном корпусе, модель вынуждена внутри своих представлений выучить **аппроксимацию того, как устроен мир** — каузальные связи, физические закономерности, социальные нормы, ментальные состояния других агентов (theory of mind). Это близко по духу принципу «predicting compresses understanding»: компрессия требует моделирования.

**World model** в RL/cognitive science — это внутренняя модель среды, позволяющая агенту прогнозировать последствия действий и планировать. У человека это «mental model»: мы умеем мысленно симулировать, что произойдёт, не делая физического действия.

### Эмпирические свидетельства

- **Othello-GPT** (Li et al., 2023) — LLM, обученная только предсказывать ходы в Othello по их записи, во внутренних активациях имеет **явную репрезентацию состояния доски** (probes выявляют 8×8 матрицу клеток);
- **Chess-GPT** — аналогичный результат для шахмат;
- **Code generation** — для написания корректного кода нужна модель вычислительной семантики;
- **Math/physics reasoning** — необходима имплицитная каузальная модель;
- **Theory of mind** — современные LLM решают false-belief tasks (Sally-Anne style);
- **Linear probes** на активациях восстанавливают: координаты на карте, time-of-day, sentiment, syntactic structure, named entities.

### Применения

1. **Планирование (planning)**: PlanBench, Blocksworld; агенты Voyager в Minecraft; LLM-as-world-model в model-based RL для симуляции «что будет, если...» (RAP, Hao et al., 2023). LLM используется внутри MCTS как оценщик исходов.

2. **Reasoning**: Chain-of-Thought, Tree-of-Thoughts, ReAct — LLM генерирует промежуточные шаги рассуждения, фактически разворачивая внутреннюю модель причинности.

3. **Симуляция последствий**: LLM может предсказать, что ответит человек, как изменится состояние системы, последствия политического решения — всё требует имплицитной world model.

4. **Tool-augmented agents**: LLM как «контроллер», понимающий **когда** применять калькулятор, поиск, код-исполнитель — это требует модели возможностей и ограничений инструментов.

5. **Embodied AI / robotics**: SayCan (Google, 2022), RT-2, PaLM-E — LLM выполняет high-level reasoning, world model связывает язык с действиями робота.

6. **Multi-agent simulation**: «Generative Agents» (Park et al., 2023) — симулятивный город из LLM-агентов, у каждого своя world model, психология, память.

### Ограничения

1. **Hallucinations** — модель уверенно генерирует ложные факты, потому что её world model — статистическая аппроксимация, а не grounded истина. Особенно критично с редкими фактами и недавними событиями.

2. **Отсутствие grounding** — нет прямой связи токенов с физическим миром, сенсорным опытом, актуальным состоянием реальности.

3. **Несогласованность (inconsistency)** — внутренняя модель может противоречить себе в разных контекстах: на один вопрос ответить так, на переформулированный — иначе.

4. **Compositionality** — тяжёлые многошаговые рассуждения, особенно про новые комбинации концепций, ломаются. Композиции, не виденные в обучении, плохо обобщаются.

5. **Causality vs correlation** — LLM учится на корреляциях из текста и не всегда отделяет каузальные структуры от просто частых совпадений. В counterfactual reasoning часто ошибается.

6. **Static world** — мир, выученный из корпуса, фиксирован на момент cutoff. Без RAG/retrieval модель не «знает» о свежих событиях.

7. **Bias** — повторяет стереотипы и предубеждения корпуса.

### Связь с alignment

Хорошая внутренняя world model — необходимое (но не достаточное) условие для alignment:
- **helpful** — нужна модель того, чего хочет пользователь (theory of mind);
- **harmless** — нужна модель последствий своих ответов;
- **honest** — нужна способность отличать «что я знаю» от «что я придумываю».

RLHF-сигнал, по сути, корректирует именно те части world model, которые касаются человеческих предпочтений и норм поведения.

### Текущие исследовательские направления

- **Mechanistic interpretability / sparse autoencoders** (Anthropic) — извлечение явной world model из активаций (концепты «Golden Gate», «deception» и т. п.);
- **Multimodal grounding** — добавление визуального/аудио input'а для grounding (GPT-4o, Gemini);
- **Tool-augmented reasoning** — компенсация ограничений world model через внешние инструменты;
- **Continual learning** — обновление world model без полной переобучения (LoRA, retrieval).

---

## Q2. Direct Preference Optimization (DPO)

**DPO** (Rafailov et al., NeurIPS 2023, «Direct Preference Optimization: Your Language Model is Secretly a Reward Model») — метод alignment LLM, **обходящий явную reward-модель и RL-цикл PPO**. Превращает RLHF-задачу в обычный supervised classification-лосс на парах предпочтений.

### Идея

Классический RLHF: **SFT → RM → PPO**. Сложно (двухэтапная тренировка), нестабильно (PPO чувствителен к гиперпараметрам), дорого (RL-цикл с rollout'ами и value-сетью).

Авторы DPO заметили: **оптимальная полиси** под Bradley-Terry моделью предпочтений и KL-штрафом имеет аналитическую форму, в которой награду можно выразить через лог-отношение $\pi_\theta / \pi_{ref}$. Эта подстановка превращает RLHF-цель в **прямой classification-лосс на парах предпочтений** — без сэмплирования, без критика, без отдельного RM.

### Аналитический вывод (кратко)

В RLHF задача: $\max_{\pi} \mathbb{E}_{x, y \sim \pi}[r(x, y)] - \beta \mathrm{KL}(\pi \,\|\, \pi_{ref})$.

Решение этой задачи имеет вид:

$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{ref}(y \mid x) \exp\!\bigl(r(x, y) / \beta\bigr).$$

Отсюда выражение награды через оптимальную полиси:

$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{ref}(y \mid x)} + \beta \log Z(x).$$

При подстановке этого выражения в Bradley-Terry-лосс RM нормирующий $\log Z(x)$ **сокращается** (одинаков для $y_w, y_l$), и получается лосс прямо на полиси:

### Имплицитная награда

В DPO определяется:

$$\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{ref}(y \mid x)}.$$

«Policy сама и есть reward model» — отсюда подзаголовок статьи. Параметр $\beta$ контролирует силу регуляризации к референсной модели (типично 0.1–0.5).

### Лосс DPO

На датасете предпочтений $\mathcal{D} = \{(x, y_w, y_l)\}$ (chosen / rejected):

$$\mathcal{L}_{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \Bigl[\log \sigma\bigl(\hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l)\bigr)\Bigr]$$

$$= -\mathbb{E} \Bigl[\log \sigma\Bigl(\beta \log \tfrac{\pi_\theta(y_w \mid x)}{\pi_{ref}(y_w \mid x)} - \beta \log \tfrac{\pi_\theta(y_l \mid x)}{\pi_{ref}(y_l \mid x)}\Bigr)\Bigr].$$

Это та же Bradley-Terry log-loss, что в обучении RM, но применённая прямо к полиси.

### Градиент и интерпретация

$$\nabla_\theta \mathcal{L}_{DPO} \propto -\sigma\bigl(-\hat{r}_\theta(x, y_w) + \hat{r}_\theta(x, y_l)\bigr) \cdot \bigl[\nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x)\bigr].$$

Интерпретация:
- Увеличиваем log-вероятность $y_w$ (winner);
- Уменьшаем log-вероятность $y_l$ (loser);
- Вес обновления $\sim \sigma(-\Delta \hat r)$ велик, когда модель **ошибается** (присваивает rejected больше вероятности, чем chosen);
- Когда модель уже хорошо различает $y_w$ и $y_l$ — обновления маленькие (нет переобучения).

### Преимущества и недостатки

**+** Нет RL-цикла, нет явного RM, нет sampling полиси на лету — только forward по фиксированному датасету; **+** проще, стабильнее, дешевле; **+** конкурентное качество на большинстве бенчмарков; **+** применяется в Zephyr, Llama 2 Chat, Mistral, многих open-weight моделях.

**−** Только offline (требует фиксированный датасет, без exploration); **−** чувствителен к качеству $\pi_{ref}$ и покрытию датасета; **−** сложнее интегрировать множественные сигналы награды; **−** при сильном distributional shift между $\pi_{ref}$ и реальными запросами теряет преимущество; **−** reward hacking всё равно возможен (length bias и пр.).

### Развитие

- **IPO** (Identity PO, Azar 2023) — поправка против переобучения;
- **KTO** (Kahneman-Tversky Optimization, Ethayarajh 2024) — работает с unpaired данными;
- **ORPO** (Hong 2024) — без $\pi_{ref}$, объединяет SFT и preference learning;
- **SimPO** (Meng 2024) — упрощение без $\pi_{ref}$;
- **Iterative / Online DPO** — несколько раундов с пересборкой данных и переобучением.

### Сравнение PPO и DPO

| Свойство | PPO | DPO |
|---|---|---|
| Reward model | явная (отдельная сеть) | имплицитная (через $\pi_\theta$) |
| RL-цикл | да | нет |
| Value-сеть (critic) | да | нет |
| Online sampling | да (rollout) | нет (offline датасет) |
| Сложность кода | большая | малая |
| Память | actor + critic + RM + ref (4 модели) | actor + ref (2 модели) |
| Стабильность | средняя | высокая |
| Гиперпараметры | $\varepsilon, \beta, \gamma, \lambda$, lr | $\beta$, lr |
| Качество | comparable | comparable |
