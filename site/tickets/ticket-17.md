# Билет 17

## Q1. Key components of a reinforcement learning system

**Reinforcement Learning (RL)** — формальная парадигма обучения, в которой агент учится действовать в окружении, оптимизируя отложенную награду через взаимодействие. Все компоненты RL описываются через **Markov Decision Process (MDP)**.

### MDP — формальная основа

MDP — кортеж $(S, A, P, R, \gamma)$:
- $S$ — пространство состояний;
- $A$ — пространство действий;
- $P(s' \mid s, a)$ — функция переходов (марковская: будущее зависит только от текущего состояния);
- $R(s, a)$ — функция награды;
- $\gamma \in [0, 1)$ — дисконтирующий фактор.

В RLHF файнтюнинг LLM кастуется как **token-level MDP**: state — промпт + сгенерированные токены, action — следующий токен из словаря, reward — скаляр от RM в конце последовательности.

### Базовые сущности

#### Agent

Обучаемая сущность, принимающая решения. В классическом RL — робот / Atari-player. В RLHF — сама LLM.

#### Environment

Всё, что снаружи агента. Реагирует на действия, возвращает следующее состояние и награду. В RLHF environment — пользовательские промпты + reward model.

#### State $s_t$

Текущая ситуация. Идеально — Марковское наблюдение, на практике partial observability требует история / RNN / стеков кадров.

#### Action $a_t$

Выбор агента в момент $t$. Дискретный (Atari, токены LLM) или непрерывный (моторика робота).

#### Policy $\pi(a \mid s)$

Стратегия агента — вероятностное распределение действий в состоянии. Может быть детерминированной $\pi: S \to A$ или стохастической. В LLM это распределение над словарём.

#### Reward $r_t$

Мгновенный сигнал от среды. Часто **sparse** (только в конце эпизода) или **dense** (на каждом шаге). В RLHF — scalar в конце последовательности от RM, плюс per-token KL-penalty.

### Возвраты и value-функции

#### Return

$$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}.$$

Дисконтированная сумма будущих наград. Цель агента — максимизировать $\mathbb{E}[G_t]$.

#### State-value function

$$V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s].$$

Ожидаемый return из состояния $s$ при следовании полиси $\pi$.

#### Action-value function

$$Q^\pi(s, a) = \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a].$$

Ожидаемый return при выборе действия $a$ в $s$ и далее $\pi$.

#### Bellman equations

$$V^\pi(s) = \mathbb{E}_a[r + \gamma V^\pi(s')], \quad Q^\pi(s,a) = \mathbb{E}[r + \gamma \mathbb{E}_{a'}[Q^\pi(s', a')]].$$

#### Advantage

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s).$$

Насколько действие $a$ лучше «среднего». Используется в PPO/GRPO для уменьшения дисперсии policy gradient'а.

### Ключевые дихотомии

#### Model-based vs model-free

- **Model-based**: агент явно учит / использует модель $P(s' \mid s, a)$ и награду $R$, может планировать (Dyna-Q, MuZero, MCTS-based);
- **Model-free**: учит value/policy напрямую из опыта, без модели среды (PPO, DQN, REINFORCE).

PPO/GRPO в RLHF — model-free.

#### Value-based vs policy-based

- **Value-based** — учит $Q$ или $V$, полиси выводится через argmax (Q-learning, DQN);
- **Policy-based** — параметризует $\pi_\theta$ напрямую, оптимизирует policy gradient (REINFORCE, A2C, PPO);
- **Actor-Critic** — гибрид: actor $\pi$ + critic $V$.

#### On-policy vs off-policy

- **On-policy** — обучается на данных только текущей полиси (REINFORCE, vanilla PG);
- **Off-policy** — может учиться на данных из любой полиси (Q-learning, DQN, replay buffer);
- **PPO** — слегка off-policy через importance sampling (ratio $r_t(\theta)$ внутри clip-региона).

#### Exploration vs Exploitation

Фундаментальный trade-off:
- **Exploitation** — использовать известное хорошее действие;
- **Exploration** — пробовать новые ради потенциально лучших.

Стратегии: $\varepsilon$-greedy, softmax (Boltzmann), entropy bonus в лоссе, UCB, intrinsic motivation. В LLM exploration регулируется температурой и entropy-бонусом в PPO-лоссе.

### Алгоритмы (по семействам)

- **Tabular Q-learning** — для маленьких MDP;
- **DQN, Double DQN, Rainbow** — value-based deep RL для Atari;
- **REINFORCE** — vanilla policy gradient;
- **A2C / A3C** — actor-critic с advantage;
- **TRPO, PPO** — on-policy, trust region / clipped objective;
- **DDPG, TD3, SAC** — off-policy для непрерывных действий;
- **MuZero, AlphaZero** — model-based + MCTS;
- **GRPO** — упрощение PPO для LLM (см. билет 20).

### RL для LLM (RLHF)

В RLHF MDP устроен специфически:
- State = промпт + уже сгенерированные токены;
- Action = следующий токен (словарь ~50K-128K — огромное дискретное action space);
- Reward = скаляр от reward model в конце эпизода + per-token KL-штраф к референсной модели;
- Episode length = 128–1024 токена;
- Policy = LLM $\pi_\theta(a_t \mid s_t)$.

Используется **PPO-clipped** для стабильности с такой огромной полиси и sparse терминальной наградой.

---

## Q2. Alignment vs. Instruction Tuning vs. Fine-tuning

Эти три понятия часто путают, но они означают разные вещи и охватывают разные стадии пост-обучения LLM.

### Fine-tuning (тонкая настройка) — самое широкое

**Fine-tuning** — дообучение предобученной модели на размеченных парах `(input, output)` для адаптации к конкретной задаче или домену через стандартный supervised loss.

Примеры:
- Дообучить BERT для NER на медицинских текстах;
- Дообучить GPT для генерации SQL по описанию;
- Адаптировать LLM к юридическому домену через продолжение претрейна на legal-корпусе.

Цель — **перенос общих знаний на узкую задачу**. Не требует, чтобы данные были в формате инструкций или отражали человеческие предпочтения.

#### Виды fine-tuning

- **Full fine-tuning** — обновляются все параметры модели;
- **PEFT** — обучается малая часть (LoRA, QLoRA, IA³, prefix-tuning);
- **Continued pretraining (CPT)** — продолжение претрейна на доменных текстах с language modeling loss.

### Instruction tuning — частный случай fine-tuning

**Instruction tuning** — это fine-tuning, где данные имеют формат `(instruction, response)` и покрывают **много разнообразных задач** (QA, summarization, brainstorming, code, classification, transformation). Цель — научить модель **следовать инструкциям пользователя в общем виде**, чтобы она обобщала на инструкции, не виденные при обучении.

Технически — тот же cross-entropy лосс на следующем токене, но датасет специально сконструирован вокруг инструкций.

#### Известные instruction-датасеты и модели

- **FLAN** (Wei et al., 2021) — Google, ~60 NLP-задач переоформлены как инструкции;
- **T0** (Sanh et al., 2022) — BigScience, multitask prompted training;
- **Self-Instruct** (Wang et al., 2022) — bootstrapping инструкций через GPT-3;
- **Alpaca** (Stanford, 2023) — 52K self-instruct-инструкций для LLaMA;
- **InstructGPT SFT** — ~13K human-written demonstrations;
- **Dolly, OpenAssistant, ShareGPT, OpenHermes** — последующие открытые датасеты.

#### Что даёт

После instruction tuning модель умеет:
- понимать команды («summarize this», «translate to French»);
- следовать многоступенчатым инструкциям;
- работать в zero-shot режиме на новых задачах;
- держать формат ответа (JSON, list, и т. д.).

Это и есть **этап 1 (SFT) в RLHF-пайплайне**.

### Alignment — более широкая концепция

**Alignment (согласование)** — процесс приведения поведения модели в соответствие с человеческими ценностями, предпочтениями и нормами безопасности (helpful, harmless, honest).

Instruction tuning обучает **форме** («следуй инструкции»), но не **качеству и ценностям** ответа: модель после SFT всё ещё может галлюцинировать, давать вредные советы, быть многословной, льстивой или бесполезной.

Alignment-методы используют **сигнал предпочтений или принципов**.

#### Главные методы alignment

- **RLHF** (Ouyang et al., 2022) — reward model + PPO. См. билет 2;
- **DPO** (Rafailov et al., 2023) — прямая оптимизация на парах preference. См. билет 9;
- **GRPO** (DeepSeek, 2024) — без value-сети, групповая нормализация. См. билет 20;
- **RLAIF** (Lee et al., 2023) — Reinforcement Learning from AI Feedback, разметчиков заменяет LLM;
- **Constitutional AI** (Bai et al., 2022, Anthropic) — модель направляется набором принципов («конституцией»), цикл self-critique → revision → RM на AI-разметке;
- **KTO, IPO, ORPO** — другие preference-optimization-варианты;
- **Adversarial / red-teaming** — поиск jailbreak'ов и патчинг;
- **Process supervision** (OpenAI, 2023) — reward не только по финальному ответу, но и по каждому шагу рассуждения.

### Pipeline post-training в современных LLM

```
Pretrained base model
     │
     ▼
Continued pretraining (опц.)
     │
     ▼
SFT (instruction tuning)        ← обучение «следовать инструкциям»
     │
     ▼
RM training / Preference data
     │
     ▼
PPO / DPO / GRPO (alignment)    ← согласование с предпочтениями
     │
     ▼
Red-teaming + safety patches
     │
     ▼
Released model
```

### Сравнительная таблица

| Свойство | Fine-tuning | Instruction tuning | Alignment |
|---|---|---|---|
| Что улучшает | task performance | следование инструкциям | helpful/harmless/honest |
| Тип данных | (input, output) пары | (instruction, response) пары | preferences / pairs / principles |
| Метод | supervised CE | supervised CE | RLHF / DPO / GRPO / RLAIF |
| Степень общности | узкая задача | широкий спектр инструкций | поведенческая |
| Когда применять | каждая узкая задача | для chat / instruct LLM | финальная стадия LLM-product |

### Границы и пересечения

- Все три — это **post-pretraining**;
- **Instruction tuning ⊂ fine-tuning** (по технике);
- **Alignment** часто включает SFT-этап + дополнительный preference-этап;
- В лекциях курса фраза «alignment ≠ instruction tuning» означает: одного SFT недостаточно — нужен второй этап с preference signal.

### «Alignment tax»

Историческое наблюдение: после RLHF модель может слегка просесть на стандартных NLP-бенчмарках (T0, FLAN, GLUE), потому что оптимизируется под другой критерий. В InstructGPT этот «налог» оказался **минимальным** — RLHF-1.3B даже превзошёл базовый 175B GPT-3 на пользовательских задачах. В современных моделях alignment-tax эффективно нейтрализуется добавлением diversity / regularization.

### Если коротко

- **Fine-tuning** = «как делать конкретную задачу»;
- **Instruction tuning** = «как понимать инструкции в общем»;
- **Alignment** = «как делать это хорошо, безопасно и в соответствии с ценностями».
