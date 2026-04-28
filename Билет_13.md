# Билет 13

## Q1. Types of LLM agent actions

LLM-агент выполняет **действия (actions)** — это всё, что меняет состояние агента или окружения. Действия — основной способ агента влиять на мир, поэтому от их типов и качества зависит, что агент в принципе умеет.

### Базовая классификация

Действия LLM-агента условно разбиваются на несколько групп:

#### 1. Языковые действия (text generation)

- **Final answer** — ответ пользователю в естественной форме;
- **Explanation / reasoning** — изложение хода мысли (CoT, scratchpad);
- **Clarification question** — уточняющий вопрос пользователю;
- **Summary / report** — сжатие промежуточных результатов в финальный текст.

Это «по умолчанию» действия — то, что LLM делала всегда. В рамках агентного цикла они часто служат финальным шагом или промежуточной коммуникацией.

#### 2. Tool calls (использование инструментов)

Это центральная группа действий — то, что отличает агента от чатбота. Через function calling / tool use агент вызывает внешний код:

- **Search** — веб-поиск (Google, Bing API), внутренний поиск по документам;
- **Calculator / math** — точные вычисления;
- **Code execution** — Python/SQL/JS interpreter;
- **API calls** — REST API сторонних сервисов (Slack, Stripe, Notion, Github);
- **Database queries** — SQL/NoSQL;
- **File I/O** — чтение/запись файлов в локальной FS;
- **Image / audio / video processing** — генерация, классификация, OCR;
- **Domain-specific tools** — медицинские расчёты, симуляторы, CAD-системы.

Формат: LLM генерирует структурированный вызов (JSON), оркестратор парсит, исполняет, возвращает результат как **observation**.

#### 3. Memory operations

Агент управляет своей памятью — это тоже действия:

- **Write to short-term memory** (scratchpad, current chain-of-thought);
- **Write to long-term memory** — vector store, journal, knowledge graph;
- **Read / retrieve** — поиск релевантных фактов в долгой памяти (RAG для агентов);
- **Forget / decay** — выкидывание устаревшей или нерелевантной информации;
- **Summarize history** — сжатие длинного контекста в саммари (memory consolidation).

#### 4. Communication actions (multi-agent)

Когда система состоит из нескольких агентов:

- **Send message to agent X** — передача задачи/вопроса другому агенту;
- **Broadcast** — сообщение всей группе;
- **Request critique** — попросить Critic-агента оценить промежуточный результат;
- **Delegate task** — назначить подзадачу sub-agent'у.

#### 5. Web / browser actions

Агенты типа WebGPT, Browser-Use, Anthropic Computer Use, Operator:

- **Navigate URL**;
- **Click element / fill form / scroll**;
- **Read DOM / screenshot**;
- **Type / press key**.

#### 6. OS / Computer actions (computer-using agents)

- **Open / close application**;
- **Mouse move / click**;
- **Keyboard input**;
- **Screenshot и vision-based reasoning о экране**;
- **File operations** на уровне ОС.

Анти-паттерн: дать агенту `rm -rf` без человеческого подтверждения. На таких действиях критически важен **human-in-the-loop** или sandboxing.

#### 7. Embodied / physical actions

Для роботов и embodied AI:

- **Motor commands** (move arm, walk, grasp);
- **Sensor-based control loops**;
- **Speech output**.

LLM используется как high-level планировщик; low-level моторика обычно отдельная.

#### 8. Planning / control actions

Это «мета»-действия, которые управляют ходом самого агента:

- **Replan** — отказаться от текущего плана и построить новый;
- **Reflect** — посмотреть на свою историю и сделать выводы (Reflexion);
- **Stop / declare done** — заявить о завершении цели;
- **Ask for help** — эскалация к человеку.

### Структура одного действия

Типичный action в современных LLM-API имеет формат:

```json
{
  "name": "search_web",
  "arguments": {
    "query": "RoPE rotary position embedding",
    "top_k": 5
  }
}
```

То есть: имя, типизированные аргументы (под schema). Это позволяет программно валидировать вызов перед исполнением. В Anthropic Tool Use, OpenAI Function Calling, MCP схема инструментов описывается через JSON Schema.

### Action space (пространство действий)

В одной агентной системе действия задаются её **action space** — набором tools, plus языковые действия (Thought, Final answer). Дизайн action space — ключевой инженерный вопрос:
- слишком узкое → агент не справится со сложной задачей;
- слишком широкое → агент путается, выбирает неверный tool, плохо генерализует.

Часто помогает **иерархия**: сначала router / planner выбирает категорию инструмента (research / coding / data), потом конкретный tool внутри.

### Связь с типами агентов

- **ReAct-агент**: action — это либо tool call, либо final answer;
- **Plan-and-Execute**: actions — шаги плана + tool calls;
- **CodeAct** (Wang et al., 2024) — действие = Python-код, который может содержать любые вычисления и tool-вызовы внутри. Унифицирует action space до одной модальности.

### Безопасность действий

С действиями связаны главные риски агентов:
- **Side effects** — действие изменяет состояние мира (отправлено письмо, удалён файл, потрачены деньги);
- **Irreversibility** — нельзя откатить;
- **Authorization** — у агента может быть доступ к ресурсам, которые он не должен использовать.

Меры: human-in-the-loop для критичных действий, allowlists, sandboxing, dry-run (`plan` перед `apply`), audit-логи, ограничение бюджета.

---

## Q2. BERT architecture and training

**BERT** (Bidirectional Encoder Representations from Transformers; Devlin et al., NAACL 2019) — encoder-only трансформер, обучаемый двунаправленно через masked language modeling. Принципиально важно: каждый токен в BERT'е видит **весь контекст** (и левый, и правый), а не только левый, как в GPT.

### Архитектура

#### Энкодер-стек

BERT — это **только encoder-стек** оригинального трансформера (Vaswani et al., 2017):

- **BERT-base**: $L=12$ слоёв, hidden size $H=768$, $A=12$ heads, intermediate size 3072, **~110M параметров**;
- **BERT-large**: $L=24$, $H=1024$, $A=16$, intermediate 4096, **~340M параметров**.

Каждый слой:

$$h^{(\ell)} = \mathrm{LN}\bigl(\mathrm{MultiHead}(h^{(\ell-1)}) + h^{(\ell-1)}\bigr),$$
$$h^{(\ell)} = \mathrm{LN}\bigl(\mathrm{FFN}(h^{(\ell)}) + h^{(\ell)}\bigr).$$

Используется **post-LayerNorm** (как в оригинальной статье 2017 года). FFN: `Linear(H → 4H) → GELU → Linear(4H → H)`.

#### Self-attention без маски

Главное отличие от GPT — **нет causal mask**. Любой токен может смотреть на любой другой токен (двунаправленный self-attention). Поэтому BERT не может генерировать в авторегрессивном режиме, но даёт более богатые представления для понимания.

#### Токенизация и спецтокены

- **WordPiece** tokenizer, словарь 30K (английский) / 110K (multilingual mBERT);
- **`[CLS]`** — добавляется в начало каждой последовательности; его финальный embedding используется как «sentence-level» представление для классификации;
- **`[SEP]`** — разделитель между двумя сегментами (предложениями) в задачах с парами;
- **`[MASK]`** — заглушка для masked-токенов в претрейнинге;
- **`[PAD]`** — заполнение коротких последовательностей до длины батча.

#### Embedding-слой

Эмбеддинг входа = сумма трёх:

$$E_{\text{input}} = E_{\text{token}} + E_{\text{segment}} + E_{\text{position}},$$

- **Token embedding** — WordPiece-эмбеддинг;
- **Segment embedding** — A или B (нужен для NSP и QA-задач);
- **Position embedding** — **обучаемый** (не синусоидальный), поддерживает длину до 512.

### Pre-training objectives

#### 1. Masked Language Modeling (MLM)

Случайно выбираются **15%** токенов. Из этих 15%:
- **80%** заменяются на `[MASK]`;
- **10%** заменяются на случайный токен;
- **10%** оставляются неизменными.

Задача — предсказать оригинальный токен по двунаправленному контексту:

$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in M} \log P_\theta(x_i \mid x_{\setminus M}).$$

Зачем «80/10/10», а не всегда `[MASK]`: при fine-tuning токена `[MASK]` не существует, и модель не должна слишком привыкать к нему. Mix симулирует более реалистичные условия.

MLM требует двунаправленного контекста — это и оправдывает encoder-архитектуру без causal mask.

#### 2. Next Sentence Prediction (NSP)

Бинарная классификация: следует ли предложение B сразу за A в исходном тексте.
- 50% — реальные пары (positive label `IsNext`);
- 50% — случайные предложения из других мест корпуса (`NotNext`).

Используется представление `[CLS]` через классификационную голову.

$$\mathcal{L}_{\text{NSP}} = -\sum (\log P(y \mid \mathrm{CLS})).$$

Цель — научить модель моделировать связь между предложениями (полезно для QA, NLI). Позже (RoBERTa) показано, что NSP **слабо помогает** или мешает, и его чаще выкидывают.

#### Суммарный loss

$$\mathcal{L} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}.$$

### Тренировочные данные и гиперпараметры

- **Корпус**: BookCorpus (800M слов) + English Wikipedia (2.5B слов), всего ~16 GB текста;
- **Длина последовательности**: 512 токенов (с динамическим обрезанием);
- **Batch size**: 256 sequences;
- **Шаги**: 1M;
- **Optimizer**: Adam с warmup и линейным decay;
- **Learning rate**: 1e-4;
- **Compute**: 4 дня на 4 TPU pods (BERT-large).

### Fine-tuning

После претрейнинга BERT адаптируется к задаче добавлением минимальной task-specific головы:

- **Sentence classification** (GLUE, IMDB): голова поверх `[CLS]` → `Linear → softmax`;
- **Sentence pair** (NLI, MRPC): подаётся пара `[CLS] sent_A [SEP] sent_B [SEP]`, классификация на `[CLS]`;
- **Token classification** (NER, POS): голова на каждом токене;
- **QA / SQuAD**: предсказание start/end-позиций ответа в контексте — две линейные головы, по одной на каждую границу;
- **Masked LM на новом домене** — продолжение претрейна на доменных текстах (domain-adaptive pretraining).

Fine-tune обычно занимает 2–5 эпох на одной GPU.

### Результаты

На момент выхода (2018) BERT поставил SOTA на 11 NLP-бенчмарках, включая GLUE, SQuAD v1.1/v2, SWAG. До этого NLP-задачи часто решались сложными task-specific архитектурами; BERT показал, что **универсальный претрейн + минимальная голова** сильнее.

### Преемники и улучшения

- **RoBERTa**: больше данных (160 GB), больше шагов, динамическое маскирование, **выбросили NSP**;
- **ALBERT**: shared-параметры между слоями, factorized embeddings, sentence-order prediction;
- **DistilBERT**: дистилляция в 6-слойную модель;
- **DeBERTa**: disentangled attention (отдельно content и position);
- **ELECTRA**: replaced-token-detection вместо MLM (эффективнее по compute);
- **mBERT / XLM-R**: мультиязычные BERT;
- **SciBERT, BioBERT, LegalBERT**: доменные варианты.

### Где BERT сегодня

В 2025–2026 BERT всё ещё используется как:
- backbone эмбеддинг-модели для retrieval (DPR, sentence-transformers);
- классификатор и NER в продакшене (быстрый и компактный);
- основа для cross-encoder reranker'ов в RAG;
- учебная модель для понимания трансформеров.

Для генерации и для самых сложных NLU-задач его место заняли GPT-style LLM, но в роли «текстовый процессор для классификации/поиска» он остаётся стандартом.
