# Persona-Graph-Agent — Подробное техническое описание системы

---

## 1. Что это такое

Persona-Graph-Agent — это локальный AI-рантайм с архитектурой **controller-first**: каждый запрос проходит через детерминированные слои анализа, принятия решений и контроля **до** того, как языковая модель вообще вызывается. LLM — это только последний шаг, вербализатор, а не мозг системы.

Система выполняет две вещи одновременно:

1. **Персонаж-диалог** — отвечает от лица выбранного персонажа (Дракула, Катерина, Капитан Джек Воробей и др.) с психологически глубокой и стабильной личностью.
2. **Адаптивный коуч** — режим планирования, где система помогает пользователю с задачами, отслеживает его прогресс, запоминает паттерны и адаптируется к его психологическому профилю.

Всё работает **полностью локально** — никаких облачных API.

---

## 2. Общая архитектура

```
Входящий запрос
       │
       ▼
 Request Preprocessing        ← языкодетекция, нормализация, маркеры ситуации
       │
       ▼
 Controller (router)          ← route decision + capability plan
       │
       ├─── Lightweight path  ← фактические вопросы, короткие реплики
       │
       └─── Persona-aware path
                │
                ├── Memory + Graph Load    ← персонаж, граф знаний, история сессии
                │
                ├── Cognitive Pipeline V1 (P1–P6)    ← событие → регулятор → действие
                │
                ├── Cognitive Modules V2 (P1–P49)    ← 49 параллельных сигнальных детекторов
                │
                ├── Message Vector System (P1–P51)   ← 51 координата диалога
                │
                ├── Context Builder                  ← сборка промпта с бюджетами токенов
                │
                ├── Response Shaping                 ← Speech Plan / State Transition
                │
                ├── LLM Call (Qwen3.5-2B Q4_K_M)    ← вербализация
                │
                ├── Validation + Repair              ← проверка персонажности, утечек, ошибок
                │
                └── Persistence                      ← история, трейс, обновление памяти
```

---

## 3. Языковые модели (ИИ в основе)

### 3.1. Основная модель
**Модель:** `Qwen3.5-2B.Q4_K_M.gguf`
- Архитектура: Qwen 3.5, 2 миллиарда параметров
- Квантизация: Q4_K_M (4-бит, средней точности) — примерно 1.4 ГБ на диске
- Роли: `general`, `analyst`, `creative`, `planner` — используется для **всего диалога**
- Контекстное окно: до 7168 токенов (`LOCAL_GGUF_N_CTX=7168`)
- Максимум токенов на ответ: 2048
- Запуск: через `llama.cpp` (Python binding `llama-cpp-python`)

**Ограничение:** Модель 2B — маленькая. Из-за RLHF-тренировки она иногда ломает персонажа и отвечает как ИИ-ассистент. Именно поэтому система имеет несколько слоёв защиты: `ЗАПРЕЩЕНО`-правила в примерах персонажа, ремонт ответа, ситуационные реакции.

### 3.2. Модель для кода
**Модель:** `qwen2.5-coder-7b-instruct-q4_k_m.gguf`
- 7B параметров, Qwen 2.5 Coder Instruct
- Роли: `coder_architect`, `coder_reviewer`, `coder_refactor`, `coder_debug`
- Контекст: 7000 токенов

### 3.3. Детектор языка
Эвристический детектор по Unicode-диапазонам:
- Кириллица (`Ѐ-ӿ`) → русский
- Армянский (`԰-֏`) → армянский
- Арабский (`؀-ۿ`) → арабский
- CJK (`一-鿿`) → китайский
- Иначе → английский

---

## 4. PersonalityGenome — 53 обучаемых параметра

Геном — это **числовой ДНК персонажа**: 53 параметра типа `float[0..1]`, каждый из которых определяет реакции, тревожность, доминирование, стиль защиты и т.д.

### Структура LearnableParam
```python
LearnableParam:
    value:      float         # текущее значение
    prior:      float         # начальное значение (якорь)
    lr:         float = 0.03  # скорость обучения
    stability:  float = 0.5   # 0=свободный, 1=замороженный
    confidence: float         # растёт с каждым обновлением
    updates:    int           # счётчик обновлений
    history:    list          # последние 200 изменений (timestamp, delta, source)
```

**Обновление:** `effective = delta × lr × (1 - stability)`, затем clip к `[-0.08, +0.08]`.
**Регуляризация:** `elastic_pull = -0.015 × (value - prior) × (1 - stability)` — тянет параметр обратно к prior.

**Как геном связан с поведением.** Геном — это не просто набор чисел. Каждый параметр напрямую влияет на одну или несколько точек в пайплайне:
- `baseline_anxiety` → начальное значение регулятора `anxiety` в P3, то есть высокотревожный персонаж *стартует* напряжённым ещё до первого слова
- `fear_shame` → чувствительность модуля Humiliation в V2 и вес вхождения `fear_shame ← shame_trigger` в P2, то есть персонаж с высоким страхом стыда *видит* унижение там, где другой его не замечает
- `dominance_tendency` → вес оси `dominance` в P49, то есть персонаж с высоким доминированием склоняется к `attack` и `seek_control` в P6
- `trust_baseline` → чувствительность модулей MessageHonesty, Sincerity, PromiseReliability — насколько вообще персонаж считывает честность или ложь
- `impulsivity` → через P4 влияет на `action_urgency`, то есть импульсивный персонаж быстрее доходит до агрессивных действий

Геном обучается на диалогах через `LearnableParam.step()` — каждый ход, где реакция персонажа получила положительную или отрицательную оценку, слегка сдвигает связанные параметры. Параметры с высоким `stability` (страхи, базовая тревожность) меняются медленно — это «характер». Параметры с низким `stability` меняются быстро — это «настроение».

### Группы параметров

**A. Мотивационные влечения (8 параметров)**
| Параметр | Смысл |
|---|---|
| `drive_recognition` | Жажда признания |
| `drive_security` | Потребность в безопасности |
| `drive_control` | Стремление к контролю |
| `drive_closeness` | Тяга к близости |
| `drive_autonomy` | Потребность в независимости |
| `drive_superiority` | Стремление к превосходству (по умолчанию 0.3) |
| `drive_stability` | Потребность в стабильности |
| `drive_meaning` | Поиск смысла |

**B. Страхи (8 параметров, stability=0.6–0.7 — медленно меняются)**
`fear_shame`, `fear_rejection`, `fear_loss_of_control`, `fear_helplessness`, `fear_abandonment`, `fear_judgment`, `fear_chaos`, `fear_failure`

**C. Когнитивный стиль (9 параметров)**
`planning_depth`, `analysis_bias`, `impulsivity`, `ambiguity_tolerance`, `category_rigidity`, `generalization_bias`, `suspicion_bias`, `threat_first`, `hypothesis_switch_speed`

**D. Социальная регуляция (7 параметров)**
`hierarchy_sensitivity`, `approval_seeking`, `dominance_tendency`, `vulnerability_concealment`, `social_distance_default`, `trust_baseline`, `mirror_tendency`

**E. Механизмы защиты (6 параметров)**
`defense_avoidance`, `defense_rationalization`, `defense_aggression`, `defense_freeze`, `defense_humor`, `defense_hypercontrol`

**F. Энергетика / базовая регуляция (6 параметров)**
`baseline_anxiety` (stability=0.65), `baseline_drive` (stability=0.6), `fatigue_sensitivity`, `stress_recovery_speed`, `novelty_reward`, `shame_recovery_speed`

**G. Якоря памяти (4 параметра)**
`pain_memory_weight`, `shame_memory_weight`, `success_memory_weight`, `past_influence_decay`

**H. Когнитивные смещения (5 параметров)**
`threat_scan_first`, `blame_self_vs_other` (0=себя, 1=других), `feel_first` (0=думать, 1=чувствовать), `justify_vs_plan`, `pessimism_bias`

---

## 5. Когнитивный пайплайн V1 (P1–P6)

Шесть NumPy-перцептронов, работающих последовательно. **Нет LLM, нет случайности** — чистая линейная алгебра.

**Зачем это нужно.** Без этого пайплайна LLM получает голый текст пользователя и сам решает, что значит «ты идиот» — оскорбление, провокация, шутка или тест. И решает каждый раз по-разному, в зависимости от случайности сэмплирования. Пайплайн V1 убирает эту случайность: он детерминированно превращает текст в структурированное внутреннее состояние персонажа — и только потом передаёт это состояние LLM как директиву. LLM больше не решает *что делать* — только *как это сказать*.

Шесть модулей образуют причинно-следственную цепочку: **событие → какой ген генома затронут → как это меняет регуляторы → что персонаж сейчас думает → есть ли внутренний конфликт → какое действие выбрать**. Каждый слой сужает пространство возможных ответов.

### P1 — Event Encoder (EventEncoder)
**Назначение:** Текст → вероятности 14 типов событий.

**Входной вектор:** `float[48]`
- `[0:14]` — совпадения по паттернам (regex keyword scores) для каждого из 14 типов событий
- `[14:20]` — структурные: длина, вопросы, восклицания, заглавные, отрицания, команды
- `[20:26]` — референции к лицам: я/меня, ты/тебя, он/она/они, мы/нас
- `[26:32]` — временные: прошедшее, настоящее, будущее, срочность, частотность, длительность
- `[32:38]` — маркеры интенсивности: очень/крайне, всегда/всё, никогда/ничего, самый, всё ещё, полностью
- `[38:44]` — социальный контекст: публично, в одиночестве, сравнение, авторитет, группа, кто-то
- `[44:48]` — флаги сессии: prior_failure, repeated, escalating, resolved

**Архитектура:** `W[14×48] @ x + b[14]` → softmax → `event_probs[14]`

**14 типов событий:**
`criticism`, `praise`, `rejection`, `failure`, `overload`, `uncertainty`, `intimacy`, `opportunity`, `danger`, `shame_trigger`, `boredom`, `novelty`, `loss_of_control`, `neutral`

**Выход:** `(event_probs[14], intensity: float)`

**Обучение:** Офлайн через `perceptron_trainer.py` → метод градиентного спуска по кросс-энтропии, LR=0.01, 30 эпох, минимум 20 размеченных примеров.

**Зачем именно 14 событий.** Это не произвол — это минимальный набор, который покрывает психологически значимые ситуации: угроза самооценке (criticism, shame_trigger), социальные сигналы (rejection, intimacy), ресурсные состояния (failure, overload, boredom) и позитивные входы (praise, opportunity). Neutral — это «ничего не произошло», нужен для того, чтобы система не галлюцинировала событие там, где его нет.

---

### P2 — Trigger Network (TriggerNetwork)
**Назначение:** Какие гены генома активированы данным событием?

**Вход:** `event_probs[14] + ctx_flags[6] = [20]` + `genome_vec[53]` + `intensity`

**Архитектура:** `S[53×20] @ ev_extended × genome_vec × intensity`

Матрица S инициализируется prior-правилами из психологии:
- `fear_shame ← criticism` (вес 1.2)
- `fear_rejection ← rejection` (вес 1.3)
- `fear_loss_of_control ← loss_of_control` (вес 1.4)
- `fear_shame ← shame_trigger` (вес 1.5)
- `drive_recognition ← praise` (вес 0.8)
- `drive_closeness ← intimacy` (вес 0.9)

**Выход:** `triggers[53]` — активация каждого параметра генома

**Зачем это нужно.** Одно и то же событие «критика» должно по-разному задевать Катерину (высокий `fear_shame`) и Дракулу (высокий `drive_superiority`, низкий `fear_rejection`). P2 реализует именно это — он умножает событие на геном, получая персонаж-специфичный отклик. Без этого слоя все персонажи реагировали бы одинаково на одни и те же слова.

---

### P3 — Regulator Cell (RegulatorCell) — упрощённый GRU
**Назначение:** Обновить 10 внутренних регуляторов на основе триггеров.

**10 регуляторов:**
| Регулятор | Аналог |
|---|---|
| `anxiety` | тревожность |
| `drive` | побуждение к действию |
| `fatigue` | усталость |
| `motivation` | мотивация |
| `shame` | стыд |
| `confidence` | уверенность |
| `threat_sense` | восприятие угрозы |
| `control_sense` | ощущение контроля |
| `closeness` | близость |
| `frustration` | фрустрация |

**Вход:** `[triggers(53), h_prev(10), energy_genes(6)]` = вектор 69

**GRU-шаг:**
```
z = sigmoid(Wz @ x + bz)        # ворота забывания
r = sigmoid(Wr @ x + br)        # ворота сброса
xr = concat([triggers, r*h, energy])
hc = tanh(Wh @ xr + bh)         # кандидат
h_next = (1-z)*h + z*hc         # обновлённое состояние
```

Prior-правила в Wh: `fear_shame→shame`, `fear_rejection→anxiety`, `drive_recognition→motivation`, `drive_closeness→closeness`

**Выход:** `regulator_state[10]` — новые значения всех регуляторов

**Зачем GRU, а не просто линейный слой.** GRU имеет память: ворота z определяют, сколько старого состояния сохранить, ворота r — насколько прошлое влияет на новое. Это значит, что если персонаж несколько ходов подряд получает критику, тревожность накапливается, а не сбрасывается после каждой реплики. Регуляторы — это «гормоны» персонажа: они не мгновенны, они накапливаются и затухают.

---

### P4 — Thought MLP (ThoughtMLP) — 3-слойный персептрон
**Назначение:** Построить "мысль" — что персонаж думает прямо сейчас.

**Вход:** `[triggers(53), regulators(10), cognitive_genes(9)]` = вектор 72

**Архитектура:** 72 → ReLU → 32 → 16 (ThoughtVector)

**16 компонент мысли:**
```
perceived_risk        — насколько опасна ситуация
confidence_in_frame   — насколько персонаж уверен в своей позиции
dominant_need_goal    — softmax: security vs recognition vs control
dominant_need_social  — softmax: closeness vs autonomy vs stability
dominant_need_other
frame_approach        — softmax: approach / hold / retreat
frame_avoid
frame_freeze
self_deception        — степень самообмана
action_urgency        — срочность ответа
social_concern        — беспокойство о социальных последствиях
planning_horizon      — горизонт планирования (fatigue ↑ → horizon ↓)
blame_direction       — на кого направляется вина
emotional_intensity   — накал эмоций
control_assessment    — оценка контроля над ситуацией
novelty_seeking       — поиск нового
```

**Выход:** `thought_vec[16]`

---

### P5 — Conflict Scorer (ConflictScorer)
**Назначение:** Какой внутренний конфликт сейчас активен, какова стратегия его разрешения?

**Вход:** `[thought(16), goals(8), fears(8), regulators(10)]` = вектор 42

**Выход:** `conflict_vec[8]` = `[intensity, avoidance, overcompensation, attack, freeze, planning, support_seeking, self_deception]`

**Блокировки действий по стратегии:**
- `freeze` → блокирует: approach, attack, connect
- `avoidance` → блокирует: approach, connect, attack
- `attack` → блокирует: placate, ask_for_help, connect
- `self_deception` → блокирует: analyze, reframe

**Зачем нужен конфликт как отдельный слой.** Мысль (P4) говорит «я воспринимаю угрозу и хочу сближения». Это противоречие. Если его не разрешить явно, P6 получит противоречивые сигналы и выберет случайное действие. P5 диагностирует, какая стратегия сейчас доминирует, и через блокировки исключает несовместимые с ней действия — как психологический защитный механизм, который не даёт делать взаимоисключающие вещи одновременно.

---

### P6 — Action Policy (ActionPolicy) — 2-слойный персептрон
**Назначение:** Выбрать поведенческое действие.

**Вход:** `[thought(16), conflict(8), regulators(10), defenses(6)]` = вектор 40

**Архитектура:** 40 → ReLU(24) → logits[14]

**14 семейств действий:**
```
approach       — открыться, напрямую ответить
avoid          — остаться на поверхности
freeze         — пауза, признать вес ситуации
attack         — чёткий отпор
placate        — сгладить напряжение
analyze        — холодный разбор ситуации
ask_for_help   — выразить нужду
seek_control   — установить границы
reduce_exposure — краткий ответ, не раскрываться
reframe        — другой ракурс
self_protect   — минимальный ответ, держать границу
connect        — близость, эмпатия
withdraw       — отступить, нужна дистанция
plan_small_step — один конкретный маленький шаг
```

**Обучение:** `train_p6()` в `perceptron_trainer.py` — те же 30 эпох, LR=0.01.

---

### CognitiveRuntime — полный проход P1→P6

```python
CognitiveTurnOutput:
    action_id            # выбранное действие (0-13)
    action_name          # его название
    action_probs         # вероятности всех 14 действий
    thought_vec          # вектор мысли [16]
    conflict_vec         # вектор конфликта [8]
    regulator_state      # состояние регуляторов {name: float}
    event_probs          # вероятности событий [14]
    primary_event        # главное событие
    intensity            # интенсивность
    perceived_risk       # воспринимаемый риск
    dominant_resolution  # стратегия разрешения конфликта
    blocked_actions      # заблокированные действия
```

---

## 6. Когнитивные модули V2 (P1–P49)

Параллельная система из 49 детекторов — работает одновременно с V1. Каждый модуль — лёгкий линейный классификатор над общим вектором признаков.

**Зачем V2, если уже есть V1.** V1 отвечает на вопрос «что делать»: выбирает поведенческое действие через цепочку событие→регулятор→конфликт→действие. V2 отвечает на вопрос «что происходит между людьми»: читает субтекст, манипуляцию, маски, иронию, скрытую угрозу — всё то, что V1 не видит, потому что работает на уровне событий, а не отношений. Вместе они дают персонажу и внутреннее состояние, и социальное восприятие.

**Роль каждой из пяти групп детекторов:**
- **Группа I (отношение к себе)** — как к персонажу относятся прямо сейчас: уважают, унижают, используют, проявляют тепло. Это «зеркало» — персонаж видит, как его видят.
- **Группа II (намерения другого)** — честен ли собеседник, есть ли скрытый мотив, можно ли доверять. Кормит параметры `suspicion_bias` и `trust_baseline` в геноме.
- **Группа III (безопасность)** — насколько ситуация опасна физически, социально, эмоционально. Итог группы — `threat_level`, который поступает в P49 и может перекрыть другие сигналы.
- **Группа IV (смысл/ценности)** — затронут ли принцип, выгодно ли это персонажу долгосрочно, честно ли это. Влияет на то, будет ли персонаж бороться или уступит.
- **Группа V (внутреннее состояние)** — есть ли желание ответить, сомнение, срочность, внутренний конфликт. Это то, что персонаж чувствует, но не обязательно говорит.

**P49 как финальный синтез.** Все 48 сигналов сходятся в FinalPosition — пять осей (доверие, близость, принятие, открытость, речь), которые вместе дают `dominant_stance`. Этот стэнс передаётся в Speech Planner и определяет тональность ответа.

### SignalFeatureExtractor — общий вектор признаков `float[70]`

Детерминированное преобразование текста в 70-мерный вектор:

**[0:60] Семантические признаки (6 групп по 10):**
```
Группа I   (0-9)  — отношение к себе:
    humiliation_signal, care_signal, respect_signal, acceptance_signal,
    control_attempt, instrumental_use, sincerity_cue, hostility_cue,
    warmth_cue, hierarchy_cue

Группа II  (10-19) — намерения другого:
    intent_clean, manipulation_cue, hidden_agenda, honesty_cue,
    promise_cue, consistency_cue, predictability_cue, accountability_cue,
    mature_motive, conflict_subtext

Группа III (20-29) — угроза / безопасность:
    physical_threat, social_threat, emotional_threat, practical_risk,
    freedom_threat, identity_threat, dependency_trap, error_risk,
    irreversibility, threat_composite (= среднее группы × 1.5)

Группа IV  (30-39) — смысл / ценности:
    principle_trigger, value_alignment, fairness_cue, goal_visibility,
    meaningfulness, self_benefit, other_benefit, longterm_value,
    self_fidelity, compromise_cost

Группа V   (40-49) — внутреннее состояние:
    desire_signal, repulsion_signal, doubt_signal, certainty_signal,
    internal_conflict, goal_priority, action_impulse, inhibition_signal,
    urgency_signal, ambivalence_signal

Группа VI  (50-59) — тонкий человеческий подтекст:
    sarcasm_signal, passive_aggression, self_deprecation, existential_despair,
    cry_for_help, dark_humor, provocation, deflection,
    irony_marker, implicit_pain
```

**[60:70] Структурные признаки:**
```
question_count, exclamation_intensity, negation_density,
intensity_markers, self_reference, other_reference,
hedging_language, command_language, length_signal, sentiment_balance
```

**Мультиязычность:** паттерны для русского, английского, армянского, китайского.

### CognitiveModuleV2 — один детектор

```python
CognitiveModuleV2:
    module_id:   int       # 1..48
    name:        str       # e.g. "humiliation"
    feat_indices: list[int] # какие из 70 признаков использует
    W:  np.ndarray[n]      # веса
    b:  float              # смещение
    sensitivity: float     # модулируется геномом

    def process(feat, sensitivity, text) -> ModuleSignal:
        raw = W @ feat[feat_indices] + b
        value = sigmoid(raw × sensitivity)
        direction = ...    # -1..+1
        confidence = ...   # 0..1
```

**ModuleSignal:**
```python
ModuleSignal:
    module_id:  int
    name:       str
    value:      float   # интенсивность [0..1]
    direction:  float   # -1=плохо/угроза, +1=хорошо/безопасно
    confidence: float   # уверенность детектора
    evidence:   list    # что сработало
```

### 48 модулей по группам

**Группа I — отношение к себе (P1–P10):**
Humiliation, Care, Respect, Acceptance, ControlOverMe, BeingUsed, Sincerity, Hostility, Benevolence, SocialHierarchy

**Группа II — намерения (P11–P20):**
IntentPurity, Manipulation, HiddenBenefit, MessageHonesty, PromiseReliability, BehavioralConsistency, Predictability, Accountability, MatureMotive, HiddenConflict

**Группа III — безопасность (P21–P30):**
PhysicalThreat, SocialThreat, EmotionalThreat, PracticalRisk, FreedomThreat, IdentityThreat, DependencyTrap, ErrorRisk, Irreversibility, ThreatIndex

**Группа IV — смысл/ценности (P31–P40):**
Principle, ValueAlignment, Fairness, GoalVisibility, Meaningfulness, SelfBenefit, OtherBenefit, LongTermValue, SelfFidelity, CostOfCompromise

**Группа V — внутреннее состояние (P41–P49):**
Desire, Repulsion, Doubt, Certainty, InternalConflict, GoalPriority, ActionImpulse, Inhibition, FinalPosition

### P49 — FinalPositionIntegrator

Читает сигналы всех 48 модулей → строит `FinalPosition`:

```python
FinalPosition:
    # 5 осей, каждая float [0..1]
    trust / distrust        — доверие vs подозрение
    approach / distance     — близость vs дистанция
    accept / argue          — принятие vs возражение
    defend / open           — закрытость vs открытость
    speak / silence         — говорить vs молчать

    dominant_stance: str    # итоговая позиция
    threat_level: float     # из группы III
    signals: dict           # все сырые значения
```

Связь с геномом: `genome_sensitivity` определяет чувствительность каждого модуля. Например, для персонажа с высоким `fear_shame` модуль Humiliation имеет чувствительность > 0.8 → реагирует на слабые намёки на унижение.

---

## 7. Система координат сообщений (P1–P51)

Каждое сообщение получает 51 координату — структурированное описание его диалогической природы. Это не embedding, а **символьные метки** с оценками уверенности.

### Группы координат

**Группа A — Форма речи (P1–P7)**
```
P1  Форма высказывания:   question | statement | thought | quote | directive
P2  Тип смысла:           direct | figurative | masked
P3  Буквальность:         literal | rhetorical_move
P4  Ответность:           answer | non_answer | avoidance
P5  Самостоятельность:    independent | reaction
P6  Направление хода:     closing | opening_new_direction
P7  Логическая ясность:   defined | diffuse
```

**Группа B — Психологическое состояние (P8–P15)**
```
P8  Сомнение:             absent | doubt
P9  Уверенность:          absent | confidence
P10 Внутренний конфликт:  absent | inner_conflict
P11 Защита:               absent | defense
P12 Уязвимость:           absent | vulnerability
P13 Нападение:            absent | attack
P14 Сдерживание:          absent | containment
P15 Напряжённость:        calm | tense | overloaded
```

**Группа C — Отношение (P16–P23)**
```
P16 Забота:               absent | care
P17 Уважение:             absent | respect
P18 Обесценивание:        absent | devaluation
P19 Унижение:             absent | humiliation
P20 Дружелюбие:           absent | friendliness
P21 Скрытая враждебность: absent | hidden_hostility
P22 Доминирование:        neutral | dominance
P23 Уступка:              neutral | concession
```

**Группа D — Риторические маски (P24–P31)**
```
P24 Сарказм:              absent | sarcasm | dry_sarcasm | false_praise
P25 Ирония:               absent | irony
P26 Издевательство:       absent | mockery
P27 Маска похвалы:        absent | mask_of_praise
P28 Маска заботы:         absent | mask_of_care
P29 Манипуляция:          absent | manipulation
P30 Давление:             absent | pressure
P31 Ложное смягчение:     absent | false_softening
```

**Группа E — Реляционное движение (P32–P39)**
```
P32 Сближение:            neutral | approach
P33 Дистанцирование:      neutral | distancing
P34 Примирение:           neutral | reconciliation
P35 Эскалация:            neutral | escalation
P36 Обострение:           neutral | sharpening
P37 Разрыв:               neutral | rupture
P38 Смягчение:            neutral | softening
P39 Удержание контакта:   neutral | contact_maintenance
```

**Группа F — Искренность (P40–P47)**
```
P40 Искренность:          unclear | sincerity
P41 Маска/неискренность:  unclear | masking
P42 Признание:            absent | admission
P43 Отрицание:            absent | denial
P44 Переосмысление:       absent | reframing
P45 Корректирующая похвала: absent | corrective_praise
P46 Ложная похвала:       absent | false_praise
P47 Скрытый упрёк:        absent | hidden_reproach
```

**Группа G — Структура дискурса (P48–P51)**
```
P48 Реплика в цепочке:    continuation | reinterpretation | masking_shift |
                           conflict_reply | repair_attempt
P49 Итоговое направление: neutral | toward_repair | toward_distance |
                           toward_escalation | toward_masking | toward_contact
P50 Смена темы:           no_change | soft_shift | hard_shift | return_to_topic
P51 Граница контекста:    full_window | since_topic_shift | last_turn_only | anchor_only
```

**P51** особенно важен — он говорит Context Builder-у, сколько истории диалога включать в промпт:
- `full_window` → вся история
- `since_topic_shift` → только с последней смены темы
- `last_turn_only` → только последняя реплика

### Как работает предсказание координат

`MessageVectorRuntime` → для каждого из P1–P51:
1. Извлекает `float[256]`-вектор из текста (хэш-функция + структурные признаки)
2. Если есть обученная модель из annotation store — использует её
3. Иначе — детерминированная эвристика по паттернам

---

## 8. Персона-движок (PersonaEngine)

### Структура персонажной головы

Каждый персонаж хранится в `memory/heads/{slug}/`:

```
meta.json             — метаданные: name, slug, entity_type, aliases, readiness,
                        validation_status, revision, maturity_score, evidence_count,
                        adaptation_locked, persona_confidence

baseline.json         — базовые неизменяемые факты: knowledge, traits, relations

persona_form.json     — психологический профиль:
                        identity_class, core_goal, secondary_goals,
                        core_dispositions, communication_style, speech_tendencies,
                        sarcasm_profile, clarification_policy,
                        constraints_internal/social/hard_system,
                        allowed_methods, defense_mechanisms,
                        triggers, reaction_patterns, decision_patterns

learned_patterns.json — обученные паттерны:
                        examples (Q&A пары с примерами ответов),
                        situation_reactions (как реагировать на тип ситуации),
                        log_tuples (частотные паттерны из диалогов),
                        persona_form (встроенная копия — ВАЖНО: перезаписывает persona_form.json)

examples.json         — конкретные примеры реплик с ситуационными реакциями

knowledge.txt         — знания о персонаже (текстовое описание)

traits.json           — список черт характера

decision_explanation.txt — объяснение логики принятия решений персонажем

dynamic_state.json    — текущее эмоциональное состояние:
                        emotion_vector {anger, fear, curiosity, confidence, empathy},
                        last_situation, last_response_style

log_tuples.json       — частотные n-граммы из диалогов (для обучения)
structured_persona.json — структурированный психологический профиль
```

### Цикл фонового перестроения

После каждых N диалогов система запускает фоновый rebuild:
1. Читает `log_tuples.json` — находит вхождение с максимальной `frequency`
2. Из него дериверует `core_goal`, `communication_style`, `core_self_image`
3. Обновляет `persona_form.json`

**Критично:** `learned_patterns.json` содержит встроенную копию `persona_form`. При загрузке именно она перезаписывает файл `persona_form.json`. Поэтому оба файла должны быть синхронизированы.

### Активные персонажи

| Slug | Тип | Зрелость |
|---|---|---|
| `катерина` | FICTIONAL_CHARACTER | mature |
| `dracula` | FICTIONAL_CHARACTER | mature |
| `капитан_джек_воробей` | FICTIONAL_CHARACTER | mature |
| `rum` | CONCEPT | — |

---

## 9. Граф знаний (GraphStore)

### Структура узла
```json
{
    "id": "uuid",
    "name": "Дракула",
    "entity_type": "FICTIONAL_CHARACTER",   // PERSON|CONCEPT|PHENOMENON|OBJECT|FICTIONAL_CHARACTER|PROFESSION
    "aliases": ["Dracula", "Count Dracula"],
    "facts": ["...", "..."],
    "importance": 1.0,        // обучаемый параметр
    "confidence": 0.97,
    "frequency": 10,          // сколько раз встречался
    "state": "active"         // active|weak|suspect|archived|merged
}
```

### Формула quality score
```
quality = 0.45 × confidence + 0.35 × min(frequency/10, 1) + 0.2 × min(importance, 1)
```

### Интеграция с контекстом

Context Builder ранжирует узлы графа для включения в промпт по формуле:
```
context_score = 0.34 × relevance
              + 0.16 × importance
              + 0.16 × persona_alignment
              + 0.12 × confidence
              + 0.12 × recency
              + 0.10 × graph_connectivity
```

Источники контекста (в порядке приоритета):
```
persona_memory → persona_triad → social_role → mood_research →
session_graph_context → local_graph_neighborhood → global_graph_facts →
file_ingested_knowledge → session_short_term_history
```

---

## 10. Построитель промпта (ContextBuilder + PromptBuilder)

**Зачем такая сложная сборка контекста.** Модель 2B имеет контекстное окно 7168 токенов, из которых промпт занимает 3400. Это мало. Если просто свалить туда всё — историю, граф, персонажа, — самое важное окажется в конце, где модель его плохо учитывает. Поэтому контекст не просто собирается, а **ранжируется и обрезается** по бюджетам: каждый источник знает свой лимит, и если персонаж важнее старой истории — история обрезается первой.

Порядок рендеринга персонажного блока тоже не случаен: сначала идёт «кто я» (`persona_core`), потом ограничения и запреты, потом примеры реплик. Модель читает промпт сверху вниз — идентичность должна быть установлена до того, как придут конкретные инструкции.

**P51 как динамический контроль контекста.** Система координат сообщений (раздел 7) возвращает P51 — метку, сколько истории диалога включать. Если тема не менялась (`full_window`) — берём всё. Если только что была жёсткая смена темы (`since_topic_shift`) — старый контекст нерелевантен, берём только с момента смены. Это позволяет не тратить бюджет на историю, которая уже не связана с текущим разговором.

### Бюджеты токенов

```
COGNITIVE_MAX_CONTEXT_TOKENS=3400  — общий лимит
COGNITIVE_PERSONA_BLOCK_BUDGET=900 — на блок персонажа
COGNITIVE_GRAPH_CONTEXT_BUDGET=1100 — на граф и факты
COGNITIVE_RECENT_DIALOGUE_BUDGET=320 — на недавний диалог
```

### Порядок рендеринга блоков персонажа
```
1. persona_core              — основное "кто я"
2. persona_identity          — идентичность
3. persona_work_profile      — рабочий профиль
4. persona_social_role       — социальная роль в сессии
5. persona_mood_dynamics     — текущее настроение/динамика
6. persona_control           — ограничения и запреты
7. persona_knowledge         — знания о мире
8. persona_relations         — отношения с другими
9. persona_form              — психологический профиль
10. persona_decision_explanation — объяснение логики
11. persona_state            — текущее эмоциональное состояние
12. persona_reactions        — ситуационные реакции
13. persona_examples         — примеры реплик (Q&A)
14. persona_log_tuples       — обученные паттерны
```

### Локализационный движок

Не переводит, а **обеспечивает нативность тона**. Для русского языка инжектирует в промпт:
- формальность: formal | neutral | informal | intimate
- теплота: cold | reserved | warm | tender
- острота: flat | ironic | sharp | confrontational
- темп: slow | measured | brisk | rapid

---

## 11. Регулятор ситуаций (SituationRegulator) — 4 слоя

### Слой 1: Классификатор событий (14 типов)
```
threat, reward_signal, social_shame, warm_support,
challenge, failure, social_rejection, attachment_activation,
overload, novelty, planning_request, feedback_report,
neutral, crisis
```

Специальный случай: `crisis` — детектирует суицидальные маркеры и немедленно переключается на режим поддержки, подавляя планирование.

### Слой 2: Политика выброса регуляторов
Правила вида `событие → изменения регуляторов`:
- `threat → cortisol↑, adrenaline↑`
- `reward_signal → dopamine↑`
- `social_shame → cortisol↑, serotonin↓`
- `warm_support → oxytocin↑, cortisol↓`

### Слой 3: Выбор действия
Взвешенная оценка по силам личности + состоянию регуляторов → выбор семейства поведенческого ответа.

### Слой 4: LLM-вербализатор
Получает выбранное действие + весь контекст → генерирует текст.

---

## 12. Speech Planner (SpeechPlan)

Мост между `CognitiveTurnOutput` и LLM-промптом. LLM получает не «ответь на вопрос», а структурированный план:

```python
SpeechPlan:
    action_name:     str         # "withdraw", "attack", "analyze", ...
    speech_goal:     str         # "step back, reply minimally"
    tone:            str         # "sharp and direct, does not soften edges"
    perceived_risk:  float       # 0..1
    confidence:      float       # 0..1
    intensity:       float       # интенсивность события
    primary_event:   str         # "criticism", "rejection", ...
    key_points:      list[str]   # что обязательно сказать
    blocked_topics:  list[str]   # что нельзя затрагивать
    style_hints:     list[str]   # как говорить
    language:        str         # ru/en/hy
```

LLM вербализует план — не изобретает структуру, только облекает в слова.

---

## 13. Движок поведенческих действий (behavioral_action_engine)

Детерминированный слой между контроллером и LLM. Вычисляет оценку каждого возможного действия:

```
ActionScore(action) =
      Σ(fear alignment × wf)
    + Σ(desire alignment × wd)
    + Σ(goal alignment × wg)
    + Σ(need alignment × wn)
    + Σ(value alignment × wv)
    + Σ(habit alignment × wh)
    + Σ(attachment alignment × wa)
    + Σ(shame pressure × ws)
    + current_trigger_sensitivity × wt
    + current_state_modifier × wx
    - internal_constraint_penalty × ci
    - social_constraint_penalty × cs
    - hard_constraint_penalty × ch   ← если > 0, действие ЗАПРЕЩЕНО
```

Жёсткие ограничения аннулируют действие полностью.

---

## 14. Классификатор безопасности (SafetyClassifier)

**Без LLM**, детерминированный, многоуровневый:

1. **Быстрый rule-based путь** — явные нелегальные/экстремальные сигналы → немедленная блокировка
2. **Извлечение признаков** — 10-мерный вектор (плотность ключевых слов, структурные, эвристические)
3. **KNN поиск** — k=7 ближайших соседей по косинусной метрике над векторами примеров
4. **Взвешенное большинство** → метка + уверенность

**Метки:** `safe → normal_response`, `suggestive → soft_filter`, `explicit → blur_or_generalize`, `illegal → block`

Если уверенность < 0.45 — эскалация к следующей более строгой метке.

---

## 15. Движок планирования (PlanningEngine)

Основной продуктовый режим системы — **адаптивный коуч**.

### Логика одного цикла
```
Сообщение пользователя
    → классификация типа хода (репорт / вопрос / рефлексия / обратная связь)
    → загрузка заметок + профиля личности + недавних результатов
    → вычисление сил, конфликтов, перегрузки
    → выбор ОДНОГО маленького следующего шага
    → выбор ОДНОГО резервного меньшего шага
    → выбор ОДНОГО уточняющего вопроса
    → PlanningOutput
```

LLM только вербализует готовый план — не строит его.

### Классификация обратной связи
| Метка | Смысл | Реакция системы |
|---|---|---|
| `success` | шаг сработал | усилить паттерн |
| `partial` | частично сработал | уменьшить размер шага |
| `failure` | не сработал | переключиться на резервный |
| `resistance` | пользователь изменил план | адаптироваться |
| `crisis` | кризис | стоп планирование, переключение на поддержку |
| `neutral` | нет сигнала | продолжить |

---

## 16. Learner важности (ImportanceLearner)

Учится **что пользователь считает важным сохранить**.

### Механика
- `/save` → позитивный пример
- Каждый непомеченный ход → слабый негативный пример
- Оценка хода: keyword score + длина + давность + личный контент
- `SUGGESTION_THRESHOLD = 0.45` — выше этого значения предлагает сохранить

### Что важно (по дизайну)
- Личные факты (биография, отношения)
- Заявленные цели
- Эмоциональные переломные моменты
- Провалы и успехи
- Принятые обязательства
- Явные просьбы запомнить

### Что шум
Короткие подтверждения, вопросы без личного содержания, приветствия.

---

## 17. Офлайн-обучение перцептронов (PerceptronTrainer)

```python
train_p1(rt, tuples):
    # обучает EventEncoder (P1 из когнитивного пайплайна V1)
    # только на размеченных (text, event_label) парах
    # минимум 20 примеров, иначе пропускает
    # 30 эпох, LR=0.01
    # взвешивание по |feedback|

train_p6(rt, tuples, genome):
    # обучает ActionPolicy (P6)
    # только на парах (текст, целевое_действие)
    # то же самое: 30 эпох, LR=0.01
```

`TrainingTuple` = `(session_id, text, event_label, action_label, feedback_score, timestamp)`

---

## 18. Память системы (Memory Architecture)

```
memory/
├── sessions/           — история диалогов
│   ├── *.txt           — legacy формат (одиночный файл)
│   └── _messages/      — modern JSONL (по сообщению, сохраняет user_persona_name)
│
├── graphs/             — граф знаний
│   ├── nodes.json      — все узлы
│   └── edges.json      — все рёбра
│
├── heads/              — персонажи (см. раздел 8)
│
├── importance_learner/ — паттерны важности
│   ├── global_examples.jsonl
│   └── {session_id}_examples.jsonl
│
├── message_annotations/ — операторские правки координат P1-P51
│   └── *.json
│
├── message_vector_models/ — обученные модели координат
│   └── p_registry_v1.json
│
├── personalities/      — профили пользователя (психологический портрет)
│
├── proposals/          — предложения по новым персонажам (ожидают проверки)
│
├── training_examples/  — обучающие примеры для персептронов
│
└── working/            — текущий рабочий контекст сессий
```

---

## 19. API (`/api/cognitive/`)

### Основные эндпоинты

| Метод | URL | Назначение |
|---|---|---|
| GET | `/health` | Статус системы |
| GET | `/sessions` | Список сессий |
| POST | `/sessions` | Создать сессию |
| GET | `/sessions/{id}` | Получить сессию с историей |
| DELETE | `/sessions/{id}` | Удалить сессию |
| POST | `/chat/respond` | Основной чат-эндпоинт |
| GET | `/graph` | Граф знаний |
| GET | `/graph/snapshots` | Снапшоты графа |
| GET | `/genome/{persona_id}` | Геном персонажа |
| POST | `/files/upload` | Загрузить файл (PDF/DOCX/TXT) |
| POST | `/sessions/{id}/annotations` | Сохранить операторскую правку |
| GET | `/debug/metrics` | Метрики системы |
| GET | `/debug/traces` | Трейсы запросов |
| GET | `/debug/graph-health` | Здоровье графа |

### ChatRequest (POST /chat/respond)
```json
{
    "message":          "текст",
    "session_id":       "session-xxx",
    "user_persona_id":  "катерина",    // slug персонажа
    "user_persona_name": "Катерина"    // отображаемое имя
}
```

---

## 20. Frontend (React + Vite)

```
webapp/
├── src/
│   ├── components/
│   │   ├── Chat/           — чат-интерфейс, ChatGraphPanel
│   │   ├── Graph/          — визуализация графа знаний
│   │   ├── Persona/        — карточки персонажей
│   │   ├── Inspector/      — инспектор P-векторов
│   │   ├── Trace/          — трейс пайплайна
│   │   ├── Studio/         — редактирование персонажей
│   │   ├── Rebuild/        — управление фоновым перестроением
│   │   ├── Operator/       — операторские правки
│   │   ├── Controller/     — управление контроллером
│   │   ├── Hypotheses/     — система гипотез
│   │   └── Editor/         — редактор заметок
│   └── api.js              — клиент к бэкенду
└── vite.config.js
```

---

## 21. Конфигурация рантайма

Ключевые env-переменные:

```bash
# Модели
LOCAL_GGUF_MODEL=Qwen3.5-2B.Q4_K_M.gguf
LOCAL_GGUF_N_CTX=7168
LOCAL_GGUF_MAX_TOKENS=2048
LOCAL_GGUF_MAX_LOADED=1          # сколько моделей держать в памяти

# Роли
COGNITIVE_CHAT_ROLE=general
COGNITIVE_RETHINK_ROLE=analyst

# Оркестрация чата
COGNITIVE_CHAT_ORCHESTRATION=single     # single|multi
COGNITIVE_CHAT_REVIEW_MODE=never        # never|always|on_failure

# Контекстные бюджеты (токены)
COGNITIVE_MAX_CONTEXT_TOKENS=3400
COGNITIVE_PERSONA_BLOCK_BUDGET=900
COGNITIVE_GRAPH_CONTEXT_BUDGET=1100
COGNITIVE_RECENT_DIALOGUE_BUDGET=320

# Какие этапы пускать через LLM (по умолчанию — ни одного, кроме шейпера)
COGNITIVE_STAGE_MODEL_STEPS=none        # none | response_shaper | state_reader | ...

# Облако (выключено)
CLOUD_LLM_ENABLE=0
```

---

## 22. Что система умеет решать

### 22.1. Стабильный персонаж в диалоге
Обычная LLM на 2B деградирует после 3–5 ходов. Эта система держит персонажа через:
- Частотные паттерны (`log_tuples`) — не просто описание, а выученные формулировки
- Ситуационные реакции (`situation_reactions`) — как реагировать на конкретный тип хода
- ЗАПРЕЩЕНО-правила — жёсткий запрет на определённые фразы
- Ремонт ответа при деградации

### 22.2. Психологически точная модель личности
53-параметровый геном + 10 регуляторов создают персонажей, у которых:
- Есть страхи и влечения (не просто "черты")
- Защитные механизмы меняются по ситуации
- Реакция на критику отличается от реакции на похвалу

### 22.3. Граф знаний и память
Факты о мире и отношениях не теряются между сессиями. Граф обновляется при каждом диалоге — система знает, кто такой пользователь, какие у него отношения, что происходило раньше.

### 22.4. Адаптивное планирование
В режиме коуча: отслеживает прогресс, адаптируется к успехам/провалам, предлагает микрошаги. Не даёт советов в кризис — переключается на поддержку.

### 22.5. Полная локальность и приватность
Никаких внешних API. Все данные — на машине пользователя. Подходит для личных дневников, конфиденциального консультирования, закрытых корпоративных систем.

### 22.6. Мультиязычность
Русский, английский, армянский, китайский — в паттернах признаков, детекции языка, локализационном движке.

### 22.7. Операторский контроль
Через веб-интерфейс: правка P-координат сообщений, исправление геномов, управление перестроением персонажей. Правки накапливаются и влияют на следующие обучения.

---

## 23. Нерешённые проблемы и ограничения

| Проблема | Корень | Статус |
|---|---|---|
| 2B LLM ломает персонажа на провокациях | RLHF 2B-модели перебивает persona prompt | Частично решено через примеры + ЗАПРЕЩЕНО |
| Фоновый rebuild перезаписывает persona_form.json | rebuild читает log_tuples → дериверует параметры | Решено заменой доминантной записи log_tuples |
| sexual_orientation и кастомные поля стираются rebuild-ом | rebuild не знает о нестандартных полях | Известная проблема, нет автофикса |
| Медленный первый ответ | llama.cpp холодный старт 1–3 сек | Частично решено prewarm |
| Контекстное окно 3400 токенов мало для длинных сессий | Модель 2B не тянет больше | Решается через P51-обрезку контекста |

---

## 24. Заключение

Persona-Graph-Agent — это не обёртка над ChatGPT и не RAG с LangChain. Это **собственная когнитивная архитектура**, где языковая модель — только один из компонентов.

Ключевое: система знает **что сказать** (через P1–P6, поведенческий движок, Speech Plan) ещё до того, как LLM открывает рот. LLM только переводит этот план в связный текст.

Система решает проблемы, которые не решаются prompt-инжинирингом:
- **Стабильность персонажа** на дешёвой 2B-модели — через обученные паттерны, геном и ремонт
- **Психологическая точность** — не "персонаж злой", а 53-параметровая модель с регуляторами и защитными механизмами
- **Долгосрочная память** — граф + персонажные головы + история сессий
- **Приватность** — 100% локально, без интернета
- **Обучаемость** — система учится на каждом диалоге: паттерны, геном, координаты сообщений

Текущий приоритет развития: превратить её в **локального адаптивного коуча** — персонаж + память + планирование + психологический профиль пользователя в одном рантайме.
