# ԳԼՈՒԽ 3. `PERSONA-GRAPH-AGENT` ՀԱՄԱԿԱՐԳԻ ՃԱՐՏԱՐԱՊԵՏԱԿԱՆ ԵՎ ԾՐԱԳՐԱՅԻՆ ՆԿԱՐԱԳՐՈՒԹՅՈՒՆԸ

## 3.1. Համակարգի նպատակը և հիմնական գաղափարը

Մշակված համակարգի նպատակը սովորական `LLM`-հենված chat պատասխանից անցնելն է դեպի վերահսկվող agent runtime, որտեղ յուրաքանչյուր հարցում անցնում է հստակ սահմանված ծրագրային փուլերով։ Համակարգը միավորում է՝

- երկխոսության հիշողություն,
- կառուցվածքային persona հիշողություն,
- knowledge graph,
- կոգնիտիվ վարքային pipeline,
- փաստաթղթերի ընդունում և մշակում,
- route-based response generation,
- validation և repair։

Հիմնական գաղափարն այն է, որ model-ը ինքնին համակարգը չէ։ Այն օգտագործվում է միայն այն տեղերում, որտեղ պետք է՝

- կառուցվածքային knowledge extraction,
- սահմանափակված response generation,
- reviewer/rewrite գործառույթ։

Մնացած որոշումները կատարվում են deterministic controller-ի կողմից։

Կանոնական գործառնական հոսքը հետևյալն է․

```text
request
-> controller interpretation
-> route selection
-> capability planning
-> cognitive pipeline (P1–P6)
-> bounded context assembly
-> state transition / response shaping
-> generation
-> validation
-> repair
-> persistence
```

Այս մոտեցումը կանխում է մի շարք կարևոր խնդիրներ՝

- persona description-ի essay-ի վերածումը,
- lightweight հարցումների անցումը ծանր graph pipeline-ով,
- prompt-ի գերբեռնվածությունը,
- persona registry-ի աղտոտումը file label-երով կամ prompt debris-ով,
- model-ի կողմից route-ի ինքնուրույն ընտրությունը,
- model-ի կողմից վարքային action-ի ինքնուրույն հորինումը։

## 3.2. `controller-first` runtime-ի կառուցվածքը

### 3.2.1. Request intake

Յուրաքանչյուր turn-ի համար ստեղծվում է `RequestEnvelope`, որը պարունակում է՝

- `request_id`
- `session_id`
- `raw_text`
- `normalized_text`
- `timestamp`

Այս շերտը ապահովում է request traceability և հետագա observability-ի հիմքը։

### 3.2.2. Controller interpretation

Նոր ճարտարապետության առանցքային շերտը `controller_runtime.py` մոդուլն է, որը մեկ հոսքի մեջ համադրում է՝

- `interaction frame`
- `message analysis`
- `request preprocessing`
- `route decision`
- `capability plan`

Արդյունքում route logic-ը այլևս ցրված չէ տարբեր մոդուլներով և հակասական կրկնություն չի ստեղծում։

### 3.2.3. Request preprocessing

`request_pipeline.py` մոդուլում request-ը ստանում է հետևյալ բնութագրերը՝

- `detected_language`
- `intent_type`
- `interaction_mode`
- `request_type`
- `persona_style_traits`
- `speech_style_hints`
- `clarification_needed`

Համակարգը տարբերակում է առնվազն հետևյալ request type-երը՝

- `factual_query`
- `roleplay_prompt`
- `persona_specification`
- `persona_assignment`
- `persona_analysis`
- `persona_chat`
- `document_request`
- `meta_previous_answer`
- `general_chat`
- `project_architecture_request`
- `clarification_request`

Սա կարևոր է, քանի որ, օրինակ, հարուստ persona dossier-ը պետք է ճանաչվի որպես `persona_specification`, ոչ թե սովորական chat turn։

### 3.2.4. Route selection

Յուրաքանչյուր request-ի համար ձևավորվում է `RouteDecision`, որը ներառում է՝

- `selected_route`
- `request_type`
- `requires_history`
- `requires_graph`
- `requires_persona`
- `requires_llm`
- `strict_grounding`
- `response_style`
- `validation_mode`
- `fast_path`

Ակտիվ հիմնական route-երն են՝

- `factual_answer`
- `lightweight_conversation`
- `hypothetical_roleplay`
- `persona_chat_fast_path`
- `persona_specification`
- `persona_assignment`
- `persona_dialogue_analysis`
- `persona_graph_reasoning`
- `project_document_analysis`
- `meta_previous_answer`
- `clarification_request`

Route-ը որոշում է ոչ միայն generation-ի ձևը, այլ նաև այն, թե որ context layer-երն են պետք և ինչ validation պետք է կիրառվի։

### 3.2.5. Capability planning

`CapabilityPlan` շերտը որոշում է, թե տվյալ request-ի համար ինչ է անհրաժեշտ՝

- history-ի բեռնում,
- graph retrieval,
- persona-ի բեռնում,
- heavy persona pipeline,
- context builder,
- deterministic reply,
- `LLM` generation,
- reviewer pass։

Այս մեխանիզմը թույլ է տալիս lightweight turn-երը չուղարկել անիմաստ ծանր հաշվարկի։

## 3.3. Համակարգի հիմնական ծրագրային մոդուլները

### 3.3.1. `chat_engine.py`

`chat_engine.py`-ը runtime orchestration layer-ն է։ Այն՝

- ստանում է request-ը,
- աշխատում է controller state-ի հետ,
- early fast path-ով սպասարկում է note command-երը,
- ընտրում է fast path կամ heavy path,
- **pre-flight-ով ստուգում է runtime-ի վիճակը** — եթե degraded, `LLM` call-ը շրջանցվում է andb behavioral fallback-ին ուղղակի անցնում,
- planning/regulator/safety/localization շերտերը միացնում է `route_guidance`-ի հետ,
- կառուցում է context-ը,
- կանչում է generation և reviewer փուլերը,
- կիրառում է validation և repair,
- գրում է session history,
- թարմացնում է route memory-ը,
- գրանցում է trace և response metadata observability store-ում։

### 3.3.2. `controller_runtime.py`

Այս մոդուլը կենտրոնացնում է request interpretation-ը մեկ հոսքի մեջ։ Այն ապահովում է, որ analysis, preprocessing, route selection և capability planning փուլերը օգտագործեն նույն input view-ը և միմյանց հետ չմտնեն հակասության մեջ։

### 3.3.3. `request_pipeline.py`

Այս մոդուլը պատասխանատու է՝

- request classification-ի,
- route selection-ի,
- route guidance-ի,
- deterministic fast-path reply-ների,
- validation rule-երի համար։

### 3.3.4. `cognitive_pipeline.py` և `genome.py`

Կոգնիտիվ pipeline-ը 6-փուլ deterministic ճարտարապետություն է, որն աշխատում է ամեն heavy persona turn-ի վրա։

**P1 — EventEncoder**․ Ինկոդում է incoming text-ը event probability vector-ի և intensity signal-ի մեջ։ 14 event type՝ `neutral`, `threat`, `reward`, `overload`, `shame_trigger`, `loss_of_control`, `failure`, `criticism`, `rejection`, `intimacy`, `opportunity`, `boredom`, `novelty`, `uncertainty`։

**P2 — TriggerNetwork**․ Ակտիվացնում է genome-derived trigger weights-ը event vector-ի և intensity-ի դեմ։

**P3 — RegulatorCell**․ GRU-like cell, որը թարմացնում է 10-չափ regulator state-ը (anxiety, motivation, fatigue, shame, frustration, guilt, closeness, hope, emptiness) trigger activations-ից և genome-ից։

**P4 — ThoughtMLP**․ Արտադրում է 16-չափ thought vector trigger-ներից և regulator state-ից։ Ներառում է perceived risk, confidence, need dimensions (connection / achievement / safety) և frame dimensions (approach / hold / retreat)։

**P5 — ConflictScorer**․ Scoring 8 resolution strategies-ի (avoidance, overcompensation, attack, freeze, planning, support-seeking, self-deception)։

**P6 — ActionPolicy**․ MLP, ընտրում է 14 action family-ներից մեկը (approach, avoid, freeze, attack, placate, analyze, ask_for_help, seek_control, reduce_exposure, reframe, self_protect, connect, withdraw, plan_small_step)։

`PersonalityGenome`-ը պահում է persona-ի կայուն trait-ները float field-երով՝ fears, defense mechanisms, vulnerabilities, drive profile։

`CognitiveRuntime.__init__`-ը օգտագործում է fixed seed weight initialization-ի համար, isolated from global numpy state, ապահովելով deterministic արդյունքներ test order-ից անկախ։

### 3.3.5. `cognitive_authority.py` և `speech_planner.py`

`CognitiveAuthority`-ը scoring-ի հիման վրա ընտրում է generation mode-ը՝

- `pure_llm` (score < 0.20): pipeline-ը բավական ակտիվ չէ, legacy prompt-ն է օգտագործվում
- `hint` (score < 0.55): cognitive hint-ը inject է արվում `route_guidance`-ի մեջ
- `planner` (score ≥ 0.55): `SpeechPlanner` path — ամբողջ կառուցվածքը pipeline-ից է գալիս

`SpeechPlanner.build()`-ը `CognitiveTurnOutput`-ը վերածում է structured `SpeechPlan`-ի, ներառելով action name, speech goal, tone, key points, blocked topics, style hints, max tokens։

`verbalizer_prompt(plan)`-ը կառուցում է verbalization prompt-ը, ուր PERSONA + RECENT EXCHANGE → system role, SPEECH DIRECTIVE → user role (разделено `"User question:"` separator-ով)։

Planner mode-ը ակտիվ է միայն score ≥ 0.55 դեպքում — session history-ի և graph data-ի կուտակումից հետո։ Ранние turn-երի համար legacy prompt + cognitive hint-ն ավելի հուսալի է 2B model-ների համար։

### 3.3.6. `state_transition_runtime.py`

Այս մոդուլը orchestrate է անում LLM-guided enrichment stages-ի շարք, որոնք աշխատում են context building-ի և generation-ի միջև։

Փուլերն են՝

- `state_reader` — կարդում է persona state snapshot-ը
- `persona_update` — թարմացնում է emotion vector-ը և situation reactions-ը
- `bounded_state_transition` — ընտրում է next active role, risk posture, mood signals
- `context_curator` — curate է working context layer-ը
- `context_reviewer` — review-ում է context-ը հակասությունների և priorities-ի համար
- `response_shaping` — ձևավորում է response style, behavior mode, constraints

Ամեն փուլ կանչում է `call_json_model_for_role` և gracefully fallback է deterministic output-ի, երբ `LLM`-ը useful բան չի վերադարձնում։ Ամեն փուլ independently gate է `COGNITIVE_STAGE_MODEL_STEPS` env variable-ով։

### 3.3.7. `reliability.py`

`reliability.py`-ը ապահովում է atomic rollback guarantees և runtime health reporting։

`StorageWriteFailure` — raise-ի է, երբ graph կամ persona write-ը ձախողվում է ճանապարհի կեսին։ Caller-ը restore է անում նախորդ snapshot-ը։ `graph_store.save_graph()` և `persona_engine.materialize_persona()` snapshot-ի են state-ը write-ից առաջ և rollback-ի են ձախողման դեպքում։

`MutationRejectedFailure` — raise-ի է, երբ node rethink-ը apply-ի է description update, բայց հաջորդ graph mutation-ը (link connection) ձախողվում է։ Node description-ը rollback-ի է, snapshot path-ը include-ի է `details`-ում։

`runtime_status_snapshot()` — inspect-ի է local `LLM` provider-ը, վերադարձնում է `mode` (`full` կամ `degraded`) dict — pre-flight check generation-ից առաջ։

### 3.3.8. `head_caller.py`

`head_caller.py`-ը պատասխանատու է persona head-ի ընտրության համար։

Այն՝

- prepare-ի է candidate heads-ը analysis-ից, classifications-ից, graph store-ից,
- ընտրում է primary head-ը,
- normalize-ի է personality name-ը։

### 3.3.9. `node_rethinker.py`

`node_rethinker.py`-ը graph node-ի intelligent rethink-ի engine-ն է։

Այն՝

- կանչում է `LLM`-ը node description-ը բարելավելու համար,
- suggest-ի է նոր link-եր,
- apply-ի է changes graph-ում,
- rollback-ի է ամբողջ mutation-ը, եթե link connection-ը ձախողվում է (reliability layer-ի կողմից)։

### 3.3.10. `history_store.py`

`history_store.py`-ը պահպանում է՝

- session log-երը,
- route state sidecar file-երը,
- session continuity-ի համար անհրաժեշտ տվյալները,
- session delete lifecycle-ը։

### 3.3.11. `persona_engine.py`

`persona_engine.py`-ը համակարգի առանցքային շերտերից է։ Այն իրականացնում է՝

- persona creation,
- persona validation,
- registry hygiene,
- rejected candidate-ների quarantine,
- persona storage և activation,
- snapshot-based rollback `materialize_persona`-ում։

Persona-ն պահվում է որպես կառուցվածքային object, որի հիմնական դաշտերն են՝

- `identity`
- `core_goal`
- `secondary_goals`
- `fears`
- `needs`
- `constraints_internal`
- `constraints_social`
- `constraints_hard_system`
- `allowed_methods`
- `maladaptive_methods`
- `core`
- `conflict`
- `defense`
- `behavior`
- `dynamics`
- `meta`

Believable behavior-ը ստացվում է հետևյալ կոմբինացիայից․

```text
goal + fears + constraints + methods + trigger
```

### 3.3.12. `graph_store.py`

`graph_store.py`-ը knowledge graph memory-ի file-first շերտն է։ Այն ապահովում է՝

- nodes/edges merge,
- duplicate resolution,
- lifecycle hygiene,
- session-aware provenance,
- graph retrieval,
- node explanation view,
- snapshot-before-write + rollback-on-failure (reliability layer-ի կողմից)։

### 3.3.13. `context_builder.py`

`context_builder.py`-ը իրականացնում է bounded context packing։

Հիմնական տրամաբանությունը հետևյալն է․

```text
collect -> score -> rank -> pack
```

Context priority-ն հետևյալն է՝

1. ընթացիկ turn-ի պահանջը,
2. current session evidence,
3. active persona block,
4. local graph evidence,
5. միայն դրանից հետո global graph evidence։

### 3.3.14. `file_ingestion.py`

Աջակցվող ձևաչափերն են՝ `txt`, `md`, `json`, `csv`, `pdf`, `docx`, `odt`, `fb2`։

`pdf` ֆայլերի դեպքում կիրառվում է section-aware extraction։

### 3.3.15. `llm.py` և `prompt_builder.py`

`llm.py`-ը unified inference wrapper-ն է, իսկ `prompt_builder.py`-ը կատարում է compact prompt construction։

Այս շերտը ներառում է՝

- token budgeting,
- section budget-եր,
- reserved output budget,
- compatibility-aware role selection,
- reviewer orchestration,
- thinking model support (`_strip_think_blocks`)։

Thinking model-ները (Qwen3.5-2B, Nanbeige4.1-3B) emit-ի են internal reasoning block-ներ։ `_strip_think_blocks()` handle-ի է 3 output format — full `<think>...</think>`, template-hidden (`reasoning\n</think>\n\nAnswer`), truncated unclosed block — և վերադարձնում է միայն answer portion-ը։

### 3.3.16. `notes_store.py` և `planning_engine.py`

`notes_store.py`-ը ապահովում է lightweight manual note system, totally separated from graph memory-ից և session history-ից։

Command-երը՝ `/save <text>`, `/notes`, `/del_note <index|id>`, `/clear_notes` — мошакатывается безо LLM call-ի։

`planning_engine.py`-ը ստեղծում է bounded planning structure, ուր `LLM`-ը verbalize-ի է արդեն ընտրված frame-ը, ոչ թե ինքն ընտրում է կառուցվածքը։

### 3.3.17. `situation_regulator.py` և `behavioral_action_engine.py`

`situation_regulator.py`-ը event → regulator → action architecture-ն է legacy path-ի համար (non-persona routes)։

`behavioral_action_engine.py`-ը ապահովում է behavioral fallback decisions, ընտրելով strategy, confidence, risk level, uncertainty level — degraded կամ model-failure վիճակներում։

### 3.3.18. `importance_learner.py`

`importance_learner.py`-ը collect-ի է positive signals-ը `/save` command-ից, weak negative signals-ը non-saved turn-երից, ու stage 2-ի համար suggestion candidates-ը prepare-ի է per-session importance profile-ի հիման վրա։

### 3.3.19. `personality_schema.py` և `personality_store.py`

Ավելի խոր `PersonalityObject` construction model, ուր psychological profile field-երը, biography fact-երը, temporary state-ը, provenance-ը, conflict record-երը ամբողջությամբ տարանջատված են։

### 3.3.20. `observability.py` և `training_examples_store.py`

`observability.py`-ը գրանցում է request trace-եր, stage timing-եր, fallback reason-եր, context token estimate-ներ, response metadata, rebuild/rethink counter-ներ։

`training_examples_store.py`-ը թույլ է տալիս պահել `(input, correct_output)` օրինակներ և JSONL ձևաչափով export-ի անել fine-tuning dataset-երի համար։

## 3.4. Persona architecture և behavior control

### 3.4.1. Persona registry hygiene

Համակարգում persona registry-ը չի ընդունում file/media label-եր, ontology junk (`Human`, `File`, `PDF`), raw prompt fragment-եր, behaviorless noun-եր, extraction debris-ը կամ random entity leftover-ներ։ Rejected candidate-ները պահվում են quarantine log-երում։

### 3.4.2. Persona readiness

Persona object-ները տարբերակվում են ըստ readiness-ի՝ `seed`, `draft`, `full`։

### 3.4.3. Fragile persona validator

Fragile կամ shame-based persona-ների համար համակարգը ստուգում է, որ պատասխանը չափազանց խելացի, lecture-like, երկար կամ ինքնավստահ չլինի։

## 3.5. Prompt packing և generation orchestration

### 3.5.1. Compact prompt packing

Համակարգը compact-ի է persona block-ը, session history-ն, graph evidence-ը, instruction block-ը — maksimum useful density-ի, ոչ թե maximum mass-ի։

### 3.5.2. Primary + reviewer generation

Runtime-ը supports-ի է `single`, `primary_with_reviewer`, `alternate`, `randomized` orchestration mode-եր։

Reviewer-ն օգտագործվում է route mismatch detection-ի, style mismatch repair-ի, truncation repair-ի, invalid draft rewrite-ի համար։

### 3.5.3. Degradation detection

`runtime_status_snapshot()`-ի pre-flight check-ը generation-ից **ԱՌԱՋ** detect-ի է degraded LLM runtime-ը։ Degraded դեպքում `LLM` call-ը skip-ի է entirely, behavioral fallback-ն է ընտրվում — `fallback_reason = 'dependency_unavailable'`։ String comparison-ը (`reply == generic_fallback`) ամբողջությամբ eliminated-ն է։

### 3.5.4. Validation and repair

Generation-ից հետո՝ route consistency, persona consistency, truncation, repetition loop, grounding mismatch, style mismatch, fallback reason correctness ստուգումները։ Ձախողման դեպքում pipeline-ը apply-ի է regeneration, style-guard regeneration, reviewer rewrite, կամ deterministic fallback։

## 3.6. Storage layout

```text
memory/
  sessions/
    {session_id}.txt
    _route_state/
  notes/
    {session_id}.jsonl
  personalities/
    personalities_index.json
    {personality_id}.json
  training_examples/
    global.jsonl
    {session_id}.jsonl
  importance_learner/
    global_examples.jsonl
    {session_id}_examples.jsonl
  files/
    uploaded_documents/
      {session_id}/
  graphs/
    nodes.json
    edges.json
  heads/
    index.json
    {persona_slug}/
  archive/

runtime/
  current_context/
  logs/
  system_realism_reports/
```

## 3.7. Operator interface

Frontend-ը կառուցված է operator workspace-ի տեսքով, ներառելով session list, persona selection/inspection, personality delete, training-example curation, file upload, chat surface, graph workspace, graph rethink preview/apply workflow, debug traces, diagnostics։

Ներկայում frontend-ն առանձին UI surface չունի `/notes` և planning-mode output-ի համար, բայց backend API-ն արդեն հասանելի է։

## 3.8. Թեստավորում և ընթացիկ արդյունքները

Backend test suite-ը գտնվում է `tests/agent_system/` շերտում և охватывает՝

- routing correctness,
- cognitive pipeline — attractor tests-ը real datasets-ի վրա,
- persona creation, validation, registry hygiene,
- graph merge, retrieval, lifecycle, hygiene, localizer,
- node rethinker — rollback correctness,
- reliability — StorageWriteFailure, MutationRejectedFailure rollback-ներ,
- state transition runtime,
- behavioral fallback decisions,
- social persona system,
- task procedures,
- trace learning,
- file ingestion,
- compact context packing,
- reviewer orchestration,
- planning mode և notes fast path,
- behavior regulation,
- personality construction integrity,
- observability и trace learning,
- training example lifecycle,
- API behavior,
- memory lifecycle,
- LLM runtime — fallback_chat_reply contract, thinking model stripping,
- local LLM provider policy,
- semantic routing, interaction routing, request pipeline, controller runtime,
- fallback, degradation, repair logic։

```text
.venv/bin/python -m pytest --collect-only -q tests/agent_system
382 tests collected

.venv/bin/python -m pytest -q tests/agent_system
382 passed
```

Ամբողջ suite-ն անցնում է 0 failure-ով։

Frontend build-ի ստուգման արդյունքը՝

```text
npm --prefix webapp run build
passed
```

## 3.9. Գլխի եզրակացություն

Կատարված նախագծային և ծրագրային աշխատանքի արդյունքում ձևավորվել է controller-first `Persona-Graph-Agent` runtime, որտեղ՝

- request-ը նախ դասակարգվում է,
- route-ն որոշում է ամբողջ հետագա հոսքը,
- **cognitive pipeline P1–P6**-ը deterministic ձևով հաշվում է behavioral state-ն իր turn-ում,
- `CognitiveAuthority`-ը ընտրում է `pure_llm`, `hint`, կամ `planner` generation mode,
- `SpeechPlanner`-ը `planner` mode-ում կառուցում է structured speech plan, LLM-ը verbalize-ի է անում,
- persona-ն պահվում է կառուցվածքային object-ի տեսքով,
- graph memory-ն ծառայում է long-term semantic layer-ի տեսքով,
- **reliability layer**-ը atomic rollback-ներ ապահովում է graph/persona write ձախողումների դեպքում,
- **degradation detection**-ն pre-flight check-ի է runtime-ի վիճակը LLM call-ից ԱՌԱՋ,
- **thinking model support**-ն handle-ի է `<think>...</think>` blocks-ը local 2B–3B model-ների output-ում,
- compact prompt packing-ը կանխում է context overload-ը,
- validator և reviewer շերտերը բարձրացնում են final answer-ի վերահսկելիությունը։

Ստացվել է ոչ թե chat interface, այլ բազմաշերտ bounded runtime, հարմարեցված փոքր context window ունեցող local thinking model-ների գործնական օգտագործման համար։
