# ԳԼՈՒԽ 3. PERSONA-GRAPH-AGENT ՀԱՄԱԿԱՐԳԻ ՃԱՐՏԱՐԱՊԵՏԱԿԱՆ ԵՎ ԾՐԱԳՐԱՅԻՆ ՆԿԱՐԱԳՐՈՒԹՅՈՒՆԸ

## 3.1. Առաջարկվող համակարգի ընդհանուր գաղափարը

Այս աշխատանքում ներկայացվում է `Persona-Graph-Agent` համակարգը, որը կառուցված է ոչ թե անմիջական `հաղորդագրություն -> պատասխան` սկզբունքով, այլ վիճակային անցումների վերահսկվող ճարտարապետությամբ։

Համակարգի հիմնական գաղափարն այն է, որ օգտատիրոջ հաղորդագրությունը դիտարկվում է որպես ազդեցություն համակարգի ընթացիկ վիճակի վրա։ Հաղորդագրությունը պետք է պարզի.

- ինչ է փոխվում ընթացիկ վիճակում,
- որ persona-ն է խոսողը,
- որն է քննարկվող թեման,
- ինչ ռիսկեր և առաջնահերթություններ են ակտիվանում,
- ինչ նյութեր պետք է մտնեն ընթացիկ աշխատանքային համատեքստ։

Միայն այս փուլերից հետո է թույլատրվում ձևավորել վերջնական պատասխանը։

Համակարգի ընդհանուր սկզբունքը հետևյալն է.

```text
օգտատիրոջ հաղորդագրություն
-> ընթացիկ վիճակի ընթերցում
-> ազդեցության մեկնաբանում
-> սահմանափակ state transition
-> working context-ի կառուցում
-> working context-ի review
-> response shaping
-> final generation
-> transition logging
-> current context persistence
```

Այստեղ լեզվային մոդելը դիտարկվում է որպես ստորադաս հաշվարկային բաղադրիչ, ոչ թե որպես ամբողջ համակարգը ղեկավարող օբյեկտ։

Համակարգի կառուցման նպատակը երեք հիմնական խնդիր լուծելն է.

- persona-ն ներկայացնել որպես կառուցվածքային և երկարատև օբյեկտ,
- հիշողությունը բաժանել հստակ շերտերի,
- պատասխանն ստանալ ոչ թե անմիջապես մուտքային հարցից, այլ վերանայված աշխատանքային համատեքստից։

Գործնական տեսանկյունից սա նշանակում է, որ runtime-ը պետք է կարողանա.

- պահպանել նույնականությունը երկար session-ների ընթացքում,
- տարբերակել խոսող persona-ն և քննարկվող թեման,
- չխառնել long-term memory-ն և current context-ը,
- թույլ չտալ, որ model-ը ինքնուրույն վերագրի իրեն նոր բնութագրեր կամ վերագրի չստուգված փաստեր graph-ին,
- պահել պատճառական կապը state, memory և final answer-ի միջև։

Այս մոտեցումը հատկապես կարևոր է persona-driven համակարգերի դեպքում, քանի որ այստեղ պատասխանի աղբյուրը միայն knowledge base-ը չէ։ Պատասխանի ձևավորման աղբյուր են նաև.

- persona traits-ը,
- decision patterns-ը,
- learned interaction patterns-ը,
- session continuity-ը,
- current role-ը,
- mood dynamics-ը։

## 3.2. Persona-Graph-Agent համակարգի կառուցվածքային նկարագրությունը

### 3.2.1. Համակարգի հիմնական շերտերը

Գործող համակարգը կազմված է հետևյալ հիմնական շերտերից.

1. `persona structure`
2. `graph logic`
3. `file-based memory`
4. `state transition runtime`
5. `current working context layer`
6. `staged prompt system`
7. `mood research layer`
8. `operator UI և diagnostics`

Այս շերտերը համատեղ ապահովում են, որ համակարգը չդառնա սովորական polite assistant, այլ գործի որպես վերահսկվող persona-runtime։

### 3.2.2. Persona կառուցվածքը

Persona-ն համակարգում ներկայացված է որպես բազմաշերտ stateful կառուցվածք։ Այն բաժանված է երեք հիմնական շերտերի.

- `baseline definition`
- `dynamic emotional state`
- `learned interaction patterns`

Այս բաժանումը թույլ է տալիս տարբերակել.

- what the persona is,
- how the persona currently feels,
- what the persona has learned from interaction.

Persona-ի կառուցվածքում առկա են.

- traits,
- roles,
- habits,
- reaction patterns,
- values,
- conflicts,
- topic affinities,
- speech tendencies,
- memories,
- relations,
- decision patterns,
- local graph links։

Persona subsystem-ի կարևոր առանձնահատկությունն այն է, որ այն չի պահվում մեկ միասնական “description” դաշտում։ Յուրաքանչյուր շերտ ունի իր դերը.

- baseline-ը պահպանում է համեմատաբար կայուն նույնականությունը,
- dynamic state-ը պահպանում է ընթացիկ հուզական և իրավիճակային փոփոխությունները,
- learned patterns-ը պահպանում է փոխազդեցությունից ստացված սահմանափակ սովորած վարքագիծը։

Այս բաժանումը թույլ է տալիս միաժամանակ պահպանել persona-ի կայունությունը և թույլատրել սահմանափակ ադապտացիա։

### 3.2.3. Graph memory

Graph memory-ն հանդիսանում է երկարաժամկետ կառուցվածքային հիշողության հիմնական շերտը։ Այն պահվում է `memory/graphs/` կատալոգում և բաղկացած է.

- `nodes.json`
- `edges.json`

Graph layer-ը պատասխանատու է.

- գիտելիքի կառուցվածքային պահպանման,
- entity և relation grounding-ի,
- duplicate resolution-ի,
- node lifecycle-ի,
- hygiene-ի,
- rethink preview/apply հոսքերի համար։

Գրաֆի հանգույցների համար կիրառվում են lifecycle states.

- `active`
- `weak`
- `suspect`
- `archived`
- `merged`

Graph-ը համակարգում կատարում է միանգամից մի քանի դեր.

- entity grounding,
- knowledge organization,
- persona-local relation storage,
- context support,
- operator inspection surface։

Համակարգում կարևոր է, որ graph-ը դիտվում է որպես semantic memory layer, ոչ թե միայն visualization resource։

### 3.2.4. Հիշողության շերտերը

Համակարգի հիշողությունը բաժանված է մի քանի մակարդակի.

- `working memory`
- `session memory`
- `persona memory`
- `graph knowledge memory`
- `archive / cold memory`

Այս շերտավորումը թույլ է տալիս մեկտեղել.

- ընթացիկ երկխոսության ակտիվ բովանդակությունը,
- երկարաժամկետ persona knowledge-ը,
- graph-grounded facts-ը,
- արխիվային և ցուրտ storage-ը։

Layered memory model-ը կանխում է երկու հիմնական խառնաշփոթ.

- երբ ամբողջ անցյալը միանգամից լցվում է prompt-ի մեջ,
- երբ համակարգը չի տարբերակում, թե որն է ընթացիկ ակտիվ context-ը, իսկ որը՝ արխիվային նյութ։

### 3.2.5. Mood research layer

Համակարգում առկա է նաև ֆոնային mood research շերտ, որի նպատակն է ուսումնասիրել.

- օգտատիրոջ վարքային և հուզական ազդակները,
- persona-ի ներքին դինամիկան,
- mood cluster-ները,
- role choice-ի և response style-ի կապերը։

Այս շերտը պահվում է `memory/mood_research/` կատալոգում և ապահովում է.

- snapshots,
- clustering,
- transition analysis,
- interpretable summaries,
- role-effect reports։

Mood research layer-ը runtime-ում օգտագործվում է ոչ թե որպես ազատ մեկնաբանող agent, այլ որպես աջակցող analytical layer, որը կարող է ազդել.

- social role ընտրության,
- uncertainty posture-ի,
- response style-ի,
- diagnostics-ի վրա։

### 3.2.6. Operator interface

Frontend-ը կառուցված է որպես operator workspace, ոչ թե որպես պարզ chat window։ Այն ներառում է առանձին մակերեսներ հետևյալ խնդիրների համար.

- chat,
- graph workspace,
- persona inspection,
- file ingestion,
- diagnostics։

Operator interface-ի առկայությունը ճարտարապետական տեսանկյունից կարևոր է, քանի որ համակարգը նախատեսված է ոչ միայն runtime execution-ի, այլ նաև դիտարկելիության, ստուգման և վերահսկման համար։

## 3.3. Persona-Graph-Agent համակարգի ծրագրային նկարագրությունը

### 3.3.1. Գլխավոր runtime path-ը

Համակարգի հիմնական գործարկման կետը `start.py` ֆայլն է։ Գործող runtime path-ը հետևյալն է.

```text
start.py
  -> bootstrap_runtime_environment()
  -> get_runtime_config()
  -> src.web.combined_app.create_combined_app()
    -> agent_system.api.create_app()
    -> src.web.api.attach_frontend_routes()
```

Այսպիսով, տեղային գործարկման դեպքում համակարգը իրենից ներկայացնում է combined app, որտեղ միևնույն runtime միջավայրում միավորված են.

- backend API,
- frontend routes,
- operator UI։

Այս runtime path-ը կարևոր է, որովհետև այն ցույց է տալիս, որ համակարգի իրական աշխատանքային միջավայրը միասնական է. backend-ը, diagnostics-ը և operator surface-ը միմյանցից անջատ չեն։

### 3.3.2. Backend-ի հիմնական մոդուլները

Գլխավոր backend տրամաբանությունը կենտրոնացած է `agent_system/` կատալոգում։

Հիմնական ակտիվ մոդուլներն են.

- `chat_engine.py`
- `interaction_routing.py`
- `message_analyzer.py`
- `situation_engine.py`
- `state_transition_runtime.py`
- `context_builder.py`
- `persona_engine.py`
- `graph_store.py`
- `llm.py`
- `prompt_builder.py`
- `history_store.py`
- `mood_research.py`
- `observability.py`
- `reliability.py`

Այս մոդուլների պատասխանատվությունները պայմանականորեն կարելի է բաժանել հինգ խմբի.

1. orchestration and runtime,
2. state interpretation,
3. persona and behavior,
4. graph and memory,
5. observability and failure handling։

### 3.3.3. Chat runtime-ի հիմնական քայլերը

Մեկ chat turn-ի ընթացքում համակարգը կատարում է հետևյալ քայլերը.

1. ստեղծում կամ բեռնում է session,
2. բեռնում է ընթացիկ state snapshot-ը,
3. route-ավորում է interaction-ը,
4. վերլուծում է հաղորդագրությունը,
5. կառուցում է structured situation,
6. ընտրում կամ materialize է persona-head-ը,
7. թարմացնում է persona dynamic state-ը,
8. կառուցում է bounded context,
9. review-ում է context-ը,
10. ձևավորում է response plan-ը,
11. կանչում է final generator-ը,
12. գրում է session history, current context և transition log։

Այս հերթականությունը կարևոր է, որովհետև final answer-ը ստացվում է արդեն վերափոխված և review եղած context-ից, ոչ թե անմիջապես raw user input-ից։

### 3.3.4. Interaction routing

`interaction_routing.py` մոդուլը առանձնացնում է.

- խոսող persona-ն,
- քննարկվող entity-ն,
- follow-up mode-ը,
- explicit persona switch-ը։

Այս մոտեցումը կարևոր է, քանի որ համակարգը պետք է տարբերակի օրինակ հետևյալ երկու դեպքերը.

- երբ օգտատերը խոսում է տվյալ persona-ի հետ,
- երբ օգտատերը խոսում է մեկ persona-ի հետ, բայց հարցնում է մեկ այլ թեմայի կամ կերպարի մասին։

Այս բաժանումը թույլ է տալիս լուծել context continuity-ի այն խնդիրները, որոնք սովորաբար առաջանում են pronoun follow-up հարցերում, persona switch-ի դեպքերում և topic continuity-ի ընթացքում։

### 3.3.5. Context builder

`context_builder.py` մոդուլը աշխատում է հետևյալ deterministic pipeline-ով.

```text
collect -> score -> rank -> compress -> pack
```

Context sources-ը ներառում են.

- session short-term history,
- persona memory,
- persona triad,
- global graph facts,
- local graph neighborhood,
- file-ingested knowledge,
- social role,
- mood research։

Context builder-ի նպատակը ոչ թե “շատ բան հավաքելն” է, այլ ճիշտ բաները սահմանափակ քանակով ընտրելն ու pack անելն այնպես, որ final generator-ը ստանա պատճառականորեն օգտակար context։

### 3.3.6. Staged prompt system

Համակարգում չի օգտագործվում մեկ մեծ prompt։ Փոխարենը կիրառվում է staged prompt architecture, որի հիմնական փուլերն են.

- `INTERACTION_ROUTER`
- `STATE_READER`
- `INFLUENCE_INTERPRETER`
- `STATE_TRANSITION_GUIDE`
- `CONTEXT_CURATOR`
- `CONTEXT_REVIEWER`
- `RESPONSE_SHAPER`
- `FINAL_GENERATOR`

Այս փուլերի prompt-երը պահվում են `agent_system/prompts/` կատալոգում։

Staged prompt system-ը առանձնացնում է.

- state reading,
- influence interpretation,
- bounded transition guidance,
- context curation,
- context review,
- response shaping,
- final generation։

Այսպիսով, model-ը չի ստանում մեկ խառնված prompt, այլ մասնակցում է սահմանափակ և ստուգելի փուլերի։

### 3.3.7. Current working context-ի առանձին պահպանում

Համակարգում current working context-ը պահվում է առանձին operational շերտում.

- `runtime/current_context/current_context.json`
- `runtime/current_context/current_context.txt`

Իսկ state transition history-ն պահվում է.

- `runtime/logs/state_transitions.jsonl`

Այս լուծումը թույլ է տալիս չխառնել.

- long-term memory-ն,
- session history-ն,
- և ընթացիկ ակտիվ context-ը։

Current context layer-ի առանձին պահպանումը նաև թույլ է տալիս operator-ին տեսնել, թե կոնկրետ որ context-ն է օգտագործվել տվյալ turn-ի ժամանակ։

### 3.3.8. Reliability և observability

Համակարգում ներդրված են նաև reliability և observability շերտեր։

`reliability.py` ապահովում է.

- degraded mode support,
- snapshot-before-mutation logic,
- rollback / recovery paths,
- operator-visible failure signaling։

`observability.py` ապահովում է.

- request tracing,
- stage timing,
- fallback tracking,
- graph health diagnostics,
- debug endpoints։

## 3.4. Պահոցների և ֆայլային կառուցվածքի նկարագրությունը

### 3.4.1. Գլխավոր ակտիվ կատալոգները

Գործող runtime-ի համար հիմնական ակտիվ շերտերն են.

- `start.py`
- `agent_system/`
- `src/web/`
- `src/utils/`
- `webapp/`
- `memory/`
- `runtime/`
- `tests/agent_system/`
- `tests/system_realism/`

Այս կատալոգներն են կազմում current canonical runtime-ի հիմնական տարածքը, և հենց դրանց վրա պետք է հենվել համակարգի ծրագրային նկարագրությունը գրելու ժամանակ։

### 3.4.2. Persona storage

Persona head-երը պահվում են `memory/heads/{head_slug}/` կառուցվածքով։

Տիպային persona bundle-ը ներառում է.

- `baseline.json`
- `dynamic_state.json`
- `learned_patterns.json`
- `traits.json`
- `relations.json`
- `examples.json`
- `emotion_vector.json`
- `knowledge.txt`
- `log_tuples.json`
- `persona_form.json`
- `decision_explanation.txt`
- `revisions.json`
- `meta.json`
- `local_graph.json`

Այս կառուցվածքը ցույց է տալիս, որ persona-ն համակարգում ոչ թե մեկ տեքստային նկարագրություն է, այլ բազմաֆայլ state bundle։

### 3.4.3. Session storage

Session history-ն պահվում է `memory/sessions/` կատալոգում։

### 3.4.4. Graph storage

Graph storage-ի հիմնական ֆայլերն են.

- `memory/graphs/nodes.json`
- `memory/graphs/edges.json`

### 3.4.5. File ingestion storage

Բեռնված փաստաթղթերը պահվում են.

- `memory/files/uploaded_documents/`

### 3.4.6. Archive storage

Սառը և վերականգնելի storage շերտերը պահվում են.

- `memory/archive/`

### 3.4.7. Runtime operational artifacts

Runtime-ի ընթացքում ստեղծվող գործող ֆայլերը պահվում են առանձին շերտում.

- `runtime/current_context/current_context.json`
- `runtime/current_context/current_context.txt`
- `runtime/logs/state_transitions.jsonl`
- `runtime/system_realism_reports/`

Այս շերտը կարևոր է, քանի որ այստեղ պահպանվում են ոչ թե long-term memory օբյեկտները, այլ ընթացիկ աշխատանքի և ստուգման արտեֆակտները։

## 3.5. Թեստավորում և ընթացիկ փորձնական արդյունքները

### 3.5.1. Backend regression tests

Համակարգի հիմնական backend test layer-ը գտնվում է.

- `tests/agent_system/`

Այստեղ ստուգվում են.

- chat runtime-ը,
- state transition-ը,
- interaction routing-ը,
- memory lifecycle-ը,
- graph lifecycle-ը,
- context pipeline-ը,
- reliability-ը,
- API failures-ը։

Վերջին փաստացի արդյունքը.

```text
python3 -m pytest tests/agent_system -q
91 passed, 6 skipped
```

### 3.5.2. End-to-end realism tests

Կենդանի runtime-level ստուգումները կատարվում են.

- `tests/system_realism/`

Այստեղ ստուգվում են.

- իրական startup path-ը,
- endpoint reachability-ը,
- persona materialization-ը,
- live dialogue behavior-ը,
- persona fidelity-ը,
- memory continuity-ը,
- contradiction handling-ը,
- mutation/evolution scenarios-ը։

Վերջին փաստացի արդյունքը.

```text
python3 -m pytest tests/system_realism -q
18 passed
```

Այս երկու test layer-երը միասին տալիս են հետևյալ պատկերը.

- backend logic-ը ստուգվում է deterministic regression-ներով,
- live runtime behavior-ը ստուգվում է realism harness-ով,
- persona behavior, memory continuity և mutation flows-ը ստանում են առանձին verification surface։

## 3.6. Գլխի եզրակացություն

Կատարված ուսումնասիրությունը ցույց է տալիս, որ մեծ համատեքստով և ընդհանրացված օգնականային համակարգերը բավարար չեն, եթե դրանցում բացակայում է persona-ի, հիշողության, graph grounding-ի և state transition-ի խիստ կազմակերպված շերտավորումը։

`Persona-Graph-Agent` համակարգում առաջարկված է այլ ճարտարապետական մոտեցում, որտեղ.

- persona-ն ներկայացվում է որպես կառուցվածքային stateful օբյեկտ,
- graph-ը գործում է որպես երկարաժամկետ կառուցվածքային հիշողություն,
- current working context-ը առանձնացված է long-term storage-ից,
- պատասխանն առաջանում է staged runtime pipeline-ից,
- transition history-ն պահվում է առանձին,
- LLM-ը ենթարկվում է սահմանափակված prompt stages-ի։

Այսպիսով, համակարգը նպատակ ունի բարձրացնել ոչ թե միայն պատասխանների արտաքին բնականությունը, այլ դրանց պատճառական կապը ներքին վիճակի, persona structure-ի, graph memory-ի և session continuity-ի հետ։ Հենց այս հատկությունն է այն դարձնում ոչ թե սովորական prompt-based chat interface, այլ վերահսկվող persona-graph runtime։
