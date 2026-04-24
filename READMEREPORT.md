# `Persona-Graph-Agent` համակարգի մանրամասն հաշվետվություն

## 1. Ներածություն

Սույն փաստաթուղթը նկարագրում է `Persona-Graph-Agent` համակարգի ներկա, փաստացի աշխատանքային վիճակը։ Այն այլևս չի դիտարկվում որպես սովորական chat wrapper կամ մեկ մեծ `LLM`-ի շուրջ կառուցված միջավայր։ Համակարգի հիմքում դրված է `controller-first` մոտեցումը, որտեղ յուրաքանչյուր հարցում անցնում է ծրագրայինորեն վերահսկվող փուլերով, իսկ լեզվային մոդելը օգտագործվում է միայն սահմանափակ և հստակ տեղերում։

Համակարգի նպատակն է ապահովել՝

- persona-կենտրոնացված պատասխանների ձևավորում,
- graph-ով և հիշողությամբ հիմնավորված reasoning,
- deterministic route ընտրություն,
- կոգնիտիվ վարքային pipeline,
- հաղորդագրությունների վեկտորային մեկնաբանման ուսուցանվող շերտ,
- operator-facing correction interface,
- observability, validation և repair։

Այսպիսով, համակարգը չի աշխատում հետևյալ պարզ սխեմայով՝

```text
message -> LLM -> answer
```

Փոխարենը կիրառվում է վերահսկվող գործառնական հոսք՝

```text
request
-> controller interpretation
-> route selection
-> capability planning
-> cognitive pipeline
-> context assembly
-> response shaping
-> generation
-> validation
-> repair
-> persistence
```

Այս ճարտարապետության հիմնական գաղափարն այն է, որ `LLM`-ը որոշումներ կայացնող կենտրոն չէ։ Որոշումների զգալի մասը կայացվում է runtime-ի կողմից, իսկ մոդելը ներգրավվում է այնտեղ, որտեղ անհրաժեշտ է լեզվային ձևակերպում, սահմանափակ enrichment կամ կառուցվածքային արտածում։

## 2. Համակարգի ընդհանուր ճարտարապետությունը

### 2.1. `controller-first` մոտեցումը

Համակարգի առանցքային տարբերությունը սովորական chatbot-ներից այն է, որ այստեղ request-ը նախ անցնում է controller շերտով։ Այդ controller-ը որոշում է՝

- ինչ տեսակի դիմում է ստացվել,
- պետք է արդյոք persona,
- անհրաժեշտ է արդյոք graph,
- բավարար է արդյոք lightweight path-ը,
- պետք է արդյոք ծանր persona reasoning,
- ինչ validation է պահանջվում,
- ինչքան context կարելի է թույլատրել։

Սա թույլ է տալիս նույն backend-ում սպասարկել ինչպես պարզ lightweight հարցումներ, այնպես էլ persona-հիմնված ծանր turn-եր՝ առանց ամբողջ համակարգը անընդհատ ծանր path-ով անցկացնելու։

### 2.2. Գործառնական հոսքը

Ընդհանուր runtime հոսքը կարելի է ներկայացնել այսպես՝

```text
run_chat_turn()
-> request envelope
-> controller_runtime
-> route decision
-> capability plan
-> simple path կամ persona path
-> runtime status pre-flight
-> LLM generation կամ deterministic fallback
-> validation / repair
-> history persistence
-> trace logging
```

Հիմնական branch-երը երկուսն են՝

1. `simple path`
   Օգտագործվում է այն դեպքերում, երբ բավարար է արագ context assembly-ը և bounded generation-ը։

2. `persona path`
   Օգտագործվում է այն դեպքերում, երբ պետք է persona selection, cognitive pipeline, state transition, response shaping և ավելի խորը reasoning։

### 2.3. Backend-ի հիմնական ակտիվ մոդուլները

Համակարգի ակտիվ backend-ը գտնվում է `agent_system/` պանակում։ Իրական աշխատանքային շերտում հատկապես կարևոր են հետևյալ մոդուլները՝

- `chat_engine.py`
- `controller_runtime.py`
- `request_pipeline.py`
- `context_builder.py`
- `prompt_builder.py`
- `llm.py`
- `history_store.py`
- `cognitive_pipeline.py`
- `cognitive_authority.py`
- `speech_planner.py`
- `state_transition_runtime.py`
- `head_caller.py`
- `persona_engine.py`
- `graph_store.py`
- `reliability.py`
- `node_rethinker.py`
- `behavioral_fallback.py`
- `safety_classifier.py`
- `api.py`
- `message_vector_registry.py`
- `message_vector_runtime.py`
- `message_annotation_store.py`

Վերջին երեք մոդուլները կազմում են հաղորդագրությունների վեկտորային մեկնաբանման և correction-layer-ի ներկայիս հիմքը, և հենց այդ շերտն է տարբերակում համակարգը սովորական persona-chat runtime-ից։

## 3. Request intake, դասակարգում և route ընտրություն

### 3.1. Request envelope

Յուրաքանչյուր նոր chat turn-ի համար ստեղծվում է envelope, որը պահում է առնվազն հետևյալ դաշտերը՝

- `request_id`
- `session_id`
- `raw_text`
- `analysis_text`
- `timestamp`

Այստեղ կարևոր սկզբունքը հետևյալն է. սկզբնական user text-ը այլևս չպետք է կորցվի կամ վերագրվի։ Վերլուծական normalize-ը կիրառվում է միայն առանձին պատճենի վրա, ոչ թե այն տեքստի վրա, որը պահվում է history-ում կամ ցուցադրվում է UI-ում։

### 3.2. Request preprocessing

`request_pipeline.py`-ը յուրաքանչյուր հարցման համար փորձում է կառուցել կառուցվածքային նկարագրություն, որը ներառում է՝

- լեզվի որոշում,
- intent,
- interaction mode,
- request type,
- clarification անհրաժեշտություն,
- persona-style hints,
- response style guidance։

Համակարգը տարբերակում է, օրինակ՝

- `factual_query`
- `general_chat`
- `persona_chat`
- `persona_specification`
- `persona_assignment`
- `persona_dialogue_analysis`
- `project_document_analysis`
- `meta_previous_answer`

Այս տարբերակումը կարևոր է, որովհետև persona dossier, սովորական հարց, meta-analysis և file request-ը չեն կարող նույն route-ով գնալ։

### 3.3. RouteDecision և CapabilityPlan

Route ընտրությունից հետո ձևավորվում է `RouteDecision`, որը որոշում է՝

- `selected_route`
- `requires_history`
- `requires_graph`
- `requires_persona`
- `requires_llm`
- `strict_grounding`
- `validation_mode`
- `fast_path`

Դրան զուգահեռ `CapabilityPlan`-ը որոշում է, թե տվյալ turn-ի համար ինչ իրական գործողություններ են պետք՝

- history load,
- graph retrieval,
- persona load,
- heavy persona runtime,
- deterministic reply,
- LLM call,
- reviewer / repair։

Այս մեխանիզմի առավելությունն այն է, որ lightweight հարցումը չի անցնում անիմաստ ծանր graph կամ persona pipeline-ով, իսկ ծանր persona turn-ը չի մնա չափազանց պարզ chat պատասխանողի հույսին։

## 4. Կոգնիտիվ pipeline-ը

### 4.1. Ընդհանուր գաղափարը

Heavy persona path-ում համակարգը օգտագործում է deterministic 6-փուլ կոգնիտիվ pipeline, որը մոդելավորում է ոչ թե վերջնական բառերը, այլ ներքին վիճակի որոշ մասը։ Այդ pipeline-ը կառուցված է այնպես, որ persona-ի վարքը կախված չլինի միայն language model-ի ազատ ձևակերպումից։

### 4.2. P1–P6 փուլերը

Pipeline-ը բաղկացած է հետևյալ փուլերից՝

1. `P1 — EventEncoder`
   Մուտքային տեքստը վերածում է event probability vector-ի և intensity signal-ի։

2. `P2 — TriggerNetwork`
   Event vector-ը համադրում է genome-derived trigger weights-ի հետ։

3. `P3 — RegulatorCell`
   Թարմացնում է ներքին regulator state-ը՝ anxiety, motivation, fatigue, shame, frustration, guilt, closeness, hope, emptiness ուղղություններով։

4. `P4 — ThoughtMLP`
   Ստեղծում է thought vector, որտեղ արտացոլվում են perceived risk, confidence, needs և interaction frame-ը։

5. `P5 — ConflictScorer`
   Գնահատում է conflict / resolution ռազմավարությունները։

6. `P6 — ActionPolicy`
   Ընտրում է action family՝ օրինակ `approach`, `avoid`, `freeze`, `attack`, `analyze`, `connect`, `withdraw` և այլն։

Արդյունքում ստացվում է `CognitiveTurnOutput`, որը պարունակում է՝

- `action_name`
- `dominant_resolution`
- `perceived_risk`
- `intensity`
- `thought_vec`
- `conflict_vec`
- `blocked_actions`

### 4.3. `CognitiveAuthority` և `SpeechPlanner`

Pipeline-ի արդյունքը անմիջապես չի վերածվում վերջնական պատասխանի։ Նախ աշխատում է `CognitiveAuthority`-ը, որը որոշում է generation mode-ը՝

- `pure_llm`
- `hint`
- `planner`

Եթե pipeline-ի ազդեցությունը քիչ է, runtime-ը կարող է անցնել `pure_llm` կամ `hint` mode-ի։ Եթե pipeline-ը բավականաչափ վստահելի է, կիրառվում է `planner` mode, որտեղ `SpeechPlanner`-ը նախ կառուցում է structured `SpeechPlan`, և միայն հետո `LLM`-ը verbalize է անում այդ plan-ը։

Այսպիսով, ծանր persona mode-ում `LLM`-ը հաճախ արդեն ոչ թե ինքնուրույն որոշող է, այլ նախապես ձևավորված հոսքի լեզվական արտահայտիչ։

## 5. Հաղորդագրության վեկտորային մեկնաբանման շերտը

### 5.1. Նպատակը

Համակարգի կարևոր նոր շերտերից մեկը հաղորդագրության վեկտորային մեկնաբանման ենթահամակարգն է, որը նախատեսված է ոչ թե ուղղակի պատասխան ստեղծելու, այլ հաղորդագրությունների իմաստային և հարաբերական կոորդինատային ներկայացում կառուցելու համար։

Այս շերտը ձևավորում է յուրաքանչյուր message-ի համար առանձին vector, որը պահում է մի քանի անկախ մեկնաբանիչների արդյունքները։

Այստեղ կիրառվում է հետևյալ սկզբունքը՝

```text
message_t + context_matrix_t -> Pn interpreters -> vector_t
```

որտեղ `context_matrix_t`-ը նախորդ հաղորդագրությունների վեկտորների պատուհանն է, ոչ թե մեկ scalar flag։

### 5.2. `P1..P51` registry

Ներկայիս իրական implementation-ում registry-ն արդեն ընդլայնված է մինչև `P1..P51`։ Սկզբնական գաղափարը եղել է `P1..P49`, սակայն համակարգի ներկա տարբերակում ավելացվել են նաև՝

- `P50` — թեմայի շեղման կամ topic shift-ի արձանագրում,
- `P51` — կիրառելի context boundary-ի արձանագրում։

Registry-ն խմբավորված է հետևյալ բաժիններով։

Գործնական ստուգմամբ հաստատվում է, որ թե՛ registry payload-ը, թե՛ annotation workspace-ում վերադարձվող vector-ները ներկայումս արդեն աշխատում են հենց `51` interpreter-ով, այսինքն այստեղ փաստաթուղթը նկարագրում է ոչ թե ապագա մտադրություն, այլ live runtime-ի ընթացիկ ձևը։

#### A. Խոսքային ձև

- `P1` — հարց / պնդում / միտք / մեջբերում / հորդոր
- `P2` — ուղիղ իմաստ կամ քողարկված / փոխաբերական իմաստ
- `P3` — բառացի արտահայտություն կամ ռետորիկական քայլ
- `P4` — պատասխան / չպատասխան / խուսափում
- `P5` — ինքնուրույն միտք կամ նախորդ replica-ի ռեակցիա
- `P6` — թեմայի փակո՞ւմ, թե նոր ուղղության բացում
- `P7` — տրամաբանական հստակություն / մշուշոտություն

#### B. Հոգեբանական ձև

- `P8` — կասկած
- `P9` — վստահություն
- `P10` — ներքին կոնֆլիկտ
- `P11` — պաշտպանություն
- `P12` — խոցելիություն
- `P13` — հարձակում
- `P14` — զսպում
- `P15` — հուզական լարվածություն

#### C. Հարաբերություն զրուցակցի նկատմամբ

- `P16` — հոգատարություն
- `P17` — հարգանք
- `P18` — արժեզրկում
- `P19` — նվաստացում
- `P20` — բարեհաճություն
- `P21` — թաքնված թշնամանք
- `P22` — գերիշխում
- `P23` — զիջում / ենթարկում

#### D. Թաքնված իմաստային կառուցվածք

- `P24` — սարկազմ
- `P25` — հեգնանք
- `P26` — ծաղր / mockery
- `P27` — գովասանքի դիմակ
- `P28` — հոգատարության դիմակ
- `P29` — մանիպուլյացիա
- `P30` — ճնշում
- `P31` — կեղծ մեղմացում

#### E. Զրույցի շարժման ուղղություն

- `P32` — մոտեցում
- `P33` — հեռացում
- `P34` — հաշտեցում
- `P35` — էսկալացիա
- `P36` — սրում
- `P37` — կապի խզում
- `P38` — մեղմացում
- `P39` — կապի պահպանում

#### F. Ճշմարտություն և դիրք

- `P40` — անկեղծություն
- `P41` — դիմակ / ոչ անկեղծություն
- `P42` — ընդունում
- `P43` — հերքում
- `P44` — վերաիմաստավորում
- `P45` — ուղղող գովասանք
- `P46` — կեղծ գովասանք
- `P47` — թաքնված նախատինք

#### G. Մետա-մեկնաբանում

- `P48` — ընթացիկ replica-ի ընթերցում նախորդ շղթայի համատեքստում
- `P49` — turn-ի ընդհանուր ուղղությունը երկխոսության պատմության նկատմամբ
- `P50` — թեմայի կտրուկ շեղում / topic shift
- `P51` — համապատասխան context boundary

Այս registry-ն այլևս պարզապես տեքստային նկարագրություն չէ։ Այն իրականում պահվում է առանձին կոնֆիգուրացիոն շերտով և օգտագործվում է ինչպես prediction-ի, այնպես էլ UI խմբագրման ընթացքում։

### 5.3. Վեկտորի տվյալային ձևը

Յուրաքանչյուր interpreter-ի արդյունքը պահվում է երկիմաստությունը թույլատրող կառուցվածքով՝

```json
{
  "main": "statement",
  "extra": ["question"]
}
```

Այս մոտեցումը կարևոր է, որովհետև հաղորդագրությունը շատ հաճախ չի տեղավորվում մեկ միանշանակ պիտակի մեջ։ Օրինակ՝ մի replica-ն կարող է ձևով լինել հարց, բայց գործառույթով՝ պնդում կամ ճնշում։

Ընդհանուր message-vector շերտը կարող է ունենալ այսպիսի տեսք՝

```json
{
  "message_id": "m17",
  "role": "assistant",
  "raw_text": "Դե ապրես, երբ ուզես կարողանում ես։",
  "display_text": "Դե ապրես, երբ ուզես կարողանում ես։",
  "analysis_text": "Դե ապրես, երբ ուզես կարողանում ես։",
  "vector": {
    "P1": {"main": "statement", "extra": []},
    "P24": {"main": "false_praise", "extra": ["sarcasm"]},
    "P45": {"main": "corrective_praise", "extra": []}
  },
  "context_window": ["m13", "m14", "m15", "m16"],
  "context_matrix_ref": "ctx_17"
}
```

### 5.4. `context_matrix` հասկացությունը

Համակարգի այս շերտում context-ը չի ներկայացվում մեկ թվով, մեկ probability score-ով կամ մեկ summary flag-ով։ Փոխարենը օգտագործվում է նախորդ հաղորդագրությունների վեկտորների պատուհան՝

```text
context_matrix_t = [vector_(t-k), ..., vector_(t-1)]
```

Այդ պատճառով նույն բառերը կարող են տարբեր կերպ մեկնաբանվել տարբեր պատմությունների մեջ։ Օրինակ՝

- հաշտեցումից հետո `ապրես` կարող է կարդացվել որպես իրական գովասանք,
- սրված երկխոսությունից հետո նույն `ապրես`-ը կարող է կարդացվել որպես սարկազմ, կեղծ գովասանք կամ ծաղր։

Սա հենց այն պատճառներից է, որ correction layer-ը չի սահմանափակվում միայն ընթացիկ message-ի label խմբագրմամբ։ Եթե նախորդ replica-ն սխալ է նշված եղել որպես `care`, իսկ իրականում եղել է `pressure`, ապա ընթացիկ turn-ի մեկնաբանումը նույնպես կարող է խեղաթյուրվել։

### 5.5. Ուսուցանվող runtime

`message_vector_runtime.py`-ը կառուցված է որպես trainable inference layer։ Յուրաքանչյուր `Pn` դիտարկվում է որպես առանձին classifier-like մեկնաբանիչ, որի վրա կարող են ազդել՝

- current text,
- role,
- persona name,
- context matrix,
- operator correction-երը։

Bootstrap փուլում կարող են գոյություն ունենալ պարզ նախնական հուշումներ, բայց այդ շերտը նախագծված է correction data-ով աստիճանաբար բարելավվելու համար։ Այսինքն՝ այստեղ նպատակը rule-engine կառուցելը չէ, այլ ուսուցանվող մեկնաբանման շերտ ունենալը։

Կարևոր է նաև, որ այս շերտը ներկայումս արդեն օգտագործվում է ոչ միայն annotation UI-ում, այլ նաև live chat runtime-ում։ `chat_engine.py`-ը կանչում է `build_runtime_message_vector_payload()`-ը, ստանում է ընթացիկ `context_matrix` և `current_vector`, այնուհետև այդ տվյալը ներառում է prompt/planner guidance-ի և trace preview-ի մեջ։ Այսինքն message-vector layer-ը ներկայումս արդեն դեկորատիվ interface չէ, այլ պատասխան ձևավորող runtime-ի ակտիվ մասն է։

## 6. Annotation և correction layer

### 6.1. Ընդհանուր գաղափարը

Համակարգի correction layer-ը առանձին է սովորական chat history-ից։ Սա կարևոր է, որովհետև operator-ը պետք է կարողանա ուղղել ոչ միայն վերջնական պատասխանը, այլ նաև՝

- message vector-ը,
- context window-ի անցյալ replica-ների vector-ները,
- transition interpretation-ը։

Այս ուղղումները չեն պետք վերագրեն chat history-ն այնպես, կարծես սկզբից հենց այդպես է եղել։ Դրանք պետք է պահվեն որպես առանձին annotation շերտ, որը հետագայում կարող է օգտագործվել որպես ուսուցողական dataset։

### 6.2. `message_annotation_store.py`

Այս մոդուլը պատասխանատու է՝

- session history-ից annotation workspace հավաքելու,
- predicted vector-ը կառուցելու,
- correction-ով effective vector-ը ստանալու,
- `context_window` և `context_matrix` կազմելու,
- `transition_interpretation` հաշվարկելու,
- correction-երը session-level և global dataset շերտերում պահելու համար։

Արդյունքում համակարգը յուրաքանչյուր հաղորդագրության համար կարող է ցուցադրել ոչ միայն տեքստը, այլ նաև նրա շուրջ կառուցված իմաստային դիրքը։

### 6.3. Correction storage

Correction-երը պահվում են առանձին շերտերով, որպեսզի չխառնվեն սովորական history-ի հետ։ Գաղափարական մակարդակում առանձնացվում են՝

- session-level annotation store,
- global correction dataset,
- trainable vector-model state։

Սա թույլ է տալիս մի կողմից անմիջապես կիրառել operator ուղղումները տվյալ workspace-ում, մյուս կողմից՝ հետագայում օգտագործել դրանք որպես ուսուցանման օրինակներ։

Ներկայիս implementation-ում այդ շերտերը ֆիզիկապես առանձնացված են հետևյալ ուղիներով՝

- `memory/message_annotations/<session_id>.json`
- `memory/message_annotations/global.jsonl`
- `memory/message_vector_models/`

Այս տարանջատումը կարևոր է, որովհետև սովորական session history-ն չի վերագրվում «կարծես սկզբից այդպես էր», իսկ correction շերտը պահպանվում է որպես ուսուցման և հետագա վերամշակման նյութ։

### 6.4. Annotation API

Annotation workspace-ը և correction save-ը հասանելի են նաև backend API-ի մակարդակում, ոչ միայն UI state-ի ներսում։ Ներկայիս ակտիվ endpoint-ներն են՝

- `GET /api/cognitive/sessions/{session_id}/annotation-workspace`
- `POST /api/cognitive/sessions/{session_id}/annotations`

Այսպիսով frontend correction interface-ը կապակցված է իրական persistence շերտի հետ և չի հանդիսանում mock կամ purely local editor։

## 7. Chat frontend և operator workspace

### 7.1. Frontend-ի դերը

`webapp/`-ը նախագծված չէ որպես սովորական վերջնական օգտագործողի chat shell։ Այն operator-facing միջավայր է, որտեղ միավորվում են՝

- session management,
- persona selection,
- chat surface,
- graph inspection,
- diagnostics,
- training / correction գործիքներ։

Այսինքն frontend-ը ցույց է տալիս ոչ միայն վերջնական պատասխանը, այլ նաև runtime-ի մի շարք ներքին աշխատանքային շերտեր։

### 7.2. Persona ընտրությունը որպես source of truth

Ներկայիս տարբերակում chat surface-ում persona ընտրության dropdown-ը հանդիսանում է assistant speaker label-ի հիմնական աղբյուրը։ Սա կարևոր ուղղում է, որովհետև նախկինում UI-ն հաճախ generic fallback-ով assistant reply-ը ցուցադրում էր պարզապես `Assistant` կամ `Ассистент` պիտակով։

Այժմ speaker label-ի առաջնահերթությունը կառուցված է այս տրամաբանությամբ՝

1. operator-ի կողմից dropdown-ով ընտրված persona,
2. session history-ից եկող persona name,
3. եթե առկա է meaningful speaker name,
4. fallback label, եթե persona ընդհանրապես ընտրված չէ։

Արդյունքում chat thread-ում assistant reply-ը պետք է ներկայանա persona-ի անունով, ոչ թե generic assistant label-ով։

Եթե persona ընդհանրապես ընտրված չէ և meaningful stored label չկա, ներկայիս frontend fallback-ը `LLM`-ն է, ոչ թե `Assistant`։

### 7.3. Անմիջական chat UX

Frontend-ի chat composer-ում կատարվել է կարևոր վարքային ուղղում. user-ը submit անելուն պես իր հաղորդագրությունը անմիջապես պետք է՝

- անհետանա input field-ից,
- հայտնվի chat thread-ում որպես user bubble,
- backend reply-ին սպասելու ընթացքում մնա տեսանելի որպես ուղարկված հաղորդագրություն։

Այս փոփոխությունը կարևոր է, որովհետև նախկին վարքում user-ի տեքստը մնում էր textarea-ում մինչև backend-ի պատասխանը, ինչը ստեղծում էր այն տպավորությունը, որ submit-ը դեռ չի կատարվել։

Նոր հոսքը հետևյալն է՝

```text
textarea input
-> submit
-> composer reset
-> optimistic user message in thread
-> await backend
-> assistant reply appended
```

Սա իրականում արդեն կապված է UI state-ի կոնկրետ մեխանիզմի հետ՝ optimistic message insertion և composer reset token, այսինքն խոսքը ոչ թե ցանկալի UX-ի, այլ արդեն ներդրված հոսքի մասին է։

### 7.4. Prompt-leak cleanup

Chat UI-ում և backend validation layer-ում կիրառվում է պաշտպանություն այն դեպքերի դեմ, երբ language model-ը պատասխանի մեջ արտածում է ծառայողական scaffold, օրինակ՝

- `# Answer`
- `Review Notes`
- `Analyze the Request`
- `Issues Identified`

Նպատակն այն է, որ operator thread-ում չցուցադրվեն reviewer prompt-ի, internal plan-ի կամ chain-like boilerplate-ի պատահական արտահոսքերը։

### 7.5. Training / annotation editor

Chat surface-ի training / annotation workspace-ը թույլ է տալիս՝

- ընտրել ընթացիկ message-ը,
- տեսնել դրա speaker-ը և text-ը,
- տեսնել predicted vector-ը,
- խմբագրել յուրաքանչյուր interpreter-ի `main` և `extra`,
- բացել context window-ի նախորդ replica-ները,
- տեսնել դրանց vector-ները,
- անհրաժեշտության դեպքում ուղղել նաև context replica-ները։

Այսպիսով correction layer-ը սահմանափակված չէ միայն վերջին assistant reply-ի վերագրությամբ։ Այն աշխատում է երկխոսության պատմության վրա։

## 8. History persistence և տվյալների ամբողջականություն

### 8.1. `raw_text`, `display_text`, `analysis_text`

Վերջին ուղղումներից կարևորագույններից մեկը user message-ի ամբողջականության պահպանումն է։ Համակարգը այժմ տրամաբանականորեն տարբերակում է՝

- `raw_text` — սկզբնական մուտքագրված տեքստը,
- `display_text` — UI-ում ցուցադրվող տեքստը,
- `analysis_text` — normalize / preprocessing-ի համար նախատեսված պատճենը։

User message-ի դեպքում `display_text`-ը պետք է լռությամբ համընկնի `raw_text`-ի հետ։ Այս բաժանումը կանխում է այն սխալը, երբ մուտքային տեքստը պատահաբար կտրատվում կամ վերաձևակերպվում էր preprocessing logic-ի պատճառով։

### 8.2. History store

`history_store.py`-ը պահպանում է session data-ն այնպես, որ հաղորդագրությունները լինեն վերարտադրելի, traceable և annotation-layer-ի հետ համադրելի։ Սա կարևոր է ոչ միայն UI-ի համար, այլ նաև հետագա correction dataset-ի ձևավորման տեսանկյունից։

Ներկայիս implementation-ում structured message rows-ը պահվում են session sidecar ձևաչափով, ինչը թույլ է տալիս վերականգնել `raw_text`, `display_text`, `analysis_text`, `persona_name` և այլ օժանդակ դաշտերը առանց հին plain-text history-ի սահմանափակումների։

## 9. Reliability, validation և degradation handling

### 9.1. Reliability layer

`reliability.py`-ը նախատեսված է storage mutation-ների անվտանգ իրականացման համար։ Այն ապահովում է snapshot-before-write մոտեցում, rollback և structured failure reporting։

Դա հատկապես կարևոր է հետևյալ դեպքերում՝

- graph mutation,
- persona materialization,
- node rethink apply,
- partial write failure։

### 9.2. Runtime degradation

Եթե local provider-ը կամ անհրաժեշտ dependency-ն անհասանելի է, runtime-ը pre-flight փուլում դա ճանաչում է և ըստ անհրաժեշտության անցնում fallback շերտի։ Կարևոր սկզբունքն այն է, որ degraded path-ի որոշումը կատարվում է նախքան generation փորձը, ոչ թե հետադարձ string-matching-ով։

Այս մոտեցումը զգալիորեն ավելի հուսալի է, քան դատել այն բանից հետո, երբ արդեն ստացվել է դատարկ կամ անորակ reply։

Ներկայիս chat fallback-ը նույնպես բերված է ավելի հստակ պայմանագրի. base reply-ը այլևս չի ձևանում persona-introduction կամ context-explanation, այլ պահպանում է չեզոք ներկայության ձև (`I'm here.` / `Go ahead.` և դրանց թարգմանված տարբերակները)՝ կախված persona selection-ի առկայությունից։

### 9.3. Validation և repair

Generation-ից հետո համակարգը կարող է կիրառել՝

- format validation,
- grounding validation,
- persona consistency check,
- repair / regeneration։

Այս շերտը նպատակ ունի թույլ չտալ, որ ակնհայտ prompt leak-ը, empty response-ը կամ route-ին չհամապատասխանող reply-ը անմիջապես գրանցվի որպես վերջնական արդյունք։

## 10. Knowledge graph, persona և rethink շերտեր

### 10.1. Persona engine

`persona_engine.py`, `head_caller.py` և հարակից մոդուլները ապահովում են persona-ների ընտրությունը, նյութականացումը և normalized naming-ը։ Persona-ն այստեղ միայն text description չէ. այն կապված է հիշողության, graph data-ի և behavioral state-ի հետ։

### 10.2. Graph runtime

Graph layer-ը պահպանում է entity-ները, կապերը, localized views-ը և retrieval-ի համար անհրաժեշտ կառուցվածքը։ Այս շերտը կարևոր է հատկապես factual կամ graph-grounded persona reasoning-ի ժամանակ։

### 10.3. Node rethink

`node_rethinker.py`-ը փորձում է բարելավել graph node-երի նկարագրությունները և կապերը, բայց դա անում է reliability layer-ի հսկողությամբ, որպեսզի անհաջող mutation-ի դեպքում ամբողջ փոփոխությունը rollback արվի։

## 11. Թեստավորում և որակի վերահսկում

Համակարգը ունի մեծածավալ backend test tree, որը ներառում է routing, persona, graph, reliability, chat engine, context pipeline, local LLM runtime և մի շարք այլ ենթահամակարգեր։

Թեստային ծածկույթը ներառում է՝

- route runtime,
- request pipeline,
- controller runtime,
- cognitive pipeline,
- reliability,
- node rethink rollback,
- graph lifecycle,
- behavioral fallback,
- local LLM policy,
- chat engine,
- API behavior,
- annotation / memory lifecycle։

Առանձին ստուգվում են նաև հետևյալ նոր behavior-ները՝

- user `raw_text`-ի ամբողջական պահպանում,
- annotation workspace-ի կառուցում `context_matrix`-ով,
- correction layer-ի առանձին պահպանում,
- corrected context-ի օգտագործումը runtime message-vector payload-ում։

Առանձնապես կարևոր է, որ ուղղումները կատարվել են ոչ թե միայն test patching-ի միջոցով, այլ նաև իրական ճարտարապետական սխալների շտկմամբ, օրինակ՝

- degradation detection-ի refactor,
- deterministic seeding cognitive runtime-ում,
- factual grounding contract-ի հստակեցում,
- history contamination-ի կանխում,
- prompt leak-երի մաքրում։

## 12. Ճարտարապետական էվոլյուցիան

Համակարգի զարգացման ընթացքում աստիճանաբար պարզ է դարձել, որ միայն route-երի բազմապատկումը և prompt-երի շերտավորումը բավարար չեն։ Եթե wrapper-ը չափազանց շատ է, բայց իրական runtime որոշումը մնում է `LLM`-ի վրա, ապա համակարգը դառնում է և՛ բարդ, և՛ փխրուն։

Դրա արդյունքում ձևավորվել է հետևյալ անցումը՝

### Նախորդ վիճակ

- չափազանց շատ հաջորդական routing քայլեր,
- context-ի չափազանց մեծ փաթեթավորում,
- model-ի կողմից behavior invention,
- թույլ observability,
- fast path-երի բացակայություն,
- correction layer-ի բացակայություն։

### Ներկա վիճակ

- controller-first route որոշում,
- persona heavy path միայն անհրաժեշտության դեպքում,
- deterministic cognitive pipeline,
- structured `SpeechPlan`,
- runtime degradation pre-flight check,
- message-vector annotation layer,
- operator correction workspace,
- ավելի հստակ persistence և traceability։

Այս փոփոխությունները համակարգը մոտեցրել են ոչ թե «ավելի խելացի chatbot»-ի, այլ վերահսկվող cognitive-agent runtime-ի գաղափարին։

## 13. Ընթացիկ սահմանափակումները

Թեև համակարգը զգալիորեն առաջ է գնացել, մի շարք խնդիրներ դեռ բաց են և համարվում են հետագա աշխատանքի ուղղություններ։

### 13.1. Regulator state-ի երկարաժամկետ պահպանում

Կոգնիտիվ pipeline-ի regulator state-ը դեռ ամբողջությամբ չի պահվում session-level persistent ձևաչափով բոլոր սցենարների համար։

### 13.2. Annotation dataset-ի խորացված ուսուցում

Message-vector correction layer-ը արդեն գործում է, սակայն երկարաժամկետ տեսանկյունից անհրաժեշտ է խորացնել trainable interpreter-ների վերաուսուցման pipeline-ը, որպեսզի session-level ուղղումները ավելի արդյունավետ կերպով տեղափոխվեն global learning state։

### 13.3. Frontend-ի լրացուցիչ operator գործիքներ

Թեև chat annotation UI-ն արդեն գոյություն ունի, notes/planning/advanced correction flow-երի ամբողջական visualization-ը դեռ կարելի է ընդլայնել։

### 13.4. Small-model calibration

Փոքր տեղային մոդելների դեպքում planner mode-ի չափից ավելի կառուցվածքային հուշումները երբեմն կարող են ծանրաբեռնել verbalization փուլը։ Հետևաբար պետք է շարունակական calibration։

### 13.5. Persona-grounded style reliability

Persona ընտրությունն այժմ ավելի լավ է կապված chat UI-ի հետ, սակայն տարբեր degraded կամ rare fallback path-երում դեռ պետք է շարունակել վերահսկել, որ generic assistant label-երը և persona-ից կտրված պատասխանները ամբողջությամբ չվերադառնան։

## 14. Եզրակացություն

`Persona-Graph-Agent` համակարգի ներկա տարբերակը ներկայացնում է controller-driven, graph-aware, persona-aware և correction-aware միջավայր, որտեղ `LLM`-ը այլևս չի դիտարկվում որպես միակ որոշող ուժ։

Համակարգի հիմնական ձեռքբերումներն են՝

- request-երի controller-first կազմակերպում,
- heavy persona path-ի հստակ տարանջատում lightweight chat-ից,
- deterministic P1–P6 cognitive pipeline,
- `CognitiveAuthority` և `SpeechPlanner`-ով կառավարվող verbalization,
- message-vector մեկնաբանման `P1..P51` շերտ,
- `context_matrix`-ով աշխատող correction workspace,
- operator-facing UI, որտեղ հնարավոր է ուղղել ոչ միայն վերջնական պատասխանը, այլ նաև դրա համատեքստային մեկնաբանումը,
- history և annotation տվյալների տարանջատում,
- reliability, validation և degradation handling-ի հասունացում։

Ամենակարևոր եզրահանգումն այն է, որ այս համակարգը կառուցված է ոչ թե մեկ մեծ model-ի «ամեն ինչ հասկանալու» գաղափարի վրա, այլ մի քանի իրար հետ կապված, բայց ծրագրայինորեն վերահսկվող շերտերի վրա։ Այդ շերտերի համադրությունը թույլ է տալիս ոչ միայն պատասխան ստեղծել, այլ նաև հասկանալ, թե ինչ route-ով, ինչ context-ով և ինչ ներքին մեկնաբանությամբ է այդ պատասխանը ստացվել։

Այդ պատճառով համակարգի հետագա զարգացումը պետք է շարունակվի նույն ուղղությամբ՝

- մեծացնել trainable interpretation layer-ի որակը,
- խորացնել correction dataset-ի կիրառելիությունը,
- ավելի լավ կապել context history-ն և runtime behavior-ը,
- պահել UI-ն operator-centric, թափանցիկ և խմբագրվող։

Այս փաստաթուղթը ներկայացնում է համակարգի հենց այդ ներկա վիճակը և կարող է դիտարկվել որպես նախագծի համապարփակ տեխնիկական հաշվետվություն։
