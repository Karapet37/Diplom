# Դիպլոմային Աշխատանքի Տեխնիկական Նկարագիր

Repository: `https://github.com/Karapet37/Diplom`

## Ներածություն

Տվյալ նախագիծը ներկայացնում է file-first սկզբունքով կառուցված `graph-grounded dialogue system`, որի նպատակը պարզապես chat կազմակերպելը չէ, այլ երկարաժամկետ իմաստային հիշողություն ձևավորելը։ Համակարգը նախատեսված է այնպես, որ օգտատիրոջ հաղորդագրությունները և կցված ֆայլերը չկորչեն միայն ընթացիկ prompt-ի սահմաններում, այլ պահպանվեն, վերամշակվեն `LLM`-ի միջոցով, վերածվեն կառուցվածքային `graph memory`-ի և հետագայում օգտագործվեն ավելի ճշգրիտ, ավելի կայուն և ավելի անձնավորված պատասխաններ ստանալու համար։

Այս աշխատանքի կենտրոնական գաղափարը հետևյալն է.

- `chat`-ը տալիս է համակարգին նոր տվյալներ,
- `history`-ը պահում է այդ տվյալների հաջորդականությունը,
- `knowledge_extractor`-ը վերածում է text-ը կառուցվածքային առաջարկների,
- `graph_store`-ը պահում է իմաստային կապերը,
- `context_builder`-ը ընտրում է տվյալ պահի համար կարևոր knowledge-ը,
- `llm`-ը վերջնական պատասխանը ձևավորում է արդեն ոչ թե դատարկ context-ում, այլ graph-ով և personality-ով ուղղորդված միջավայրում։

Այսպիսով, վերջնական նպատակը հետևյալ համակարգն է.

`chat_engine -> llm -> session file -> knowledge_extractor -> memory/graphs -> context_builder -> next answer`

Հիմնական գաղափարական բաժանումը այստեղ հետևյալն է.

- `personality`-ը պետք է գործի որպես **մշտական context**,
- `history`-ը պետք է գործի որպես **փոփոխվող context**,
- `graph`-ը պետք է գործի որպես **կառուցվածքային իմաստային հիշողություն**,
- `LLM`-ը պետք է գործի որպես **analysis + extraction + response generation** շերտ, բայց ոչ որպես graph-ը ուղիղ կառավարող subsystem։

Սա կարևոր տարբերակում է. `LLM`-ը չի գրում `nodes.json` և `edges.json` ֆայլերը անմիջապես։ Այն վերադարձնում է միայն `proposal`-ներ, իսկ graph-ի փոփոխությունը կատարվում է միայն validate և merge անելուց հետո։

---

# Գլուխ 1. Համակարգի Նպատակն ու Լոգիկական Մոդելը

## 1.1. Նախագծի հիմնական խնդիրը

Շատ դասական chat համակարգեր ունեն մեկ հիմնարար սահմանափակում. նրանք պատասխանում են միայն ընթացիկ prompt-ի հիման վրա և չեն ձևավորում կայուն իմաստային հիշողություն։ Նույնիսկ եթե նախորդ հաղորդագրությունները փոխանցվում են model-ին, այդ մեխանիզմը ունի մի քանի խոշոր խնդիր.

- context window-ը սահմանափակ է,
- մեծ history-ը թանկ է փոխանցել յուրաքանչյուր հարցի համար,
- model-ը հաճախ չի տարբերակում կարևոր knowledge-ը երկրորդական text-ից,
- անցյալից ստացված knowledge-ը չի վերածվում կայուն կառուցվածքի,
- personality behavior-ը մնում է մակերեսային, եթե այն պահվում է միայն prompt-ի տեսքով։

Այս նախագծի խնդիրը հենց սա է լուծում. կառուցել մի համակարգ, որտեղ օգտատիրոջ և համակարգի միջև հաղորդակցությունը վերածվում է երկարաժամկետ file-backed memory-ի, իսկ այդ memory-ից կառուցվում է `graph system of contexts`։

## 1.2. Համակարգի վերջնական նպատակը

Համակարգի վերջնական նպատակը կարելի է ձևակերպել այսպես.

1. ընդունել data ինչպես `chat`-ից, այնպես էլ attached files-ից,
2. այդ data-ն պահպանել որպես ֆայլեր,
3. վերամշակել այդ data-ն `LLM`-ի միջոցով,
4. ստեղծել `graph`-ի վրա հիմնված knowledge/context system,
5. ձևավորել `personality`-ներ, որոնք օգտագործվում են որպես կայուն ուղղորդող context,
6. օգտագործել `history`-ը որպես փոփոխվող context,
7. կառուցել պատասխաններ, որոնք հիմնված են թե՛ graph knowledge-ի, թե՛ personality knowledge-ի, և թե՛ ընթացիկ session-ի վրա։

Այս մոտեցումը համակարգը դարձնում է ոչ թե մեկ քայլով պատասխանող chat, այլ file-backed knowledge-driven runtime։

## 1.3. Constant context և dynamic context բաժանումը

Համակարգում context-ը բաժանված է երկու հիմնական տիպի.

### Constant context

Սա այն շերտն է, որը փոփոխվում է դանդաղ և պետք է պահպանի համակարգի կայուն ինքնությունը։ Այդ մասի մեջ մտնում են.

- `personality profile`
- `personality graph`
- graph-ում պահվող կայուն entity-ներ, traits, relations, patterns

Constant context-ը սահմանում է, թե *ով է խոսում*, *ինչպիսի ձևով է խոսում* և *ինչ կայուն knowledge ունի համակարգը տվյալ entity-ի կամ personality-ի մասին*։

### Dynamic context

Սա այն շերտն է, որը փոփոխվում է յուրաքանչյուր նոր հարցով։ Այստեղ մտնում են.

- վերջին user/assistant հաղորդագրությունները,
- ընթացիկ session-ի տրամաբանությունը,
- տվյալ պահին ակտուալ user intent-ը,
- ընթացիկ հարցի հետ կապված graph subset-ը։

Dynamic context-ը սահմանում է, թե *ինչի մասին է հիմա խոսքը*։

Այս բաժանումը կարևոր է, որովհետև personality-ը չպետք է կորչի յուրաքանչյուր նոր հարցով, իսկ history-ը չպետք է դառնա ամբողջ knowledge-ի աղբյուր։

## 1.4. Ինչու է պետք graph memory

Սովորական text history-ը բավարար չէ, որովհետև text-ը դժվար է արդյունավետ փնտրել ըստ իմաստի։ `Graph memory`-ը անհրաժեշտ է հետևյալ պատճառներով.

- entity-ները պետք է պահվեն որպես առանձին node-եր,
- նրանց traits-ը պետք է պահվեն որպես առանձնացված նկարագրական knowledge,
- relations-ը պետք է լինեն որոնելի և վերահսկելի,
- context selection-ը պետք է արվի ըստ կարևորության, ոչ թե ըստ text-ի հերթականության,
- personality behavior-ը պետք է հնարավոր լինի կապել ոչ միայն description-ի, այլ նաև օրինակների և relations-ի հետ։

Այս նախագծում graph-ը պատասխանատու է երեք հիմնարար հարցերի համար.

1. `who / what is this node`
2. `what it is like`
3. `how it acts through relations`

Սա graph-ի quality-ի հիմնական չափանիշն է։ Եթե node-ը չի պատասխանում այս երեք հարցերին, ապա այն knowledge-ի տեսանկյունից թույլ node է։

## 1.5. Personality-ի դերը

`Personality`-ը այստեղ դիտարկվում է ոչ թե որպես decorative prompt, այլ որպես կայուն semantic guidance layer։ Այսինքն՝ system-ը պետք է կարողանա.

- բեռնել personality file,
- բեռնել personality graph,
- քաղել traits, patterns, examples,
- օգտագործել դրանք որպես persistent speaking frame։

Օրինակ, եթե ընտրված է `Sheldon Cooper`, ապա model-ը չպետք է պատասխանի պարզապես “Sheldon is a character...”, այլ պետք է պատասխանի Sheldon-ի perspective-ից։ Սա նշանակում է, որ personality-ը պետք է իրականում ներարկվի final prompt-ի մեջ և դառնա response generation-ի մասնակից։

## 1.6. Attached files-ի նշանակությունը

Նախագիծը նախատեսում է, որ knowledge-ի աղբյուրը միայն chat-ը չէ։ Կարևոր մասը նաև attached files-ն են.

- `.txt`
- `.md`
- `.json`
- `.csv`

Այս ֆայլերը պետք է պահպանվեն, կտրտվեն chunk-երի, մշակվեն `LLM`-ի միջոցով և վերածվեն `graph proposal`-ների։ Այսպիսով system-ը կարող է սովորել ոչ միայն երկխոսությունից, այլ նաև արտաքին փաստաթղթերից։

---

# Գլուխ 2. Ծրագրային Ճարտարապետությունը և Ներքին Կազմակերպումը

## 2.1. Canonical runtime-ի կառուցվածքը

Ներկայիս ակտիվ runtime-ը կառուցված է վեց հիմնական business module-ների շուրջ.

- `roaches_viz/roaches_viz/chat_engine.py`
- `roaches_viz/roaches_viz/llm.py`
- `roaches_viz/roaches_viz/history_store.py`
- `roaches_viz/roaches_viz/knowledge_extractor.py`
- `roaches_viz/roaches_viz/graph_store.py`
- `roaches_viz/roaches_viz/context_builder.py`
- `roaches_viz/roaches_viz/api.py`

Օժանդակ thin bootstrap շերտը.

- `start.py`
- `src/web/combined_app.py`
- `src/web/api.py`

Frontend-ը կենտրոնացված է `webapp/`-ի մեջ։

Այս կառուցվածքը թույլ է տալիս հստակ բաժանել պատասխանատվությունները.

- `chat_engine` — իրականացնում է dialog orchestration,
- `history_store` — պահպանում է session history-ը,
- `knowledge_extractor` — կատարում է structure extraction,
- `graph_store` — իրականացնում է graph persistence և merge logic,
- `context_builder` — հավաքում է prompt-ի համար անհրաժեշտ context-ը,
- `llm` — ապահովում է inference wrapper-ը։

## 2.2. File-first storage մոդելը

Համակարգի հիմքում դրված է file-first գաղափարը։ Սա նշանակում է, որ memory-ի հիմնական source of truth-ը database չէ, ոչ էլ in-memory cache-ը, այլ իրական filesystem-ը։

Հիմնական կառուցվածքը հետևյալն է.

```text
memory/
  sessions/
    {session_id}.txt
  files/
    uploaded_documents/
      {session_id}/
        {safe_filename}
  personalities/
    index.json
    {name}.json
    {name}_graph.json
    proposals/
      {name}.json
  graphs/
    nodes.json
    edges.json
```

### `memory/sessions/`

Այստեղ պահվում է յուրաքանչյուր session-ի պատմությունը text ձևաչափով։

### `memory/files/uploaded_documents/`

Այստեղ պահվում են օգտատիրոջ կցած փաստաթղթերը ըստ session-ի։

### `memory/personalities/`

Այստեղ պահվում են personality profile-ները, personality graph-երը և proposal file-երը։

### `memory/graphs/`

Այստեղ գտնվում են ամբողջ համակարգի knowledge graph-ի երկու հիմնական storage files-ը.

- `nodes.json`
- `edges.json`

## 2.3. `history_store.py`-ի դերը

`history_store.py`-ը պատասխանատու է session history-ի file-backed պահպանման համար։ Այն ապահովում է հետևյալ հնարավորությունները.

- `create_session`
- `append_turn`
- `parse_session`
- `list_sessions`
- `recent_dialogue`
- `infer_current_entity`
- `session_files_dir`

Session file-ի ձևաչափը շատ պարզ է.

```text
[2026-03-14T10:00:00Z]
user: message

[2026-03-14T10:00:01Z]
assistant: response
```

Այս լուծումը ընտրված է մի քանի պատճառներով.

- հեշտ է debug անել,
- հեշտ է մարդու կողմից կարդալ,
- հեշտ է նորից parse անել,
- extractor-ի համար text source-ը հստակ է,
- session state-ը չի թաքնվում ոչ տեսանելի storage-ի մեջ։

`recent_dialogue`-ը history-ից վերցնում է միայն վերջին հատվածը և սահմանափակում է այն մոտավորապես `~1200 tokens equivalent` չափով, որպեսզի prompt-ը չծանրաբեռնվի։

## 2.4. `chat_engine.py`-ի դերը

`chat_engine.py`-ը համակարգի հիմնական orchestration module-ն է։ Այն ապահովում է ամբողջ user-facing response lifecycle-ը.

1. ստանում է `message`, `session_id`, `language`, `personality_name`,
2. բեռնում է session-ը,
3. վերցնում է recent dialogue-ը,
4. փորձում է հասկանալ current entity-ն,
5. որոշում է, թե selected personality կա, թե ոչ,
6. եթե personality-ն missing է, ստեղծում է proposal file,
7. եթե personality-ն գոյություն ունի, կառուցում է լիարժեք context,
8. կանչում է `llm.generate_chat_reply(...)`,
9. պահպանում է պատասխանը session file-ում,
10. պլանավորում է background extraction։

Եթե system-ը տեսնում է personality, որը բացակայում է, այն վերադարձնում է անվտանգ fallback.

`Контекст личности не найден. Запрошено создание анкеты по описанию.`

Սա կարևոր է, որովհետև system-ը չպետք է “ձևացնի”, թե personality context ունի, եթե իրականում այն չկա։

## 2.5. `llm.py`-ի դերը

`llm.py`-ը միասնական inference wrapper-ն է։ Այն ապահովում է երեք հիմնական աշխատանքային ռեժիմ.

1. `chat reply`
2. `graph proposal`
3. `personality proposal`

### `chat reply`

Այս ռեժիմում system-ը կառուցում է plain-text prompt, որի մեջ մտնում են.

- `personality prompt`
- `graph context`
- `recent dialogue`
- `user message`

### `graph proposal`

Այս ռեժիմում `LLM`-ին փոխանցվում է text chunk կամ session text, և այն պետք է վերադարձնի strict JSON, օրինակ.

```json
{
  "proposals": [
    {
      "entity": "Dracula",
      "type": "PERSON",
      "traits": ["vampire", "aristocratic", "predatory"],
      "relations": [
        {"type": "FEEDS_ON", "target": "humans"},
        {"type": "FEARS", "target": "sunlight"}
      ]
    }
  ]
}
```

### `personality proposal`

Այս ռեժիմում model-ը պետք է ստեղծի personality profile-ի JSON նկարագիրը։

Կարևոր սահմանափակում.

- եթե output-ը invalid JSON է, այն discard է արվում,
- եթե output-ը ներքին prompt leak է, այն մաքրվում է,
- `LLM`-ը graph files-ը երբեք ուղիղ չի գրում։

## 2.6. `knowledge_extractor.py`-ի դերը

`knowledge_extractor.py`-ը ամբողջ knowledge materialization pipeline-ի սիրտն է։ Այն միավորում է.

- session extraction,
- file extraction,
- chunking,
- proposal validation,
- graph merge orchestration,
- personality proposal processing,
- personality graph construction։

### Session extraction

Session-ի text-ը վերցվում է `history_store`-ից և ամբողջությամբ փոխանցվում է proposal generation logic-ին։

### File extraction

Կցված file-երը պահպանվում են և հետո մշակվում են ըստ file type-ի.

- `.txt`, `.md` — raw text,
- `.json` — pretty-printed JSON text,
- `.csv` — rows converted to `key=value` lines։

### Chunking

Քանի որ local model-ը սահմանափակ context ունի, մեծ text-ը կտրատվում է.

- մոտ `7000` symbol chunk,
- մոտ `500` symbol overlap,
- գործնականում սա ծառայում է որպես `1500–2000 token` proxy։

### Validation

Յուրաքանչյուր proposal անցնում է validate փուլով.

- `entity` պետք է լինի,
- `type` պետք է լինի,
- `relations`-ը պետք է ունենան `type` և `target`,
- invalid structure-ը discard է արվում։

### Personality proposal flow

Եթե personality-ն բացակայում է, extractor-ը գրում է.

- `memory/personalities/proposals/{name}.json`

Հետո `process_personality_proposals()`-ը.

1. կարդում է proposal file-երը,
2. `LLM`-ից ստանում է personality profile JSON,
3. validate է անում այն,
4. materialize է անում.
   - `memory/personalities/{name}.json`
   - `memory/personalities/{name}_graph.json`
5. թարմացնում է `memory/personalities/index.json`։

## 2.7. `graph_store.py`-ի դերը

`graph_store.py`-ը պատասխանատու է graph persistence-ի և merge semantics-ի համար։

Այն պահպանում է.

- `memory/graphs/nodes.json`
- `memory/graphs/edges.json`

### Graph node contract

Յուրաքանչյուր node normalize է արվում այնպես, որ ունենա առնվազն.

- `id`
- `name`
- `type`
- `description`
- `attributes`
- `context`
- `importance`
- `confidence`
- `frequency`

### Graph edge contract

Յուրաքանչյուր edge ունի.

- `from`
- `to`
- `type`
- `weight`

### Merge-only update

Graph-ը երբեք չի `replace`-վում ամբողջությամբ `LLM` output-ով։ Կիրառվում է միայն.

`graph = merge(graph, proposals)`

### Empty overwrite protection

Եթե նոր payload-ը դատարկ է, բայց graph files-ը արդեն պարունակում են data, system-ը չի overwrite անում դրանք։ Սա պաշտպանում է graph-ը պատահական ջնջումից։

### Duplicate handling

Base duplicate key-ը հետևյալն է.

- normalized `name + type`

`PERSON` node-երի համար հավելյալ հաշվի են առնվում.

- `profession`
- `location`
- `context`

Այս մոտեցումը թույլ է տալիս չխառնել, օրինակ, նույն անունով, բայց տարբեր context ունեցող մարդկանց։

### Scoring

Graph node relevance-ը հաշվարկվում է մոտավորապես այս բանաձևով.

`score = importance * confidence * log1p(frequency)`

Սա կարևոր է `context_builder`-ի համար։

## 2.8. `context_builder.py`-ի դերը

`context_builder.py`-ը միավորում է տարբեր knowledge աղբյուրները և կառուցում է final prompt context-ը։

Այն աշխատում է հետևյալ շերտերով.

1. բեռնում է `personality profile`
2. բեռնում է `personality graph`
3. փնտրում է համապատասխան graph nodes
4. վերցնում է recent dialogue-ը
5. կազմավորում է final context block-երը

### Persona prompt

Եթե personality ընտրված է և գոյություն ունի, կառուցվում է persona prompt, որը ներառում է.

- `You are ...`
- first-person instruction
- traits
- patterns
- examples
- personality graph facts

### Graph context

Graph-ից ընտրվում են միայն առավել համապատասխան node-երը և edge-երը։ Սահմանափակումները.

- առավելագույնը `8` node,
- առավելագույնը `12` edge,
- մոտավորապես `~1800 tokens equivalent` չափի graph context։

Այս մոտեցումը պաշտպանում է `context collapse`-ից, երբ չափազանց շատ knowledge փոխանցելը նվազեցնում է model-ի ուշադրությունը կարևոր facts-ի նկատմամբ։

## 2.9. API և Frontend շերտերը

Համակարգը ունի մեկ միասնական API surface։ Minimal API-ը ներառում է.

- `GET /api/health`
- `GET /api/cognitive/health`
- `GET /api/cognitive/sessions`
- `POST /api/cognitive/sessions`
- `GET /api/cognitive/sessions/{session_id}`
- `POST /api/cognitive/chat/respond`
- `POST /api/cognitive/files/upload`
- `POST /api/cognitive/rebuild`
- `GET /api/cognitive/graph`
- `GET /api/cognitive/graph/subgraph`
- `GET /api/cognitive/personalities`
- `GET /api/cognitive/personalities/{name}`

`Frontend`-ը պահպանում է միայն այն մասերը, որոնք անհրաժեշտ են MVP-ի համար.

- `Chat`
- `Graph`
- `Session list`
- `Personality dropdown`
- `File upload`

---

# Գլուխ 3. Համակարգի Օգտագործումը, Աշխատանքային Սցենարները և Ապագա Զարգացումը

## 3.1. Օգտագործման հիմնական սցենարը

Օգտատիրոջ հիմնական աշխատանքային սցենարը հետևյալն է.

1. բացել web interface-ը,
2. ստեղծել կամ ընտրել session,
3. գրել message chat-ում,
4. անհրաժեշտության դեպքում ընտրել personality,
5. անհրաժեշտության դեպքում attach անել files,
6. ստանալ assistant reply,
7. դիտել graph-ը և node relations-ը,
8. շարունակել conversation-ը՝ արդեն graph-ով հարստացված context-ի հիման վրա։

Այս սցենարում կարևոր է հասկանալ, որ user-ը չի աշխատում անմիջապես graph files-ի հետ։ Նա աշխատում է chat-ի և files-ի մակարդակում, իսկ համակարգը ներքին կերպով կառուցում է knowledge layer-ը։

## 3.2. Chat-ից graph ձևավորման սցենարը

Երբ user-ը գրում է հաղորդագրություն.

1. message-ը ուղարկվում է `chat_engine`-ին,
2. response-ը ստացվում է `llm`-ից,
3. հաղորդագրությունն ու պատասխանը պահվում են `session file`-ում,
4. background extraction-ը կարդում է այդ file-ը,
5. `knowledge_extractor`-ը ստանում է `graph proposal`-ներ,
6. validate-ից հետո `graph_store`-ը merge է անում դրանք,
7. հաջորդ հարցին պատասխանելիս `context_builder`-ը արդեն օգտագործում է այդ graph-ը։

Այսպիսով, chat-ը դառնում է ոչ միայն interaction channel, այլ նաև knowledge acquisition source։

## 3.3. Attached files-ից graph ձևավորման սցենարը

Երբ user-ը attach է անում file.

1. file-ը պահվում է `memory/files/uploaded_documents/{session_id}/` տակ,
2. file type-ը որոշվում է ըստ suffix-ի,
3. content-ը վերածվում է text-ի,
4. մեծ text-ը բաժանվում է chunk-երի,
5. յուրաքանչյուր chunk-ի համար `LLM`-ը ստեղծում է JSON proposal,
6. valid proposals-ը merge են արվում graph-ի մեջ,
7. արդյունքում file-ի knowledge-ը հասանելի է դառնում future answers-ի համար։

Սա հատկապես կարևոր է այն դեպքերում, երբ knowledge-ը գալիս է ոչ թե dialog-ից, այլ արտաքին փաստաթղթերից։

## 3.4. Personality-ի օգտագործման սցենարը

Եթե user-ը ընտրում է personality dropdown-ից որևէ personality, system-ը կատարում է հետևյալ քայլերը.

1. բեռնում է personality profile-ը,
2. բեռնում է personality graph-ը,
3. persona prompt-ը ներառում է final context-ի մեջ,
4. `LLM`-ը պատասխանում է արդեն տվյալ personality-ի perspective-ից։

Եթե ընտրված personality-ն գոյություն չունի, system-ը.

- proposal file է ստեղծում,
- user-ին վերադարձնում է fallback message,
- հետին պլանում փորձում է materialize անել personality profile-ը։

Այս behavior-ը թույլ է տալիս համակարգին աստիճանաբար աճեցնել personality library-ն։

## 3.5. Graph view-ի օգտագործումը

Graph workspace-ը նախատեսված է ոչ թե raw JSON նայելու, այլ knowledge-ը հասկանալու համար։ Յուրաքանչյուր node-ի համար user-ը պետք է կարողանա տեսնել.

1. ով կամ ինչ է տվյալ node-ը,
2. ինչ նկարագրություն և ինչ հատկություններ ունի,
3. ինչ relations ունի այլ node-երի հետ։

Այս մոտեցումը graph-ը դարձնում է ոչ միայն storage, այլ նաև explainable interface։

## 3.6. Համակարգի գործարկման կարգը

Backend-ի գործարկում.

```bash
cd /home/karapet/agent_project
pip install -e .[dev]
python start.py --host 127.0.0.1 --port 8008
```

Frontend-ի գործարկում.

```bash
cd /home/karapet/agent_project
VITE_API_BASE_URL=http://127.0.0.1:8008 npm --prefix webapp run dev
```

Frontend-ը հասանելի է.

- `http://127.0.0.1:5173`

## 3.7. Վերահսկում և verification

Backend test run.

```bash
cd /home/karapet/agent_project
PYTHONPATH=. roaches_viz/.venv/bin/python -m pytest -q
```

Frontend build check.

```bash
cd /home/karapet/agent_project
npm --prefix webapp run build
```

Գործող համակարգը համարվում է ճիշտ, եթե.

- chat-ը միշտ գրում է `session file`,
- attached files-ը պահվում են ճիշտ folder-ում,
- graph-ը կառուցվում է միայն validate եղած proposals-ից,
- graph files-ը դատարկ payload-ով չեն կորչում,
- personality dropdown-ը սնվում է `memory/personalities/index.json`-ից,
- missing personality-ը ստեղծում է proposal file,
- հաջորդ պատասխանները օգտագործում են graph/personality context։

## 3.8. Նախագծի գործնական արժեքը

Այս համակարգի գործնական արժեքը կայանում է նրանում, որ այն միավորում է երեք կարևոր շերտ.

- `LLM`
- `Dialogue Memory`
- `Graph Memory`

Այս միավորումը թույլ է տալիս կառուցել համակարգ, որը.

- հիշում է ոչ միայն text-ը, այլ նաև իմաստային կառուցվածքը,
- տարբերակում է մշտական և փոփոխվող context-ը,
- կարող է աշխատել personality-driven mode-ում,
- կարող է սովորել ինչպես chat-ից, այնպես էլ files-ից,
- ունի explainable graph surface,
- չի կախված ամբողջ history-ը ամեն անգամ prompt-ի մեջ ուղարկելու անարդյունավետ մեթոդից։

## 3.9. Հետագա զարգացման ուղղությունները

Հետագա փուլերում համակարգը կարող է զարգացվել հետևյալ ուղղություններով.

- ավելի խորը duplicate detection,
- multilingual normalization,
- stronger entity linking,
- richer personality vectors,
- better graph ranking,
- source provenance tracking,
- selective rebuild policies,
- stronger evaluation metrics for personality consistency։

Սակայն նույնիսկ ներկա MVP տարբերակով համակարգն արդեն ունի հստակ և պրակտիկ նպատակային ճարտարապետություն.

- file-first storage,
- validated proposal pipeline,
- graph-backed context selection,
- personality as constant context,
- history as dynamic context։

---

## Եզրակացություն

Նախագծի վերջնական նպատակը մի այնպիսի intelligent dialogue system կառուցելն է, որը օգտատիրոջ chat messages-ը և attached files-ը ընդունում է որպես raw input, պահպանում է դրանք files-ի մեջ, մշակում է `LLM`-ի միջոցով, ստեղծում է `graph system of contexts` և դրա հիման վրա կազմակերպում է հաջորդ պատասխանները։

Այստեղ `personality`-ը դառնում է մշտական ուղղորդող context, իսկ `history`-ը՝ փոփոխվող context, որը նկարագրում է ներկա conversation state-ը։ Այս երկու շերտերի միավորումը graph memory-ի հետ ստեղծում է այն հիմքը, որի վրա հնարավոր է կառուցել ոչ միայն chat assistant, այլ նաև երկարաժամկետ, կառուցվածքային և անձնավորված reasoning system։
