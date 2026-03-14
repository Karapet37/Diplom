# Դիպլոմային Աշխատանքի Համառոտ Տեխնիկական Հաշվետվություն

Repository: `https://github.com/Karapet37/Diplom`

## 1. Նախագծի նպատակը

Նախագիծը file-first `graph-grounded dialogue system` է, որի նպատակը հետևյալն է.

- ընդունել input `chat`-ից և attached files-ից,
- պահպանել այդ input-ը files-ի մեջ,
- մշակել text-ը `LLM`-ի միջոցով,
- ստեղծել `graph memory`,
- ձևավորել `personality`-ներ որպես կայուն ուղղորդող context,
- օգտագործել `history`-ը որպես փոփոխվող context,
- հաջորդ պատասխանները կառուցել graph-ի, personality-ի և session state-ի հիման վրա։

Հիմնական աշխատանքային շղթան.

`chat_engine -> llm -> session file -> knowledge_extractor -> memory/graphs -> context_builder -> next answer`

Հիմնական գաղափարը.

- `personality` = constant context
- `history` = dynamic context
- `graph` = structured long-term memory
- `LLM` = analysis + extraction + response generation, բայց ոչ graph writer

## 2. Համակարգի կառուցվածքը

Ակտիվ business modules-ը.

- `roaches_viz/roaches_viz/chat_engine.py`
- `roaches_viz/roaches_viz/llm.py`
- `roaches_viz/roaches_viz/history_store.py`
- `roaches_viz/roaches_viz/knowledge_extractor.py`
- `roaches_viz/roaches_viz/graph_store.py`
- `roaches_viz/roaches_viz/context_builder.py`
- `roaches_viz/roaches_viz/api.py`

Thin bootstrap.

- `start.py`
- `src/web/combined_app.py`
- `src/web/api.py`

Frontend.

- `webapp/`

Մոդուլների պատասխանատվությունը.

### `chat_engine.py`
- ընդունում է user message,
- բեռնում է session state,
- որոշում է current entity/personality,
- հավաքում է context,
- կանչում է `llm`,
- գրում է reply-ը history-ում,
- запускает background extraction։

### `history_store.py`
- ստեղծում է sessions,
- գրում է turn-երը `memory/sessions/{session_id}.txt`-ում,
- parse է անում session history-ը,
- վերադարձնում է `recent_dialogue`,
- օգնում է որոշել current entity-ը,
- պահում է uploaded files-ի folder structure-ը։

### `knowledge_extractor.py`
- կարդում է session text և uploaded files,
- text-ը բաժանում է chunk-երի,
- `LLM`-ից ստանում է strict JSON proposals,
- validate է անում proposals-ը,
- փոխանցում է դրանք `graph_store`-ին,
- materialize է անում personality proposal-ները։

### `graph_store.py`
- պահում է `nodes.json` և `edges.json`,
- normalize է անում node-երն ու edge-երը,
- merge է անում valid proposals-ը,
- թույլ չի տալիս դատարկ payload-ով overwrite անել գոյություն ունեցող graph-ը,
- տալիս է search/subgraph/node-view հնարավորություններ։

### `context_builder.py`
- բեռնում է `personality profile` և `personality graph`,
- ընտրում է համապատասխան graph nodes,
- կառուցում է `persona prompt`,
- սահմանափակում է context-ը, որպեսզի prompt-ը չծանրաբեռնվի,
- վերադարձնում է final context chat response-ի համար։

### `llm.py`
- կառուցում է chat prompt,
- կառուցում է graph proposal prompt,
- կառուցում է personality proposal prompt,
- մաքրում է prompt leaks-ը,
- JSON invalid output-ը discard է անում։

## 3. File-first memory մոդելը

Source of truth-ը filesystem-ն է, ոչ թե database-ը։

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
Պահվում է chat history-ը.

```text
[2026-03-14T10:00:00Z]
user: message

[2026-03-14T10:00:01Z]
assistant: response
```

### `memory/files/uploaded_documents/`
Պահվում են user-ի կցած files-ը ըստ session-ի։

### `memory/personalities/`
Պահվում են.

- available personalities list,
- personality profile,
- personality graph,
- missing personality proposals։

### `memory/graphs/`
Պահվում են համակարգի հիմնական knowledge graph files-ը.

- `nodes.json`
- `edges.json`

## 4. Context-ի մոդելը

Համակարգում context-ը բաժանված է երկու շերտի.

### Constant context
- `personality profile`
- `personality graph`
- graph-ում պահված կայուն knowledge

Այս շերտը որոշում է.
- ով է խոսում,
- ինչ traits և patterns ունի,
- ինչ style կամ relation knowledge կա։

### Dynamic context
- վերջին user/assistant messages,
- ընթացիկ session logic,
- ընթացիկ հարցին համապատասխան graph subset։

Այս շերտը որոշում է.
- ինչի մասին է այս պահին խոսքը,
- որ facts-երն են հիմա կարևոր։

Այս բաժանումը նախագծի հիմքն է. personality-ը պետք է մնա կայուն, իսկ history-ը պետք է փոխվի conversation-ի հետ։

## 5. Graph model

Graph-ը պահում է ոչ թե raw text, այլ knowledge structure։

Node-ը պետք է պատասխանի երեք հարցի.

1. `who / what is this node`
2. `what it is like`
3. `how it acts through relations`

### Node contract
Յուրաքանչյուր node normalize է արվում և ունի առնվազն.

- `id`
- `name`
- `type`
- `description`
- `attributes`
- `context`
- `importance`
- `confidence`
- `frequency`

### Edge contract
Յուրաքանչյուր edge ունի.

- `from`
- `to`
- `type`
- `weight`

### Relevance scoring
Graph context selection-ի համար օգտագործվում է մոտավոր scoring.

`score = importance * confidence * log1p(frequency)`

## 6. Knowledge extraction pipeline

Համակարգը knowledge-ը քաղում է երկու աղբյուրից.

- session history
- attached files

### Session pipeline
1. user/assistant turns-ը գրվում են session file-ում,
2. extractor-ը կարդում է session text-ը,
3. `LLM`-ը վերադարձնում է JSON proposals,
4. proposals-ը validate են արվում,
5. valid data-ն merge է արվում graph-ի մեջ։

### File pipeline
1. file-ը պահվում է `memory/files/uploaded_documents/{session_id}/`,
2. content-ը վերածվում է text-ի,
3. text-ը բաժանվում է chunk-երի,
4. յուրաքանչյուր chunk-ի համար `LLM`-ը վերադարձնում է graph proposals,
5. valid proposals-ը merge են արվում graph-ի մեջ։

### Supported file types
- `.txt`
- `.md`
- `.json`
- `.csv`

### Chunking rules
- մոտ `7000` symbol chunk,
- մոտ `500` symbol overlap,
- գործնականում սա ծառայում է որպես `1500–2000 token` proxy։

## 7. Personality համակարգը

Personality-ը պահվում է որպես file-backed knowledge layer.

Հիմնական files.

- `memory/personalities/index.json`
- `memory/personalities/{name}.json`
- `memory/personalities/{name}_graph.json`
- `memory/personalities/proposals/{name}.json`

### Read path
`context_builder.py`-ը.
- բեռնում է profiles,
- բեռնում է personality graph,
- կազմում է `persona prompt`։

### Write path
`knowledge_extractor.py`-ը.
- գրում է proposal files,
- ստանում է personality JSON `LLM`-ից,
- materialize է անում profile-ը,
- materialize է անում personality graph-ը,
- թարմացնում է `index.json`-ը։

### Missing personality behavior
Եթե system-ը տեսնում է personality, որը չկա.

`вижу личность, ищу в графе, не нашел, запрашиваю создание файла анкеты под его описание`

Այս դեպքում.

1. ստեղծվում է `proposal file`,
2. user-ը ստանում է fallback message,
3. background extraction-ը հետագայում materialize է անում personality-ն։

User-facing fallback.

`Контекст личности не найден. Запрошено создание анкеты по описанию.`

## 8. LLM-ի սահմանափակումները և կանոնները

`LLM`-ը նախագծում ունի երեք դեր.

1. chat response generation
2. graph proposal generation
3. personality proposal generation

Բայց `LLM`-ը **չի կառավարում graph files-ը ուղիղ**։

Կանոններ.

- `LLM`-ը վերադարձնում է միայն `proposal`,
- invalid JSON-ը discard է արվում,
- invalid structure-ը discard է արվում,
- prompt leak-ը մաքրվում է,
- graph files-ը երբեք չեն overwrite արվում դատարկ data-ով։

## 9. API և օգտագործման ձևը

Minimal API surface.

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

Frontend-ը պահում է միայն.

- Chat
- Graph
- Session list
- Personality dropdown
- File upload

## 10. Համակարգի օգտագործման սցենարը

### Սցենար 1. Սովորական chat
1. user-ը ստեղծում է session,
2. ուղարկում է message,
3. ստանում է reply,
4. message/reply-ը գրվում են session file-ում,
5. graph-ը հետին պլանում թարմացվում է,
6. հաջորդ հարցը արդեն կարող է օգտագործել այդ knowledge-ը։

### Սցենար 2. File upload
1. user-ը attach է անում file,
2. file-ը պահվում է համապատասխան folder-ում,
3. extractor-ը text-ը բաժանում է chunk-երի,
4. `LLM`-ը տալիս է proposals,
5. valid proposals-ը մտնում են graph,
6. file-ի knowledge-ը դառնում է future context-ի մաս։

### Սցենար 3. Personality-driven answer
1. user-ը ընտրում է personality,
2. system-ը բեռնում է personality profile-ը և graph-ը,
3. `context_builder`-ը ավելացնում է persona prompt,
4. `LLM`-ը պատասխանում է այդ personality-ի perspective-ից։

## 11. Run և verification

Backend:

```bash
cd <project_root>
pip install -e .[dev]
python start.py --host 127.0.0.1 --port 8008
```

Frontend:

```bash
cd <project_root>
VITE_API_BASE_URL=http://127.0.0.1:8008 npm --prefix webapp run dev
```

Backend tests:

```bash
cd <project_root>
PYTHONPATH=. roaches_viz/.venv/bin/python -m pytest -q
```

Frontend build:

```bash
cd <project_root>
npm --prefix webapp run build
```

## 12. Եզրակացություն

Նախագիծը ներկայացնում է մի համակարգ, որտեղ.

- chat-ը և attached files-ը դառնում են raw input,
- այդ input-ը պահվում է files-ի մեջ,
- `LLM`-ը text-ը վերածում է validate եղած proposals-ի,
- graph-ը ձևավորում է երկարաժամկետ structured memory,
- personality-ը դառնում է constant context,
- history-ը դառնում է dynamic context,
- հաջորդ պատասխանները կառուցվում են արդեն ոչ թե դատարկ prompt-ի, այլ knowledge-driven context-ի հիման վրա։

Այսպիսով համակարգի վերջնական նպատակը chat assistant կառուցելը չէ միայն, այլ `graph-based contextual reasoning system` ձևավորելը, որը կարող է հիշել, կազմակերպել, բացատրել և օգտագործել knowledge-ը ժամանակի ընթացքում։
