# Project Report

Repository: `https://github.com/Karapet37/Diplom`

## 1. Ընթացիկ Նպատակ

Repository-ը բերվել է դեպի մեկ minimal աշխատող product.

`chat_engine -> llm -> session file -> knowledge_extractor -> memory/graphs -> context_builder -> next answer`

Project-ը այլևս չի դիտարկվում որպես multi-engine research sandbox։ Ակտիվ նպատակը file-first MVP-ն է։

## 2. Canonical Runtime

Ակտիվ business modules-ը.

- `roaches_viz/roaches_viz/chat_engine.py`
- `roaches_viz/roaches_viz/llm.py`
- `roaches_viz/roaches_viz/history_store.py`
- `roaches_viz/roaches_viz/knowledge_extractor.py`
- `roaches_viz/roaches_viz/graph_store.py`
- `roaches_viz/roaches_viz/context_builder.py`
- `roaches_viz/roaches_viz/api.py`

Միայն thin bootstrap.

- `start.py`
- `src/web/combined_app.py`
- `src/web/api.py`

Միայն Frontend.

- `webapp/`

## 3. File-First Memory Layout

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

Այս ֆայլերը MVP-ի source of truth-ն են։

## 4. Dialogue Pipeline

Ակտիվ dialogue flow-ը սա է.

1. user-ը ուղարկում է message
2. `chat_engine.py`-ը բեռնում է.
   - recent session text
   - selected personality file
   - graph context
3. `llm.py`-ը կառուցում է մեկ plain-text chat prompt
4. վերադարձվում է assistant reply
5. `history_store.py`-ը երկու turn-երն էլ append է անում `memory/sessions/{session_id}.txt`
6. background extraction-ը կարդում է session file-ը և uploaded files-ը
7. graph/personality updates-ը առաջարկվում և validate են արվում
8. հաջորդ reply-ը կարող է օգտագործել graph/personality context

Chat path-ը graph files-ը ուղիղ չի գրում։

## 5. Extractor Contract

`knowledge_extractor.py`-ը միակ տեղն է, որը text-ը վերածում է graph կամ personality updates-ի։

Նրա contract-ը.

- կարդալ session/source text
- LLM-ից խնդրել structured JSON proposals
- validate անել proposals-ը
- merge անել դեպի.
  - `memory/graphs/nodes.json`
  - `memory/graphs/edges.json`
- validated personality proposals-ը materialize անել file-backed profiles-ի և personality graphs-ի մեջ

Invalid JSON-ը discard է արվում։

Դատարկ graph payload-ները երբեք չեն overwrite անում ոչ դատարկ graph files-ը։

## 6. Personality Contract

Personalities-ը file-backed են։

Հիմնական files-ը.

- `memory/personalities/index.json`
- `memory/personalities/{name}.json`
- `memory/personalities/{name}_graph.json`
- `memory/personalities/proposals/{name}.json`

`context_builder.py`-ը տնօրինում է read path-ը.

- listing personalities
- loading profiles
- loading personality graphs
- building persona prompt blocks

`knowledge_extractor.py`-ը տնօրինում է write path-ը.

- writing proposal files
- examples -> signals -> patterns -> traits
- personality graph updates
- materializing validated personality proposal files

## 7. Ամենակարևոր Added Missing Behavior-ը

System-ը պետք է սա անի, երբ տեսնում է personality, որը գոյություն չունի.

`вижу личность, ищу в графе, не нашел, запрашиваю создание файла анкеты под его описание`

Ակտիվ implementation-ը սա է.

1. detect անել candidate personality/entity
2. check անել.
   - `memory/personalities/{name}.json`
   - graph files
3. եթե missing է.
   - գրել `memory/personalities/proposals/{name}.json`
   - վերադարձնել.
     - `Контекст личности не найден. Запрошено создание анкеты по описанию.`
4. background extraction-ը հետո materialize է անում file-backed personality-ն

## 8. Graph UI Contract

Graph workspace-ը node-ի համար պատասխանում է միայն երեք գործնական հարցի.

1. who / what is this node
2. what it is like
3. how it acts through relations

## 9. Bootstrap և Frontend

`src/web/combined_app.py`-ը մնում է միայն որպես thin integration shell։

`webapp/`-ը մնում է միակ Frontend-ը և կրճատված է մինչև.

- Chat
- Graph
- Session list
- Personality dropdown
- File upload

## 10. Tests

Կրճատված test set-ը կենտրոնացած է իրական MVP behavior-ի վրա.

- chat-ը գրում է session file
- file upload-ը documents-ը պահում է `memory/files/uploaded_documents/{session_id}/` տակ
- graph-ը կառուցվում է session-ից և files-ից
- graph files-ը մնում են ոչ դատարկ
- personality files-ը բեռնվում են `index.json`-ից
- missing personalities-ը ստեղծում են proposal files
- հաջորդ answers-ը կարող են օգտագործել graph/personality context
- prompt leaks-ը չեն հասնում user output

## 11. Run

Backend:

```bash
cd /home/karapet/agent_project
pip install -e .[dev]
python start.py --host 127.0.0.1 --port 8008
```

Frontend:

```bash
cd /home/karapet/agent_project
VITE_API_BASE_URL=http://127.0.0.1:8008 npm --prefix webapp run dev
```

## 12. Verification

Backend:

```bash
cd /home/karapet/agent_project
PYTHONPATH=. roaches_viz/.venv/bin/python -m pytest -q
```

Frontend build:

```bash
cd /home/karapet/agent_project
npm --prefix webapp run build
```
