# ԳԼՈՒԽ 3. ՄՇԱԿՎԱԾ PERSONA-GRAPH ԱԳԵՆՏԱՅԻՆ ՀԱՄԱԿԱՐԳԻ ՃԱՐՏԱՐԱՊԵՏՈՒԹՅՈՒՆԸ, ԻՐԱԿԱՆԱՑՈՒՄԸ ԵՎ ՎԵՐԼՈՒԾՈՒԹՅՈՒՆԸ

## 3.1 Խնդրի ձևակերպումը

Մեծ լեզվային մոդելների վրա հիմնված սովորական զրուցային համակարգերը վերջին տարիներին ցույց են տվել բարձր արդյունավետություն բնական լեզվի հասկացման և տեքստի գեներացման խնդիրներում, սակայն դրանց գործնական կիրառման ընթացքում ի հայտ են գալիս մի շարք հիմնարար սահմանափակումներ։ Առաջին սահմանափակումը երկարաժամկետ հիշողության բացակայությունն է։ Սովորական `LLM`-ը, եթե չունի արտաքին հիշողության մեխանիզմ, յուրաքանչյուր հարց մշակվում է գրեթե միայն ընթացիկ prompt-ի սահմաններում։ Արդյունքում համակարգը չի կարող կայուն կերպով պահել նախկին երկխոսություններից ստացված գիտելիքը, օգտատիրոջ նախընտրությունները, փաստական կապերը կամ դերային վարքագծի օրինաչափությունները։

Երկրորդ կարևոր խնդիրը հալյուցինացիաների վտանգն է։ Եթե մոդելը ստիպված է պատասխան կառուցել առանց վերահսկվող փաստական հիմքի, այն հաճախ արտադրում է վստահ հնչող, բայց ոչ բավարար հիմք ունեցող արտահայտություններ։ Այս խնդիրը հատկապես նկատելի է այնպիսի միջավայրերում, որտեղ համակարգից ակնկալվում է ոչ միայն լեզվական սահունություն, այլ նաև կառուցվածքային հիշողություն, ստուգելի կապեր և պատճառաբանելի պատասխաններ։

Երրորդ խնդիրը persona վարքագծի անկայունությունն է։ Շատ համակարգեր կարող են prompt-ի միջոցով ժամանակավորապես մոդելին տալ որոշակի դեր կամ ոճ, սակայն այդ persona-ն հաճախ չի պահպանվում ժամանակի ընթացքում։ Եթե persona-ն պահվում է միայն որպես մեկ prompt block, ապա երկար երկխոսությունների, նոր փաստերի ավելացման կամ տարբեր թեմաների միջև անցումների դեպքում վարքագիծը սկսում է տատանվել։ Այդ դեպքում համակարգը կարող է նույն սուբյեկտի մասին տարբեր շրջադարձերում խոսել տարբեր տոնով, տարբեր գիտելիքային հիմքով կամ նույնիսկ հակասական ինքնաներկայացմամբ։

Այս խնդիրների լուծման նպատակով մշակվել է persona-graph agent համակարգ, որի նպատակը երեք հատկությունների համադրումն է` հիշողություն, վերահսկելիություն և դերային կայունություն։ Նախագծվող համակարգը կառուցված է այնպես, որ `LLM`-ը չդառնա ամբողջ համակարգի տրամաբանական ղեկավարող տարրը, այլ օգտագործվի միայն այն խնդիրների համար, որոնցում այն առավել ուժեղ է, այսինքն` գիտելիքի քաղում (`knowledge extraction`) և վերջնական բնական լեզվով պատասխանների գեներացում (`response generation`)։ Մնացած շերտերը` երթուղավորումը, դասակարգումը, graph-ի թարմացումը, persona-ի ընտրությունը, context-ի կառուցումը և graph hygiene-ը, իրականացվում են deterministic code-ով։

Ուստի աշխատանքի հիմնական նպատակը եղել է կառուցել այնպիսի AI համակարգ, որը կլինի ոչ միայն «խոսող մոդել», այլ նաև բացատրելի, հիշողություն ունեցող, persona-կայուն և փաստականորեն վերահսկելի գործակալային միջավայր։

## 3.2 Գոյություն ունեցող մոտեցումների վերլուծություն

Ժամանակակից հետազոտություններում և կիրառական համակարգերում լայն տարածում են ստացել այնպիսի մոտեցումներ, որոնք փորձում են `LLM`-երի սահմանափակումները հաղթահարել արտաքին հիշողության, retrieval-ի և կառուցվածքային ներկայացումների միջոցով։ Այս համատեքստում հղումային համակարգերի շարքում արժե առանձնացնել `GraphRAG`, `PersonaAgent with GraphRAG` գաղափարախոսությունը և graph-based agent memory համակարգերը։

`GraphRAG` մոտեցումների հիմնական գաղափարն այն է, որ retrieval-augmented generation համակարգերում միայն vector similarity-ը բավարար չէ։ Գիտելիքը ներկայացվում է հանգույցների և կապերի տեսքով, իսկ retrieval-ը կատարվում է ոչ միայն փաստաթղթերի chunk-երի, այլ նաև graph topology-ի հիման վրա։ Արդյունքում ստացվում է բազմաքայլ պատճառաբանական կապերի համար ավելի հարմար context, քանի որ համակարգը կարող է միավորել ոչ միայն առանձին փաստեր, այլև դրանց միջև եղած հարաբերությունները։ `GraphRAG` համակարգերի առավելությունն այն է, որ դրանք բարձրացնում են բացատրելիության աստիճանը և բարելավում retrieval-ի որակը այն դեպքերում, երբ թեման տարածված է բազմաթիվ փոխկապակցված փաստերի մեջ։ Սակայն նման համակարգերը հիմնականում կենտրոնանում են կորպուսի գիտելիքային կազմակերպման վրա և համեմատաբար քիչ ուշադրություն են դարձնում persona-ի վիճակին, էմոցիոնալ պրոֆիլին կամ դերային վարքագծի կայունությանը։

`PersonaAgent with GraphRAG` տիպի համակարգերը փորձում են համատեղել persona-ի մոդելավորումը և graph-grounded retrieval-ը։ Այս մոտեցումներում սովորաբար առկա է persona-ի նկարագիր, որը սահմանում է գործակալի բնութագրերը, ինչպես նաև knowledge graph կամ retrieval layer, որն ապահովում է փաստական աջակցություն։ Նման համակարգերը հատկապես օգտակար են այն խնդիրներում, որտեղ գործակալը պետք է խոսի որոշակի կերպարի կամ մասնագիտական դերի անունից։ Այնուամենայնիվ, շատ դեպքերում persona-ն շարունակում է մնալ հիմնականում prompt-ի մակարդակի կառուցվածք, այսինքն` նկարագրված է տեքստային պրոֆիլով, բայց չի ներկայացվում որպես առանձին հիշողության միավոր` սեփական կյանքի ցիկլով, հուզական վիճակով և situation-reaction վարքագծով։

Graph-based AI agent memory համակարգերը կազմում են ավելի ընդհանուր մոտեցումների խումբ։ Դրանք փորձում են agent memory-ը ներկայացնել որպես entities, events, relations և երբեմն նաև episodic traces։ Նման համակարգերի առավելությունն այն է, որ գիտելիքը դառնում է վերամշակելի, որոնելի և սեղմելի։ Սակայն այս կատեգորիայի շատ իրականացումներում memory layer-ը պահվում է database-ներում կամ ոչ թափանցիկ պահոցներում, որտեղ հետազոտական վերարտադրելիությունն ու սխալների վերլուծությունը բարդանում են։ Բացի այդ, երկարատև օգտագործման դեպքում graph memory-ն կարող է արագ կուտակել աղմուկ, duplicate nodes և ցածր արժեք ունեցող տարրեր, եթե համակարգում ներառված չեն graph hygiene-ի մեխանիզմներ։

Գոյություն ունեցող արտադրական և հետազոտական մոտեցումների համեմատությունից երևում է, որ դրանցից շատերը կամ լավ են աշխատում retrieval-ի հետ, կամ persona-ի մոդելավորման, կամ graph memory-ի հետ առանձին։ Սակայն հաճախ բացակայում է միասնական ճարտարապետությունը, որտեղ persona-ի վիճակը, graph memory-ն, routing logic-ը և context construction-ը միավորվում են մեկ վերահսկելի, բացատրելի և file-first համակարգում։ Մշակված նախագիծը միտված է լրացնել հենց այս բացը։

Տեխնիկական համեմատությունն ամփոփված է հետևյալ աղյուսակում.

| Մոտեցում | Գլխավոր նպատակը | Հիշողության ձևը | Ուժեղ կողմը | Սահմանափակումը |
| --- | --- | --- | --- | --- |
| `GraphRAG` | retrieval-ի բարելավում graph topology-ի միջոցով | graph + chunk retrieval | multi-hop knowledge access | թույլ persona layer |
| `PersonaAgent with GraphRAG` | persona-կայուն պատասխան + factual retrieval | persona-ի նկարագիր + graph context | role-aware generation | persona-ն հաճախ մնում է prompt-ի մակարդակում |
| Graph-based agent memory systems | երկարաժամկետ agent memory | entities, events, relations | memory explainability | graph entropy, storage opacity |
| Մշակված persona-graph agent system | persona, graph memory և deterministic orchestration-ի համադրում | file-first graph + folder-based heads | explainable, stable, debuggable pipeline | պահանջում է հիշողության ավելի խիստ կարգապահություն |

## 3.3 Մշակված համակարգի ճարտարապետությունը

`persona-graph` ճարտարապետությամբ մշակված AI գործակալային համակարգի կենտրոնական նպատակը օգտատիրոջ հարցումը persona-կայուն, graph-grounded և պատճառաբանական կապերը հաշվի առնող պատասխանի վերածելն է։ Համակարգի մշակման ընթացքում ընտրվել է խողովակաձև ճարտարապետություն, որտեղ յուրաքանչյուր փուլ ունի հստակ պատասխանատվություն և չի խառնվում հարակից շերտերի տրամաբանությանը։ Այս մոտեցումը բարձրացնում է համակարգի կայունությունը, ստուգելիությունն ու բացատրելիությունը։

Հիմնական մշակման շղթան հետևյալն է.

`chat -> message analyzer -> feature extractor -> classifier forest -> head caller -> persona head -> context builder -> LLM -> response`

Այս շղթայի սկզբում գտնվում է chat շերտը, որը ընդունում է օգտատիրոջ հաղորդագրությունը և session context-ը։ `Message analyzer`-ը կատարում է նախնական իմաստաբանական վերլուծություն. հայտնաբերում է entity-ների հիշատակումները, կառուցում է իրավիճակի (`situation`) նկարագրությունը և առանձնացնում cues, օրինակ հարցականություն, ագրեսիվություն, վախ կամ էմպատիկ տոն։ Այս շերտը կարևոր է, քանի որ հենց այստեղից հետո համակարգը պետք է որոշի, թե որ entity-ներն են պետք դիտարկել և որ persona head-ը կարող է լինել առաջնային։

`Feature extractor`-ը յուրաքանչյուր candidate entity-ի համար հաշվարկում է deterministic հատկանիշներ, օրինակ բառային ազդակներ, title-case կառուցվածք, fictional, professional, conceptual կամ object-oriented բնույթի նշաններ։ Այդ հատկանիշները ձևավորում են հաջորդ շերտի մուտքային տվյալները։

`Classifier forest`-ը random forest մեխանիզմ է, որտեղ մի քանի decision tree-եր քվեարկում են entity type-ի շուրջ։ Աջակցվող դասերը վեցն են` `PERSON`, `CONCEPT`, `PHENOMENON`, `OBJECT`, `FICTIONAL_CHARACTER`, `PROFESSION`։ Իրականացման ընթացիկ տարբերակում օգտագործվում է 7 ծառից կազմված անտառ, իսկ վստահության աստիճանը հաշվարկվում է հետևյալ կերպ.

```text
confidence = votes(winner) / tree_count
```

Քվեարկության արդյունքում համակարգը ստանում է ոչ միայն class, այլ նաև confidence և votes բաշխում։ Սա կարևոր է, որովհետև persona routing-ը այլևս հիմնված չէ միայն մեկանգամյա հեյուրիստիկ կանոնների կամ `LLM`-ի ազատ եզրակացության վրա, այլ ստանում է հաշվետու և վերարտադրելի որոշում։

`Head caller`-ը classifier-ի արդյունքների հիման վրա որոշում է, թե տվյալ entity-ն պետք է մնա միայն graph node, թե materialize արվի որպես persona head։ Այստեղ հիմնական գաղափարն այն է, որ persona head չպետք է ստեղծվի յուրաքանչյուր պատահական entity-ի համար։ Այդպիսի սահմանափակումը պաշտպանում է համակարգը graph entropy-ից և persona storage-ի աղտոտումից։ Head materialization-ը թույլատրվում է հիմնականում այն դեպքերում, երբ entity-ն համապատասխանում է head դառնալու չափանիշներին կամ օգտատիրոջ կողմից explicit ընտրված է որպես persona։

`Persona head` շերտը ներկայացնում է կերպարային հիշողությունը։ Այստեղ պահվում են traits-ը, relations-ը, օրինակային արտահայտությունները, հուզական վիճակն ու situation-reaction knowledge-ը։ Persona head-ը ոչ թե prompt-ի առանձին հատված է, այլ հիշողության ինքնուրույն միավոր, որն ունի սեփական թղթապանակային կառուցվածքը, մետատվյալները և global graph-ի հետ համաժամեցման տրամաբանությունը։

`Context builder`-ը համակարգի կենտրոնական ինտեգրացիոն շերտերից մեկն է։ Այն միավորում է persona-ի վիճակը, recent dialogue-ը, graph subgraph-ը և օգտատիրոջ հարցը։ Այս բաղադրիչի կարևոր հատկությունն այն է, որ այն կառուցում է bounded context. այսինքն` մոդելին ուղարկվող context-ը սահմանափակվում է token-ային բյուջեով և կազմվում է դասակարգման ու փուլային կրճատման միջոցով։ Այդ կերպ համակարգը խուսափում է prompt-ի չափի անկառավարելի աճից և միաժամանակ պահպանում է առավել կարևոր nodes-ը և persona signals-ը։

Միայն այս բոլոր deterministic փուլերից հետո `LLM`-ը կանչվում է վերջնական response generation-ի համար։ Ճարտարապետական առումով սա առանցքային որոշում է, քանի որ թույլ է տալիս օգտագործել `LLM`-ի լեզվական և extraction-ային կարողությունները` առանց նրան հանձնելու կառավարման շերտը։

### 3.3.1 Բաղադրիչների փոխգործակցության դիագրամ

```text
                    +----------------------+
                    |   User / Web Client  |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |       Chat API       |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |    chat_engine.py    |
                    +----------+-----------+
                               |
         +---------------------+----------------------+
         |                                            |
         v                                            v
+--------------------+                    +----------------------+
| message_analyzer   |                    |   history_store      |
+---------+----------+                    +----------+-----------+
          |                                            |
          v                                            |
+--------------------+                                 |
| feature_extractor  |                                 |
+---------+----------+                                 |
          |                                            |
          v                                            |
+--------------------+                                 |
| classifier_forest  |                                 |
+---------+----------+                                 |
          |                                            |
          v                                            |
+--------------------+                                 |
|    head_caller     |                                 |
+---------+----------+                                 |
          |                                            |
          v                                            |
+--------------------+                    +----------------------+
|   persona_engine   |<------------------>|      graph_store     |
+---------+----------+                    +----------+-----------+
          |                                            ^
          v                                            |
+--------------------+                                 |
|  context_builder   |---------------------------------+
+---------+----------+
          |
          v
+--------------------+
|        LLM         |
| extraction/reply   |
+---------+----------+
          |
          v
+--------------------+
|      Response      |
+--------------------+
```

Այս դիագրամը ցույց է տալիս, որ `LLM`-ը գտնվում է մշակման շղթայի վերջին փուլում և չի զբաղվում routing, head spawning կամ graph-ի սպասարկման խնդիրներով։

### 3.3.2 Հարցման մշակման բարձր մակարդակի ալգորիթմ

```text
Ալգորիթմ 3.1. Հարցման մշակումը
Մուտք: message, session_id, selected_persona, explicit_context
Ելք: assistant_reply, persona_name, graph_context

1. session <- create_or_load_session(session_id)
2. current_entity <- infer_current_entity(session)
3. analysis <- analyze_message(message, current_entity, selected_persona, explicit_context)
4. entities <- analysis.entities
5. յուրաքանչյուր entity-ի համար
6.     features <- extract_features(entity, analysis)
7.     decision <- classifier_forest.classify(features)
8. prepared <- head_caller.prepare_heads(analysis, decisions)
9. primary_head <- head_caller.select_primary_head(prepared)
10. եթե primary_head գոյություն ունի, ապա
11.     update_emotion_vector(primary_head, analysis.cues)
12. context <- build_context(message, session, primary_head, explicit_context)
13. prompt <- build_prompt(context, message)
14. assistant_reply <- LLM.generate_response(prompt)
15. append_turn(session, message, assistant_reply)
16. եթե primary_head գոյություն ունի, ապա
17.     record_situation_reaction(primary_head, analysis.situation, assistant_reply)
18. schedule_background_rebuild(session)
19. վերադարձնել final payload
```

Այս ալգորիթմից երևում է, որ `LLM`-ին փոխանցվող prompt-ը ստեղծվում է արդեն այն ժամանակ, երբ persona-ի ընտրությունը, graph retrieval-ը և token-ային բյուջեի կառավարումը ավարտված են։

## 3.4 File-First հիշողության ճարտարապետությունը

Մշակված համակարգի ամենակարևոր նախագծային որոշումներից մեկը file-first հիշողության ճարտարապետության ընտրությունն է։ Ի տարբերություն այն լուծումների, որոնք հիմնական վիճակը պահում են միայն database-ում կամ ոչ թափանցիկ service layer-ներում, այս համակարգում հիշողության հիմնական աղբյուրը filesystem-ն է։ Այդ ընտրությունը պայմանավորված է ոչ միայն պարզությամբ, այլև հետազոտական վերահսկելիությամբ։

Հիշողության ընդհանուր կառուցվածքը հետևյալն է.

```text
memory/
  graphs/
    nodes.json
    edges.json
  heads/
    {head_slug}/
      traits.json
      relations.json
      examples.json
      knowledge.txt
      emotion_vector.json
      meta.json
      local_graph.json
    index.json
  sessions/
    {session_id}.txt
  files/
    uploaded_documents/
      {session_id}/
        {filename}
  proposals/
    {head}.json
```

`graphs/` թղթապանակում պահվում է global knowledge graph-ը, որը ներկայացված է `nodes.json` և `edges.json` ֆայլերով։ `heads/` թղթապանակում պահվում են persona heads-ը` առանձին թղթապանակային կառուցվածքով։ `sessions/` թղթապանակը ծառայում է session history-ի պահպանման համար, իսկ `files/` թղթապանակը պահում է օգտատիրոջ կողմից ներբեռնված փաստաթղթերը ըստ session-ի։ `proposals/` թղթապանակը նախատեսված է այն դեպքերի համար, երբ համակարգը պետք է գրանցի persona materialization-ի նոր հարցում կամ միջանկյալ առաջարկ։

File-first մոտեցման հիմնական առավելություններն են թափանցիկությունը, վերարտադրելիությունը և սխալների վերլուծության դյուրությունը։ Քանի որ վիճակը պահվում է ընթեռնելի ֆայլերում, հետազոտողը կամ մշակողը կարող է անմիջապես տեսնել, թե որ nodes-ն են ավելացվել graph-ում, ինչպես է փոխվել persona-ի emotion vector-ը, ինչ relations են materialize արվել և ինչ փաստաթղթերից են դրանք առաջացել։ Սա կարևոր է հատկապես հետազոտական նախագծերի համար, որտեղ միայն վերջնական պատասխանը բավարար չէ. անհրաժեշտ է հասկանալ նաև, թե այդ պատասխանը ինչ հիշողության և ինչ կանոնների հիման վրա է կառուցվել։

Ավելին, filesystem-ի վրա հիմնված մոտեցումը հեշտացնում է տարբեր փորձերի համեմատությունը, backup-երի ստեղծումը, վիճակի պահված տարբերակների պահպանումը և սխալների տեղայնացումը։ Թեև մեծ մասշտաբի բաշխված արտադրական միջավայրերի համար database-backed համակարգերը հաճախ ավելի հարմար են, տվյալ նախագծի նպատակների համար file-first model-ը ավելի համապատասխան է, քանի որ այն առաջնահերթություն է տալիս բացատրելիությանը և ճարտարապետական վերահսկելիությանը։

### 3.4.1 Հիշողության գրառման և ընթերցման հոսքերը

```text
User message
   |
   v
sessions/{session_id}.txt  -----> background rebuild -----> extraction -----> graphs/nodes.json, edges.json
   |
   +-----> current dialogue retrieval

Uploaded file
   |
   v
files/uploaded_documents/{session_id}/{filename}
   |
   v
text conversion -> chunking -> extraction -> validation -> graph merge

Persona selection / spawning
   |
   v
heads/{head_slug}/traits.json
heads/{head_slug}/relations.json
heads/{head_slug}/examples.json
heads/{head_slug}/emotion_vector.json
heads/{head_slug}/meta.json
```

### 3.4.2 Ֆայլային համահունչություն և ատոմիկ գրառումներ

Տվյալ համակարգում file persistence-ը իրականացվում է ոչ թե ուղիղ overwrite-ով, այլ ժամանակավոր ֆայլի ստեղծման և `replace` գործողության միջոցով։ Այս մոտեցումը նվազեցնում է այն ռիսկը, որ ծրագրի ընդհատման պահին կստացվի մասնակի գրված graph կամ persona file։ Ակադեմիական և ինժեներական տեսանկյունից սա կարևոր է, քանի որ memory corruption-ը կարող է խաթարել ոչ միայն ընթացիկ session-ը, այլ նաև հետագա բոլոր retrieval արդյունքները։

Ընդհանուր գրառման սխեման հետևյալն է.

```text
payload -> serialize -> path.tmp -> atomic replace(path.tmp, path)
```

Այս սխեման հատկապես կարևոր է `nodes.json`, `edges.json`, `meta.json` և `examples.json` նման վիճակ պահող ֆայլերի համար։

## 3.5 Persona head մոդելը

Մշակված համակարգի հիմնարար հասկացություններից մեկը persona head-ն է։ Persona head-ը տվյալ համակարգում չի նույնացվում պարզապես prompt-ում գրված «դերի նկարագրության» հետ։ Այն հանդես է գալիս որպես գիտելիքի ինքնուրույն միավոր, որը ներկայացնում է որոշակի անձ, կերպար, մասնագիտական դերի կրող կամ այլ գործակալային բնույթ ունեցող սուբյեկտ, որի անունից համակարգը կարող է խոսել։

Յուրաքանչյուր head պահվում է առանձին թղթապանակում։ Օրինակ.

```text
memory/heads/dracula/
  traits.json
  relations.json
  examples.json
  knowledge.txt
  emotion_vector.json
  meta.json
  local_graph.json
```

`traits.json` ֆայլում պահվում են head-ի հիմնական բնութագրերը և entity type-ը։ Այս traits-ը նկարագրում են տվյալ persona-ի կայուն կողմերը, օրինակ` տրամաբանական, արիստոկրատական, վամպիրային, էմպատիկ կամ վերլուծական հատկանիշներ։ `relations.json` ֆայլը պահում է aliases-ը և այլ entities-ի հետ հարաբերությունները։ Այս շերտը կարևոր է, քանի որ persona-ն սովորաբար մեկուսացված չի գոյատևում. այն մշտապես կապվում է այլ entities-ի, իրադարձությունների կամ թեմաների հետ։

`examples.json` ֆայլն ունի երկակի դեր։ Մի կողմից այն պահում է persona-ի օրինակային արտահայտությունները, որոնք օգնում են վերականգնել խոսքի ոճը և ինքնաներկայացումը։ Մյուս կողմից այն ներառում է `situation_reactions` բաժին, որտեղ պահվում է իրավիճակների և արձագանքների կապը։ Այդ պատճառով persona model-ը ստանում է ոչ միայն ստատիկ նկարագիր, այլ նաև վարքագծային գիտելիք։

`emotion_vector.json` ֆայլում պահվում է head-ի հուզական պրոֆիլը։ Ընտրվել են հինգ հիմնական առանցքներ` `anger`, `fear`, `curiosity`, `confidence`, `empathy`։ Յուրաքանչյուր արժեք ներկայացվում է `0..1` միջակայքում։ Այսպիսի ներկայացումը թույլ է տալիս persona-ի վիճակը ներկայացնել թվայնացված և context builder-ի ու response generation-ի համար հեշտ կիրառելի ձևով։ Էմոցիոնալ վիճակը թարմացվում է deterministic կանոններով` հիմնվելով օգտատիրոջ հաղորդագրության cues-ի վրա։ Արդյունքում persona-ն ստանում է վերահսկվող վիճակային անցումներ, այլ ոչ թե ամբողջությամբ թողնվում է model improvisation-ին։

`knowledge.txt` ֆայլը պահում է head-ի ամփոփ գիտելիքը բնական լեզվով։ Այն օգտագործվում է persona block-ի կառուցման ժամանակ և ծառայում է որպես հակիրճ, բայց կայուն փաստական կամ ոճական ամփոփում։ `meta.json` ֆայլը պահում է head-ի canonical name-ը, slug-ը, entity type-ը, folder path-ը, importance-ը, frequency-ն, aliases-ը և timestamps-ը։ Վերջապես, `local_graph.json` ֆայլը ներկայացնում է տվյալ head-ի ներքին graph ներկայացումը, որը կարող է ներառել traits, relation targets և example nodes։

Այս մոդելի կարևոր լրացուցիչ շերտերից են emotion vectors-ը և situation-reaction learning-ը։ Եթե persona-ն ունի միայն traits-ի և knowledge-ի նկարագիր, ապա այն մնում է համեմատաբար ստատիկ։ Երբ ավելացվում են հուզական վիճակ և իրավիճակային արձագանքներ, persona head-ը սկսում է հանդես գալ որպես վարքագծային հիշողության միավոր։ Դա բարելավում է persona consistency-ը, քանի որ պատասխանը սկսում է կախված լինել ոչ միայն «ով է persona-ն», այլ նաև «ինչ վիճակում է persona-ն» և «ինչպես է սովորաբար արձագանքում նման իրավիճակներում»։

### 3.5.1 Persona head-ի ստեղծման և ակտիվացման ալգորիթմ

```text
Ալգորիթմ 3.2. Head spawning և materialization
Մուտք: entity_name, entity_type, aliases, payload, explicit_flag
Ելք: persona head կամ graph-only node

1. եթե entity_type ∈ {PERSON, FICTIONAL_CHARACTER, PROFESSION}, ապա
2.     allow_head <- True
3. այլ եթե explicit_flag = True, ապա
4.     allow_head <- True
5. այլ
6.     allow_head <- False
7. եթե allow_head = False, ապա
8.     պահել entity-ն միայն global graph-ում
9.     ավարտել
10. ստեղծել head folder-ը, եթե այն գոյություն չունի
11. initialize անել traits, relations, examples, emotion_vector, meta files
12. validate անել payload-ը
13. merge անել aliases-ը, examples-ը, relations-ը
14. sync անել head-ը global graph-ի հետ
15. վերադարձնել materialized head
```

Այս ալգորիթմը ցույց է տալիս, որ persona storage-ը պաշտպանված է անկառավարելի ընդլայնումից։ Օրինակ, ընդհանուր concepts կամ generic objects-ը կարող են մնալ graph-only nodes և չվերածվել heads-ի։

### 3.5.2 Emotion update-ի կանոնները

Իրականացման ընթացիկ տարբերակում emotion vector-ը թարմացվում է deterministic կանոններով։ Դրանք կարելի է ներկայացնել հետևյալ կերպ.

```text
anger'      = clamp(anger + 0.18 * insult_signal - 0.02 * (1 - insult_signal))
fear'       = clamp(fear + 0.12 * fear_signal - 0.01 * (1 - fear_signal))
curiosity'  = clamp(curiosity + 0.08 * question_signal - 0.01 * (1 - question_signal))
confidence' = clamp(confidence - 0.04 * fear_signal + 0.02 * (1 - fear_signal))
empathy'    = clamp(empathy + 0.12 * empathy_signal - 0.01 * (1 - empathy_signal))
```

Այստեղ `clamp(x)`-ը սահմանափակում է արժեքը `[0, 1]` միջակայքում։ Այս կանոնները կարևոր են նրանով, որ հուզական վիճակը փոփոխվում է պատճառաբանելի ձևով, այլ ոչ թե թողնվում է model-ի ազատ ներքին դինամիկային։

### 3.5.3 Situation-reaction retrieval

Ստացված արձագանքները պահվում են `situation -> reaction` զույգերով, իսկ համապատասխան reaction ընտրելիս օգտագործվում է ընթացիկ situation-ի և պահված situation-ների token-ային համընկնումը.

```text
reaction_score(s_mem, s_now) = |tokens(s_mem) ∩ tokens(s_now)|
```

Ավելի բարձր score ունեցող արձագանքները ներառվում են persona block-ի մեջ։ Այսպիսով, համակարգը կարող է ոչ միայն հիշել persona-ի նախապես սահմանված նկարագիրը, այլ նաև վերականգնել վարքագծային նախադեպեր։

## 3.6 Գիտելիքի գրաֆը և graph hygiene-ը

Համակարգի global memory շերտը ներկայացված է knowledge graph-ի միջոցով։ Graph-ը պահում է այնպիսի կառուցվածքային գիտելիք, որը հնարավոր չէ վստահորեն և վերարտադրելիորեն պահպանել միայն ազատ տեքստային history-ի օգնությամբ։ Այս graph-ում nodes-ը ներկայացնում են entities, concepts, phenomena, professions կամ fictional characters, իսկ edges-ը ներկայացնում են դրանց միջև կապերը։

Node-ի մակարդակում ներկայացվում են հետևյալ հիմնական դաշտերը` `id`, `name`, `type`, `aliases`, `description`, `facts`, `folder`, `importance`, `confidence`, `frequency`, `context`։ Այս կառուցվածքը թույլ է տալիս յուրաքանչյուր node-ին վերագրել ոչ միայն անվանում և տիպ, այլ նաև փաստական հենակետ, վստահության աստիճան, օգտագործման հաճախականություն և ծագման մասին տեղեկություն։ Edge-երը ներառում են `from`, `to`, `type`, `weight`, `confidence`, `source` դաշտերը և այդպիսով արտացոլում են թե՛ հարաբերության իմաստաբանական տեսակը, թե՛ այդ կապի ուժը։

Graph memory-ի երկարատև օգտագործման դեպքում առաջանում է entropy-ի կուտակման խնդիր։ Եթե յուրաքանչյուր extraction արդյունք անմիջապես և առանց հիգիենիկ մեխանիզմների կուտակվի graph-ում, ժամանակի ընթացքում համակարգը կսկսի պարունակել duplicate nodes, հնացած հարաբերություններ, ցածրարժեք փաստեր և անունների տարբերակների պատճառով տրոհված entities։ Այդ պատճառով մշակված համակարգում ներդրվել են graph hygiene մեխանիզմներ։

Առաջին մեխանիզմը `importance decay`-ն է։ Յուրաքանչյուր commit-ի ընթացքում node importance-ը բազմապատկվում է `0.99` գործակցով։ Այս մոտեցումն ապահովում է, որ երկար ժամանակ չօգտագործվող nodes-ը աստիճանաբար կորցնեն առաջնահերթությունը։ Երկրորդ մեխանիզմը duplicate detection-ն է, որը հաշվի է առնում name similarity-ը, aliases overlap-ը և context similarity-ը։ Եթե երկու nodes-ը բարձր հավանականությամբ ներկայացնում են նույն entity-ն, դրանք merge են արվում մեկ canonical node-ի մեջ։

Երրորդ մեխանիզմը garbage collection-ն է։ Եթե node-ի կարևորությունը ընկնում է `0.05`-ից ցածր և հաճախականությունը `2`-ից փոքր է, ապա այն հեռացվում է graph-ից։ Այս կանոնը պաշտպանում է graph-ը ցածր արժեք ունեցող աղմուկից։ Չորրորդ մեխանիզմը graph compression-ն է։ Եթե նույն source node-ից առաջանում են միևնույն relation type-ով բազմաթիվ նման կապեր, համակարգը կարող է ստեղծել summary node, որն ամփոփում է այդ կապերի խումբը։ Այս լուծումը կարևոր է context building-ի համար, որովհետև թույլ է տալիս կապերի մեծ խմբերը ներկայացնել ավելի կոմպակտ ձևով։

Graph hygiene-ի ներդրումը տվյալ նախագծի կարևոր նորամուծություններից մեկն է։ Շատ graph-backed agent համակարգեր ունակ են գիտելիք ավելացնել, բայց քիչ համակարգեր ունեն հիշողության մաքրման և սեղմման հստակ կանոններ։ Առանց այդ շերտի memory graph-ը դառնում է անընդհատ աճող և գնալով պակաս օգտագործելի կառուցվածք։ Մշակված համակարգում hygiene-ը դիտարկվում է ոչ թե որպես օժանդակ գործիք, այլ որպես հիմնական architectural layer։

### 3.6.1 Node և edge representation-ի դիագրամ

```text
                +--------------------------------------+
                | Node: Dracula                        |
                |--------------------------------------|
                | id = fictional_character:dracula     |
                | type = FICTIONAL_CHARACTER           |
                | aliases = [Count Dracula]            |
                | facts = [feeds on humans, fears ...] |
                | importance = 0.82                    |
                | confidence = 0.95                    |
                | frequency = 7                        |
                +-----------------+--------------------+
                                  |
                          FEARS / weight=0.8
                                  |
                                  v
                +--------------------------------------+
                | Node: sunlight                       |
                | type = PHENOMENON                    |
                +--------------------------------------+
```

### 3.6.2 Duplicate resolution-ի տեխնիկական կանոնը

Տվյալ իրականացման մեջ nodes-ի միավորման համար օգտագործվում է համակցված նմանության չափում.

```text
S_total = 0.6 * S_name + 0.3 * S_context + 0.1 * S_alias
```

որտեղ.

- `S_name` հաշվի է առնում token similarity և string similarity,
- `S_context` հաշվի է առնում description, context blob, aliases և folder-related metadata,
- `S_alias` համեմատում է aliases-ի տեքստային նմանությունը։

Եթե nodes-ի տիպերը համատեղելի են և `S_total >= 0.88`, ապա դրանք համարվում են միավորման ենթակա։

### 3.6.3 Graph hygiene-ի ալգորիթմ

```text
Ալգորիթմ 3.3. Graph hygiene
Մուտք: nodes, edges
Ելք: normalized, compacted graph

1. յուրաքանչյուր node-ի համար կիրառել importance_decay:
       importance <- 0.99 * importance
2. գտնել duplicate node զույգերը
3. merge անել duplicate nodes-ը և վերահասցեավորել edges-ը
4. հեռացնել nodes, որոնց համար
       importance < 0.05 և frequency < 2
5. գտնել relation clusters նույն source node-ի համար
6. ստեղծել summary nodes, եթե cluster size >= 2
7. validate անել վերջնական graph-ը
8. պահպանել compacted graph-ը
```

Այս ալգորիթմը հիշողության շերտը դարձնում է ինքնամաքրվող և երկարաժամկետ օգտագործման համար ավելի պիտանի։

## 3.7 Context-ի կառուցումը

`LLM`-ով պատասխան կառուցելու որակը մեծապես կախված է նրանից, թե ինչ context է տրվում նրան։ Եթե context-ը չափազանց աղքատ է, համակարգը կորցնում է factual grounding-ը և persona consistency-ը։ Եթե context-ը չափազանց մեծ է, մոդելը ստանում է խիտ, աղմկոտ prompt, որի արդյունքում retrieval-ի որակն ու generation-ի կայունությունը նվազում են։ Այդ պատճառով մշակված համակարգում context construction-ը նախագծված է որպես առանձին վերահսկվող փուլ։

Context builder-ը ներառում է մի քանի աղբյուր.

- persona traits,
- emotion vector,
- persona relations,
- example utterances,
- learned situation reactions,
- relevant graph nodes,
- recent dialogue,
- user question։

Այս աղբյուրներից ստացված ամբողջ տեղեկատվությունը չի ուղարկվում մոդելին ամբողջ ծավալով։ Փոխարենը կիրառվում է ranking և սահմանափակ ծավալով հավաքում։ Graph retrieval-ը հիմնված է node-ների գնահատականների վրա, որտեղ օգտագործվում են importance-ը, confidence-ը և frequency-ից ստացվող log-գործոնը։ Բացի այդ retrieval-ը հաշվի է առնում aliases-ը, description-ը, facts-ը և relation neighborhood-ը։ Այսպիսով context builder-ը փորձում է ընտրել ոչ թե պարզապես բառային համընկնող nodes, այլ այն nodes-ը, որոնք առավել հավանական է օգտակար լինեն ընթացիկ հարցին պատասխանելու համար։

Ստացված context-ը բաժանվում է մի քանի մասի` `persona_block`, `graph_context`, `recent_dialogue` և question։ Յուրաքանչյուր մաս ունի իր բյուջեն, իսկ ամբողջ prompt-ի համար սահմանվում է առավելագույն token limit։ Այս նախագծում այդ upper bound-ը սահմանվել է `5000 token`։ Եթե context-ի որևէ մաս չափից ավելի մեծանում է, կիրառվում է փուլային կրճատման ռազմավարություն։ Նախ կրճատվում են երկրորդային հատվածները, ապա անհրաժեշտության դեպքում սահմանափակվում են graph context-ը կամ recent dialogue-ը, բայց փորձ է արվում պահպանել persona-ի վիճակի և առանցքային factual nodes-ի ամբողջականությունը։

Այս մոտեցման նպատակն է կանխել երկու ծայրահեղություն. մի կողմից context starvation-ը, մյուս կողմից prompt overload-ը։ Արդյունքում `LLM`-ը ստանում է համեմատաբար կոմպակտ, բայց semantic-որեն հագեցած prompt, որն ավելի հարմար է persona-consistent և graph-grounded պատասխանների համար։

### 3.7.1 Node ranking-ի բանաձևը

Retrieval փուլում graph nodes-ի առաջնահերթությունը հաշվարկվում է մոտավորապես հետևյալ ձևով.

```text
score(node) = importance * confidence * max(log(frequency), 0.1)
```

Այս score-ը լրացվում է բառային համընկնմամբ, aliases-ի, facts-ի և relation neighborhood-ի հաշվառմամբ։ Արդյունքում ընտրվում են առավել կարևոր և համատեքստին առավել համապատասխան nodes-ը։

### 3.7.2 Token budgeting-ի դիագրամ

```text
MAX_CONTEXT_TOKENS = 5000
        |
        +--> question budget        ~ 1200
        +--> prompt overhead        ~  180
        +--> persona_block budget   ~ 1600
        +--> graph_context budget   ~ 2200
        +--> recent_dialogue budget ~  900
```

Եթե ընդհանուր բյուջեն գերազանցվում է, համակարգը սկսում է կրճատումը հետևյալ հերթականությամբ.

```text
graph_context -> recent_dialogue -> persona_block
```

Սակայն յուրաքանչյուր բաժնի համար կա նվազագույն շեմ, օրինակ persona block-ը չպետք է ընկնի չափազանց փոքր արժեքի, որպեսզի persona consistency-ը չկորչի։

### 3.7.3 Context construction-ի ալգորիթմ

```text
Ալգորիթմ 3.4. Bounded context construction
Մուտք: question, session_id, selected_persona
Ելք: persona_block, graph_context, recent_dialogue

1. question <- clip(question, question_budget)
2. recent <- read_recent_dialogue(session_id)
3. current_entity <- infer_current_entity(session_id)
4. resolved_persona <- infer_persona_name(question, selected_persona, current_entity)
5. query <- concatenate(question, recent, current_entity, resolved_persona, persona_hints)
6. subgraph <- search_top_ranked_nodes(query)
7. persona_block <- render_persona_block(resolved_persona)
8. graph_context <- render_graph_context(subgraph)
9. sections <- fit_to_budget(persona_block, graph_context, recent)
10. վերադարձնել bounded context
```

## 3.8 Փաստաթղթերից սովորելու մեխանիզմը

Համակարգի կարևոր հնարավորություններից մեկը document learning-ն է, այսինքն` նոր գիտելիք ստանալը ոչ միայն chat history-ից, այլ նաև օգտատիրոջ ներբեռնած ֆայլերից։ Այս մեխանիզմը թույլ է տալիս համակարգը դարձնել գիտելիքով հարստացվող գործակալ, որը կարող է իր հիշողությունը համալրել արտաքին աղբյուրներից։

Document ingestion-ի շղթան հետևյալն է.

`file -> chunk -> extraction -> validation -> graph merge`

Նախ ներբեռնված ֆայլը պահվում է համապատասխան session folder-ում։ Այնուհետև ֆայլի բովանդակությունը վերածվում է տեքստային ներկայացման։ Համակարգը աջակցում է `txt`, `md`, `json`, `csv` ձևաչափերին։ Քանի որ մեծ փաստաթղթերը ամբողջությամբ prompt-ով ուղարկելը արդյունավետ չէ, text-ը բաժանվում է մասերի (`chunks`)։ Chunking-ի ռազմավարությունն ընտրված է այնպես, որ յուրաքանչյուր կտոր մնա մոտավոր `2000 token`-ից ցածր սահմանաչափում և միևնույն ժամանակ պահպանի որոշ overlap հարևան հատվածների հետ։

Յուրաքանչյուր chunk-ի համար կառուցվում է extraction prompt, որը `LLM`-ից ակնկալում է JSON ձևաչափով entities և relations։ Սակայն extraction output-ը անմիջապես memory չի դառնում։ Այն անցնում է validation փուլով, որտեղ ստուգվում են structure-ը, relation contract-ը և entity typing-ը։ Ընդհանուր, աղմկային կամ չհիմնավորված տեսակավորումները չեն materialize արվում որպես persona heads։ Միայն այն դեպքում, երբ extraction-ը համարվում է բավարար վստահելի, այն merge է արվում graph memory-ի մեջ։

Այս մեխանիզմի շնորհիվ համակարգը կարող է փաստաթղթերից վերցնել facts, entities և relations, ապա դրանք վերածել որոնելի graph memory-ի։ Հետագա երկխոսությունների ընթացքում այդ knowledge-ը կարող է վերականգնվել retrieval-ի միջոցով և օգտագործվել պատասխանների grounding-ի համար։ Այսպիսով document ingestion-ը դառնում է agent-ի ուսուցման գործնական մեխանիզմ, որը չի պահանջում մոդելի լիարժեք վերաուսուցում։

### 3.8.1 Document ingestion-ի տեխնիկական դիագրամ

```text
      +-------------------+
      | Uploaded document |
      +---------+---------+
                |
                v
      +-------------------+
      | text conversion   |
      +---------+---------+
                |
                v
      +-------------------+
      | chunk_text()      |
      | max_tokens=2000   |
      | overlap=200       |
      +---------+---------+
                |
                v
      +-------------------+
      | LLM extraction    |
      | JSON proposals    |
      +---------+---------+
                |
                v
      +-------------------+
      | validation        |
      | typing/relation   |
      +---------+---------+
                |
                v
      +-------------------+
      | graph merge       |
      +---------+---------+
                |
                +-------> selective head materialization
```

### 3.8.2 Chunking-ի հաշվարկային կանոնը

Chunking-ի ընթացիկ իրականացման մեջ token budget-ը գնահատվում է սիմվոլների միջոցով.

```text
max_chars     = max_tokens * 4
overlap_chars = overlap_tokens * 4
```

Այս մոտարկումն օգտագործվում է, որովհետև այն արագ է, deterministic է և չի պահանջում tokenizer-ի մակարդակի ծանր հաշվարկ յուրաքանչյուր chunk-ի համար։

### 3.8.3 Փաստաթղթերից head materialization-ի սահմանափակում

Extraction-ից հետո բոլոր entities-ները չեն materialize արվում որպես heads։ Դրա փոխարեն կիրառվում է head-gating logic.

```text
եթե entity_type ∈ {PERSON, FICTIONAL_CHARACTER, PROFESSION}
    և classification evidence-ը բավարար է,
ապա create/update head
այլ
    պահել միայն graph-ում
```

Այս քայլը կարևոր է, քանի որ փաստաթղթերից քաղված ընդհանուր բառերը, օրինակ `sunlight`, `dust`, `objects`, չպետք է վերածվեն persona heads-ի։

## 3.9 Իրականացման առանձնահատկությունները

Մշակված համակարգի ծրագրային իրականացումը կառուցված է մի քանի պատասխանատվական շերտերի շուրջ։ Dialogue orchestration-ը կենտրոնացած է այն շերտում, որն ընդունում է օգտատիրոջ հարցումը, որոշում է ընթացիկ entity-ն, այն ուղղորդում է համապատասխան persona layer-ին, գրանցում է session history-ը և գործարկում է ետին պլանի վերականգնման ցիկլը։ Այս շերտը կատարում է համակարգի «վզիկի» (`neck`) դերը, քանի որ այստեղից է սկսվում բոլոր ենթամոդուլների հաջորդական գործարկումը։

Semantic preprocessing շերտը ներկայացված է analyzer և feature extraction բաղադրիչներով։ Analyzer-ը կատարում է նախնական entity detection, situation detection և cue extraction, իսկ feature extractor-ը ձևավորում է դասակարգման համար պատրաստ հատկանիշների տարածք classifier forest-ի համար։ Դասակարգման շերտը կազմված է մի քանի decision trees-ից, որոնք միասին ապահովում են քվեարկությամբ կատարվող դասակարգում։ Այս լուծումը ընտրվել է այն պատճառով, որ այն ավելի բացատրելի է, քան վերջից-վերջ նեյրոնային կառավարման տրամաբանությունը, և միևնույն ժամանակ բավականաչափ ճկուն է տարբեր entity classes տարբերակելու համար։

Persona management շերտը պատասխանատու է head materialization-ի, head file structure-ի, emotional state-ի, learned reactions-ի և local graph representation-ի համար։ Այստեղ առանցքային է այն, որ persona-ների ստեղծումը սահմանափակված է հստակ կանոններով, որպեսզի ցանկացած entity չդառնա persona։ Դա պահպանում է պահպանման կարգապահությունը և բարձրացնում է համակարգի իմաստաբանական կայունությունը։

Knowledge memory շերտը իրագործված է global graph store-ի տեսքով։ Այն ոչ միայն պահում է nodes և edges, այլ նաև իրականացնում է graph validation, merge logic, duplicate resolution, graph compression և garbage collection։ Այս շերտը ներկայացնում է համակարգի երկարաժամկետ կառուցվածքային հիշողությունը։

Document ingestion և extraction շերտը կապում է արտաքին փաստաթղթերը հիշողության համակարգի հետ։ Այն ապահովում է text conversion, chunking, `LLM`-ի միջոցով extraction proposal generation, validation և merge։ API շերտը կառուցված է այնպես, որ համակարգը հասանելի լինի որպես վեբ ծառայություն, իսկ session, file upload, graph inspection և rebuild գործողությունները կառավարվեն միասնական ինտերֆեյսով։

Տեխնոլոգիական տեսանկյունից համակարգում օգտագործվել են `Python`, `FastAPI`, `Pydantic`, JSON/text file storage, filesystem-based persistence, atomic file replacement և pluggable `LLM` provider abstraction։ Այս ընտրությունները համապատասխանում են նախագծի նպատակին` ունենալ հետազոտական, ստուգելի և բացատրելի ծրագրային ճարտարապետություն։

### 3.9.1 Իրականացման շերտերի քարտեզը

```text
API Layer
  -> request validation
  -> session and file endpoints

Dialogue Orchestration Layer
  -> chat engine
  -> head selection
  -> response lifecycle

Semantic Control Layer
  -> message analyzer
  -> feature extractor
  -> classifier forest

Persona Layer
  -> head storage
  -> emotion update
  -> situation-reaction learning

Memory Layer
  -> graph store
  -> duplicate resolution
  -> graph hygiene

Learning Layer
  -> entity extraction
  -> file ingestion
  -> rebuild / repair
```

Այս քարտեզը ցույց է տալիս, որ համակարգը մոդուլացված է ոչ միայն ֆայլային մակարդակում, այլ նաև ինժեներական պատասխանատվությունների մակարդակում։

## 3.10 Թեստավորում և վավերացում

Մշակվող համակարգի ճարտարապետական արժեքը չի կարող հիմնավորվել միայն տեսական նկարագրությամբ, ուստի նախագծման ընթացքում նախատեսվել է բազմաշերտ վավերացման մոտեցում։ Միաժամանակ պետք է ընդգծել, որ համակարգը դեռևս գտնվում է ակտիվ մշակման փուլում։ Նրա հիմնական ճարտարապետական շերտերը և առանցքային մոդուլները արդեն ձևավորված են, սակայն լիարժեք փորձարարական փորձարկումները և ավարտուն քանակական գնահատումները դեռևս ամբողջությամբ չեն ավարտվել։

Այս պատճառով տվյալ ենթաբաժնում նկարագրվող ստուգման մեխանիզմները ներկայացվում են որպես համակարգի վավերացման նախատեսվող և մասամբ իրագործված մեթոդաբանություն, այլ ոչ թե որպես վերջնական, ամբողջությամբ փակված փորձարարական ցիկլի արդյունք։ Այստեղ առաջնայինը ոչ թե պատրաստի ցուցանիշներ ներկայացնելն է, այլ ցույց տալը, թե ինչ սկզբունքներով պետք է ստուգվի նման ճարտարապետության կայունությունը, վերահսկելիությունը և ընդարձակելիությունը։

Նախատեսվող ստուգման ուղղությունները ներառում են մի քանի մակարդակ։ Առաջին մակարդակը ավտոմատացված կոդային ստուգումներն են, որոնք պետք է կիրառվեն classifier routing-ի, persona head materialization-ի, graph hygiene-ի, duplicate merging-ի, garbage collection-ի, file ingestion-ի, context token limit-ի և background rebuild logic-ի նկատմամբ։ Այս ստուգումները կարևոր են, որովհետև համակարգը բաղկացած է փոխկապակցված deterministic շերտերից, և ճարտարապետական մակարդակի փոքր շեղումը կարող է ազդել ամբողջ մշակման շղթայի վրա։

Երկրորդ մակարդակը կատարման ընթացքում վավերացումն է։ Մշակված ճարտարապետության մեջ նախատեսված է, որ graph store-ը յուրաքանչյուր էական պահպանման գործողությունից առաջ normalize և validate անի graph state-ը։ Նման ստուգումները պետք է օգնեն վաղ փուլում հայտնաբերել orphan edges, duplicate nodes, invalid structures կամ empty overwrite-ի վտանգավոր դեպքերը։ Թեև նման մեխանիզմների ճարտարապետական հիմքը արդեն առկա է, դրանց լրիվ փորձարարական գնահատումը դեռ շարունակվող աշխատանքի մաս է։

Երրորդ մակարդակը վերականգնմանն ուղղված ստուգումն է։ Նախատեսվում է, որ chat response-ից հետո համակարգը կարող է ետին պլանում կրկին անցնել session-ի և փաստաթղթերի extraction path-ով, կիրառել graph hygiene և վերադարձնել վերականգնման վիճակի մասին տեղեկություն։ Այս մոտեցումը կարևոր է հատկապես agent memory համակարգերի համար, քանի որ այստեղ անհրաժեշտ է գնահատել ոչ միայն անմիջական պատասխանների որակը, այլ նաև այն, թե ինչպես է համակարգը պահպանում կամ վերականգնում իր ներքին վիճակը երկարաժամկետ աշխատանքի ընթացքում։

Այս պահին արդեն տեսանելի է, որ համակարգի ստուգման ճարտարապետությունը նախագծված է և հիմնական մոդուլները մեծ մասամբ իրականացված են, սակայն ամբողջական փորձարարական վավերացումը դեռևս մնում է շարունակվող հետազոտական և ինժեներական աշխատանքի մաս։

### 3.10.1 Վավերացման հոսքի դիագրամ

```text
code change
   |
   v
unit / subsystem tests
   |
   v
graph validation
   |
   v
full active suite
   |
   +--> failure detected? ---- yes ----> repair -> re-run validation
   |                                ^
   |                                |
   +--------------- no -------------+
```

Այս հոսքը ներկայացնում է նախատեսվող վավերացման և շարունակական ուղղման տրամաբանությունը։ Այն արտացոլում է այն սկզբունքը, ըստ որի համակարգը չի կարելի համարել բավարար չափով կայուն, եթե ճարտարապետական մակարդակի անհամապատասխանությունը շարունակում է գոյություն ունենալ, նույնիսկ այն դեպքում, երբ առանձին ֆունկցիոնալ մասերը ժամանակավորապես աշխատունակ են։

## 3.11 Առաջարկվող համակարգի առավելությունները

Նման տիպի persona-graph agent ճարտարապետությունների հիմնական առավելությունն այն է, որ դրանք բաժանում են լեզվային գեներացիայի խնդիրը և համակարգային կառավարման խնդիրը։ Այսպիսի տարանջատումը բարձրացնում է վերահսկելիությունը, քանի որ համակարգի կազմակերպումը, entity routing-ը և memory management-ը այլևս ամբողջությամբ չեն թողնվում գեներատիվ մոդելի ներքին վիճակին։

Առաջին ընդհանուր առավելությունը deterministic control layer-ի առկայությունն է։ Երբ routing-ը, classification-ը, graph update-ը և context assembly-ը իրականացվում են կանոնակարգված ծրագրային շերտերով, համակարգը դառնում է ավելի բացատրելի և հեշտ վերլուծելի։ Սա կարևոր է հատկապես այն միջավայրերում, որտեղ պահանջվում է հասկանալ ոչ միայն արդյունքը, այլ նաև արդյունքին բերած մեխանիզմը։

Երկրորդ առավելությունը explainable memory-ի հնարավորությունն է։ Եթե agent memory-ն կառուցված է graph representation-ի և բաց storage-ի հիման վրա, հետազոտողը կարող է ուսումնասիրել գիտելիքի կառուցվածքը, entities-ի կապերը և retrieval-ի հիմքերը։ Այսպիսի մոտեցումը նպաստում է վերարտադրելիությանը և բարդ համակարգերի գիտական վերլուծությանը։

Երրորդ առավելությունը persona-consistent վարքագծի ձևավորման հնարավորությունն է։ Եթե persona-ն ներկայացվում է ոչ միայն տեքստային նկարագրությամբ, այլ առանձին վիճակ պահող կառուցվածքով, ապա հնարավոր է ավելի կայուն կերպով միավորել traits-ը, relations-ը, emotion state-ը և վարքագծային patterns-ը։

Չորրորդ առավելությունը graph hygiene-ի գաղափարական ներառումն է։ Knowledge graph-ով աշխատող շատ համակարգերի դեպքում հիմնական խնդիրը ոչ թե գիտելիքի ավելացումն է, այլ դրա երկարաժամկետ պահպանման կարգապահությունը։ Importance decay-ը, duplicate detection-ը, garbage collection-ը և graph compression-ը հիշողության շերտը դարձնում են ավելի հարմար reasoning-ի և retrieval-ի հետագա փուլերի համար։

Հինգերորդ առավելությունը փաստաթղթերից սովորելու վերահսկելի մեխանիզմն է։ Երբ փաստաթղթերից ստացված տեղեկատվությունը անցնում է extraction, validation և graph merge փուլերով, հնարավոր է նվազեցնել անմիջական և չստուգված text-to-response անցումների ռիսկը։ Այս մոտեցումը բարելավում է հետագա ինտեգրման հնարավորությունները և ընդլայնում է համակարգի կիրառելիությունը գիտելիք-կենտրոնացված խնդիրներում։

### 3.11.1 Մշակված համակարգի առանձնահատկությունները

Մշակվող համակարգի ճարտարապետական առանձնահատկությունները առկա ընդհանուր մոտեցումների համեմատ հստակ դրսևորվում են մի քանի ուղղություններով։ Առաջին առանձնահատկությունը deterministic orchestration-ն է։ Համակարգի հիմնական մշակման շղթան կառուցված է որպես հաջորդական, հստակ առանձնացված փուլերի շարք, որտեղ chat մուտքը անցնում է message analyzer, feature extractor, classifier forest, head caller և context builder փուլերով, և միայն դրանից հետո փոխանցվում է `LLM`-ին։ Սա նվազեցնում է այն կախվածությունը, երբ ամբողջ կառավարման հոսքը որոշվում է մեկ գեներատիվ մոդելի կողմից։

Երկրորդ առանձնահատկությունը file-first memory architecture-ն է։ Global graph-ը, session history-ը, uploaded files-ը և persona heads-ը պահվում են filesystem-ում ընթեռնելի կառուցվածքով։ Այդ ընտրությունը ոչ միայն տեխնիկական պարզեցում է, այլ նաև հետազոտական առավելություն, որովհետև այն հեշտացնում է վիճակի անմիջական դիտարկումը, փորձերի վերարտադրումը և սխալների հետագծումը։

Երրորդ առանձնահատկությունը persona heads-ի folder-based ներկայացումն է։ Յուրաքանչյուր persona ներկայացվում է ոչ թե մեկ ընդհանուր նկարագրությամբ, այլ առանձին ֆայլային փաթեթով, որը ներառում է traits, relations, examples, knowledge, emotion vector և meta-information։ Այս ճարտարապետությունը persona-ն վերածում է հիշողության առանձին միավորի, այլ ոչ թե միայն prompt template-ի։

Չորրորդ առանձնահատկությունը graph hygiene mechanisms-ի ներառումն է որպես հիմնական շերտ։ Մշակվող համակարգում duplicate detection-ը, importance decay-ը, garbage collection-ը և summary-node compression-ը դիտարկվում են որպես knowledge graph-ի կյանքի ցիկլի բնական մաս։ Այս լուծումը կարևոր է, քանի որ թույլ է տալիս memory graph-ը պահել ավելի կարգապահ և reasoning-ի համար օգտագործելի վիճակում։

Հինգերորդ առանձնահատկությունը `LLM`-ի և համակարգային տրամաբանության միջև հստակ տարանջատումն է։ Ճարտարապետության մեջ ընդունված է, որ `LLM`-ը ծառայում է հիմնականում երկու նպատակների` knowledge extraction և response generation։ Routing-ը, graph modifications-ը, head spawning-ը և context budgeting-ը կատարվում են ծրագրային կանոններով։ Այսպիսի բաժանումը համակարգին տալիս է ավելի կանխատեսելի և ինժեներապես վերահսկվող բնույթ։

Վեցերորդ առանձնահատկությունը emotion vectors-ի և situation-reaction learning-ի ներառումն է persona layer-ում։ Թեև այս բաղադրիչների հետագա կայունացումը դեռևս շարունակվող աշխատանքի մաս է, արդեն ճարտարապետական մակարդակում նրանք ապահովում են, որ persona-ն կարողանա ներկայացվել ոչ միայն որպես փաստական պրոֆիլ, այլ նաև որպես սահմանափակ վարքագծային, վիճակային մեքենա հիշեցնող կառուցվածք։

## 3.12 Եզրակացություններ

Գլխի շրջանակում դիտարկվեց persona-graph agent համակարգի տեսական հիմքը, համեմատական համատեքստը և առաջարկվող ճարտարապետական կառուցվածքը։ Ցույց տրվեց, որ սովորական `LLM`-ահեն զրուցային համակարգերի երկարաժամկետ հիշողության, factual grounding-ի և persona stability-ի խնդիրները կարելի է մեղմել բազմաշերտ agent architecture-ի օգնությամբ, որտեղ graph memory-ն, persona heads-ը և context construction-ը հանդես են գալիս որպես առանձին նախագծային շերտեր։

Քննարկված լուծումները ցույց են տալիս, որ file-first memory model-ը, folder-based persona representation-ը, deterministic routing-ը, bounded context construction-ը և graph hygiene-ի մեխանիզմները ձևավորում են հետազոտական տեսանկյունից խոստումնալից ճարտարապետական շրջանակ։ Այդ շրջանակը արժեքավոր է նրանով, որ այն լեզվային մոդելին տրամադրում է ավելի կառուցվածքային, պատճառաբանելի և persona-ն հաշվի առնող միջավայր։

Միաժամանակ հարկ է ընդգծել, որ գլխում նկարագրված համակարգը ներկայում հանդես է գալիս որպես մշակվող հետազոտական նախագիծ, որի ճարտարապետական հիմքերը արդեն ձևավորված են, բայց որի լիարժեք կայունացումը և ամբողջական փորձարարական գնահատումը շարունակում են մնալ հետագա աշխատանքի առարկա։ Հետևաբար այս գլխի հիմնական արդյունքը պատրաստի արտադրական համակարգի ամփոփումը չէ, այլ այն ինժեներական սկզբունքների և կառուցվածքային լուծումների ներկայացումը, որոնց վրա նման համակարգը կարող է զարգացվել։

### 3.12.1 Համակարգի հետագա զարգացումը

Համակարգի հետագա զարգացումը նախատեսում է մի քանի փոխկապակցված ուղղություններ։ Նախ և առաջ, այն ներկայում բնութագրվում է որպես հետազոտական նախատիպ (`research prototype`)։ Սա նշանակում է, որ ճարտարապետական հիմնական գաղափարները, տվյալների հոսքերը և հիշողության մոդելը արդեն մշակված են, սակայն համակարգի վերջնական կայունացումը, ամբողջական ինտեգրացիոն փորձարկումները և երկարաժամկետ շահագործման գնահատումը դեռևս բաց խնդիրներ են։

Հետագա աշխատանքի առաջին ուղղությունը համակարգի ընդհանուր ստաբիլացումն է։ Այստեղ ներառվում են հիշողության կյանքի ցիկլի հետագա կարգաբերում, graph hygiene-ի վարքագծի երկարատև դիտարկում, ետին պլանի վերականգնման ցիկլի լրացուցիչ ստուգումներ և persona-ի վիճակային դինամիկայի հավասարակշռում։ Նպատակն է ապահովել, որ ճարտարապետության առանձին շերտերը ոչ միայն տեսականորեն համատեղելի լինեն, այլ նաև կայուն աշխատեն փոխադարձ ներգործության պայմաններում։

Երկրորդ ուղղությունը լեզվային մոդելի ինտեգրման ավարտուն կազմակերպումն է։ Ճարտարապետական տեսանկյունից համակարգը նախագծված է ճկուն ձևով և չի կապվում միայն մեկ կոնկրետ մոդելային ենթակառուցվածքի հետ։ Հետագա զարգացման ընթացքում նախատեսվում է ավարտին հասցնել այդ շերտի ինտեգրումը այնպես, որ համակարգը կարողանա աշխատել առնվազն երկու հիմնական ռեժիմներով։

Առաջին հնարավոր ռեժիմը տեղային լեզվային մոդելի օգտագործումն է, օրինակ `*.gguf` ձևաչափով տեղակայված մոդելների միջոցով։ Այս մոտեցումը կարևոր է այն դեպքերում, երբ պահանջվում է օֆֆլայն աշխատանք, տվյալների տեղային վերահսկում կամ հետազոտական փորձերի ամբողջական վերարտադրելիություն նույն հաշվարկային միջավայրում։

Երկրորդ հնարավոր ռեժիմը արտաքին ծառայությունների օգտագործումն է, օրինակ `OpenAI API`-ի կամ համանման հարթակների միջոցով։ Այս ռեժիմը ճարտարապետորեն օգտակար է, քանի որ հնարավորություն է տալիս համեմատել տարբեր մոդելային հետնամասային ծառայություններ, օգտագործել ավելի հզոր արտաքին inference ծառայություններ և գնահատել, թե ինչպես է նույն deterministic control layer-ը աշխատում տարբեր լեզվային մոդելների հետ։

Կարևոր է ընդգծել, որ այս երկակի հնարավորությունը ներկայացնում է համակարգի ճարտարապետական ճկունությունը, այլ ոչ թե արդեն ամբողջությամբ ավարտված ինտեգրումը։ Հետագա հետազոտական և ինժեներական աշխատանքը պետք է ուղղված լինի հենց այս ճկունությունը գործնականորեն կայուն և վերահսկելի դարձնելուն։
