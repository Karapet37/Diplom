# Persona-Graph-Agent. Stateful, Persona-Aware, Graph-Grounded Ինտելեկտուալ Համակարգի Նախնական Տարբերակը

> **Ծանոթություն.** Այս ֆայլը նախատեսված է copy-paste-ի համար։ Վերնագրի էջերը (4 կայուն էջ) առկա են КК (1).docx ֆայլում և ձեռքով պետք է ավելացվեն։ Գրականության ցանկում `[N]` հղումները ստորև տեքստում նշված են անմիջապես համապատասխան կետի մոտ։

---

## ՆԵՐԱԾՈՒԹՅՈՒՆ

Վերջին տարիներին մեծ լեզվային մոդելների և դրանց հիմքով կառուցվող agent համակարգերի զարգացումը զգալիորեն ընդլայնել է արհեստական բանականության կիրառման հնարավորությունները։ Այդպիսի համակարգերը կիրառվում են ծրագրավորման, վերլուծության, փաստաթղթերի մշակման, հարցուպատասխանի, թվային օգնականների և մասնագիտական տարբեր միջավայրերի մեջ։ Սակայն նույնիսկ այս արագ առաջընթացի պայմաններում պահպանվում է մի հիմնարար գործնական խնդիր. օգտագործողը հաճախ ստիպված է երկար ժամանակ ծախսել, իր պահանջը մի քանի անգամ տարբեր ձևերով բացատրել, սխալ մեկնաբանությունները ուղղել և համակարգը «մոտեցնել» իր իրական նպատակին, մինչև ստանում է օգտակար պատասխան։

Այս խնդիրը միայն հալյուցինացիաների խնդիր չէ։ Շատ հաճախ համակարգը տալիս է արտաքուստ ճիշտ, լեզվապես վստահ, նույնիսկ լավ կառուցված պատասխան, որը, սակայն, լիովին չի համապատասխանում օգտագործողի ակնկալիքին։ Այն կարող է չափազանց ընդհանուր լինել, սխալ հասկանալ խնդրի ձևը, խառնել բովանդակությունն ու կառուցվածքը, սխալ ընտրել պատասխանի ձևաչափը կամ պարզապես չըմբռնել, թե իրականում ինչ է ուզում մարդը։ Հետևաբար խնդիրը պետք է դիտարկել ոչ միայն որպես factual accuracy-ի խնդիր, այլ նաև որպես նպատակի ընկալման, վարքի կայունության և runtime կազմակերպման խնդիր։

Այս սահմանափակումները հաղթահարելու համար AI ոլորտում առաջարկվել են մի շարք հետազոտական ուղղություններ։ Retrieval համակարգերը փորձում են grounding տալ արտաքին փաստաթղթերի միջոցով։ GraphRAG մոտեցումները փորձում են գիտելիքը դարձնել ավելի կառուցվածքային։ Reasoning/prompting մեթոդները ստիպում են մոդելին անցնել միջանկյալ քայլերով։ Tool-using agent-ները լուծումը բաժանում են պլանավորման, գործիքների օգտագործման և վերջնական պատասխանի փուլերի։ Persona-oriented dialogue համակարգերը փորձում են նվազեցնել չափազանց ընդհանուր վարքը, իսկ memory-oriented architectures-ը՝ ապահովել continuity և երկարատև փոխազդեցություն։ Verification և correction մոտեցումները, իրենց հերթին, փորձում են ոչ միայն գեներացնել պատասխան, այլ նաև ստուգել այն մինչև վերջնական ներկայացումը։

Սակայն այս ուղղությունների համեմատական ուսումնասիրությունը ցույց է տալիս, որ դրանց մեծ մասը ուժեղացնում է խնդրի առանձին կողմերը, բայց չի վերակազմակերպում ամբողջ AI runtime-ը։ Retrieval-ը բարելավում է knowledge grounding-ը, բայց ինքնին չի լուծում persona continuity-ը։ Prompting-ը բարելավում է reasoning-ի տեսքը, բայց չի ստեղծում իրական operational state։ Tool use-ը օգնում է բազմափուլ կատարմանը, բայց առանց persona/state layer-ի համակարգը շարունակում է մնալ չափազանց ընդհանուր։ Memory-ն կարևոր է continuity-ի համար, բայց եթե այն խիստ չի կառավարվում, կարող է ինքն էլ դառնալ նոր սխալների աղբյուր։

Այս աշխատանքի հիմքում ընկած է այն գաղափարը, որ մեծ լեզվային մոդելի շուրջ պետք է կառուցել ոչ թե պարզապես ավելի մեծ prompt, այլ ավելի խիստ կազմակերպված wrapper/runtime համակարգ։ Այդ wrapper-ը պետք է ապահովի՝ օգտագործողի հարցման նախնական ուսումնասիրություն, նպատակի պարզում, persona-ի և թեմայի տարանջատում, համապատասխան working context-ի սահմանափակ կառուցում, պատասխանի review/shaping մինչև վերջնական գեներացում, ինչպես նաև հիշողության և graph-ի վերահսկվող թարմացում։

Սույն աշխատանքի նպատակն է առաջարկել Persona-Graph-Agent համակարգի նախնական տարբերակը՝ որպես graph-grounded, file-backed, state-transition runtime, որը պետք է մեղմի մեծ լեզվային մոդելների չափազանց ընդհանրական, անկայուն և երբեմն հալյուցինացիոն վարքը։ Աշխատանքի շրջանակում ուսումնասիրվում են համապատասխան հետազոտական ուղղությունները, վերլուծվում են դրանց սահմանափակումները, ձևակերպվում է առաջարկվող համակարգի կառուցվածքային գաղափարը և նկարագրվում են դրա հիմնական ծրագրային շերտերը։

Ուսումնասիրության օբյեկտը stateful, persona-aware, graph-grounded ինտելեկտուալ համակարգն է, որը մշակում է օգտագործողի բնական լեզվով հարցումները։ Ուսումնասիրության առարկան այն ճարտարապետական և ծրագրային մեխանիզմներն են, որոնց միջոցով հնարավոր է նվազեցնել մեծ լեզվային մոդելների չափազանց ընդհանրական և թույլ վերահսկվող վարքը՝ այն փոխարինելով ավելի խիստ կազմակերպված state-transition մոտեցմամբ։

---

## ԳԼՈՒԽ 1. ՄԵԾ ԼԵԶՎԱՅԻՆ ՄՈԴԵԼՆԵՐԻ ՉԱՓԱԶԱՆՑ ԸՆԴՀԱՆՐԱԿԱՆ, ԱՆԿԱՅՈՒՆ ԵՎ ՀԱԼՅՈՒՑԻՆԱՑԻԱՆԵՐՈՎ ԼԻ ՎԱՐՔԸ ՀԱՂԹԱՀԱՐԵԼՈՒ ՀԵՏԱԶՈՏԱԿԱՆ ՄՈՏԵՑՈՒՄՆԵՐԸ

### 1.1. Խնդրի ընդհանուր բնութագիրը

Մեծ լեզվային մոդելների հիմնական գործնական խնդիրներից մեկը նրանց չափազանց ընդհանրական վարքն է։ Շատ դեպքերում մոդելը կարող է տալ լեզվապես գեղեցիկ, վստահ և ամբողջական թվացող պատասխան, որը, սակայն, քիչ է համապատասխանում օգտագործողի իրական մտադրությանը։ Նման պատասխաններում հաճախ միաժամանակ առկա են մի քանի թերություններ՝

- չափազանց ընդհանուր ձևակերպում,
- խնդրի նպատակի մասնակի սխալ ընկալում,
- reasoning-ի անկայունություն,
- persona continuity-ի բացակայություն,
- հալյուցինացիոն կամ չհիմնավորված պնդումների ներմուծում։

Հենց այս խնդրի շուրջ էլ ձևավորվել են մի շարք հետազոտական ուղղություններ, որոնք տարբեր ճանապարհներով փորձում են բարձրացնել LLM-ների օգտակարությունն ու վստահելիությունը [1]։

### 1.2. Retrieval և grounding մոտեցումներ

Այս ուղղության հիմնական գաղափարն այն է, որ մոդելը չպետք է պատասխան ձևավորի միայն իր պարամետրային գիտելիքի հիման վրա, այլ պետք է grounding ստանա արտաքին աղբյուրներից։ Այդ պատճառով լայնորեն կիրառվել են Retrieval-Augmented Generation (RAG) համակարգերը, որտեղ նախ որոնվում են հարցման հետ կապված փաստաթղթեր կամ գիտելիքային հատվածներ, ապա դրանք տրվում են մոդելին որպես supporting context [8]։

Այս մոտեցման ուժեղ կողմն այն է, որ այն բարելավում է factual grounding-ը և հատկապես օգտակար է այն դեպքերում, երբ անհրաժեշտ է հենվել կոնկրետ կամ արտաքին knowledge source-երի վրա։ Սակայն retrieval-ը հիմնականում լուծում է «ինչ տեղեկատվություն տալ մոդելին» խնդիրը, ոչ թե ամբողջությամբ «ինչպես պետք է հասկանալ իրավիճակը և ինչ կերպ պետք է կառուցել պատասխանը» խնդիրը։ Այդ պատճառով retrieval-ով հագեցած համակարգն անգամ կարող է մնալ generic assistant, եթե չունի persona/state continuity և current working context-ի խիստ կառավարում։

### 1.3. GraphRAG և կառուցվածքային գիտելիքի օգտագործումը

Retrieval ուղղության զարգացած տարբերակը GraphRAG-ն է, որտեղ grounding-ը կառուցվում է ոչ միայն փաստաթղթերի, այլ նաև graph կառուցվածքների հիման վրա [2]։ Այս մոտեցումը կարևոր է այն դեպքերում, երբ անհրաժեշտ է ոչ միայն գտնել փաստ, այլ նաև հասկանալ entity-ների միջև կապերը, բազմաքայլ հարաբերությունները և կառուցվածքային evidence-ը [14]։

GraphRAG-ի առավելությունն այն է, որ այն կարող է բարելավել knowledge representation-ը և reasoning-ի հիմքը։ Սակայն նույնիսկ graph-aware retrieval-ը ինքնին դեռ չի ապահովում persona continuity, behavioral consistency կամ stateful response shaping։ Այսինքն՝ graph-ը կարող է լավացնել գիտելիքի կառուցվածքը, բայց դեռ պարտադիր չի դարձնում ամբողջ runtime-ը լավ կազմակերպված։

### 1.4. Reasoning և prompting ռազմավարություններ

Հետազոտությունների մեկ այլ մեծ ուղղություն փորձում է ստիպել մոդելին չանցնել անմիջապես վերջնական պատասխանի, այլ նախ կառուցել reasoning-ի միջանկյալ քայլեր [9]։ Այստեղ մտնում են chain-of-thought, self-consistency, decomposition, verification prompting և նման ռազմավարություններ։ Ավելի ուշ աշխատանքներ ցույց են տալիս, որ reasoning-ը կարելի է համատեղել action-ի հետ ReAct ձևաչափով [10]։

Այս մեթոդների առավելությունն այն է, որ դրանք նվազեցնում են լիովին անմիջական և մակերեսային պատասխանների հավանականությունը։ Սակայն դրանց հիմնական սահմանափակումն այն է, որ reasoning prompting-ը հաճախ բարելավում է reasoning-ի արտաքին ներկայացումը, բայց պարտադիր չի ստեղծում իրական operational state։ Այլ կերպ ասած՝ մոդելը կարող է «գրել մտածելու նմանվող քայլեր», բայց դեռ չունենալ հստակ տարբերակում՝ որն է նպատակը, որն է ձևը, որն է persona-ն, և որ knowledge-ն է այժմ իրականում կարևոր։

### 1.5. Tool use և agent frameworks

Գործիքների օգտագործման և բազմափուլ execution-ի ուղղությամբ կարևոր տեղ են զբաղեցնում agent framework-ները և tool-using համակարգերը [3][10]։ Այս մոտեցումների հիմնական գաղափարն այն է, որ խնդիրն անհրաժեշտ է բաժանել փուլերի՝ պլանավորում, tool call, ստացված արդյունքի ինտեգրում և վերջնական պատասխան [4]։

Այս ուղղության առավելությունն այն է, որ մոդելը դադարում է լինել միայն «տեքստ գրող» և սկսում է աշխատել նաև որպես orchestration engine։ Սակայն առանց հստակ persona/state layer-ի նույնիսկ բազմափուլ agent-ները հաճախ պահպանում են չափազանց ընդհանուր վարք։ Նրանք կարող են ավելի լավ tool call անել կամ ավելի լավ բաժանել խնդիրը քայլերի, բայց դեռ չհասկանալ՝ օգտագործողը ինչ է ակնկալում պատասխանի ձևի, խորության և կառուցվածքի տեսանկյունից։

### 1.6. Persona-oriented dialogue և personalization

Persona-ի և personalization-ի ուղղությունը փորձում է հաղթահարել մեծ լեզվային մոդելների «միևնույն բոլորը համար» բնույթը [11]։ Persona-հիմքով համակարգերը փորձում են մոդելին տալ կայուն ինքնություն, խոսքի նախընտրելի ձև, հարաբերություններ, background knowledge կամ վարքային գիծ, որպեսզի պատասխանը լինի ոչ միայն ընդհանուր, այլ նաև անձնավորված և ավելի consistent։

Այս ուղղությունը ցույց է տվել, որ persona-ն կարող է բարելավել style consistency-ն և երկխոսության կապակցվածությունը։ Սակայն այստեղ կա կարևոր սահմանափակում. շատ աշխատանքներում persona-ն մնում է պարզապես prompt-ի մաս կամ տեքստային profile block։ Այդ դեպքում այն կարող է օգնել ոճին, բայց չի դառնում լիարժեք operational state։ Եթե persona-ն չունի իր առանձին state-ը, revision history-ն, reaction pattern-երը, decision pattern-երը և local graph-ը, ապա այն շարունակում է մնալ ավելի շատ գեներացիոն հուշում, քան համակարգի իրական վարքային հիմք։

### 1.7. Memory-oriented systems

Memory ուղղությունը փորձում է լուծել այն խնդիրը, որ մոդելը սովորաբար չունի իրական երկարաժամկետ աշխատանքային հիշողություն և յուրաքանչյուր նոր turn-ում մասամբ սկսում է նորից [12]։ Այդ պատճառով առաջարկվել են long-term profile storage, episodic memory, semantic memory, compressed summaries և այլ մեխանիզմներ։

Այս մոտեցումների կարևորությունն այն է, որ continuity-ի բացակայության պատճառով օգտվողը հաճախ ստիպված է նույն սահմանափակումը, preference-ը կամ context-ը նորից ու նորից բացատրել։ Սակայն memory ունենալը դեռ բավարար չէ։ Եթե write-path-ը թույլ է վերահսկվում, current working context-ը չի տարբերակվում long-term memory-ից, կամ սխալ եզրակացություններն էլ են պահվում որպես հիշողություն, ապա memory-ն կարող է դառնալ ոչ թե օգնություն, այլ նոր սխալների աղբյուր։

### 1.8. Verification, detection և correction մոտեցումներ

Հետազոտությունների մեկ այլ կարևոր ուղղություն կապված է պատասխանի ստուգման հետ [13]։ Այս մոտեցումների հիմքում ընկած է այն գաղափարը, որ մոդելին չպետք է լիովին վստահել վերջնական պատասխանի մակարդակում։ Փոխարենը անհրաժեշտ է ունենալ երկրորդ շերտ, որը կարող է ստուգել, հայտնաբերել կամ ուղղել թույլ, չհիմնավորված կամ հալյուցինացիոն հատվածները։

Այս ուղղության մեջ մտնում են self-checking, claim verification, external fact-checking, critique loop և correction pipeline-ները։ Այս մոտեցումների առավելությունն այն է, որ դրանք բացահայտորեն ընդունում են AI-ի սահմանափակելիությունը և փորձում են ոչ թե միայն ավելի լավ պատասխան գեներացնել, այլ նաև ավելի խիստ վերահսկել արդեն գեներացված արդյունքը։

### 1.9. Ընդհանուր եզրակացություն առաջին գլխի համար

Վերոհիշյալ բոլոր հետազոտական ուղղությունները ցույց են տալիս, որ ոլորտում արդեն գոյություն ունեն բազմաթիվ լուրջ փորձեր՝ նվազեցնելու LLM-ների չափազանց ընդհանրական, անկայուն և հալյուցինացիաներով հարուստ վարքը։

Դրանք հիմնականում ուժեղացնում են հետևյալ գործոններից մեկը կամ մի քանիսը՝

- grounding,
- reasoning,
- tool use,
- persona consistency,
- continuity,
- verification։

Սակայն գործնականում օգտագործողը հաճախ շարունակում է տանջվել մինչև ստանում է իր ուզածին մոտիկ պատասխան, որովհետև այս ուղղությունները շատ դեպքերում ուժեղացնում են խնդրի առանձին կողմերը, բայց չեն վերակազմակերպում ամբողջ runtime-ը որպես խիստ վերահսկվող, state-aware, persona-aware և graph-aware operational համակարգ։

Հենց այս բացն է հիմք տալիս առաջարկելու այլ մոտեցում, որտեղ հիմնական հարցը դառնում է ոչ թե միայն «ինչքան շատ բան տալ մոդելին», այլ՝

- ինչ հերթականությամբ պետք է այն աշխատի,
- ինչ state-ով պետք է պատասխան ձևավորի,
- ինչպես պետք է առանձնացվի current working context-ը,
- ինչպես պետք է վերահսկվի write-path-ը,
- և ինչպես կարելի է մոդելը պահել ավելի կայուն վարքի սահմաններում։

---

## ԳԼՈՒԽ 2. PERSONA-GRAPH-AGENT ՀԱՄԱԿԱՐԳԻ ՃԱՐՏԱՐԱՊԵՏԱԿԱՆ ԵՎ ԾՐԱԳՐԱՅԻՆ ՆԿԱՐԱԳՐՈՒԹՅՈՒՆԸ

### 2.1. Առաջարկվող համակարգի ընդհանուր գաղափարը

Այս աշխատանքում ներկայացվում է `Persona-Graph-Agent` համակարգը, որը կառուցված է ոչ թե անմիջական `հաղորդագրություն → պատասխան` սկզբունքով, այլ վիճակային անցումների վերահսկվող ճարտարապետությամբ [1]։

Համակարգի հիմնական գաղափարն այն է, որ օգտատիրոջ հաղորդագրությունը դիտարկվում է որպես ազդեցություն համակարգի ընթացիկ վիճակի վրա։ Հաղորդագրությունը պետք է պարզի.

- ինչ է փոխվում ընթացիկ վիճակում,
- որ persona-ն է խոսողը,
- որն է քննարկվող թեման,
- ինչ ռիսկեր և առաջնահերթություններ են ակտիվանում,
- ինչ նյութեր պետք է մտնեն ընթացիկ աշխատանքային համատեքստ։

Միայն այս փուլերից հետո է թույլատրվում ձևավորել վերջնական պատասխանը։

Համակարգի ընդհանուր սկզբունքը հետևյալն է.

```
օգտատիրոջ հաղորդագրություն
→ ընթացիկ վիճակի ընթերցում
→ ազդեցության մեկնաբանում
→ սահմանափակ state transition
→ working context-ի կառուցում
→ working context-ի review
→ response shaping
→ final generation
→ transition logging
→ current context persistence
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

### 2.2. Persona-Graph-Agent համակարգի կառուցվածքային նկարագրությունը

#### 2.2.1. Համակարգի հիմնական շերտերը

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

#### 2.2.2. Persona կառուցվածքը

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

Persona subsystem-ի կարևոր առանձնահատկությունն այն է, որ այն չի պահվում մեկ միասնական "description" դաշտում։ Յուրաքանչյուր շերտ ունի իր դերը.

- baseline-ը պահպանում է համեմատաբար կայուն նույնականությունը,
- dynamic state-ը պահպանում է ընթացիկ հուզական և իրավիճակային փոփոխությունները,
- learned patterns-ը պահպանում է փոխազդեցությունից ստացված սահմանափակ սովորած վարքագիծը։

Այս բաժանումը թույլ է տալիս միաժամանակ պահպանել persona-ի կայունությունը և թույլատրել սահմանափակ ադապտացիա։

#### 2.2.3. Graph memory

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

#### 2.2.4. Հիշողության շերտերը

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

#### 2.2.5. Mood research layer

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

#### 2.2.6. Operator interface

Frontend-ը կառուցված է որպես operator workspace, ոչ թե որպես պարզ chat window։ Այն ներառում է առանձին մակերեսներ հետևյալ խնդիրների համար.

- chat,
- graph workspace,
- persona inspection,
- file ingestion,
- diagnostics։

Operator interface-ի առկայությունը ճարտարապետական տեսանկյունից կարևոր է, քանի որ համակարգը նախատեսված է ոչ միայն runtime execution-ի, այլ նաև դիտարկելիության, ստուգման և վերահսկման համար։

### 2.3. Persona-Graph-Agent համակարգի ծրագրային նկարագրությունը

#### 2.3.1. Գլխավոր runtime path-ը

Համակարգի հիմնական գործարկման կետը `start.py` ֆայլն է։ Գործող runtime path-ը հետևյալն է.

```
start.py
  → bootstrap_runtime_environment()
  → get_runtime_config()
  → src.web.combined_app.create_combined_app()
    → agent_system.api.create_app()
    → src.web.api.attach_frontend_routes()
```

Այսպիսով, տեղային գործարկման դեպքում համակարգը իրենից ներկայացնում է combined app, որտեղ միևնույն runtime միջավայրում միավորված են.

- backend API,
- frontend routes,
- operator UI։

Այս runtime path-ը կարևոր է, որովհետև այն ցույց է տալիս, որ համակարգի իրական աշխատանքային միջավայրը միասնական է. backend-ը, diagnostics-ը և operator surface-ը միմյանցից անջատ չեն։

#### 2.3.2. Backend-ի հիմնական մոդուլները

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

#### 2.3.3. Chat runtime-ի հիմնական քայլերը

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

#### 2.3.4. Interaction routing

`interaction_routing.py` մոդուլը առանձնացնում է.

- խոսող persona-ն,
- քննարկվող entity-ն,
- follow-up mode-ը,
- explicit persona switch-ը։

Այս մոտեցումը կարևոր է, քանի որ համակարգը պետք է տարբերակի օրինակ հետևյալ երկու դեպքերը.

- երբ օգտատերը խոսում է տվյալ persona-ի հետ,
- երբ օգտատերը խոսում է մեկ persona-ի հետ, բայց հարցնում է մեկ այլ թեմայի կամ կերպարի մասին։

Այս բաժանումը թույլ է տալիս լուծել context continuity-ի այն խնդիրները, որոնք սովորաբար առաջանում են pronoun follow-up հարցերում, persona switch-ի դեպքերում և topic continuity-ի ընթացքում։

#### 2.3.5. Context builder

`context_builder.py` մոդուլը աշխատում է հետևյալ deterministic pipeline-ով.

```
collect → score → rank → compress → pack
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

Context builder-ի նպատակը ոչ թե «շատ բան հավաքելն» է, այլ ճիշտ բաները սահմանափակ քանակով ընտրելն ու pack անելն այնպես, որ final generator-ը ստանա պատճառականորեն օգտակար context։

#### 2.3.6. Staged prompt system

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

#### 2.3.7. Current working context-ի առանձին պահպանում

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

#### 2.3.8. Reliability և observability

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

#### 2.3.9. Կոգնիտիվ pipeline. PersonalityGenome և P1–P6 ճշգրտող ենթահամակարգ

Համակարգի ամենանոր ճարտարապետական շերտը `cognitive_pipeline.py` մոդուլն է՝ վեց perceptron մոդուլից կազմված կոգնիտիվ pipeline-ը (P1–P6)։ Այն մշակված է մաքուր NumPy-ով՝ առանց deep learning frameworks-ների կախվածության, ընդհանուր 7 700-ից ավելի ուսումնական պարամետրով։

**P1 — EventEncoder.** Բնական լեզվի դասակարգիչ, որը վերլուծում է մուտքային տեքստը և կշռված keyword-pattern-ների հիման վրա հաշվարկում է 12 event type-ների վրա softmax հավանականությունների բաշխումը (threat, intimacy, praise, criticism, loss_of_control, abandonment, failure, reward, intimacy, conflict, neutral, և այլն)։ P1-ի տրամաչափումը կատարվում է offline՝ label-ված training tuple-ների վրա cross-entropy կորստի gradients-ներով (gradient descent, 30 epoch)։

**P2 — EmotionalProcessor.** Ստանում է event-ի one-hot կոդավորումը և persona-ի baseline affective vector-ը, ձևավորում է ավելի բարձր, 32-ծավալ հուզական ներկայացում՝ ReLU ակտիվացմամբ։

**P3 — DefenseActivator.** 32→8 թաքնված շերտ, որի 8 ելքերը համապատասխանում են ութ psychodynamic defense մեխանիզմներին (avoidance, intellectualization, regression, splitting, reaction_formation, sublimation, projection, displacement)։ Ակտիվացումները sigmoid-ի միջոցով soft probability-ների են վերածվում, ապա ազդում persona-ի defensive posture-ի վրա։

**P4 — CognitiveBias.** 40→16 perceptron, որն ինտեգրում է event representation-ը genome-ի bias\_parameters-ների հետ և արտադրում 16-ծավալ cognitive filter vector, որն ազդում P5-ի ներդրումի վրա։

**P5 — ThoughtConstructor.** 88→32 երկշերտ ցանց (ReLU + linear), արտադրում է persona-ի «thought vector»-ը՝ պայմանավորված event-ից, defense activation-ից, cognitive bias-ից և genome regulator-ների snapshot-ից։

**P6 — ActionPolicy.** Երկշերտ MLP (input 88→64→12, ReLU), ընտրում է action family-ն (approach, withdraw, maintain, deflect, attack, self_protect, freeze, seek_support, explore, negotiate, placate, observe) softmax cross-entropy objective-ով։ P6-ը train-ի ընթացքում feedback-scaled gradient-ներ է ստանում բոլոր session-ների action_target label-ների վրա (30 epoch)։

**PersonalityGenome.** P-ները սնուցող genome-ը բաղկացած է 53 `LearnableParam` օբյեկտից, բաժանված 8 gene group-ի.

1. affective regulation (baseline\_anxiety, baseline\_mood, mood\_volatility),
2. drive system (drive\_closeness, drive\_achievement, drive\_control, drive\_autonomy),
3. fear system (fear\_rejection, fear\_shame, fear\_loss\_of\_control, fear\_abandonment, fear\_failure),
4. defense system (defense\_avoidance, defense\_regression, defense\_splitting, …),
5. cognitive style (analysis\_bias, feel\_first, impulsivity, …),
6. social style (social\_distance\_default, trust\_default, vulnerability\_concealment),
7. attachment (attachment\_style\_score, separation\_anxiety, …),
8. constitutional (stress\_threshold, recovery\_speed, …)։

Յուրաքանչյուր `LearnableParam`-ն ունի ձգողական regularization (elastic restoring force), drift log և version counter, որոնք թույլ են տալիս genome-ը medically-plausible trait drift-ի սահմաններում պահել։

**Calibration channels (3 ուղի).**

1. *Online reward signal* — յուրաքանչյուর turn-ի վերջում implicit feedback value-ն (±1.0) գրվում է `TrainingStore`-ի JSONL ֆայլ։
2. *Batch perceptron training* — `PerceptronTrainer`-ը գործարկում է P1-ի (event label supervised) և P6-ի (action target supervised) gradient training-ը 30 epoch-ով, feedback-scaled learning rate-ով (LR × (1 + |feedback|))։
3. *Explicit named feedback* — operator-ը կամ auto-evaluator-ը կարող է ուղղել `apply_explicit_feedback("too_cold", intensity=0.8, genome)` կանչով, որն անմիջապես step-ում է genome-ի համապատասխան `LearnableParam`-ները (12 predefined feedback type)։

**Attractor test harness.** Cognitive pipeline-ի regression verification-ի համար ստեղծված է `DataSets/idea_attractors/idea_attractors_seed.jsonl` dataset-ը (69 sample — ասացվածքներ, առակներ, propaganda slogan-ներ)։ `tests/agent_system/test_cognitive_pipeline_attractors.py`-ն ստուգում է 19 attractor property՝ ներառյալ defensive action-ների ի հայտ գալը, trust/intimacy pattern-ների ճիշտ դասակարգումը, QAnon-ոճ loss\_of\_control ազդանշանների վերծանումը, genome drift-ի կանխարգելումը, calibration loop-ի ավարտը և trainer converge-ումը։

### 2.4. Պահոցների և ֆայլային կառուցվածքի նկարագրությունը

#### 2.4.1. Գլխավոր ակտիվ կատալոգները

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

#### 2.4.2. Persona storage

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

#### 2.4.3. Session storage

Session history-ն պահվում է `memory/sessions/` կատալոգում։

#### 2.4.4. Graph storage

Graph storage-ի հիմնական ֆայլերն են.

- `memory/graphs/nodes.json`
- `memory/graphs/edges.json`

#### 2.4.5. File ingestion storage

Բեռնված փաստաթղթերը պահվում են.

- `memory/files/uploaded_documents/`

#### 2.4.6. Archive storage

Սառը և վերականգնելի storage շերտերը պահվում են.

- `memory/archive/`

#### 2.4.7. Runtime operational artifacts

Runtime-ի ընթացքում ստեղծվող գործող ֆայլերը պահվում են առանձին շերտում.

- `runtime/current_context/current_context.json`
- `runtime/current_context/current_context.txt`
- `runtime/logs/state_transitions.jsonl`
- `runtime/system_realism_reports/`

Այս շերտը կարևոր է, քանի որ այստեղ պահպանվում են ոչ թե long-term memory օբյեկտները, այլ ընթացիկ աշխատանքի և ստուգման արտեֆակտները։

### 2.5. Թեստավորում և ընթացիկ փորձնական արդյունքները

#### 2.5.1. Backend regression tests

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
- API failures-ը,
- cognitive pipeline attractor-ները (P1–P6, PersonalityGenome, calibration loop)։

Վերջին փաստացի արդյունքը.

```
python3 -m pytest tests/agent_system --collect-only -q
382 passed, 18 warnings
```

Cognitive pipeline-ի attractor test harness-ը (19 test) ստուգում է.

- defensive action-ների ի հայտ գալն իդեա-attractor input-ի պայմաններում,
- event-ի trust/intimacy pattern-ների ճիշտ դասակարգումը,
- propaganda-slogan-ների loss\_of\_control event ճանաչումը,
- genome drift-ի elastic regularization-ի կողմից կանխարգելումը,
- P1 և P6 calibration loop-ի ավարտն ու կոնվերգումը։

#### 2.5.2. End-to-end realism tests


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

```
python3 -m pytest tests/system_realism -q
18 passed
```

Այս երկու test layer-երը միասին տալիս են հետևյալ պատկերը.

- backend logic-ը ստուգվում է deterministic regression-ներով,
- live runtime behavior-ը ստուգվում է realism harness-ով,
- persona behavior, memory continuity և mutation flows-ը ստանում են առանձին verification surface,
- cognitive pipeline-ի psychological attractor properties-ը ստուգվում են specialized seed dataset-ով։


#### 2.5.3. Ուսուցման և թեստավորման տվյալների հավաքածուները

Համակարգի մշակման ընթացքում օգտագործվել են մի քանի արտաքին և սեփական dataset-ներ՝ interaction classification-ի, context quality-ի, empathy modeling-ի, clarification detection-ի և cognitive pipeline calibration-ի նպատակով։

**DailyDialog** [15] — բազմահերթ երկխոսությունների հավաքածու (76 116 record), ձեռք բերված Kaggle-ից։ Յուրաքանչյուր record ներկայացնում է dialogue turn-ի ոճ, intention tag (inform, question, directive, commissive) և emotion label (anger, disgust, fear, happiness, sadness, surprise, neutral)։ Կիրառվել է interaction routing-ի ու emotional tone detection-ի communicative pattern-ների ուսումնասիրության համար [15]։

**DialogStudio** [16] — Salesforce Research-ի dialogue research platform-ի consolidated collection-ից ստացված sub-dataset-ների converted bundle (`DataSets/DialogStudio/`), ներառյալ.

- `ConvAI2_converted_examples.json` — persona-conditioned dialogue-ներ, ConvAI2 competition-ից (~650 record),
- `empathetic_converted_examples.json` — empathy-tagged dialogue-ներ EmpatheticDialogues-ից [17] (~280 record),
- `hh_rlhf_converted_examples.json` — Anthropic HH-RLHF dataset-ից helpful/harmless label-ված exchange-ներ (~112 record),
- `PLACES3_5_converted_examples.json` — social situational dialogue-ներ (~288 record),
- `Prosocial_converted_examples.json` — prosocial reasoning dialog-ներ (~383 record)։

Ընդամենը ~1 728 local format-ի converted record, ձեռք բերված GitHub-ի պաշտոնական repository-ից [16]։

**EmpatheticDialogues** [17] — Facebook Research-ի EmpatheticDialogues dataset-ը (84 169 record, train split), պահվում է `DataSets/empatheticdialogues/` կատալոգում։ Ներառում է 32 հուզական կատեգորիայի label-ված dialogue-ներ և ծառայում է empathic reaction modeling-ի reference data-ի դեր։

**ClariQ** [18] — Microsoft Research-ի clarification question generation dataset-ը (`DataSets/ClariQ/`), ուղղված ambiguous information-seeking запросlar-ի clarification modeling-ին։ Ներբեռնված GitHub repository-ից [18]։ Ներառում է.

- train.tsv: 9 176 record,
- dev.tsv: 2 313 record,
- question_bank.tsv: 3 941 clarification question,
- multi_turn_human_generated_data.tsv: 499 record։

Ընդամենը ~15 929 record։

**idea_attractors seed** [19] — 69 JSONL record-ից կազմված attractor corpus (`DataSets/idea_attractors/idea_attractors_seed.jsonl`), ստեղծված Codex AI-ի կողմից, ապա ստուգված, ճշտված և (idea_attractors_seed.jsonl-ի բովանդակության մասով) համauthor-ված Claude-ի մասնակցությամբ։ Ներառում է.

- ժողովրդական ասացվածքներ (proverbs),
- առակային ձևակերպումներ (fable-style statements),
- propaganda slogan-ներ (QAnon-ոճ, authoritarian rhetoric),
- folk wisdom text pattern-ներ,

annotated-ված form, cluster, crowd_pull, quality_label և source_url դաշտերով։ Dataset-ի նպատակն է ապահովել reproducible attractor test harness-ը cognitive pipeline-ի P1-P6 ենթահամակարգի regression ստուգման համար [19]։


### 2.6. Գլխի եզրակացություն

Կատարված ուսումնասիրությունը ցույց է տալիս, որ մեծ համատեքստով և ընդհանրացված օգնականային համակարգերը բավարար չեն, եթե դրանցում բացակայում է persona-ի, հիշողության, graph grounding-ի և state transition-ի խիստ կազմակերպված շերտավորումը։

`Persona-Graph-Agent` համակարգում առաջարկված է այլ ճարտարապետական մոտեցում, որտեղ.

- persona-ն ներկայացվում է որպես կառուցվածքային stateful օբյեկտ,
- graph-ը գործում է որպես երկարաժամկետ կառուցվածքային հիշողություն,
- current working context-ը առանձնացված է long-term storage-ից,
- պատասխանն առաջանում է staged runtime pipeline-ից,
- transition history-ն պահվում է առանձին,
- LLM-ը ենթարկվում է սահմանափակված prompt stages-ի,
- cognitive pipeline-ը (P1–P6) ապահովում է persona-ի հուզական, պաշտպանական և ճանաչողական dynamika-ի lightweight perceptron մոդելավորումը։

Այսպիսով, համակարգը նպատակ ունի բարձրացնել ոչ թե միայն պատասխանների արտաքին բնականությունը, այլ դրանց պատճառական կապը ներքին վիճակի, persona structure-ի, graph memory-ի և session continuity-ի հետ։ Հենց այս հատկությունն է այն դարձնում ոչ թե սովորական prompt-based chat interface, այլ վերահսկվող persona-graph runtime։

---

## ԵԶՐԱԿԱՑՈՒԹՅՈՒՆ

Աշխատանքի ընթացքում մշակվել է Persona-Graph-Agent համակարգ, որը նախատեսված է երկխոսության կազմակերպման, կառուցվածքային հիշողության պահպանման, փաստաթղթերի մշակման և persona-driven պատասխանների ձևավորման համար։ Համակարգի հիմնական նպատակը եղել է անցումը սովորական message → answer սխեմայից դեպի վերահսկվող ճարտարապետություն, որտեղ յուրաքանչյուր հարցում անցնում է հստակ սահմանված և տրամաբանորեն փոխկապակցված փուլերով։

Մշակված համակարգում իրականացվել են հետևյալ հիմնական լուծումները.

1. Կառուցվել է հարցումների մշակման pipeline, որը ներառում է հետևյալ փուլերը՝ request intake, request preprocessing, route selection, capability planning, context building, response generation, response validation։

2. Ներդրվել է հարցումների դասակարգման մեխանիզմ, որը հնարավորություն է տալիս տարբերակել հարցումների հիմնական տեսակները, մասնավորապես՝ factual\_query, roleplay\_prompt, persona\_specification, persona\_assignment, document\_request, general\_chat։

3. Իրականացվել է persona-ների կառուցվածքային ներկայացում։ Persona-ն այլևս չի դիտարկվում որպես ազատ տեքստային նկարագրություն, այլ ներկայացվում և պահպանվում է որպես կառուցվածքային օբյեկտ, որը ներառում է identity, traits, behavior, internal conflict, communication style և dynamics բաղադրիչները։

4. Կազմակերպվել է knowledge graph memory շերտ, որի միջոցով համակարգը պահպանում է entities, relations, context-linked facts և persona-related graph structures [2][14]։

5. Ներդրվել է file ingestion pipeline, որը հնարավորություն է տալիս մշակել հետևյալ ձևաչափերով փաստաթղթեր՝ txt, md, json, csv, pdf, docx, odt, fb2։

6. Իրականացվել է session-aware context building, որի շնորհիվ պատասխանների ձևավորման ընթացքում առաջնահերթություն է տրվում ընթացիկ session-ի նյութերին, իսկ ընդհանուր graph memory-ն օգտագործվում է միայն թեմատիկ համապատասխանության դեպքում [8]։

7. Ներդրվել է response validation շերտ, որը վերահսկում է route consistency, context relevance, truncation detection և persona style consistency ցուցանիշները [13]։

8. Ավելացվել է token budgeting մեխանիզմ, որը թույլ է տալիս վերահսկել տեղային մոդելի (local model runtime) սահմանափակումները և կառավարել մուտքային ու ելքային ռեսուրսները՝ ըստ հարցման տեսակի [5][6]։

9. Ճարտարապետության վերջին փուլում ավելացվել է cognitive pipeline (P1–P6)՝ persona-ի հուզական, պաշտպանական, ճանաչողական և действие-policy դինամիկայի lightweight perceptron ներկայացմամբ (≈7 700 պարամետր), ինչպես նաև PersonalityGenome-ի 53 LearnableParam-ով [11][12]։

Այսպիսով, համակարգը ձևավորվել է որպես բազմաշերտ agent architecture, որտեղ history-ը գործում է որպես փոփոխական context, persona-ն հանդես է գալիս որպես համեմատաբար կայուն context, graph memory-ն ապահովում է երկարաժամկետ semantic հիմքը, իսկ routing-ը որոշում է պատասխանի կառուցման տրամաբանական հոսքը։

### Հիմնական արդյունքներ

Կատարված աշխատանքի արդյունքում մշակված համակարգը ապահովում է հետևյալ հիմնական ֆունկցիոնալ հնարավորությունները.

- երկխոսության պահպանում ըստ session-ների,
- persona օբյեկտների ստեղծում և ընտրություն,
- փաստաթղթերի ընդունում և մշակում,
- LLM-assisted knowledge extraction,
- graph update միայն merge սկզբունքով,
- route-based response generation,
- persona-aware պատասխանների ձևավորում,
- cognitive pipeline-ի միջոցով persona հուզական դինամիկայի lightweight modealing,
- ավտոմատ վավերացում (automatic validation) և regeneration policy։

Արդյունքում ստացվել է կիրառական հարթակ, որը հարմար է LLM + graph memory + persona runtime մոտեցման հիման վրա հետագա ընդլայնման և կատարելագործման համար։

Միաժամանակ անհրաժեշտ է ընդգծել, որ համակարգը դեռևս չի ապահովում մարդկային մտածողության ամբողջական ճշգրտությունն ու խորությունը։ Այն շարունակում է կախված մնալ ընտրված մոդելների սահմանափակումներից, context window-ի չափից, extraction-ի որակից և կառուցվածքային վերահսկման մեխանիզմների լիարժեքությունից։ Այդ պատճառով համակարգը ներկայումս պետք է դիտարկել ոչ թե որպես լիարժեք «խելացի փոխարինող», այլ որպես վերահսկվող գործիքային հարթակ, որը մոտենում է մարդու տրամաբանության և persona-կենտրոնացված արձագանքների մասնակի մոդելավորմանը։

### Սահմանափակումներ և հետագա աշխատանք

Համակարգի հետագա զարգացման տեսանկյունից նպատակահարմար է կատարել հետևյալ աշխատանքները.

1. ուժեղացնել invented facts հայտնաբերման մեխանիզմները,
2. զարգացնել OCR-հիմքով ingestion-ը scan-տեսակի pdf փաստաթղթերի համար,
3. խորացնել persona drift detection-ը երկարատև session-ների ընթացքում,
4. կատարելագործել context compression-ը սահմանափակ context window ունեցող մոդելների համար,
5. բարձրացնել local inference runtime-ի արագությունն ու կայունությունը,
6. ընդլայնել cognitive pipeline-ի P1–P6 ենթահամակարգի training dataset-ը՝ PersonalityGenome-ի calibration-ի ճշգրտության բարձրացման նպատակով։

### Ամփոփում

Մշակված համակարգը ներկայացնում է կիրառական ծրագրային ճարտարապետություն, որը միավորում է dialogue memory, knowledge extraction, graph memory, persona modeling, bounded context generation, cognitive pipeline (P1–P6) և route-controlled response policy բաղադրիչները։

Այս ամենի արդյունքում ձևավորվել է համակարգ, որը կարող է ոչ միայն պատասխանել հարցումներին, այլև ապահովել context continuity, կառուցել և կիրառել persona structures, ինչպես նաև օգտագործել long-term graph memory և lightweight cognitive dynamics մոդելավորում երկխոսային միջավայրում։

---

## ՕԳՏԱԳՈՐԾՎԱԾ ԳՐԱԿԱՆՈՒԹՅԱՆ ՑԱՆԿ

1. PersonaAgentwGraphRAG research repository. — 2024–2025. URL: https://anonymous.4open.science/r/PersonaAgentwGraphRAG-DE6F/README.md

2. Microsoft GraphRAG. — Microsoft Research: documentation and repository. URL: https://github.com/microsoft/graphrag

3. Fu S. AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning. — GitHub repository, 2024.

4. Li H., He R., Zhang Q. et al. Combee: Scaling Prompt Learning for Self-Improving Language Model Agents. — arXiv preprint arXiv:2404.01823, 2024.

5. FastAPI official documentation. — Sebastián Ramírez, 2018–2025. URL: https://fastapi.tiangolo.com

6. OpenAI. ChatGPT: Optimizing Language Models for Dialogue. — OpenAI official materials and documentation, 2022–2025. URL: https://openai.com/chatgpt

7. Anthropic. Claude AI: Constitutional AI and Large Language Model Assistant. — Anthropic official materials and documentation, 2023–2025. URL: https://claude.ai

8. Lewis P., Perez E., Piktus A. et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. — Advances in Neural Information Processing Systems (NeurIPS), 2020. — Vol. 33. — P. 9459–9474.

9. Wei J., Wang X., Schuurmans D. et al. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. — Advances in Neural Information Processing Systems (NeurIPS), 2022. — Vol. 35.

10. Yao S., Zhao J., Yu D. et al. ReAct: Synergizing Reasoning and Acting in Language Models. — International Conference on Learning Representations (ICLR), 2023.

11. Park J.S., O'Brien J.C., Cai C.J. et al. Generative Agents: Interactive Simulacra of Human Behavior. — ACM Symposium on User Interface Software and Technology (UIST), 2023.

12. Packer C., Wooders S., Lin K. et al. MemGPT: Towards LLMs as Operating Systems. — arXiv preprint arXiv:2310.08560, 2023.

13. Madaan A., Tandon N., Gupta P. et al. Self-Refine: Iterative Refinement with Self-Feedback. — Advances in Neural Information Processing Systems (NeurIPS), 2023. — Vol. 36.

14. Edge D., Trinh H., Cheng N. et al. From Local to Global: A Graph RAG Approach to Query-Focused Summarization. — Microsoft Research, arXiv preprint arXiv:2404.16130, 2024.
15. Li Y., Su H., Shen X. et al. DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset. — Proceedings of the 8th International Joint Conference on Natural Language Processing (IJCNLP), 2017. — P. 986–995. [Kaggle mirror: https://www.kaggle.com/datasets/thedevastator/dailydialog-multi-turn-dialog-with-intention-and]

16. Sun H., Zhu W., Zhang Y. et al. DialogStudio: Towards Richest and Most Diverse Unified Dataset Collection for Conversational AI. — arXiv preprint arXiv:2307.10172, 2023. URL: https://github.com/salesforce/DialogStudio

17. Rashkin H., Smith E.M., Li M., Boureau Y-L. I Know, I Know! Towards Empathetic Open-domain Conversation. — Proceedings of ACL 2019. — P. 5370–5381.

18. Aliannejadi M., Zamani H., Crestani F., Croft W.B. Asking Clarifying Questions in Open-Domain Information-Seeking Conversations. — Proceedings of SIGIR 2019. URL: https://github.com/aliannejadi/ClariQ

19. Karapetyan K., Claude (Anthropic). idea_attractors_seed — Attractor Corpus for Cognitive Pipeline Regression Testing. — Self-published dataset, GitHub, 2025. URL: https://github.com/karapet/PersonaAgentwGraphRAG-DE6F (DataSets/idea_attractors/)
