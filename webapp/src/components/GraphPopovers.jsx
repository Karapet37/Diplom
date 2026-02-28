import React from "react";

function normalizeUiLanguageCode(code) {
  const token = String(code || "en").trim().toLowerCase();
  const base = token.split(/[-_]/)[0] || "en";
  return base;
}

function getCatalogValue(catalog, code, key) {
  const lang = normalizeUiLanguageCode(code);
  const bucket = catalog[lang] || catalog.en || {};
  return bucket[key] ?? (catalog.en || {})[key] ?? key;
}

const NODE_ACTION_LABELS = {
  hy: { explain: "Բացատրել", improve: "Լավացնել", risks: "Ռիսկեր", tasks: "Առաջադրանքներ", memory: "Հիշողություն" },
  ru: { explain: "Объяснить", improve: "Улучшить", risks: "Риски", tasks: "Задачи", memory: "Память" },
  en: { explain: "Explain", improve: "Improve", risks: "Risks", tasks: "Tasks", memory: "Memory" },
  fr: { explain: "Expliquer", improve: "Améliorer", risks: "Risques", tasks: "Tâches", memory: "Mémoire" },
  es: { explain: "Explicar", improve: "Mejorar", risks: "Riesgos", tasks: "Tareas", memory: "Memoria" },
  pt: { explain: "Explicar", improve: "Melhorar", risks: "Riscos", tasks: "Tarefas", memory: "Memória" },
  ar: { explain: "شرح", improve: "تحسين", risks: "مخاطر", tasks: "مهام", memory: "ذاكرة" },
  hi: { explain: "समझाएँ", improve: "सुधारें", risks: "जोखिम", tasks: "कार्य", memory: "स्मृति" },
  zh: { explain: "说明", improve: "优化", risks: "风险", tasks: "任务", memory: "记忆" },
  ja: { explain: "説明", improve: "改善", risks: "リスク", tasks: "タスク", memory: "記憶" },
};

const EDGE_ACTION_LABELS = {
  hy: { explain: "Բացատրել կապը", improve: "Լավացնել կապը", risks: "Ռիսկերը", merge: "Միաձուլել", split: "Բաժանել" },
  ru: { explain: "Объяснить связь", improve: "Улучшить связь", risks: "Риски", merge: "Слить", split: "Разделить" },
  en: { explain: "Explain Edge", improve: "Improve Edge", risks: "Edge Risks", merge: "Merge", split: "Split" },
  fr: { explain: "Expliquer le lien", improve: "Améliorer le lien", risks: "Risques", merge: "Fusionner", split: "Diviser" },
  es: { explain: "Explicar relación", improve: "Mejorar relación", risks: "Riesgos", merge: "Unir", split: "Separar" },
  pt: { explain: "Explicar aresta", improve: "Melhorar aresta", risks: "Riscos", merge: "Mesclar", split: "Dividir" },
  ar: { explain: "شرح العلاقة", improve: "تحسين العلاقة", risks: "المخاطر", merge: "دمج", split: "تقسيم" },
  hi: { explain: "एज समझाएँ", improve: "एज सुधारें", risks: "जोखिम", merge: "मर्ज", split: "स्प्लिट" },
  zh: { explain: "说明关系", improve: "优化关系", risks: "风险", merge: "合并", split: "拆分" },
  ja: { explain: "関係を説明", improve: "関係を改善", risks: "リスク", merge: "統合", split: "分割" },
};

const EDGE_TEXT_LABELS = {
  hy: {
    relationSummary: "Կապի ամփոփում",
    relationRisk: "Կապի ռիսկ",
    strengthen: "Ուժեղացնել",
    weaken: "Թուլացնել",
    mergeHint: "Միաձուլման հուշում",
    splitHint: "Բաժանման հուշում",
    strengthenLog: "ուժեղացնել կապը",
    weakenLog: "թուլացնել կապը",
    stable: "Այս կապը կառուցվածքայինորեն կայուն է։ Նշումներ ավելացրեք, եթե իմաստը փոխվում է։",
    medium: "Այս կապը աշխատում է, բայց ավելի հստակ semantics կամ ապացույց է պետք։",
    high: "Այս կապը չափազանց թույլ, լայն կամ վատ բացատրված է։ Ստուգեք իմաստը, մինչև հենվեք դրա վրա։",
    mergeSame: (fromName, toName) => `${fromName}-ը և ${toName}-ը նույն տիպի են։ Միաձուլեք միայն եթե կրկնում են նույն միտքը։`,
    mergeDifferent: (fromName, toName) => `${fromName}-ը և ${toName}-ը առանձին պահեք, եթե նույն արդյունքը չեն նկարագրում։`,
    splitGeneric: (relationLabel) => `«${relationLabel}» կապը բաժանեք, եթե այն խառնում է ապացույց, կախվածություն և սեփականություն։`,
    splitSpecific: "Բաժանեք այս կապը միայն եթե այն միաժամանակ մի քանի իմաստ կամ փուլ է փոխանցում։",
  },
  ru: {
    relationSummary: "Сводка связи",
    relationRisk: "Риск связи",
    strengthen: "Усилить",
    weaken: "Ослабить",
    mergeHint: "Подсказка по слиянию",
    splitHint: "Подсказка по разделению",
    strengthenLog: "усилить связь",
    weakenLog: "ослабить связь",
    stable: "Связь структурно стабильна. Оставьте ее, но добавляйте пояснение, если смысл меняется.",
    medium: "Связь рабочая, но ей не хватает более точной семантики или подтверждения.",
    high: "Связь слишком слабая, слишком широкая или плохо объяснена. Проверьте смысл, прежде чем на нее опираться.",
    mergeSame: (fromName, toName) => `${fromName} и ${toName} одного типа. Объединяйте только если это дубликат одной и той же идеи.`,
    mergeDifferent: (fromName, toName) => `Оставьте ${fromName} и ${toName} раздельно, если они не описывают один и тот же итог.`,
    splitGeneric: (relationLabel) => `Разделите связь «${relationLabel}», если в ней смешаны доказательство, зависимость и владение.`,
    splitSpecific: "Разделяйте эту связь только если она одновременно несет несколько смыслов или этапов.",
  },
  en: {
    relationSummary: "Relation Summary",
    relationRisk: "Relation Risk",
    strengthen: "Strengthen",
    weaken: "Weaken",
    mergeHint: "Merge Hint",
    splitHint: "Split Hint",
    strengthenLog: "strengthen edge",
    weakenLog: "weaken edge",
    stable: "This relation is structurally stable. Keep it, but annotate it if the meaning changes.",
    medium: "This relation is usable, but it still needs clearer semantics or stronger evidence.",
    high: "This relation is likely too weak, too broad, or under-explained. Verify meaning before relying on it.",
    mergeSame: (fromName, toName) => `${fromName} and ${toName} share the same node type. Merge only if they duplicate the same concept.`,
    mergeDifferent: (fromName, toName) => `Keep ${fromName} and ${toName} separate unless both branches now describe the same outcome.`,
    splitGeneric: (relationLabel) => `Split '${relationLabel}' into clearer links if it mixes evidence, dependency, and ownership in one edge.`,
    splitSpecific: "Split this relation into sub-edges only if the current link carries multiple meanings or timelines.",
  },
  fr: {
    relationSummary: "Résumé de la relation",
    relationRisk: "Risque de la relation",
    strengthen: "Renforcer",
    weaken: "Affaiblir",
    mergeHint: "Conseil de fusion",
    splitHint: "Conseil de séparation",
    strengthenLog: "renforcer la relation",
    weakenLog: "affaiblir la relation",
    stable: "Cette relation est structurellement stable. Gardez-la, mais annotez-la si son sens change.",
    medium: "Cette relation est utilisable, mais elle a encore besoin d'une sémantique plus claire ou de preuves plus solides.",
    high: "Cette relation est trop faible, trop large ou insuffisamment expliquée. Vérifiez son sens avant de vous y fier.",
    mergeSame: (fromName, toName) => `${fromName} et ${toName} ont le même type. Fusionnez seulement s'ils dupliquent le même concept.`,
    mergeDifferent: (fromName, toName) => `Gardez ${fromName} et ${toName} séparés sauf s'ils décrivent maintenant le même résultat.`,
    splitGeneric: (relationLabel) => `Scindez « ${relationLabel} » si ce lien mélange preuve, dépendance et propriété.`,
    splitSpecific: "Scindez cette relation seulement si elle transporte plusieurs sens ou étapes à la fois.",
  },
  es: {
    relationSummary: "Resumen de la relación",
    relationRisk: "Riesgo de la relación",
    strengthen: "Reforzar",
    weaken: "Debilitar",
    mergeHint: "Sugerencia de fusión",
    splitHint: "Sugerencia de división",
    strengthenLog: "reforzar relación",
    weakenLog: "debilitar relación",
    stable: "Esta relación es estructuralmente estable. Consérvala, pero anótala si cambia el significado.",
    medium: "Esta relación es usable, pero aún necesita una semántica más clara o evidencia más fuerte.",
    high: "Esta relación es demasiado débil, demasiado amplia o está mal explicada. Verifica el significado antes de confiar en ella.",
    mergeSame: (fromName, toName) => `${fromName} y ${toName} comparten el mismo tipo. Fusiónalos solo si duplican el mismo concepto.`,
    mergeDifferent: (fromName, toName) => `Mantén ${fromName} y ${toName} separados a menos que ambas ramas describan el mismo resultado.`,
    splitGeneric: (relationLabel) => `Divide '${relationLabel}' en enlaces más claros si mezcla evidencia, dependencia y propiedad.`,
    splitSpecific: "Divide esta relación solo si el enlace actual carga varios significados o etapas.",
  },
  pt: {
    relationSummary: "Resumo da relação",
    relationRisk: "Risco da relação",
    strengthen: "Reforçar",
    weaken: "Enfraquecer",
    mergeHint: "Sugestão de fusão",
    splitHint: "Sugestão de divisão",
    strengthenLog: "reforçar aresta",
    weakenLog: "enfraquecer aresta",
    stable: "Esta relação é estruturalmente estável. Mantenha-a, mas adicione contexto se o significado mudar.",
    medium: "Esta relação é utilizável, mas ainda precisa de semântica mais clara ou evidência mais forte.",
    high: "Esta relação está fraca demais, ampla demais ou mal explicada. Verifique o significado antes de confiar nela.",
    mergeSame: (fromName, toName) => `${fromName} e ${toName} compartilham o mesmo tipo. Mescle apenas se duplicarem o mesmo conceito.`,
    mergeDifferent: (fromName, toName) => `Mantenha ${fromName} e ${toName} separados, a menos que ambos descrevam o mesmo resultado.`,
    splitGeneric: (relationLabel) => `Divida '${relationLabel}' em vínculos mais claros se ela mistura evidência, dependência e posse.`,
    splitSpecific: "Divida esta relação apenas se o vínculo atual carregar vários significados ou etapas.",
  },
  ar: {
    relationSummary: "ملخص العلاقة",
    relationRisk: "مخاطر العلاقة",
    strengthen: "تقوية",
    weaken: "إضعاف",
    mergeHint: "تلميح دمج",
    splitHint: "تلميح تقسيم",
    strengthenLog: "تقوية العلاقة",
    weakenLog: "إضعاف العلاقة",
    stable: "هذه العلاقة مستقرة بنيوياً. احتفظ بها، لكن أضف شرحاً إذا تغيّر المعنى.",
    medium: "هذه العلاقة قابلة للاستخدام، لكنها ما زالت تحتاج إلى دلالة أوضح أو دليل أقوى.",
    high: "هذه العلاقة ضعيفة جداً أو عامة جداً أو غير مفسرة جيداً. تحقق من المعنى قبل الاعتماد عليها.",
    mergeSame: (fromName, toName) => `${fromName} و ${toName} من النوع نفسه. ادمجهما فقط إذا كانا يكرران المفهوم نفسه.`,
    mergeDifferent: (fromName, toName) => `أبقِ ${fromName} و ${toName} منفصلين ما لم يصفا النتيجة نفسها.`,
    splitGeneric: (relationLabel) => `قسّم '${relationLabel}' إلى روابط أوضح إذا كانت تخلط بين الدليل والاعتماد والملكية.`,
    splitSpecific: "قسّم هذه العلاقة فقط إذا كانت تحمل عدة معانٍ أو مراحل في الرابط نفسه.",
  },
  hi: {
    relationSummary: "संबंध सारांश",
    relationRisk: "संबंध जोखिम",
    strengthen: "मजबूत करें",
    weaken: "कमज़ोर करें",
    mergeHint: "मर्ज संकेत",
    splitHint: "स्प्लिट संकेत",
    strengthenLog: "एज मजबूत करें",
    weakenLog: "एज कमज़ोर करें",
    stable: "यह संबंध संरचनात्मक रूप से स्थिर है। इसे रखें, लेकिन अर्थ बदलने पर टिप्पणी जोड़ें।",
    medium: "यह संबंध उपयोगी है, लेकिन इसे अभी भी अधिक स्पष्ट अर्थ या मजबूत प्रमाण चाहिए।",
    high: "यह संबंध बहुत कमजोर, बहुत व्यापक, या कम समझाया गया है। भरोसा करने से पहले अर्थ जाँचें।",
    mergeSame: (fromName, toName) => `${fromName} और ${toName} का node type समान है। केवल तभी मर्ज करें जब दोनों एक ही अवधारणा की डुप्लिकेट हों।`,
    mergeDifferent: (fromName, toName) => `${fromName} और ${toName} को अलग रखें, जब तक दोनों एक ही परिणाम न बताने लगें।`,
    splitGeneric: (relationLabel) => `यदि '${relationLabel}' एक ही edge में प्रमाण, निर्भरता और स्वामित्व मिला रहा है, तो इसे अलग करें।`,
    splitSpecific: "इस संबंध को तभी विभाजित करें जब वर्तमान लिंक कई अर्थ या समय-चरण साथ ले जा रहा हो।",
  },
  zh: {
    relationSummary: "关系摘要",
    relationRisk: "关系风险",
    strengthen: "增强",
    weaken: "减弱",
    mergeHint: "合并建议",
    splitHint: "拆分建议",
    strengthenLog: "增强边",
    weakenLog: "减弱边",
    stable: "这条关系在结构上是稳定的。保留它，但在含义变化时补充注释。",
    medium: "这条关系可以使用，但仍需要更清晰的语义或更强的证据。",
    high: "这条关系过弱、过泛，或解释不足。在依赖它之前先确认含义。",
    mergeSame: (fromName, toName) => `${fromName} 和 ${toName} 属于相同节点类型。只有在它们重复同一概念时才合并。`,
    mergeDifferent: (fromName, toName) => `除非 ${fromName} 和 ${toName} 现在描述的是同一结果，否则保持分离。`,
    splitGeneric: (relationLabel) => `如果“${relationLabel}”同时混合了证据、依赖和归属，请把它拆成更清晰的关系。`,
    splitSpecific: "只有在当前关系同时承载多种含义或时间阶段时，才拆分这条边。",
  },
  ja: {
    relationSummary: "関係の要約",
    relationRisk: "関係のリスク",
    strengthen: "強化",
    weaken: "弱化",
    mergeHint: "統合ヒント",
    splitHint: "分割ヒント",
    strengthenLog: "エッジを強化",
    weakenLog: "エッジを弱化",
    stable: "この関係は構造的に安定しています。意味が変わる場合だけ注釈を追加してください。",
    medium: "この関係は使えますが、より明確な意味付けか強い根拠がまだ必要です。",
    high: "この関係は弱すぎるか、広すぎるか、説明不足です。依存する前に意味を確認してください。",
    mergeSame: (fromName, toName) => `${fromName} と ${toName} は同じノード型です。同じ概念の重複である場合だけ統合してください。`,
    mergeDifferent: (fromName, toName) => `${fromName} と ${toName} が同じ結果を表していない限り、分けたままにしてください。`,
    splitGeneric: (relationLabel) => `「${relationLabel}」が根拠・依存・所有を1本のエッジに混在させているなら、より明確に分割してください。`,
    splitSpecific: "現在のリンクが複数の意味や時系列を同時に持つ場合にのみ、この関係を分割してください。",
  },
};

function getEdgeTextBucket(code) {
  const lang = normalizeUiLanguageCode(code);
  return EDGE_TEXT_LABELS[lang] || EDGE_TEXT_LABELS.en;
}

function humanizeToken(value) {
  return String(value || "")
    .replace(/[_-]+/g, " ")
    .trim();
}

export function buildGraphNodeAssistActionOptions(uiLanguage) {
  const lang = normalizeUiLanguageCode(uiLanguage);
  const bucket = NODE_ACTION_LABELS[lang] || NODE_ACTION_LABELS.en;
  return ["explain", "improve", "risks", "tasks", "memory"].map((value) => ({
    value,
    label: String(bucket[value] || value),
  }));
}

export function buildGraphEdgeAssistActionOptions(uiLanguage) {
  const lang = normalizeUiLanguageCode(uiLanguage);
  const bucket = EDGE_ACTION_LABELS[lang] || EDGE_ACTION_LABELS.en;
  return ["explain", "improve", "risks", "merge", "split"].map((value) => ({
    value,
    label: String(bucket[value] || value),
  }));
}

export function buildGraphEdgeAssistInsights({ edge, nodes = [], nodeNameById, uiLanguage }) {
  if (!edge) return null;
  const fromNode = (Array.isArray(nodes) ? nodes : []).find((node) => Number(node?.id) === Number(edge.from)) || null;
  const toNode = (Array.isArray(nodes) ? nodes : []).find((node) => Number(node?.id) === Number(edge.to)) || null;
  const fromName =
    (nodeNameById && typeof nodeNameById.get === "function" ? nodeNameById.get(Number(edge.from)) : "") ||
    `#${Number(edge.from || 0)}`;
  const toName =
    (nodeNameById && typeof nodeNameById.get === "function" ? nodeNameById.get(Number(edge.to)) : "") ||
    `#${Number(edge.to || 0)}`;
  const fromType = String(fromNode?.type || "generic").trim();
  const toType = String(toNode?.type || "generic").trim();
  const relationType = String(edge.relation_type || "related_to").trim() || "related_to";
  const logicRule = String(edge.logic_rule || "explicit").trim() || "explicit";
  const weight = Math.max(0, Number(edge.weight || 0));
  const metadata = edge?.metadata && typeof edge.metadata === "object" ? edge.metadata : {};
  let riskScore = 0.18;
  if (weight < 0.25) riskScore += 0.28;
  if (weight > 1.05) riskScore += 0.18;
  if (logicRule !== "explicit") riskScore += 0.12;
  if (!Object.keys(metadata).length) riskScore += 0.12;
  if (String(edge.direction || "directed") === "undirected") riskScore += 0.05;
  if (String(relationType).includes("conflict") || String(relationType).includes("contradict")) riskScore += 0.18;
  riskScore = Math.max(0, Math.min(1, riskScore));
  const riskLevel = riskScore >= 0.7 ? "high" : riskScore >= 0.4 ? "medium" : "low";
  const genericRelation = ["related_to", "linked_to", "connected_to", "depends_on"].includes(
    relationType.toLowerCase()
  );
  const edgeTexts = getEdgeTextBucket(uiLanguage);
  return {
    fromName,
    toName,
    relationType,
    weight,
    riskScore,
    riskLevel,
    mergeSuggestion:
      fromType === toType
        ? edgeTexts.mergeSame(fromName, toName)
        : edgeTexts.mergeDifferent(fromName, toName),
    splitSuggestion: genericRelation
      ? edgeTexts.splitGeneric(humanizeToken(relationType) || relationType)
      : edgeTexts.splitSpecific,
    riskSummary:
      riskLevel === "high"
        ? edgeTexts.high
        : riskLevel === "medium"
          ? edgeTexts.medium
          : edgeTexts.stable,
    summary: `${fromName} -> ${toName} · ${humanizeToken(relationType)} · ${weight.toFixed(2)}`,
  };
}

export function graphEdgePopoverText(uiLanguage) {
  const bucket = getEdgeTextBucket(uiLanguage);
  return {
    relationSummary: bucket.relationSummary,
    relationRisk: bucket.relationRisk,
    strengthen: bucket.strengthen,
    weaken: bucket.weaken,
    mergeHint: bucket.mergeHint,
    splitHint: bucket.splitHint,
    strengthenLog: bucket.strengthenLog,
    weakenLog: bucket.weakenLog,
  };
}

export function GraphNodeAssistPopover({
  title,
  meta,
  summary,
  actionLabel,
  messageLabel,
  action,
  actionOptions,
  prompt,
  onActionChange,
  onPromptChange,
  busy = false,
  disabled = false,
  onRun,
  onUseRequest,
  onUseTask,
  onUseRisk,
  onUseNote,
  runLabel,
  requestLabel,
  taskLabel,
  riskLabel,
  noteLabel,
  resultTitle,
  resultSummary,
  emptyResultLabel,
}) {
  return (
    <>
      <div className="graph-node-popover-head">
        <div>
          <strong>{title}</strong>
          <div className="graph-node-popover-meta">{meta}</div>
        </div>
      </div>
      <div className="graph-node-popover-summary">{summary}</div>
      <label className="graph-node-popover-field">
        <span>{actionLabel}</span>
        <select value={action} onChange={(event) => onActionChange(event.target.value)} disabled={disabled}>
          {(Array.isArray(actionOptions) ? actionOptions : []).map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </label>
      <label className="graph-node-popover-field">
        <span>{messageLabel}</span>
        <textarea rows={5} value={prompt} onChange={(event) => onPromptChange(event.target.value)} disabled={disabled} />
      </label>
      <div className="graph-node-popover-actions">
        <button type="button" disabled={busy || disabled} onClick={onRun}>
          {runLabel}
        </button>
        <button type="button" disabled={disabled} onClick={onUseRequest}>
          {requestLabel}
        </button>
        <button type="button" disabled={disabled} onClick={onUseTask}>
          {taskLabel}
        </button>
        <button type="button" disabled={disabled} onClick={onUseRisk}>
          {riskLabel}
        </button>
        <button type="button" disabled={disabled} onClick={onUseNote}>
          {noteLabel}
        </button>
      </div>
      <div className="graph-node-popover-summary">
        <strong>{resultTitle}</strong>
        <div>{resultSummary || emptyResultLabel}</div>
      </div>
    </>
  );
}

export function GraphEdgeAssistPopover({
  labels,
  summary,
  relationMeta,
  reasoningSummary,
  riskSummary,
  riskBadge,
  mergeSuggestion,
  splitSuggestion,
  actionLabel,
  messageLabel,
  action,
  actionOptions,
  prompt,
  onActionChange,
  onPromptChange,
  busy = false,
  disabled = false,
  onRunAssist,
  onStrengthen,
  onWeaken,
  runLabel,
  resultTitle,
  resultSummary,
  emptyResultLabel,
}) {
  const safeLabels = labels || graphEdgePopoverText("en");
  return (
    <>
      <div className="graph-node-popover-head">
        <div>
          <strong>{safeLabels.relationSummary}</strong>
          <div className="graph-node-popover-meta">{relationMeta}</div>
        </div>
      </div>
      <div className="graph-node-popover-summary">
        {summary}
        <div className="graph-node-popover-meta" style={{ marginTop: "4px" }}>
          {reasoningSummary}
        </div>
      </div>
      <div className="graph-node-popover-summary">
        <strong>{safeLabels.relationRisk}</strong>
        <div>{riskSummary}</div>
        <div className="graph-node-popover-meta" style={{ marginTop: "4px" }}>
          {riskBadge}
        </div>
      </div>
      <label className="graph-node-popover-field">
        <span>{actionLabel}</span>
        <select value={action} onChange={(event) => onActionChange(event.target.value)} disabled={disabled}>
          {(Array.isArray(actionOptions) ? actionOptions : []).map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </label>
      <label className="graph-node-popover-field">
        <span>{messageLabel}</span>
        <textarea rows={4} value={prompt} onChange={(event) => onPromptChange(event.target.value)} disabled={disabled} />
      </label>
      <div className="graph-node-popover-actions">
        <button type="button" disabled={busy || disabled} onClick={onRunAssist}>
          {runLabel}
        </button>
        <button type="button" disabled={busy || disabled} onClick={onStrengthen}>
          {safeLabels.strengthen}
        </button>
        <button type="button" disabled={busy || disabled} onClick={onWeaken}>
          {safeLabels.weaken}
        </button>
      </div>
      <div className="graph-edge-popover-grid">
        <div className="graph-edge-popover-card">
          <strong>{safeLabels.mergeHint}</strong>
          <span>{mergeSuggestion}</span>
        </div>
        <div className="graph-edge-popover-card">
          <strong>{safeLabels.splitHint}</strong>
          <span>{splitSuggestion}</span>
        </div>
      </div>
      <div className="graph-node-popover-summary">
        <strong>{resultTitle}</strong>
        <div>{resultSummary || emptyResultLabel}</div>
      </div>
    </>
  );
}
