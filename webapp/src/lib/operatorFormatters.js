export function nodeIdentity(node) {
  if (!node) return '';
  const translation = String(node.translation_line || '').trim();
  const lifecycle = String(node.lifecycle_state || '').trim();
  return [node.name || node.id, node.type ? `(${node.type})` : '', lifecycle ? `[${lifecycle}]` : '', translation].filter(Boolean).join(' ');
}

export function nodeDescription(node) {
  if (!node) return '';
  const traits = Array.isArray(node.attributes?.traits) && node.attributes.traits.length
    ? `traits=${node.attributes.traits.join(', ')}`
    : '';
  const translation = String(node.translation_line || '').trim();
  const plainExplanation = String(node.context?.plain_explanation || '').trim();
  const lifecycle = String(node.lifecycle_state || '').trim();
  const cluster = String(node.cluster_label || '').trim();
  const parts = [
    translation,
    node.description || node.short_gloss || '',
    plainExplanation,
    traits,
    lifecycle ? `state=${lifecycle}` : '',
    cluster ? `cluster=${cluster}` : '',
    `importance=${Number(node.importance ?? 0).toFixed(2)}`,
    `confidence=${Number(node.confidence ?? 0).toFixed(2)}`,
    `frequency=${Number(node.frequency ?? 0).toFixed(0)}`,
  ].filter(Boolean);
  return parts.join(' | ');
}

export function nodeViewDescription(nodeView) {
  const block = nodeView?.what_is_it_like || {};
  const context = block.context || {};
  const parts = [
    block.translation_line || '',
    block.description || '',
    context.plain_explanation || '',
    Array.isArray(block.facts) && block.facts.length ? block.facts.join(' | ') : '',
    Array.isArray(context.capabilities) && context.capabilities.length ? `can: ${context.capabilities.join(' | ')}` : '',
    Array.isArray(context.mechanisms) && context.mechanisms.length ? `how: ${context.mechanisms.join(' | ')}` : '',
    block.lifecycle_state ? `state=${block.lifecycle_state}` : '',
    block.cluster_label ? `cluster=${block.cluster_label}` : '',
    `importance=${Number(block.importance ?? 0).toFixed(2)}`,
    `confidence=${Number(block.confidence ?? 0).toFixed(2)}`,
    `frequency=${Number(block.frequency ?? 0).toFixed(0)}`,
  ].filter(Boolean);
  return parts.join(' | ');
}

export function nodeReflectionForm(nodeView) {
  return nodeView?.what_is_it_like?.context?.reinterpretation_form || null;
}

export function nodeLocalizedExplanation(nodeView) {
  return String(nodeView?.what_is_it_like?.context?.localized_explanation || '').trim();
}

export function nodeCanonicalEnglishExplanation(nodeView) {
  return String(
    nodeView?.what_is_it_like?.context?.canonical_english_explanation
      || nodeView?.what_is_it_like?.context?.plain_explanation
      || nodeView?.what_is_it_like?.description
      || ''
  ).trim();
}

export function normalizeRethinkPreview(payload) {
  if (!payload || !Array.isArray(payload.results)) return null;
  return {
    previewOnly: Boolean(payload.preview_only),
    activeMode: Boolean(payload.active_mode),
    processed: Number(payload.processed || 0),
    results: payload.results.map((item) => ({
      nodeId: String(item?.node_id || ''),
      nodeName: String(item?.node_name || ''),
      preview: item?.preview || null,
      skippedSuggestions: Array.isArray(item?.skipped_suggestions) ? item.skipped_suggestions : [],
    })),
  };
}
