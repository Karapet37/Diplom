# Persona Lifecycle

Этот документ описывает текущий production-oriented lifecycle для persona heads в `agent_system`.

## 1. Слои persona

Каждый persona head теперь рассматривается как три явно разделённых слоя:

- `baseline definition`
  - хранит устойчивую persona-основу: `entity_type`, traits, aliases, relations, knowledge;
  - меняется только при явной materialization/update операции;
  - не должен дрейфовать от случайных chat-событий.
- `dynamic emotional state`
  - хранит текущее `emotion_vector`, `last_situation`, `last_response_style`;
  - обновляется на hot path чата;
  - зависит от `situation`, а не от raw user emotion.
- `learned interaction patterns`
  - хранит `examples`, `situation_reactions`, `log_tuples`, `persona_form`, `decision_explanation`, `learned_traits`;
  - обновляется bounded way через reviewable deterministic code;
  - не получает права переписывать baseline напрямую.

## 2. Файловое представление

Для каждого head в `memory/heads/{slug}/` используются:

- `baseline.json`
- `dynamic_state.json`
- `learned_patterns.json`
- `revisions.json`

Legacy-файлы сохраняются для обратной совместимости:

- `traits.json`
- `relations.json`
- `examples.json`
- `emotion_vector.json`
- `knowledge.txt`
- `log_tuples.json`
- `persona_form.json`
- `decision_explanation.txt`
- `meta.json`

## 3. Ревизии и inspectability

`revisions.json` хранит:

- текущие revision counters;
- bounded history последних persona updates;
- snapshots изменённых слоёв по каждой ревизии.

`meta.json` дублирует operational summary:

- `revision`
- `baseline_revision`
- `dynamic_revision`
- `learned_revision`
- `last_*_update_at`
- `confidence_score`
- `maturity_score`
- `maturity_level`
- `adaptation_locked`

Это позволяет:

- увидеть, какой слой менялся;
- быстро оценить зрелость persona;
- восстановить предыдущее состояние вручную через revision snapshots или через utility `restore_persona_revision(...)`.

## 4. Правила bounded adaptation

Главное правило:

```text
случайные interaction events могут обновлять learned patterns,
но не должны тихо переписывать persona baseline
```

Практически это означает:

- `materialize_persona(..., adaptation_mode="baseline_refresh")`
  - может обновлять baseline, dynamic state и learned patterns;
- `update_persona_from_examples(..., adaptation_mode="learned_update")`
  - сохраняет baseline traits/knowledge/relations;
  - записывает только learned-pattern updates и revision trail;
  - новые suggested traits попадают в `learned_traits`, а не в baseline traits.

## 5. Explainability

На chat path система теперь явно отдаёт:

- `persona_selection`
  - почему была выбрана именно эта persona;
  - из какого источника пришёл выбор;
  - какие evidence использовались.
- `persona_response`
  - какой response style был выбран;
  - какая ситуация его вызвала;
  - какие state, traits и learned patterns повлияли на стиль ответа.

## 6. Диагностика зрелости

`indicators` помогают отличать сырой head от устойчивого:

- `confidence_score`
- `maturity_score`
- `maturity_level`
- `evidence_count`
- `learned_pattern_count`
- `adaptation_locked`

Если persona становится `stable` или `mature`, это не отключает learning полностью, но делает видимым, что baseline больше нельзя считать “свободно плавающим”.
