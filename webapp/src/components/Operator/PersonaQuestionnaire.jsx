import React, { useState } from 'react';

const EMPTY_FORM = {
  persona_name: '',
  display_name: '',
  age: '',
  gender: '',
  nationality: '',
  occupation: '',
  education: '',
  location: '',
  self_description: '',
  how_others_see_you: '',
  core_values: '',
  life_philosophy: '',
  worldview: '',
  main_life_goals: '',
  dreams: '',
  what_drives_you: '',
  main_fears: '',
  insecurities: '',
  shame_topics: '',
  triggers: '',
  childhood: '',
  key_life_events: '',
  biggest_challenge: '',
  proudest_achievement: '',
  family_description: '',
  friendship_style: '',
  romantic_patterns: '',
  communication_style: '',
  humor_style: '',
  conflict_approach: '',
  emotional_expression: '',
  hobbies: '',
  pet_peeves: '',
  guilty_pleasures: '',
  additional_context: '',
  fictional_basis: '',
};

const SECTIONS = [
  {
    id: 'identity',
    icon: '👤',
    label: 'Идентификация',
    subtitle: 'Основные данные о личности',
    fields: [
      { key: 'display_name', label: 'Отображаемое имя', placeholder: 'Как будет называться в чате', rows: 0 },
      { key: 'age', label: 'Возраст', placeholder: 'Например: 28 лет', rows: 0 },
      { key: 'gender', label: 'Пол / гендер', placeholder: 'Например: женщина, мужчина, небинарный', rows: 0 },
      { key: 'nationality', label: 'Национальность / происхождение', placeholder: 'Откуда родом, культурный контекст', rows: 0 },
      { key: 'occupation', label: 'Профессия / род деятельности', placeholder: 'Чем занимается', rows: 0 },
      { key: 'education', label: 'Образование', placeholder: 'Уровень и область знаний', rows: 0 },
      { key: 'location', label: 'Место жительства', placeholder: 'Город, страна', rows: 0 },
    ],
  },
  {
    id: 'character',
    icon: '🪞',
    label: 'Характер',
    subtitle: 'Как личность воспринимает и представляет себя',
    fields: [
      { key: 'self_description', label: 'Как я описываю себя', placeholder: 'В 2–4 предложениях: кто я такой/такая, что для меня важно, что меня отличает', rows: 4 },
      { key: 'how_others_see_you', label: 'Как меня видят другие', placeholder: 'Что обычно говорят о тебе близкие, коллеги, случайные знакомые', rows: 3 },
    ],
  },
  {
    id: 'values',
    icon: '⚖️',
    label: 'Ценности и убеждения',
    subtitle: 'Что важно для личности, во что она верит',
    fields: [
      { key: 'core_values', label: 'Главные ценности', placeholder: 'Честность, свобода, семья, справедливость — что стоит на первом месте?', rows: 2 },
      { key: 'life_philosophy', label: 'Жизненная философия', placeholder: 'Главный принцип или девиз по жизни', rows: 2 },
      { key: 'worldview', label: 'Мировоззрение и убеждения', placeholder: 'Религиозные, политические, философские взгляды (если это важно для персонажа)', rows: 2 },
    ],
  },
  {
    id: 'goals',
    icon: '🎯',
    label: 'Цели и мотивация',
    subtitle: 'Что движет личностью',
    fields: [
      { key: 'main_life_goals', label: 'Цели в жизни', placeholder: 'Что хочет достичь — глобально и в обозримом будущем', rows: 2 },
      { key: 'dreams', label: 'Мечты и желания', placeholder: 'Чего сильно хочет, но может пока не решается достичь', rows: 2 },
      { key: 'what_drives_you', label: 'Что движет и вдохновляет', placeholder: 'Источники энергии, страсти, смысл', rows: 2 },
    ],
  },
  {
    id: 'fears',
    icon: '🌑',
    label: 'Страхи и уязвимости',
    subtitle: 'Тёмная сторона, болевые точки',
    fields: [
      { key: 'main_fears', label: 'Главные страхи', placeholder: 'Чего боится по-настоящему — потери, одиночества, провала, предательства', rows: 2 },
      { key: 'insecurities', label: 'Неуверенности', placeholder: 'В чём чувствует себя недостаточным/недостаточной', rows: 2 },
      { key: 'shame_topics', label: 'Темы-табу / что стыдно', placeholder: 'О чём не говорит, что болезненно', rows: 2 },
      { key: 'triggers', label: 'Триггеры', placeholder: 'Что выводит из себя, что вызывает острую реакцию', rows: 2 },
    ],
  },
  {
    id: 'history',
    icon: '📖',
    label: 'Жизненный путь',
    subtitle: 'История, которая сформировала личность',
    fields: [
      { key: 'childhood', label: 'Детство', placeholder: 'Где вырос(ла), семья, атмосфера, ключевые воспоминания', rows: 3 },
      { key: 'key_life_events', label: 'Ключевые события жизни', placeholder: 'Поворотные моменты, которые изменили тебя или определили путь', rows: 3 },
      { key: 'biggest_challenge', label: 'Самое тяжёлое испытание', placeholder: 'С чем пришлось столкнуться — потеря, провал, кризис', rows: 2 },
      { key: 'proudest_achievement', label: 'Главная гордость', placeholder: 'Что ты считаешь своей лучшей победой или достижением', rows: 2 },
    ],
  },
  {
    id: 'relationships',
    icon: '🤝',
    label: 'Отношения',
    subtitle: 'Как строит связи с людьми',
    fields: [
      { key: 'family_description', label: 'Семья', placeholder: 'Кто важен, как складываются отношения с родителями, братьями/сёстрами, детьми', rows: 2 },
      { key: 'friendship_style', label: 'Дружба', placeholder: 'Много поверхностных или мало близких? Как дружит, на что готов(а) ради друга', rows: 2 },
      { key: 'romantic_patterns', label: 'Любовные отношения', placeholder: 'Паттерны, повторяющееся в романтических связях, что ищет', rows: 2 },
    ],
  },
  {
    id: 'communication',
    icon: '💬',
    label: 'Общение',
    subtitle: 'Голос, стиль, характерные паттерны речи',
    fields: [
      { key: 'communication_style', label: 'Стиль общения', placeholder: 'Формально или неформально? Прямо или намёками? Многословно или кратко?', rows: 2 },
      { key: 'humor_style', label: 'Юмор', placeholder: 'Самоирония, сарказм, абсурд, тёплый юмор — или вовсе не шутит?', rows: 2 },
      { key: 'conflict_approach', label: 'В конфликте', placeholder: 'Избегает, идёт напролом, защищается, ищет компромисс?', rows: 2 },
      { key: 'emotional_expression', label: 'Выражение эмоций', placeholder: 'Легко ли говорит о чувствах? Показывает или скрывает? Как реагирует на эмоции других?', rows: 2 },
    ],
  },
  {
    id: 'habits',
    icon: '🌿',
    label: 'Быт и характер',
    subtitle: 'Мелкие детали, делающие личность живой',
    fields: [
      { key: 'hobbies', label: 'Хобби и интересы', placeholder: 'Чем занимается в свободное время, что любит', rows: 2 },
      { key: 'pet_peeves', label: 'Что раздражает', placeholder: 'Мелкие вещи, которые выводят из равновесия', rows: 1 },
      { key: 'guilty_pleasures', label: 'Тайные удовольствия', placeholder: 'Что любит, но может стесняться этого', rows: 1 },
    ],
  },
  {
    id: 'extra',
    icon: '✨',
    label: 'Дополнительно',
    subtitle: 'Всё, что не вошло выше',
    fields: [
      { key: 'additional_context', label: 'Дополнительный контекст', placeholder: 'Всё важное, что не описано выше — особенности, нюансы, история', rows: 4 },
      { key: 'fictional_basis', label: 'Основан на', placeholder: 'Если персонаж вдохновлён реальной личностью или литературным героем — укажи кем', rows: 1 },
    ],
  },
];

function FormField({ fieldKey, label, placeholder, rows, value, onChange }) {
  const id = `qf-${fieldKey}`;
  return (
    <div className="qform-field">
      <label htmlFor={id} className="qform-field__label">{label}</label>
      {rows === 0 ? (
        <input
          id={id}
          type="text"
          className="qform-field__input"
          value={value}
          onChange={(e) => onChange(fieldKey, e.target.value)}
          placeholder={placeholder}
        />
      ) : (
        <textarea
          id={id}
          className="qform-field__textarea"
          value={value}
          onChange={(e) => onChange(fieldKey, e.target.value)}
          placeholder={placeholder}
          rows={rows}
        />
      )}
    </div>
  );
}

function SectionBlock({ section, form, onChange, openSections, onToggle }) {
  const isOpen = openSections.includes(section.id);
  const filledCount = section.fields.filter((f) => String(form[f.key] || '').trim()).length;

  return (
    <section className={`qform-section ${isOpen ? 'open' : ''}`}>
      <button
        type="button"
        className="qform-section__toggle"
        onClick={() => onToggle(section.id)}
      >
        <span className="qform-section__icon">{section.icon}</span>
        <span className="qform-section__title">
          <strong>{section.label}</strong>
          <small>{section.subtitle}</small>
        </span>
        {filledCount > 0 && <span className="qform-section__badge">{filledCount}</span>}
        <span className="qform-section__chevron">{isOpen ? '−' : '+'}</span>
      </button>
      {isOpen && (
        <div className="qform-section__body">
          {section.fields.map((field) => (
            <FormField
              key={field.key}
              fieldKey={field.key}
              label={field.label}
              placeholder={field.placeholder}
              rows={field.rows}
              value={form[field.key] || ''}
              onChange={onChange}
            />
          ))}
        </div>
      )}
    </section>
  );
}

export function PersonaQuestionnaire({ onSubmit, onCancel, submitting, error }) {
  const [form, setForm] = useState({ ...EMPTY_FORM });
  const [openSections, setOpenSections] = useState(['identity', 'character']);

  function handleChange(key, value) {
    setForm((prev) => ({ ...prev, [key]: value }));
  }

  function toggleSection(id) {
    setOpenSections((prev) =>
      prev.includes(id) ? prev.filter((s) => s !== id) : [...prev, id]
    );
  }

  function expandAll() {
    setOpenSections(SECTIONS.map((s) => s.id));
  }

  function collapseAll() {
    setOpenSections([]);
  }

  function handleSubmit(e) {
    e.preventDefault();
    if (!form.persona_name.trim()) return;
    onSubmit(form);
  }

  const totalFilled = Object.values(form).filter((v) => String(v || '').trim()).length;

  return (
    <form className="qform" onSubmit={handleSubmit}>
      <div className="qform-header">
        <div>
          <p className="eyebrow">Новая личность</p>
          <h2>Анкета персонажа</h2>
          <p className="qform-header__hint">
            Заполни только то, что важно. Остальное ЛЛМ выведет из контекста.
          </p>
        </div>
      </div>

      {/* Name — always visible, required */}
      <div className="qform-name-block">
        <label htmlFor="qf-persona_name" className="qform-field__label">
          Псевдоним / системное имя <span className="qform-required">*</span>
        </label>
        <input
          id="qf-persona_name"
          type="text"
          className="qform-field__input qform-name-input"
          value={form.persona_name}
          onChange={(e) => handleChange('persona_name', e.target.value)}
          placeholder="Латиницей или кириллицей, без пробелов (например: katya, dark_knight)"
          required
        />
        <p className="qform-name-hint">
          Используется как ключ. Отображаемое имя укажи в первой секции.
        </p>
      </div>

      {/* Section controls */}
      <div className="qform-controls">
        <span className="qform-controls__count">{totalFilled} / {Object.keys(EMPTY_FORM).length} полей заполнено</span>
        <button type="button" className="link-button" onClick={expandAll}>развернуть всё</button>
        <button type="button" className="link-button" onClick={collapseAll}>свернуть всё</button>
      </div>

      {/* Sections */}
      {SECTIONS.map((section) => (
        <SectionBlock
          key={section.id}
          section={section}
          form={form}
          onChange={handleChange}
          openSections={openSections}
          onToggle={toggleSection}
        />
      ))}

      {/* Footer */}
      {error && <p className="qform-error">{error}</p>}
      <div className="qform-footer">
        <button type="button" className="button-secondary" onClick={onCancel} disabled={submitting}>
          Отмена
        </button>
        <button
          type="submit"
          className="button-primary qform-submit"
          disabled={submitting || !form.persona_name.trim()}
        >
          {submitting ? 'Создаётся…' : '✨ Создать личность'}
        </button>
      </div>
    </form>
  );
}
