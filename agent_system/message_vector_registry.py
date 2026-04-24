from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CoordinateInterpreterSpec:
    interpreter_id: str
    group_id: str
    group_title: str
    title: str
    description: str
    options: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            'interpreter_id': self.interpreter_id,
            'group_id': self.group_id,
            'group_title': self.group_title,
            'title': self.title,
            'description': self.description,
            'options': list(self.options),
        }


_GROUPS = {
    'A': 'Форма речи',
    'B': 'Психологическая форма',
    'C': 'Отношение к собеседнику',
    'D': 'Скрытая структура смысла',
    'E': 'Движение разговора',
    'F': 'Правда и позиция',
    'G': 'Мета-интерпретация',
}


def _spec(
    interpreter_id: str,
    group_id: str,
    title: str,
    description: str,
    options: tuple[str, ...],
) -> CoordinateInterpreterSpec:
    return CoordinateInterpreterSpec(
        interpreter_id=interpreter_id,
        group_id=group_id,
        group_title=_GROUPS[group_id],
        title=title,
        description=description,
        options=options,
    )


P_REGISTRY: tuple[CoordinateInterpreterSpec, ...] = (
    _spec('P1', 'A', 'Форма речи', 'Вопрос / утверждение / мысль / цитата / побуждение.', ('question', 'statement', 'thought', 'quote', 'directive')),
    _spec('P2', 'A', 'Прямой или маскированный смысл', 'Прямое, переносное или маскированное высказывание.', ('direct', 'figurative', 'masked')),
    _spec('P3', 'A', 'Буквальность', 'Буквальное высказывание или риторический ход.', ('literal', 'rhetorical_move')),
    _spec('P4', 'A', 'Ответность', 'Ответ, неответ или уклонение.', ('answer', 'non_answer', 'avoidance')),
    _spec('P5', 'A', 'Самостоятельность хода', 'Самостоятельная мысль или реакция на прошлую реплику.', ('independent', 'reaction')),
    _spec('P6', 'A', 'Направление хода', 'Завершение хода или открытие нового направления.', ('closing', 'opening_new_direction')),
    _spec('P7', 'A', 'Логическая определённость', 'Определённость или расплывчатость.', ('defined', 'diffuse')),
    _spec('P8', 'B', 'Сомнение', 'Есть ли сомнение как психологическая форма.', ('absent', 'doubt')),
    _spec('P9', 'B', 'Уверенность', 'Есть ли уверенность как форма подачи.', ('absent', 'confidence')),
    _spec('P10', 'B', 'Внутренний конфликт', 'Есть ли внутренний конфликт.', ('absent', 'inner_conflict')),
    _spec('P11', 'B', 'Защита', 'Есть ли защита или оборона позиции.', ('absent', 'defense')),
    _spec('P12', 'B', 'Уязвимость', 'Есть ли уязвимость.', ('absent', 'vulnerability')),
    _spec('P13', 'B', 'Нападение', 'Есть ли нападение.', ('absent', 'attack')),
    _spec('P14', 'B', 'Сдерживание', 'Есть ли сдерживание или торможение.', ('absent', 'containment')),
    _spec('P15', 'B', 'Напряжённость', 'Эмоциональная напряжённость или перегрузка.', ('calm', 'tense', 'overloaded')),
    _spec('P16', 'C', 'Забота', 'Есть ли забота по отношению к собеседнику.', ('absent', 'care')),
    _spec('P17', 'C', 'Уважение', 'Есть ли уважение.', ('absent', 'respect')),
    _spec('P18', 'C', 'Обесценивание', 'Есть ли обесценивание.', ('absent', 'devaluation')),
    _spec('P19', 'C', 'Унижение', 'Есть ли унижение.', ('absent', 'humiliation')),
    _spec('P20', 'C', 'Дружелюбие', 'Есть ли дружелюбие.', ('absent', 'friendliness')),
    _spec('P21', 'C', 'Скрытая враждебность', 'Есть ли скрытая враждебность.', ('absent', 'hidden_hostility')),
    _spec('P22', 'C', 'Доминирование', 'Есть ли доминирование.', ('neutral', 'dominance')),
    _spec('P23', 'C', 'Уступка', 'Есть ли уступка или подчинение.', ('neutral', 'concession')),
    _spec('P24', 'D', 'Сарказм', 'Есть ли сарказм или ложная похвала как его форма.', ('absent', 'sarcasm', 'dry_sarcasm', 'false_praise')),
    _spec('P25', 'D', 'Ирония', 'Есть ли ирония.', ('absent', 'irony')),
    _spec('P26', 'D', 'Издевательство', 'Есть ли издевательство.', ('absent', 'mockery')),
    _spec('P27', 'D', 'Маска похвалы', 'Похвала как маска другого действия.', ('absent', 'mask_of_praise')),
    _spec('P28', 'D', 'Маска заботы', 'Забота как маска другого действия.', ('absent', 'mask_of_care')),
    _spec('P29', 'D', 'Манипуляция', 'Есть ли манипуляция.', ('absent', 'manipulation')),
    _spec('P30', 'D', 'Давление', 'Есть ли давление.', ('absent', 'pressure')),
    _spec('P31', 'D', 'Ложное смягчение', 'Есть ли ложное смягчение.', ('absent', 'false_softening')),
    _spec('P32', 'E', 'Сближение', 'Движение к сближению.', ('neutral', 'approach')),
    _spec('P33', 'E', 'Дистанцирование', 'Движение к дистанцированию.', ('neutral', 'distancing')),
    _spec('P34', 'E', 'Примирение', 'Движение к примирению.', ('neutral', 'reconciliation')),
    _spec('P35', 'E', 'Эскалация', 'Движение к эскалации.', ('neutral', 'escalation')),
    _spec('P36', 'E', 'Обострение', 'Движение к обострению.', ('neutral', 'sharpening')),
    _spec('P37', 'E', 'Разрыв', 'Движение к разрыву контакта.', ('neutral', 'rupture')),
    _spec('P38', 'E', 'Смягчение', 'Движение к смягчению.', ('neutral', 'softening')),
    _spec('P39', 'E', 'Удержание контакта', 'Удержание контакта без разрыва.', ('neutral', 'contact_maintenance')),
    _spec('P40', 'F', 'Искренность', 'Есть ли искренность.', ('unclear', 'sincerity')),
    _spec('P41', 'F', 'Неискренность / маска', 'Есть ли маска или неискренность.', ('unclear', 'masking')),
    _spec('P42', 'F', 'Признание', 'Есть ли признание.', ('absent', 'admission')),
    _spec('P43', 'F', 'Отрицание', 'Есть ли отрицание.', ('absent', 'denial')),
    _spec('P44', 'F', 'Переосмысление', 'Есть ли переоценка или переосмысление.', ('absent', 'reframing')),
    _spec('P45', 'F', 'Корректирующая похвала', 'Похвала как мягкая коррекция.', ('absent', 'corrective_praise')),
    _spec('P46', 'F', 'Ложная похвала', 'Похвала как ложный ход.', ('absent', 'false_praise')),
    _spec('P47', 'F', 'Скрытый упрёк', 'Есть ли скрытый упрёк.', ('absent', 'hidden_reproach')),
    _spec('P48', 'G', 'Текущая реплика в цепочке', 'Интерпретация текущей реплики относительно прошлой цепочки.', ('continuation', 'reinterpretation', 'masking_shift', 'conflict_reply', 'repair_attempt')),
    _spec('P49', 'G', 'Итоговое направление хода', 'Куда движется ход относительно истории разговора.', ('neutral', 'toward_repair', 'toward_distance', 'toward_escalation', 'toward_masking', 'toward_contact')),
    _spec('P50', 'G', 'Смена темы', 'Произошла ли смена основной темы разговора в текущей реплике.', ('no_change', 'soft_shift', 'hard_shift', 'return_to_topic')),
    _spec('P51', 'G', 'Граница релевантного контекста', 'С какого момента контекст актуален: весь, с последней смены темы, или только последняя реплика.', ('full_window', 'since_topic_shift', 'last_turn_only', 'anchor_only')),
)


P_REGISTRY_BY_ID = {item.interpreter_id: item for item in P_REGISTRY}


def coordinate_registry_payload() -> list[dict[str, Any]]:
    return [item.to_dict() for item in P_REGISTRY]
