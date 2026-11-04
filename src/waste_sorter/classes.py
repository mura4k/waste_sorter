"""Материалы и общие подсказки; приём определяется отдельным профилем оператора."""

from collections.abc import Mapping
from enum import StrEnum


class WasteClass(StrEnum):
    PLASTIC = "plastic"
    PAPER = "paper"
    GLASS = "glass"
    METAL = "metal"
    ORGANIC = "organic"
    HAZARDOUS = "hazardous"
    OTHER = "other"


DISPOSAL_HINTS: dict[WasteClass, str] = {
    WasteClass.PLASTIC: "Проверьте маркировку и условия приёма пластика у местного оператора.",
    WasteClass.PAPER: "Проверьте состав, загрязнение и правила приёма бумаги у местного оператора.",
    WasteClass.GLASS: "Проверьте вид изделия и правила приёма стекла у местного оператора.",
    WasteClass.METAL: "Проверьте вид изделия и правила приёма металла у местного оператора.",
    WasteClass.ORGANIC: "Уточните, есть ли местный сбор органики и какие отходы он принимает.",
    WasteClass.HAZARDOUS: (
        "Возможны опасные отходы: не используйте обычный контейнер, уточните специальный сбор."
    ),
    WasteClass.OTHER: "Материал вне основных категорий: уточните состав и местные правила приёма.",
}


class PlasticType(StrEnum):
    """Четыре кандидата; неизвестный полимер не получает выдуманный подтип."""

    PET = "pet"  # ♳ 1
    HDPE = "hdpe"  # ♴ 2
    PP = "pp"  # ♷ 5
    PS = "ps"  # ♸ 6


class Acceptance(StrEnum):
    """Приём предмета в контексте проверенного профиля, а не свойство полимера."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


def acceptance_for(
    plastic_type: PlasticType,
    policy: Mapping[PlasticType, Acceptance] | None = None,
) -> Acceptance:
    """Применить профиль, выбранный после проверки условий для конкретного предмета.

    Отсутствующая запись не означает отказ в приёме. Вызывающий код отвечает за
    проверку оператора, версии правил, вида изделия и его состояния. Универсального
    профиля по умолчанию нет.
    """
    if policy is None:
        return Acceptance.UNKNOWN
    return policy.get(plastic_type, Acceptance.UNKNOWN)
