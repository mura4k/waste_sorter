"""Целевые классы (фракции) и подсказки по утилизации."""

from enum import Enum


class WasteClass(str, Enum):
    PLASTIC = "plastic"
    PAPER = "paper"
    GLASS = "glass"
    METAL = "metal"
    ORGANIC = "organic"
    HAZARDOUS = "hazardous"
    OTHER = "other"


DISPOSAL_HINTS: dict[WasteClass, str] = {
    WasteClass.PLASTIC: "Сполосните и сомните, контейнер для пластика.",
    WasteClass.PAPER: "Сухая бумага и картон, контейнер для макулатуры.",
    WasteClass.GLASS: "Без крышек, контейнер для стекла.",
    WasteClass.METAL: "Банки сполосните и сплющите, контейнер для металла.",
    WasteClass.ORGANIC: "Пищевые отходы, контейнер для органики или компост.",
    WasteClass.HAZARDOUS: "Батарейки, лампы, техника: специальный пункт приёма.",
    WasteClass.OTHER: "Не перерабатывается, контейнер для смешанных отходов.",
}


class PlasticType(str, Enum):
    """Тип пластика, определяется только для класса `WasteClass.PLASTIC`."""

    PET = "pet"  # ♳ 1
    HDPE = "hdpe"  # ♴ 2
    LDPE = "ldpe"  # ♶ 4
    PP = "pp"  # ♷ 5
    PS = "ps"  # ♸ 6
    NON_RECYCLABLE = "non_recyclable"  # ♵ 3 (PVC), ♹ 7 (прочее, композиты)


RECYCLABLE_PLASTICS: frozenset[PlasticType] = frozenset(
    {PlasticType.PET, PlasticType.HDPE, PlasticType.LDPE, PlasticType.PP, PlasticType.PS}
)


def is_recyclable(plastic_type: PlasticType) -> bool:
    return plastic_type in RECYCLABLE_PLASTICS
