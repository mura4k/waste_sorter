from waste_sorter.classes import (
    DISPOSAL_HINTS,
    RECYCLABLE_PLASTICS,
    PlasticType,
    WasteClass,
    is_recyclable,
)


def test_every_class_has_hint():
    assert set(DISPOSAL_HINTS) == set(WasteClass)


def test_only_non_recyclable_plastic_is_excluded():
    assert set(PlasticType) - RECYCLABLE_PLASTICS == {PlasticType.NON_RECYCLABLE}


def test_is_recyclable():
    assert is_recyclable(PlasticType.PET)
    assert not is_recyclable(PlasticType.NON_RECYCLABLE)
