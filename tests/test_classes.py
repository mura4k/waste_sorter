import pytest

from waste_sorter.classes import (
    DISPOSAL_HINTS,
    Acceptance,
    PlasticType,
    WasteClass,
    acceptance_for,
)


def test_every_class_has_hint():
    assert set(DISPOSAL_HINTS) == set(WasteClass)


@pytest.mark.parametrize("plastic_type", list(PlasticType))
def test_material_alone_does_not_establish_acceptance(plastic_type):
    assert acceptance_for(plastic_type) == Acceptance.UNKNOWN


def test_acceptance_depends_on_selected_policy():
    accepting_operator = {PlasticType.PET: Acceptance.ACCEPTED}
    rejecting_operator = {PlasticType.PET: Acceptance.REJECTED}

    assert acceptance_for(PlasticType.PET, accepting_operator) == Acceptance.ACCEPTED
    assert acceptance_for(PlasticType.PET, rejecting_operator) == Acceptance.REJECTED
    assert acceptance_for(PlasticType.PP, accepting_operator) == Acceptance.UNKNOWN
    assert acceptance_for(PlasticType.PET, {}) == Acceptance.UNKNOWN
