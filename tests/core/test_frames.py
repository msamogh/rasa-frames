import time

import pytest

from rasa.core.events import SlotSet, FrameCreated, CurrentFrameDumped
from rasa.core.frames import Frame, FrameSet, RuleBasedFramePolicy
from rasa.core.frames.utils import push_slots_into_current_frame, pop_slots_from_current_frame, \
    frames_from_tracker_slots
from rasa.core.frames.frame_policy import FrameIntent


@pytest.fixture()
def populated_tracker(framebot_tracker):
    framebot_tracker.update(SlotSet("city", "Bengaluru"))
    return framebot_tracker


@pytest.fixture()
def rule_based_frame_policy(framebot_domain):
    rbft = RuleBasedFramePolicy(framebot_domain)
    return rbft


def test_push_slots(populated_tracker):
    events = push_slots_into_current_frame(populated_tracker)
    assert len(events) == 1
    assert isinstance(events[0], FrameCreated)

    populated_tracker.update(events[0])
    assert populated_tracker.frames.current_frame_idx == 0
    assert populated_tracker.frames.current_frame["city"] == "Bengaluru"


def test_pop_frame(populated_tracker):
    assert populated_tracker.slots["city"].value == "Bengaluru"

    events = push_slots_into_current_frame(populated_tracker)
    populated_tracker.update(events[0])
    assert populated_tracker.frames.current_frame_idx == 0

    populated_tracker.frames.current_frame["city"] = "Tumakuru"
    events = pop_slots_from_current_frame()
    assert len(events) == 1
    assert isinstance(events[0], CurrentFrameDumped)

    populated_tracker.update(events[0])
    assert populated_tracker.slots["city"].value == "Tumakuru"


def test_frames_from_tracker_slots(rule_based_frame_policy, populated_tracker):
    assert populated_tracker.frames.current_frame is None

    frames = frames_from_tracker_slots(populated_tracker)
    assert frames.current_frame_idx == 0
    assert frames.current_frame["city"] == "Bengaluru"


@pytest.fixture()
def populated_frames():
    frames = FrameSet()
    frames.add_frame(
        slots={"city": "Bengaluru", "budget": 1500},
        created=time.time(),
        switch_to=True
    )
    frames.add_frame(
        slots={"city": "Bengaluru", "budget": 2500},
        created=time.time()
    )
    frames.add_frame(
        slots={"city": "Tumakuru", "budget": 1000},
        created=time.time(),
    )
    return frames


@pytest.mark.parametrize(
    "entities, on_frame_match_failed, on_frame_ref_identified, best_matching_idx",
    [
        (
            {"city": "Bengaluru"},
            "most_recent",
            "switch",
            1
        ),
        (
            {
                "city": "Bengaluru",
                "budget": 1500
            },
            "most_recent",
            "switch",
            0
        ),
        (
            {"city": "Tumakuru"},
            "most_recent",
            "switch",
            2
        ),
        (
            {"city": "Tumakuru", "budget": 1200},
            "create_new",
            "switch",
            3
        ),
        (
            {"city": "Mangaluru"},
            "create_new",
            "switch",
            3
        ),
        (
            {"city": "Mangaluru"},
            "most_recent",
            "switch",
            2
        )
    ]
)
def test_get_best_matching_frame_idx(
    rule_based_frame_policy,
    populated_frames,
    on_frame_match_failed,
    on_frame_ref_identified,
    entities,
    best_matching_idx
):
    assert rule_based_frame_policy.get_best_matching_frame_idx(
        populated_frames,
        FrameIntent(
            can_contain_frame_ref=True,
            on_frame_match_failed=on_frame_match_failed,
            on_frame_ref_identified=on_frame_ref_identified
        ),
        entities
    ) == best_matching_idx



