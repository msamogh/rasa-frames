import time

import pytest

from rasa.core.events import SlotSet, FrameCreated, CurrentFrameDumped
from rasa.core.frames import Frame, FrameSet, RuleBasedFramePolicy
from rasa.core.frames.utils import push_slots_into_current_frame, pop_slots_from_current_frame
from rasa.core.frames.rule_based_frame_policy import most_recent_frame_idx, create_new, \
    most_recent

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


def test_get_latest_frames(rule_based_frame_policy, populated_tracker):
    assert populated_tracker.frames.current_frame is None

    frames = rule_based_frame_policy._get_latest_frames(populated_tracker)
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
    "entities, strategy, best_matching_idx",
    [
        (
            {"city": "Bengaluru"},
            most_recent, 1
        ),
        (
            {
                "city": "Bengaluru",
                "budget": 1500
            },
            most_recent, 0
        ),
        (
            {"city": "Tumakuru"},
            most_recent, 2
        ),
        (
            {"city": "Tumakuru", "budget": 1200},
            create_new, 3
        ),
        (
            {"city": "Mangaluru"},
            create_new, 3
        ),
        (
            {"city": "Mangaluru"},
            most_recent, 2
        )
    ]
)
def test_get_best_matching_frame_idx(
    rule_based_frame_policy,
    populated_frames,
    entities,
    strategy,
    best_matching_idx
):
    assert get_best_matching_frame_idx(
        populated_frames,
        entities,
        strategy
    ) == best_matching_idx


