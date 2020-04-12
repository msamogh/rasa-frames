import pytest

from rasa.core.events import SlotSet, FrameCreated, CurrentFrameDumped
from rasa.core.frames import Frame, FrameSet, RuleBasedFrameTracker, \
    pop_slots_from_current_frame, push_slots_into_current_frame


@pytest.fixture()
def populated_tracker(framebot_tracker):
    framebot_tracker.update(SlotSet("city", "Bengaluru"))
    return framebot_tracker


@pytest.fixture()
def rule_based_frame_tracker(framebot_domain):
    rbft = RuleBasedFrameTracker(framebot_domain)
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


def test_get_latest_frames(rule_based_frame_tracker, populated_tracker):
    frames = rule_based_frame_tracker._get_latest_frames(populated_tracker)
    
