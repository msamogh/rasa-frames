import time
import logging

import pytest

from rasa.core.events import (
    SlotSet,
    FrameCreated,
    FrameUpdated,
    CurrentFrameDumped,
    UserUttered,
    CurrentFrameChanged
)
from rasa.core.frames import Frame, FrameSet, RuleBasedFramePolicy
from rasa.core.frames.utils import (
    push_slots_into_current_frame,
    pop_slots_from_current_frame,
    frames_from_tracker_slots,
)
from rasa.core.frames.frame_policy import FrameIntent


logger = logging.getLogger(__name__)


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
        slots={"city": "Bengaluru", "budget": 1500}, created=time.time(), switch_to=True
    )
    frames.add_frame(slots={"city": "Bengaluru",
                            "budget": 2500}, created=time.time())
    frames.add_frame(
        slots={"city": "Tumakuru", "budget": 1000}, created=time.time(),
    )
    return frames


def test_frameset_as_dict(populated_frames):
    assert populated_frames.as_dict() == {
        'frames': [
            {
                'slots': {
                    'city': 'Bengaluru',
                    'budget': 1500
                },
                'created': populated_frames.frames[0].created
            },
            {
                'slots': {
                    'city': 'Bengaluru',
                    'budget': 2500
                },
                'created': populated_frames.frames[1].created
            },
            {
                'slots': {
                    'city': 'Tumakuru',
                    'budget': 1000
                },
                'created': populated_frames.frames[2].created
            }
        ],
        'current_frame_idx': 0
    }


def test_frameset_from_dict(populated_frames):
    frames = FrameSet.from_dict({
        'frames': [
            {
                'slots': {
                    'city': 'Bengaluru',
                    'budget': 1500
                },
                'created': populated_frames.frames[0].created
            },
            {
                'slots': {
                    'city': 'Bengaluru',
                    'budget': 2500
                },
                'created': populated_frames.frames[1].created
            },
            {
                'slots': {
                    'city': 'Tumakuru',
                    'budget': 1000
                },
                'created': populated_frames.frames[2].created
            }
        ],
        'current_frame_idx': 0
    })

    assert frames.current_frame_idx == 0
    assert frames[0]['city'] == 'Bengaluru'
    assert frames[0]['budget'] == 1500
    assert frames[1]['city'] == 'Bengaluru'
    assert frames[1]['budget'] == 2500
    assert frames[1].created > frames[0].created
    assert frames[2]['city'] == 'Tumakuru'
    assert frames[2]['budget'] == 1000
    assert frames[2].created > frames[1].created


@pytest.mark.parametrize(
    "entities, on_frame_match_failed, on_frame_ref_identified, best_matching_idx",
    [
        ({"city": "Bengaluru"}, "most_recent", "switch", 1),
        ({"city": "Bengaluru", "budget": 1500}, "most_recent", "switch", 0),
        ({"city": "Tumakuru"}, "most_recent", "switch", 2),
        ({"city": "Tumakuru", "budget": 1200}, "create_new", "switch", 3),
        ({"city": "Mangaluru"}, "create_new", "switch", 3),
        ({"city": "Mangaluru"}, "most_recent", "switch", 2),
    ],
)
def test_get_best_matching_frame_idx(
    rule_based_frame_policy,
    populated_frames,
    on_frame_match_failed,
    on_frame_ref_identified,
    entities,
    best_matching_idx,
):
    assert (
        rule_based_frame_policy.get_best_matching_frame_idx(
            populated_frames.frames,
            populated_frames.current_frame_idx,
            FrameIntent(
                can_contain_frame_ref=True,
                on_frame_match_failed=on_frame_match_failed,
                on_frame_ref_identified=on_frame_ref_identified,
            ),
            entities,
        )
        == best_matching_idx
    )


@pytest.mark.parametrize(
    "intent, on_frame_match_failed, on_frame_ref_identified",
    [
        ("inform", "create_new", "switch"),
        ("compare", "most_recent", "populate"),
        ("switch_frame", "most_recent", "switch"),
    ],
)
def test_from_intent(
    framebot_domain, intent, on_frame_match_failed, on_frame_ref_identified
):
    frame_intent = FrameIntent.from_intent(framebot_domain, intent)
    assert frame_intent.can_contain_frame_ref is True
    assert frame_intent.on_frame_match_failed == on_frame_match_failed
    assert frame_intent.on_frame_ref_identified == on_frame_ref_identified


@pytest.mark.parametrize(
    "utterances",
    [
        ([
            (
                "inform",
                [{"entity": "budget", "value": 3000}],
                FrameCreated,
                FrameUpdated,
                1,
                0,
                None,
                3000
            )
        ]),
        ([
            (
                "inform",
                [{"entity": "city", "value": "Bengaluru"}],
                FrameCreated,
                FrameUpdated,
                1,
                0,
                "Bengaluru",
                None
            ),
            (
                "inform",
                [{"entity": "budget", "value": 3000}],
                FrameUpdated,
                FrameUpdated,
                1,
                0,
                "Bengaluru",
                3000
            ),
        ]),
        ([
            (
                "inform",
                [{"entity": "city", "value": "X"}],
                FrameCreated,
                FrameUpdated,
                1,
                0,
                "X",
                None
            ),
            (
                "inform",
                [{"entity": "city", "value": "Y"}],
                FrameUpdated,
                FrameCreated,
                2,
                1,
                "Y",
                None
            ),
            (
                "inform",
                [{"entity": "budget", "value": 1500}],
                FrameUpdated,
                FrameUpdated,
                2,
                1,
                "Y",
                1500
            ),
            (
                "inform",
                [{"entity": "city", "value": "Z"}],
                FrameUpdated,
                FrameCreated,
                3,
                2,
                "Z",
                None
            ),
            (
                "inform",
                [{"entity": "city", "value": "X"}],
                FrameUpdated,
                CurrentFrameChanged,
                3,
                0,
                "X",
                None
            ),
            (
                "inform",
                [{"entity": "budget", "value": 20}],
                FrameUpdated,
                FrameUpdated,
                3,
                0,
                "X",
                20
            ),
            (
                "inform",
                [{"entity": "city", "value": "Z"}],
                FrameUpdated,
                CurrentFrameChanged,
                3,
                2,
                "Z",
                None
            )
        ])
    ],
)
def test_rule_based_frame_policy(rule_based_frame_policy, framebot_tracker, utterances):
    logger.debug(len(utterances))
    for i, utterance in enumerate(utterances):
        (
            intent,
            entities,
            exp_push_event,
            exp_frame_event,
            exp_num_frames_post_update,
            exp_current_frame_idx,
            exp_city,
            exp_budget
        ) = utterance
        logger.debug(f"Turn {i}")

        push_events = push_slots_into_current_frame(framebot_tracker)
        frame_events = rule_based_frame_policy.get_frame_events(
            framebot_tracker,
            UserUttered(
                intent={"name": intent, "confidence": 1.0}, entities=entities)
        )
        pop_events = pop_slots_from_current_frame()

        assert isinstance(push_events[0], exp_push_event)
        if exp_frame_event is None:
            assert len(frame_events) == 0
        else:
            assert isinstance(frame_events[0], exp_frame_event)
        assert isinstance(pop_events[0], CurrentFrameDumped)

        events = push_events + frame_events + pop_events
        for event in events:
            framebot_tracker.update(event)

        logger.debug(framebot_tracker.frames)

        assert len(framebot_tracker.frames) == exp_num_frames_post_update
        assert framebot_tracker.frames.current_frame_idx == exp_current_frame_idx

        assert framebot_tracker.frames.current_frame["city"] == exp_city
        assert framebot_tracker.frames.current_frame["budget"] == exp_budget
