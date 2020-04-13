import logging
from collections import namedtuple
from typing import Any, List, Text, Dict

from rasa.core.events import Event
from rasa.core.frames import FrameSet
from rasa.core.frames.utils import (
    push_slots_into_current_frame,
    pop_slots_from_current_frame,
    frames_from_tracker_slots,
)


logger = logging.getLogger(__name__)


class FrameIntent(object):
    def __init__(
        self, can_contain_frame_ref, on_frame_match_failed, on_frame_ref_identified
    ):
        self.can_contain_frame_ref = can_contain_frame_ref
        self.on_frame_match_failed = on_frame_match_failed
        self.on_frame_ref_identified = on_frame_ref_identified

    @classmethod
    def from_intent(cls, domain: "Domain", intent: Text):
        props = domain.intent_properties[intent]

        can_contain_frame_ref = props["can_contain_frame_ref"]
        on_frame_match_failed = props["on_frame_match_failed"]
        on_frame_ref_identified = props["on_frame_ref_identified"]

        return cls(
            can_contain_frame_ref, on_frame_match_failed, on_frame_ref_identified
        )


class FramePolicy(object):
    def __init__(self, domain: "Domain") -> None:
        self.domain = domain

    def predict(
        self, tracker: "DialogueStateTracker", user_utterance: "UserUttered"
    ) -> List[Event]:
        # 1. Update the current frame with slots that might have been updated
        # through other custom actions.
        # 2. Check for changes in either the slot-values within the current frame
        # or for a switching of the active frame.
        # 3. Refill the slots with the contents of the current frame
        events = (
            push_slots_into_current_frame(tracker)
            + self.get_frame_events(tracker, user_utterance)
            + pop_slots_from_current_frame()
        )

        logger.debug(f"Final events: {events}")

        return events

    def get_frame_events(
        self, tracker: "DialogueStateTracker", user_utterance: "UserUttered"
    ) -> List[Event]:
        intent = user_utterance.intent
        frame_intent = FrameIntent.from_intent(self.domain, intent["name"])
        logger.debug(user_utterance.entities)
        dialogue_entities = FrameSet.get_framed_entities(
            user_utterance.entities, self.domain
        )

        latest_frameset = frames_from_tracker_slots(tracker)
        if frame_intent.can_contain_frame_ref:
            ref_frame_idx = self.get_best_matching_frame_idx(
                frames=latest_frameset.frames,
                current_frame_idx=latest_frameset.current_frame_idx,
                framed_entities=dialogue_entities,
                frame_intent=frame_intent,
            )
            events = self.on_frame_ref_identified(
                frames=latest_frameset.frames,
                framed_entities=dialogue_entities,
                current_frame_idx=latest_frameset.current_frame_idx,
                ref_frame_idx=ref_frame_idx,
                frame_intent=frame_intent,
            )
            logger.debug(f'Current frame......{tracker.frames}')
            logger.debug(f'Current frame......{latest_frameset}')
            logger.debug(f'Events....{events}')
            return events

        return []

    def get_best_matching_frame_idx(
        self,
        frames: List["Frame"],
        frame_intent: FrameIntent,
        framed_entities: Dict[Text, Any],
    ) -> int:
        raise NotImplementedError

    def on_frame_ref_identified(
        self,
        tracker: "DialogueStateTracker",
        ref_frame_idx: int,
        frame_intent: FrameIntent,
    ) -> List[Event]:
        return NotImplementedError
