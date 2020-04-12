import logging
from collections import namedtuple
from typing import Any, List, Text, Dict

from rasa.core.events import Event
from rasa.core.frames import FrameSet
from rasa.core.frames.utils import push_slots_into_current_frame, \
    pop_slots_from_current_frame, frames_from_tracker_slots


logger = logging.getLogger(__name__)


FrameIntent = namedtuple(
    'FrameIntent',
    ['can_contain_frame_ref', 'frame_ref_fallback', 'on_frame_ref_identified']
)

def get_intent_props(domain, intent):
    props = self.domain.intent_properties[intent]

    can_contain_frame_ref = props["can_contain_frame_ref"]
    frame_ref_fallback = props["frame_ref_fallback"]
    on_frame_ref_identified = props["on_frame_ref_identified"]

    return FrameIntent(
        can_contain_frame_ref,
        frame_ref_fallback,
        on_frame_ref_identified
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
        events = push_slots_into_current_frame(tracker) + \
            self.get_frame_events(tracker, user_utterance) + \
            pop_slots_from_current_frame()

        return events

    def get_frame_events(
        self,
        tracker: "DialogueStateTracker",
        user_utterance: "UserUttered"
    ) -> List[Event]:
        intent = user_utterance.intent
        frame_intent = get_intent_props(self.domain, intent)
        dialogue_entities = FrameSet.get_framed_entities(
            user_utterance.entities, self.domain
        )

        if frame_intent.can_contain_frame_ref:
            ref_frame_idx = self.get_best_matching_frame_idx(
                frames=frames_from_tracker_slots(tracker).frames,
                frame_intent=frame_intent,
                framed_entities=framed_entities,
            )
            events = self.on_frame_ref_identified(
                tracker,
                ref_frame_idx,
                frame_intent
            )
            return events

        return []

    def get_best_matching_frame_idx(
        self,
        frames: List["Frame"],
        frame_intent: FrameIntent,
        framed_entities: Dict[Text, Any]
    ) -> int:
        raise NotImplementedError

    def on_frame_ref_identified(
        self,
        tracker: "DialogueStateTracker",
        ref_frame_idx: int,
        frame_intent: FrameIntent
    ) -> List[Event]:
        return NotImplementedError
