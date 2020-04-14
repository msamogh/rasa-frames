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
    """Wrapper for frames-related properties of an intent (extracted from the domain)."""

    def __init__(
        self, can_contain_frame_ref, on_frame_match_failed, on_frame_ref_identified
    ):
        """Initialize a FrameIntent.

        Args:
            can_contain_frame_ref: Whether or not this intent can contain a
                frame reference.
            on_frame_match_failed: The strategy to be followed in case the entities
                extracted from the user utterance does not fully match any existing
                frame in the FrameSet.
            on_frame_ref_identified: The course of action to be taken once the frame
                ref has been identified.
        """
        self.can_contain_frame_ref = can_contain_frame_ref
        self.on_frame_match_failed = on_frame_match_failed
        self.on_frame_ref_identified = on_frame_ref_identified

    @classmethod
    def from_intent(cls, domain: "Domain", intent: Text):
        """Create a FrameIntent instance from the intent name.
        
        Args:
            domain: Domain object that describes this intent.
            intent: Name of the intent.

        Returns:
            The FrameIntent object.
        """
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
        # 1. Copy the slot values in the tracker over to the current frame. This step
        # is needed since a custom action might have modified the tracker's slots
        # since the last `UserUttered` event. Essentially, we want the frames inside
        # `tracker.frames` to reflect the latest state of the tracker.
        # 2. If the latest user-utterance calls for any changes to the FrameSet - either
        # changing the active frame or creating a new frame, make those changes.
        # 3. Copy the slots from the current frame (potentially changed after step 2)
        # over to the tracker (essentially the reverse of step 1).
        events = (
            push_slots_into_current_frame(tracker)
            + self.get_frame_events(tracker, user_utterance)
            + pop_slots_from_current_frame()
        )

        return events

    def get_frame_events(
        self, tracker: "DialogueStateTracker", user_utterance: "UserUttered"
    ) -> List[Event]:
        """Update the FrameSet based on the user_utterance."""
        intent = user_utterance.intent
        frame_intent = FrameIntent.from_intent(self.domain, intent["name"])
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
            return events

        return []

    def get_best_matching_frame_idx(
        self,
        frames: List["Frame"],
        frame_intent: FrameIntent,
        framed_entities: Dict[Text, Any],
    ) -> int:
        """Identify which frame the user is talking about.
        
        There are 3 possibilities here:
        1. the current frame
        2. an existing frame that is not the current frame
        3. a new frame.

        Args:
            frames: List of all the frames in the FrameSet
            frame_intent: Frame-related properties of the intent
            framed_entities: Relevant entities in the user_utterance

        Returns:
            int: index of the frame being referenced
        """
        raise NotImplementedError

    def on_frame_ref_identified(
        self,
        tracker: "DialogueStateTracker",
        ref_frame_idx: int,
        frame_intent: FrameIntent,
    ) -> List[Event]:
        """Course of action to take once the frame ref has been identified.
        
        Args:
            tracker: The DialogueStateTracker object.
            ref_frame_idx: 
        """
        return NotImplementedError
