import logging
from collections import Counter
from typing import List, Dict, Text, Any, Callable, Tuple

from rasa.core.events import Event, FrameCreated, CurrentFrameChanged, SlotSet, \
    FrameUpdated
from rasa.core.frames import FramePolicy
from rasa.core.frames.frame_policy import FrameIntent


logger = logging.getLogger(__name__)


class RuleBasedFramePolicy(FramePolicy):

    @staticmethod
    def most_recent(
        matching_candidates, non_conflicting_candidates, all_frames, current_frame_idx
    ) -> int:
        if matching_candidates:
            return most_recent_frame_idx(matching_candidates)
        return most_recent_frame_idx(all_frames)

    @staticmethod
    def create_new(
        matching_candidates, non_conflicting_candidates, all_frames, current_frame_idx
    ) -> int:
        """Create a new frame if no perfect match found."""
        if matching_candidates:
            return most_recent_frame_idx(matching_candidates)
        if non_conflicting_candidates:
            if current_frame_idx in [f.idx for f in non_conflicting_candidates]:
                return current_frame_idx
        return len(all_frames)

    def get_best_matching_frame_idx(
        self,
        frames: List["Frame"],
        current_frame_idx: int,
        frame_intent: FrameIntent,
        framed_entities: Dict[Text, Any],
    ) -> int:
        matching_candidates = self._fully_matching_candidates(
            frames, framed_entities)
        non_conflicting_candidates = self._non_conflicting_candidates(
            frames, framed_entities
        )

        if frame_intent.on_frame_match_failed == "create_new":
            choice_fn = RuleBasedFramePolicy.create_new
        elif frame_intent.on_frame_match_failed == "most_recent":
            choice_fn = RuleBasedFramePolicy.most_recent
        else:
            raise RuntimeError(
                "on_frame_match_failed must be one of " "['create_new', 'most_recent']."
            )

        return choice_fn(
            matching_candidates, non_conflicting_candidates, frames, current_frame_idx
        )

    def _fully_matching_candidates(
        self, frames: List["Frame"], framed_entities: Dict[Text, Any]
    ) -> List["Frame"]:
        assert len(frames) > 0

        equality_counts = Counter()
        for key, value in framed_entities.items():
            for idx, frame in enumerate(frames):
                if frame[key] == value:
                    equality_counts[idx] += 1

        return [
            frames[idx]
            for idx, num_matches in equality_counts.items()
            if num_matches == len(framed_entities)
        ]

    def _non_conflicting_candidates(
        self, frames: List["Frame"], framed_entities: Dict[Text, Any]
    ) -> List["Frame"]:
        assert len(frames) > 0

        conflict_counts = Counter()
        for key, value in framed_entities.items():
            for idx, frame in enumerate(frames):
                if frame[key] is not None and frame[key] != value:                    
                    conflict_counts[idx] += 1
        return [frame for idx, frame in enumerate(frames) if conflict_counts[idx] == 0]

    def on_frame_ref_identified(
        self,
        frames: List["Frame"],
        current_frame_idx: int,
        framed_entities: Dict[Text, Any],
        ref_frame_idx: int,
        frame_intent: FrameIntent,
    ) -> List[Event]:
        assert ref_frame_idx <= len(
            frames
        ), "ref is equal to {}. It cannot violate 0 <= ref <= len(frames)".format(
            ref_frame_idx
        )

        if frame_intent.on_frame_ref_identified == "switch":
            return self._switch_or_create_frame(
                frames, current_frame_idx, framed_entities, ref_frame_idx
            )
        elif on_frame_ref_identified == "populate":
            return [SlotSet("ref", ref_frame_idx)]
        else:
            raise RuntimeError(
                "on_frame_ref_identified must be one of ['switch', 'populate']."
            )

    def _switch_or_create_frame(
        self,
        frames: List["Frame"],
        current_frame_idx: int,
        framed_entities: Dict[Text, Any],
        ref_frame_idx: int,
    ) -> List[Event]:
        if ref_frame_idx == len(frames):
            return [FrameCreated(slots=framed_entities, switch_to=True)]
        elif ref_frame_idx != current_frame_idx:
            return [CurrentFrameChanged(frame_idx=ref_frame_idx)]
        else:
            updates = []
            for key, value in framed_entities.items():
                updates.append(FrameUpdated(
                    frame_idx=current_frame_idx,
                    name=key,
                    value=value
                ))
            return updates


def most_recent_frame_idx(frames: List["Frame"]) -> int:
    most_recent_frame = list(
        sorted(frames, key=lambda frame: frame.created, reverse=True)
    )[0]
    return most_recent_frame.idx
