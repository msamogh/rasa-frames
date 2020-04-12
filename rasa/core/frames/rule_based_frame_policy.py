import logging
from typing import List, Dict, Text, Any, Callable

from rasa.core.events import Event, FrameCreated, CurrentFrameChanged, SlotSet
from rasa.core.frames import FramePolicy
from rasa.core.frames.frame_policy import FrameIntent


logger = logging.getLogger(__name__)


class RuleBasedFramePolicy(FramePolicy):
    @staticmethod
    def most_recent(matching_candidates, non_conflicting_candidates, all_frames) -> int:
        if matching_candidates:
            return most_recent_frame_idx(matching_candidates)
        return most_recent_frame_idx(all_frames)

    @staticmethod
    def create_new(matching_candidates, non_conflicting_candidates, all_frames) -> int:
        """Create a new frame if no perfect match found."""
        if matching_candidates:
            return most_recent_frame_idx(matching_candidates)
        if non_conflicting_candidates:
            if all_frames.current_frame_idx in non_conflicting_candidates:
                return all_frames.current_frame_idx
        return len(all_frames)

    def get_best_matching_frame_idx(
        self,
        frames: List["Frame"],
        frame_intent: FrameIntent,
        framed_entities: Dict[Text, Any],
    ) -> int:

        from collections import Counter

        assert len(frames) > 0

        equality_counts = Counter()
        conflict_counts = Counter()
        for key, value in framed_entities.items():
            for idx, frame in enumerate(frames):
                if frame[key] == value:
                    equality_counts[idx] += 1
                else:
                    conflict_counts[idx] += 1

        frames_sorted_by_num_equal = equality_counts.most_common()
        frames_sorted_by_num_conflicts = conflict_counts.most_common()
        # If the best match has a less-than-perfect match, simply return
        # the most recently created frame.
        matching_candidates = []
        for idx, num_matches in frames_sorted_by_num_equal:
            if num_matches == len(framed_entities):
                matching_candidates.append(frames[idx])
        non_conflicting_candidates = []
        for idx, num_conflicts in frames_sorted_by_num_conflicts:
            if num_conflicts == 0:
                non_conflicting_candidates.append(frames[idx])

        if frame_intent.on_frame_match_failed == "create_new":
            choice_fn = RuleBasedFramePolicy.create_new
        elif frame_intent.on_frame_match_failed == "most_recent":
            choice_fn = RuleBasedFramePolicy.most_recent
        else:
            raise RuntimeError(
                "on_frame_match_failed must be one of " "['create_new', 'most_recent']."
            )

        return choice_fn(matching_candidates, non_conflicting_candidates, frames)

    def on_frame_ref_identified(
        self,
        tracker: "DialogueStateTracker",
        ref_frame_idx: int,
        frame_intent: FrameIntent,
    ) -> List[Event]:
        assert ref_frame_idx <= len(
            tracker.frames
        ), "ref cannot violate 0 <= ref <= len(tracker.frames)"

        if frame_intent.on_frame_ref_identified == "switch":
            if ref_frame_idx == len(tracker.frames):
                return [FrameCreated(slots=framed_entities, switch_to=True)]
            else:
                return [CurrentFrameChanged(frame_idx=ref_frame_idx)]
        elif on_frame_ref_identified == "populate":
            return [SlotSet("ref", ref_frame_idx)]
        else:
            raise RuntimeError(
                "on_frame_ref_identified must be one of " "['switch', 'populate']."
            )


def most_recent_frame_idx(frames: List["Frame"]) -> int:
    most_recent_frame = list(
        sorted(frames, key=lambda frame: frame.created, reverse=True)
    )[0]
    return most_recent_frame.idx
