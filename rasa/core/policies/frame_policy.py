from typing import List

from rasa.core.action import ACTION_CHANGE_FRAME_NAME
from rasa.core.domain import Domain
from rasa.core.events import ActionExecuted, SlotSet, UserUttered, ActionReverted
from rasa.core.trackers import DialogueStateTracker

from rasa.core.policies.memoization import AugmentedMemoizationPolicy
from rasa.core.policies.keras_policy import KerasPolicy


class FrameMemoizationPolicy(AugmentedMemoizationPolicy):

    def _change_frame_action(self, domain):
        idx = domain.index_for_action(action)
        prediction[idx] = 1
        return prediction

    def predict_action_probabilities(
        self, tracker: DialogueStateTracker, domain: Domain
    ) -> List[float]:
        idx = -1
        events = tracker.events_after_latest_restart()
        while isinstance(events[idx], SlotSet):
            idx -= 1

        if isinstance(events[idx], UserUttered):
            self.original_prediction = super().predict_action_probabilities(tracker, domain)
            return self._change_frame_action(domain)

        if isinstance(events[idx], ActionExecuted) and \
                events[idx].action_name == ACTION_CHANGE_FRAME_NAME:
            tracker.update(ActionReverted())
            for key, value in tracker.frames.current_frame.items():
                tracker.update(SlotSet(key, value))

        return super().predict_action_probabilities(tracker, domain)
