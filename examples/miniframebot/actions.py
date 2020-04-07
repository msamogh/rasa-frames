import logging
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


logger = logging.getLogger(__name__)


class ActionSearchDB(Action):
    def name(self) -> Text:
        return "action_search_db"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        logger.debug("Uttering message")
        dispatcher.utter_message(text=f"No results found for {tracker.slots['city']}")
        logger.debug("Uttered message")
        return []
