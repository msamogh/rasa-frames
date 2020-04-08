## story 3
* inform{"city": "Bangalore"}
  - action_change_frame
  - utter_searching
  - action_search_db

## interactive_story_1
* inform{"city": "X"}
    - action_change_frame
    - create_frame{""}
    - dump_current_frame
    - utter_searching
    - action_search_db
