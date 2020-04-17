# Rasa Frames (Rasa + Microsoft Frames)
<img align="right" height="124" src="https://www.rasa.com/assets/img/sara/sara-open-source-2.0.png">

<img align="right" height="124" src="https://logos-download.com/wp-content/uploads/2016/02/Microsoft_box.png">

Decision making through conversation involves comparing items and exploring different alternatives. This requires memory.

Rasa Frames is a fork of Rasa that augments the `DialogueStateTracker` with multiple copies of slots (or Frames), each corresonding to an item of discussion.

This project is directly inspired by the [Microsoft Frames](https://www.microsoft.com/en-us/research/project/frames-dataset/) dataset.

## Introduction

Here is a typical conversation between a travel agent bot and yet another Bangalorean who ends up going to Goa for a vacation. You can see the user and the bot going through various options before narrowing down on the final one.

Rasa Frames aims to automatically manage the gory details of creating, switching, and
referencing frames so that you can focus on writing the core "business logic" of
your bot.

<img align="center" height="540" src="https://github.com/msamogh/rasa-frames/raw/master/Frames.png">

## Installation
1. Clone the repository:
```
git clone https://github.com/msamogh/rasa-frames
```

2. Install the package (preferably inside a virtualenv):
```
cd rasa-frames && pip install -e .
```

## Getting Started: Your First Framebot
If you aren't already familiar with Rasa, then check out the [Rasa tutorial](https://rasa.com/docs/rasa/user-guide/rasa-tutorial/).

### 1. Configure your domain
#### Slots
To indicate that you want each frame to have its own copy of a slot, simply add the `frame_slot: True` property under that slot in your domain file.

```yaml
slots:
  city:
    type: unfeaturized
    frame_slot: true
  budget:
    type: unfeaturized
    frame_slot: true
  ref:
    type: unfeaturized
    frame_slot: true
  name:                 # a regular slot that is shared across all frames
    type: text
```

#### Intents
For each intent, there are a bunch of properties you need to configure.

##### `can_contain_frame_ref`
indicates whether or not this intent could potentially include a reference to one or more frames

##### `on_frame_match_failed`
defines the fallback policy in the event that no existing frame matched the frame reference in the user utterance

The possible values are:
1. `create_new` - creates a new frame with the slot-values extracted from the latest user utterance
2. `most_recent` - informs the FramePolicy to switch to the most recently created frame.

##### `on_frame_ref_identified`
determines what should happen once a frame reference has been identified

The possible values are:
1. `switch` - makes the identified frame the current (active) frame
2. `populate` - simply populates the `ref` slot with the id of the reference frame; does not change the active frame.

```yaml
intents:
  - greet
  - affirm
  - inform:
      can_contain_frame_ref: true
      on_frame_match_failed: "create_new"
      on_frame_ref_identified: "switch"
  - compare:
      can_contain_frame_ref: true
      on_frame_match_failed: "most_recent"
      on_frame_ref_identified: "populate"
  - switch_frame:
      can_contain_frame_ref: true
      on_frame_match_failed: "most_recent"
      on_frame_ref_identified: "switch"
```

### 3. Custom Actions



## References
1. El Asri, L., Schulz, H., Sharma, S., Zumer, J., Harris, J., Fine, E., Mehrotra, R., & Suleman, K. (2018). Frames: a corpus for adding memory to goal-oriented dialogue systems. 207–219. https://doi.org/10.18653/v1/w17-5526
