# Rasa Frames (Rasa + Microsoft Frames)
<img align="right" height="124" src="https://www.rasa.com/assets/img/sara/sara-open-source-2.0.png">

<img align="right" height="124" src="https://logos-download.com/wp-content/uploads/2016/02/Microsoft_box.png">

Decision making through conversation involves comparing items and exploring different alternatives. This requires memory.

Rasa Frames is a fork of Rasa that augments the `DialogueStateTracker` with multiple copies of slots (or Frames), each corresonding to an item of discussion.

This project is directly inspired by the [Microsoft Frames](https://www.microsoft.com/en-us/research/project/frames-dataset/) dataset.

## Getting Started

Here is a typical conversation between a user and a travel agent bot. You can see
the user and the bot going through various options before narrowing down on the final one.

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

## References
1. El Asri, L., Schulz, H., Sharma, S., Zumer, J., Harris, J., Fine, E., Mehrotra, R., & Suleman, K. (2018). Frames: a corpus for adding memory to goal-oriented dialogue systems. 207–219. https://doi.org/10.18653/v1/w17-5526
