# V3GAN
This repo contains official code for [V3GAN: Decomposing Background, Foreground and Motion for Video Generation](https://www.bmvc2021-virtualconference.com/assets/papers/1171.pdf). The paper has been published at BMVC 2021. You can find the conference presentation [here](https://papertalk.org/papertalks/34351).
### Abstract
Video generation is a challenging task that requires modeling plausible spatial and
temporal dynamics in a video. Inspired by how humans perceive a video by grouping a
scene into moving and stationary components, we propose a method that decomposes the
task of video generation into the synthesis of foreground, background and motion. Foreground
and background together describe the appearance, whereas motion specifies how
the foreground moves in a video over time. We propose V3GAN, a novel three-branch
generative adversarial network where two branches model foreground and background
information, while the third branch models the temporal information without any supervision.
The foreground branch is augmented with our novel feature-level masking layer
that aids in learning an accurate mask for foreground and background separation. To
encourage motion consistency, we further propose a shuffling loss for the video discriminator.
Extensive quantitative and qualitative analysis on synthetic as well as real-world
benchmark datasets demonstrates that V3GAN outperforms the state-of-the-art methods
by a significant margin.

### Set-up
Write the dataset path and save path in cfg.py. Data folder should be in the following format:
Data
    |video1
          |frame1.jpg
          |frame2.jpg
          |frame3.jpg
    |video2
          |frame1.jpg
          |........
          
          
### Training
Run the following command in order to train the model:
'python train.py'

### Testing
We have followed similar strategy as G3AN for testing our model. Kindly, refer [G3AN repo](https://github.com/wyhsirius/g3an-project) for more info.

### Acknowlegment
We want to thanks G3AN authors for releasing their code, this repo borrowed 
