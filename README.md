SmoothVision: motion smoothing for YouTube (free, runs on your own GPU)

Install
  1. Unzip this folder somewhere permanent.
  2. Open chrome://extensions, switch on Developer mode, click "Load unpacked"
     and pick the smoothvision folder.
  3. Open a YouTube video and click the SmoothVision icon.
  Try "Open test page" in the popup first: it plays a 30 fps clip through the same
  engine, so you can check it works without involving YouTube.

What it does
  YouTube video at 24/25/30 fps is shown through a WebGL2 canvas that adds in-between
  frames to match your monitor's refresh rate (60 Hz = 30 -> 60 fps).
  Nothing leaves your computer: no servers, no accounts, no remote code.

Modes
  Off          Plays normally.
  Duplicate    Repeats frames. Same look as normal playback; useful as a baseline.
  Blend        Cross-fades neighbouring frames. Cheapest, but fast motion looks ghosty.
  Optical flow Estimates motion per block and builds each in-between frame by moving
               pixels along it (the same idea as TV motion smoothing). Falls back to a
               blend wherever the motion estimate is unreliable, and holds the old frame
               on scene cuts.

Quality (optical flow)
  Low / Balanced / High set the size of the motion-search image (320 / 480 / 640 px wide)
  and how far it looks for motion. Higher costs more GPU time.

Honest numbers
  The popup shows the source frame rate and the frame rate actually shown, both measured.
  "Missed refreshes" is how often the GPU failed to deliver a frame on time. If it stays
  high the extension steps down one level by itself (turn this off in the popup).

Good to know
  - Output runs about one source frame behind the video (about 33 ms at 30 fps). That is
    the cost of interpolating between two known frames; TVs do the same.
  - Videos already at 50 fps or more are left alone, since there is nothing to add.
  - Above 1080p, DRM-protected video and 4K are skipped. Pick 1080p or lower in YouTube.
  - Alt+S switches smoothing on and off while watching, handy for comparing.
  - Older GPUs: keep YouTube on H.264 (for example with h264ify) so decoding stays on
    the GPU, and use Low or Balanced.

Troubleshooting
  - Popup says WebGL2 isn't available: open chrome://gpu and check that WebGL2 is
    "Hardware accelerated", and that Chrome's hardware acceleration setting is on.
  - Popup shows an orange note about a shader: Blend is used instead. Please report the
    message shown there.
  - Reload the YouTube tab after installing or updating the extension.
