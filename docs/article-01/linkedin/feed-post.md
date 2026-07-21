When a vision-language model says an event happened “around four seconds,” how do
we know where that answer came from?

I built **Grounded Visual Intelligence** to explore a stricter contract for video
systems:

Perception models observe. Deterministic tools measure. The LLM plans and
explains—with citations.

In the first experiment, SAM 3 and Grounding DINO + SAM 2 processed the same
ten-second cup-crossing scene. Both produced one complete 40-frame track and the
same measured transition: the cup was last observed in zone A at **3.75 s** and
first observed in zone B at **4.00 s**.

Their masks reached **0.9694 mean cross-model IoU**. That is strong agreement, not
ground-truth accuracy—and that distinction matters.

The article covers the evidence graph, deterministic query tools, model-boundary
design, confidence semantics, reproducibility, and the limitations of this first
controlled run.

Live explorer: https://vlada22.github.io/grounded-visual-intelligence/

Code and notebooks: https://github.com/vlada22/grounded-visual-intelligence

Where would you draw the boundary between perception, measurement, and language
reasoning in a production visual system?

#ComputerVision #MultimodalAI #MachineLearning
