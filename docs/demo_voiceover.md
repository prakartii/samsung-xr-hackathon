# REWIND demo — voiceover script (for a human recording)

The rendered video has no voice. The only local TTS voices (Windows SAPI: David, Hazel, Zira)
sound robotic, so the video carries on-screen captions instead. Record this over
`demo/rewind_demo.mp4` if you want narration. About 250 words at a calm pace
(~2.2 words/s) fits the 1:57 timing; timings match `demo/rewind_storyboard.md`.

| Time | Scene | Narration |
|---|---|---|
| 0:00–0:09 | Hook | "Head-mounted cameras record everything. Every wave, every step, every second. But only a few moments actually matter. This is REWIND." |
| 0:09–0:18 | Problem | "Here's over nine minutes in a real kitchen. It's continuous activity — and finding what actually happened means watching all of it." |
| 0:18–0:28 | In / out | "REWIND takes a single video file and writes one structured summary: every action it detects, with a time and a confidence, the highlight windows to keep, and a label for the whole video." |
| 0:28–0:45 | How it works | "CLIP first decides what kind of video this is. Kitchen videos go through an action detector that finds verbs and nouns in time; we pair them, score how likely each action is real, group them into events, and choose the windows that cover the most distinct actions in fifteen percent of the time. Sports footage is ranked by motion and scene signals instead." |
| 0:45–1:04 | Predictions | "These are real predictions on that kitchen video. Turn on the tap. Take the sponge. Pour washing liquid. Wash the knife. Open the bin. Each label is placed where the model put it — not edited after the fact." |
| 1:04–1:16 | Summary | "Here are the fourteen windows REWIND kept. Nine minutes twenty-one becomes one minute twenty-one — and that minute is where the actions are." |
| 1:16–1:31 | Sports | "The same system on outdoor first-person video: skiing, motocross, riding, diving, climbing, surfing — each recognised and cut down to its strongest moments." |
| 1:31–1:41 | Output | "Everything ends up in one plain JSON file per video: timestamped actions, highlight segments and a label, ready for whatever comes next." |
| 1:41–1:50 | Results | "On the official metrics: thirty-six percent action coverage, fifteen percent mAP, twenty-three percent highlight F1, and eighty-five percent label accuracy — a weighted score of thirty-two point seven." |
| 1:50–1:57 | End | "REWIND. Turn long recordings into action-aware summaries." |
