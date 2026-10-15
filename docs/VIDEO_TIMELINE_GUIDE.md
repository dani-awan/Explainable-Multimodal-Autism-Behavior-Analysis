# Exploratory video behavior timeline

Team: Muhammad Danial, Hamza Rasheed, Muhammad Fareed.

Launch START_RESEARCH_WORKSPACE.cmd to keep process temporary files and Numba/Matplotlib caches under runtime_cache on F:, then open Video timeline. The settings apply to that process, not the whole Windows system. Upload an MP4 with audio, up to 120 seconds, 7200 frames and 200 MiB. Choose a requested window of 5, 10 or 15 seconds. A final remainder shorter than 2 seconds merges into the preceding window. A video shorter than the requested window has one window.

Each window uses the existing 44 audio and 14 visual feature definitions and the trusted frozen AV-ASD combined Logistic Regression artifact at threshold 0.30. Visual features sample source frames at nominal one-second intervals within the window. Audio is trimmed to a temporary PCM WAV before the existing MFCC/RMS/ZCR extraction. A whole-video window uses the source audio directly. This path does not alter the original feature extractor, model or dataset predictions. Unsupported seeking or missing audio stops the request rather than filling invented scores.

The heatmap shows ten independent label scores across windows, on a fixed 0 to 1 scale. Select a segment on the map or use the selector. Video playback starts and ends at the selected nominal interval. Browser seeking can be approximate. The page shows per-window quality and exact linear fitted-score contributions. Exports include the full timeline CSV/JSON and a selected-window PDF. Uploaded media and audio scratch files are created under results/video_timeline_temp on F: and removed after processing; the active Streamlit session retains the upload in memory.

This is an exploratory application of a clip-trained model to fixed windows. Segment-level accuracy, event localization, new-video generalization and clinical utility are not established. A timestamp is a window boundary, not a verified behavior onset. Do not present a highlighted cell as confirmed eye-contact avoidance, a detected clinical symptom, autism diagnosis or severity.

Run scripts/video_timeline/validate_timeline.py --run on Windows. It creates a synthetic two-scene video on F:, checks scene-to-window mapping and 58 finite features, compares a whole-window extraction against the existing extractor, then checks 20 label outputs and their logit explanations with the verified frozen model. The model hash is checked before and after. These are software/numerical checks, not a test of behavioral recognition accuracy. No official test predictions are regenerated.

The move script scans Downloads/Desktop/Documents (including local OneDrive Desktop/Documents) for clearly named project files or files within clearly named project folders on C:. It skips reparse points, offline placeholders, app environments and development dependency directories. Every file is copied into dataset/transferred_from_C on F:, checksum-verified and checked again before its C: source is removed. Failures preserve the source. Unclear/generic names remain in place. It does not prove that all autism-related files anywhere on C: were discovered. Logs remain with the transferred files.

UI API references: https://docs.streamlit.io/develop/api-reference/charts/st.altair_chart ; https://docs.streamlit.io/develop/api-reference/media/st.video ; https://altair-viz.github.io/user_guide/marks/rect.html .
