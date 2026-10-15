# Multimodal research interface

Team: Muhammad Danial, Hamza Rasheed, Muhammad Fareed.

Launch dashboard/research_app.py. The original dashboard/app.py remains available. The new entry point groups research workflows and evidence pages; it does not replace a predictive model.

The project theme uses a navy navigation panel, light workspace, teal controls and audio/visual/behavior motifs. The installer changes only the main theme section of .streamlit/config.toml, preserving other sections and backing up the original configuration. Restart Streamlit after installation.

The report assistant extracts uniquely matching model, score, threshold and provenance fields from the exported report format. It answers those specific questions directly in English or Roman Urdu with exact source text and PDF page references. General questions retrieve document passages. This is document retrieval and field extraction, not a generative language model. Arbitrary layouts, OCR-free scanned PDFs and synonyms beyond supported mappings can still fail; a passage match is not verification of a document's authenticity.

Report comparison shows extracted fields side by side. A score delta is calculated only for single-score reports whose task, model name, model SHA-256 and frozen threshold match in document text. The delta is in percentage points. Document compatibility is not independent artifact/input verification, clinical improvement, a longitudinal patient assessment or evidence of diagnosis. AV-ASD multi-label scores are never collapsed into an overall autism score. Different task/model versions receive a no-comparison notice.

Browser review: open Research Home; navigate to video and feature analysis; open Reports and assistant; upload the demonstration PDF; select Roman Urdu; ask Ye kia ha and Model score kitna hai; inspect exact page references; ask a missing topic; compare two copies of the demonstration PDF (zero delta); replace one with a different model/task report (no delta); confirm changing the chat PDF clears the conversation.

No remote API is used. Uploaded documents remain in the active Streamlit session; they are not written to project storage. Research conclusions and original test predictions are preserved.
