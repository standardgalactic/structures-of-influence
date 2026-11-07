Structures of Influence: categorization pipeline

This directory implements Phases I to III of the technical specification (version 0.1, 7 November 2026). It sits beside the existing scripts in ~/academic-agent and does not modify summarize.py, corpus_analysis.py, corpus-results or corpus-visualizer.html. The new dashboard is a separate file, categorization-visualizer.html, in the same green-on-black style.

Requirements are Python 3 with numpy, scipy, scikit-learn and joblib (see requirements.txt). Nothing needs a network connection, Ollama or a GPU.

Running it

From ~/academic-agent, each stage can be run on its own and resumes from cached artifacts in corpus-cache/. A full run is `python3 nlp/pipeline.py run-all`. For a limited trial use `--limit 20` or `--video-ids ID ...`, and `--dry-run` reports planned work without writing. The individual stages are ingest, segment, fit-nmf --topics 12 20 30, classify-frozen --model-id ID, similarities --method tfidf, stability, concordances and export. The search command, for example `python3 nlp/pipeline.py search "iron law" --video VIDEO_ID`, reports occurrences and distinct lectures separately and prints timestamped YouTube links. Open categorization-visualizer.html in a browser and select the folder categorization-results/RUN_ID.

What each stage does

Ingestion identifies each video by its eleven-character ID, prefers en-orig over en, hashes the source and the normalized transcript, and records a diagnostic instead of stopping when a file is malformed. Normalization removes timestamps, markup, sound tags and rolling-caption overlap at the token level. An overlap is removed only when it is at least min_overlap_tokens long (default 3), when it is the second line of a two-line rolling cue, or when it is a near-zero-length duplicate cue, so genuine repetitions such as "no no no" are kept. Segmentation cuts each lecture into passages of roughly 500 to 800 words at sentence ends where punctuation exists, with no overlap between passages, and keeps start and end times. Passage identifiers change when the transcript or the segmentation configuration changes.

Boilerplate is flagged by cross-document near-duplicate detection on word shingles, plus optional regular-expression rules in the configuration. Flags are whole-passage only, are recorded with their reason and fraction in passages.csv, and can be reversed by changing min_fraction or disabling the check. Short promotional segments inside a longer passage stay in the text.

NMF fits TF-IDF (1 to 2 grams, min_df 3, max_df 0.85, sublinear) on eligible passages with a fixed seed and deterministic nndsvd initialization, at 12, 20 and 30 topics as separate models. Topics carry neutral IDs T00, T01 and so on. Lecture-level proportions are word-count-weighted averages of row-normalized passage weights, and passages with zero total weight are marked unassigned and leave the denominator. Representative passages are ranked by raw NMF loading, at most two per lecture. A frozen classification applies an existing fitted vectorizer and model to new passages under a different model identifier, leaving the original model file untouched.

Similarity is TF-IDF cosine between lectures, labelled tfidf_cosine in every output. Stability compares the primary fit with random-initialization refits using Hungarian-matched topic cosine, adjusted Rand index and adjusted mutual information, compares different topic counts using coverage and co-assignment rather than one-to-one matching, and correlates lecture-pair similarity from NMF profiles with TF-IDF similarity.

Export writes categorization-results/RUN_ID with every file named in the specification, plus document_terms.csv, which the dashboard uses to show contributing terms. RUN_ID combines the corpus hash, the segmentation version and a configuration hash, so runs on different corpus snapshots do not overwrite one another. The manifest is written last, so its presence marks a completed run. A missing optional artifact is reported as unavailable in report.json and in the dashboard rather than filled in.

Investigator labels

Model output never carries labels. In the dashboard you can type a label for a topic, which is stored in that browser only, and export labels as topic_labels.csv. Place that file in categorization-results/ and rerun export, and topics.csv will record the labels with label_source set to investigator.

Not yet implemented

Sentence-embedding similarity, hierarchical clustering and BERTopic (Phase IV), and the corpus-growth experiments, entity recurrence, collocation, keyness and frame classification (Phase V) are not built. report.json lists embedding similarity and BERTopic as unavailable. The specification's own priority note says to finish Phases I to III first.

Tests

`python3 -m unittest nlp.tests.test_pipeline` from ~/academic-agent runs 21 tests covering the twelve cases in section 17 of the specification, using synthetic rolling-caption VTT files generated in nlp/tests/fixtures.py. The synthetic text is invented and exists only to exercise the software.
