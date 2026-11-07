# Structures of Influence

**A computational study of political, economic, historical, and cultural discourse**

Structures of Influence is a reproducible corpus-linguistic research project examining the organization, recurrence, and transformation of concepts across a collection of publicly available lectures by Academic Agent. It investigates how political and economic arguments are constructed, how historical thinkers and intellectual traditions are invoked, and how recurring explanatory vocabularies are distributed across individual lectures and the corpus as a whole.

The project combines descriptive statistics, computational text analysis, unsupervised topic modeling, document similarity, and evidence-preserving argument reconstruction. Its purpose is not to assign definitive ideological classifications to individual lectures, but to develop an inspectable empirical basis for studying the relationships among concepts, argumentative structures, intellectual references, and patterns of discourse.

The central methodological distinction is between **a measurable textual relationship and a demonstrated relationship of intellectual influence**. Lexical recurrence, topic similarity, and shared terminology provide evidence of textual association. They do not, without additional contextual evidence, establish agreement, dependence, transmission, or causation.

## Corpus and analytical coverage

The current collection contains **318 distinct lectures**, drawn from material published between 2017 and 2026, comprising approximately **2.47 million words**. The corpus includes discussions of political philosophy, economic theory, institutional organization, intellectual history, elite theory, cultural criticism, public policy, and contemporary political controversies.

Each lecture is associated with a source identifier, publication date, title, and transcript. The ingestion system selects a preferred English transcript variant where multiple caption files exist, consolidates overlapping automatic-caption windows, and preserves source information for subsequent analysis.

The collection is a research sample rather than a complete or independently verified archive of the source channel. Its composition reflects the availability of downloaded material and may be affected by missing, removed, or inaccessible lectures.

## Analytical methods

### Lexical and phrase-level analysis

The lexical analysis computes word frequencies, n-gram distributions, and distinctive vocabulary across the corpus. The exported results include unigram, bigram, trigram, and four-gram tables, alongside document metadata and comparative term statistics.

These measurements provide an initial account of the corpus's vocabulary, recurring expressions, and frequently associated concepts. They are useful for identifying candidate themes and examining rhetorical regularities, but raw frequency is not equivalent to argumentative importance.

In particular, repeated introductory phrases, promotional material, transcription artifacts, and common conversational constructions may dominate frequency-based measurements without representing the principal intellectual content of a lecture.

### Passage-level topic modeling

The NLP pipeline divides normalized transcripts into traceable passages and applies TF-IDF feature extraction followed by non-negative matrix factorization (NMF).

The analysis has produced models with **12, 20, and 30 topic components**, allowing the same corpus to be examined at different levels of statistical granularity.

Each topic is represented by weighted terms and associated passages rather than a predetermined ideological or disciplinary label. The resulting components may correspond to recognizable subject areas, but they may also reflect recurring names, conversational formulas, advertisements, or other systematic features of the source material.

For example, exploratory results have identified vocabulary associated with markets and economic theory, European political history, Soviet history, elite organization, political institutions, and contemporary political figures. Other components capture promotional language, credits, and speech-related patterns.

The presence of these latter components is itself a methodological finding: **unsupervised topic models recover regularities in the supplied text, not necessarily the conceptual distinctions that a researcher intended to investigate**.

Consequently, topic interpretation requires inspection of representative passages, comparison across model resolutions, and explicit attention to boilerplate and transcription artifacts.

### Document similarity and conceptual association

Document-level TF-IDF representations and similarity calculations identify lectures sharing comparatively similar vocabularies.

These measurements can support the discovery of related lectures, recurring conceptual clusters, and potentially productive comparisons between discussions separated by publication date or subject matter.

A high similarity score does not establish that two lectures advance the same argument. Two texts may discuss the same historical figure while disagreeing about that figure's significance, or employ the same political vocabulary in opposing theoretical contexts.

Similarity therefore functions as a retrieval and comparison instrument rather than a measure of intellectual agreement.

### Concordance and source traceability

The pipeline provides concordance retrieval and passage-level references so that statistical findings can be examined against the language from which they were derived.

This is necessary because numerical summaries compress distinctions that may be decisive for interpretation, including negation, quotation, irony, attribution, qualification, and disagreement.

A term appearing in a lecture may represent the lecturer's own position, a historical source being described, a position under criticism, or a hypothetical example. The occurrence alone cannot determine which relationship applies.

The project therefore treats the original passage as the primary evidentiary unit for interpreting lexical and model-derived results.

### Model stability and reproducibility

The analytical system records configuration information, corpus-dependent identifiers, model outputs, and diagnostics. It supports comparison across topic resolutions and repeated analytical runs.

The NLP implementation includes automated tests covering ingestion, caption normalization, segmentation, caching, topic modeling, similarity calculations, and export behavior.

**Current validation: 21 passing automated tests.**

These tests establish the expected behavior of the computational implementation under its test conditions. They do not independently validate the historical accuracy of the source lectures or the substantive interpretation of the inferred topics.

## Research results

### Corpus organization

The first substantive result is the construction of a structured, source-traceable corpus from a collection of heterogeneous subtitle files.

The system distinguishes individual lectures from duplicate caption variants, reduces inflation caused by rolling subtitle windows, and retains identifiers linking analytical outputs to their source documents.

This establishes the basis for systematic comparison across hundreds of lectures without treating repeated caption fragments as independent observations.

### Recurring vocabularies and explanatory domains

The lexical and topic-modeling results identify multiple recurring domains of discussion, including economic exchange and markets, political institutions, intellectual history, elite theory, historical regimes, and contemporary cultural controversies.

These domains are not mutually exclusive. A lecture concerning economic policy may also invoke historical precedents, institutional incentives, or competing political philosophies.

The resulting overlap is analytically useful because it permits investigation of how explanatory frameworks recur across apparently distinct subjects.

However, the current models do not by themselves demonstrate that these frameworks constitute a unified ideology or a historically transmitted intellectual lineage.

### Topic resolution and interpretive granularity

The 12-, 20-, and 30-component NMF models provide alternative statistical decompositions of the same material.

Lower-resolution models tend to combine more vocabulary into broad components, while higher-resolution models permit more differentiated patterns, subject to the actual stability and quality of the fitted solutions.

There is no presumption that a larger number of topics produces a more accurate interpretation. The appropriate resolution depends on the research question, the stability of the components, and whether representative passages support coherent interpretation.

### Boilerplate and non-substantive regularities

The exploratory results demonstrate that recurring promotional expressions, credits, and speech patterns can become statistically prominent components of a lecture corpus.

This is an important limitation of naive topic interpretation. A model may identify a highly consistent pattern that is linguistically real but substantively peripheral to the research question.

The pipeline therefore distinguishes between detecting recurring text and deciding whether that recurrence should contribute to a particular analytical interpretation.

### Institutional and cultural influence

The broader research program concerns how ideas are selected, repeated, recontextualized, and incorporated into explanations of political and social life.

The present corpus supports investigations into the distribution of intellectual references and the recurrence of explanatory vocabularies. It also provides a basis for examining changes in the subjects and conceptual associations appearing across the collection.

Demonstrating influence in the stronger historical sense would require additional evidence, such as explicit attribution, chronological dependence, documentary transmission, or independently established intellectual relationships.

The project deliberately leaves that distinction open rather than treating computational similarity as proof of influence.

## Results by Topic

The corpus was analyzed using non-negative matrix factorization at three resolutions: 12, 20, and 30 components. Together, these models provide 62 statistical topic components representing alternative decompositions of the same lecture collection.

The results include representative vocabulary, dominant passage counts, prominent lecture counts, and model stability measurements where available.

The preliminary findings distinguish substantive lexical domains, including economic theory and political history, from recurring conversational and promotional material. This distinction is essential because a statistically coherent component is not necessarily a meaningful intellectual category.

The twelve-component model illustrates the importance of stability diagnostics. Its reported mean matched-topic cosine similarity is 0.916, while its minimum is 0.096, indicating substantial variation in the reproducibility of individual matched components. The adjusted Rand index of 0.682 provides a separate measure of agreement between dominant-topic partitions.

The complete empirical tables, including the 12-, 20-, and 30-component models, are available in [Results by Topic](TOPIC_RESULTS.md).

Topic assignments are descriptive measurements rather than determinations of ideological agreement, authorial intention, or historical influence.

## Analytical outputs

The repository organizes results into two principal collections.

**`corpus-results/`** contains lexical frequency tables, n-gram statistics, distinctive-term measurements, document metadata, and a browser-based lexical explorer.

**`categorization-results/`** contains versioned outputs from the NLP pipeline, including topic-term weights, document-topic assignments, passage-topic assignments, document similarity, concordances, stability measurements, manifests, and diagnostic reports.

Intermediate document caches, fitted models, feature matrices, and source transcripts are maintained separately from the public analytical outputs.

The project also supports lecture-level argument reconstruction using a locally hosted language model. This process produces interpretive summaries intended to preserve the lecturer's claims, reasoning, intellectual references, qualifications, and unresolved questions. These summaries are secondary analytical artifacts and should not be confused with the original transcripts or independently verified historical accounts.

## Exploratory interfaces

The repository includes browser-based interfaces for inspecting lexical statistics and corpus categorization.

The interfaces are designed to operate on exported analytical data without requiring a live Python server or access to the original transcript files.

The root `index.html` provides the principal research interface, while `corpus-results/index.html` provides access to the lexical explorer.

The interfaces support exploratory investigation rather than automated conclusions about ideological classification or intellectual influence.

## Reproducing the analysis

The NLP implementation, configuration, and pipeline documentation are available in [nlp/README.md](nlp/README.md).

Install the Python dependencies:

```bash
python3 -m pip install -r nlp/requirements.txt
```

Inspect the available pipeline commands:

```bash
python3 nlp/pipeline.py --help
```

Run the automated tests:

```bash
python3 -m pytest nlp/tests -q
```

The current tests complete successfully with 21 passing cases.

Source transcripts should be placed in the `transcripts/` directory using the filename convention documented by the ingestion system. The original transcripts are not redistributed in this repository. Researchers must obtain source material independently and comply with applicable copyright and platform requirements.

The pipeline maintains separate locations for source material, computational caches, and analytical exports, permitting results to be regenerated while preserving the distinction between inputs and derived measurements.

## Methodological limitations

The analysis is subject to several important constraints.

**Corpus selection.** The available lectures do not necessarily represent the complete publication history of the source channel. Missing material and uneven chronological coverage can affect observed distributions.

**Transcription accuracy.** Automatic captions may contain recognition errors, incorrect proper names, missing punctuation, duplicated text, and unreliable segmentation. Normalization can reduce some artifacts but cannot reconstruct every intended utterance.

**Statistical interpretation.** TF-IDF and NMF describe distributions of words and passages. They do not independently determine the meaning, truth, rhetorical function, or ideological status of a statement.

**Attribution and disagreement.** Mentioning an intellectual tradition does not imply endorsement. Similarity between texts does not establish agreement, and agreement does not establish historical influence.

**Model dependence.** Topic assignments depend on preprocessing decisions, segmentation, feature selection, topic count, and the fitted model. Different defensible configurations may yield different decompositions.

**Temporal interpretation.** Changes in vocabulary across publication dates may reflect changes in subject matter, available events, or corpus composition rather than changes in the lecturer's underlying commitments.

**Inferential scope.** The computational outputs establish patterns within the analyzed corpus. Broader claims about political institutions, cultural transmission, or historical causation require additional evidence beyond those patterns.

## Research orientation

Structures of Influence is organized around a distinction between **textual recurrence, conceptual association, argumentative dependence, and historical influence**.

These relationships are connected but not interchangeable. Repeated vocabulary may indicate a shared subject without a shared interpretation. Shared interpretations may arise independently. An explicit intellectual reference may establish attribution without establishing agreement.

The project's longer-term objective is to make these distinctions empirically investigable through reproducible computation, source-level inspection, and explicit acknowledgment of the inferential boundaries separating statistical observation from historical explanation.

The resulting framework is intended to support critical research into political and economic discourse without substituting automated classification for argument analysis.
