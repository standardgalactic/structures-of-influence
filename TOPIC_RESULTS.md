# Results by Topic

**Analytical run:** `2425069e-seg-1-873ab5-c9b2a5`

The following tables reproduce the topic identifiers, highest-weighted terms, dominant passage counts, and prominent lecture counts recorded by the NLP pipeline.

Topic labels are not inferred from the vocabulary. An empty label in the source export remains unlabeled.

The lecture counts use the pipeline's existing `n_lectures_prominent` definition. They do not represent ideological agreement or endorsement.

## 12-component model

**Model:** `nmf12-2425069e-seg-1-873ab5-776199`

| Topic | Representative terms | Dominant passages | Prominent lectures |
|---|---|---:|---:|
| T00 | know,  uh,  um,  kind,  like,  okay,  don,  ve | 445 | 154 |
| T01 | super,  hours,  thousands,  courses,  sure,  buy,  pick,  academic agency | 148 | 88 |
| T02 | price,  market,  prices,  money,  demand,  cost,  costs,  goods | 134 | 76 |
| T03 | europe,  civilization,  west,  european,  french,  german,  western,  east | 160 | 52 |
| T04 | stalin,  soviet,  soviet union,  union,  uh,  communist,  party,  um | 111 | 38 |
| T05 | elite,  elite theory,  class,  ruling,  theory,  ruling class,  power,  uh | 172 | 82 |
| T06 | percent,  million,  population,  government,  rate,  workers,  dollars,  wages | 127 | 98 |
| T07 | promo,  promo code,  code,  oof,  25,  courses,  brutal,  25 courses | 79 | 48 |
| T08 | blair,  tony,  tony blair,  trump,  labour,  government,  right,  uh | 170 | 92 |
| T09 | michael,  special thanks,  crimson,  tragic vision,  spatula,  tragic,  holy spatula,  holy | 43 | 43 |
| T10 | vince,  wrestling,  wwf,  monsoon,  tv,  shows,  prime time,  allstar | 49 | 12 |
| T11 | liberal,  liberalism,  human,  freedom,  state,  society,  man,  moral | 305 | 144 |

### Model stability

| Comparison | Metric | Value |
|---|---|---:|
| repeat_fit | mean_matched_topic_cosine | 0.915916 |
| repeat_fit | min_matched_topic_cosine | 0.095524 |
| repeat_fit | adjusted_rand_index | 0.682016 |
| repeat_fit | adjusted_mutual_info | 0.758467 |
| repeat_fit | mean_matched_topic_cosine | 0.999999 |
| repeat_fit | min_matched_topic_cosine | 0.999993 |
| repeat_fit | adjusted_rand_index | 0.692144 |
| repeat_fit | adjusted_mutual_info | 0.787440 |
| repeat_fit | mean_matched_topic_cosine | 0.999999 |
| repeat_fit | min_matched_topic_cosine | 0.999994 |
| repeat_fit | adjusted_rand_index | 0.745154 |
| repeat_fit | adjusted_mutual_info | 0.815309 |
| topic_count | coverage_a_in_b | 0.951351 |
| topic_count | coverage_b_in_a | 0.796974 |
| topic_count | co_assignment_a_in_b | 0.691751 |
| topic_count | co_assignment_b_in_a | 0.791095 |
| topic_count | adjusted_rand_index | 0.485365 |
| topic_count | adjusted_mutual_info | 0.649089 |
| topic_count | coverage_a_in_b | 0.930534 |
| topic_count | coverage_b_in_a | 0.670602 |
| topic_count | co_assignment_a_in_b | 0.558936 |
| topic_count | co_assignment_b_in_a | 0.731587 |
| topic_count | adjusted_rand_index | 0.307041 |
| topic_count | adjusted_mutual_info | 0.536602 |
| alternative_representation | spearman_pairwise_similarity | 0.567251 |

These stability statistics describe comparisons between fitted models or partitions. They should not be interpreted as individual topic confidence scores unless the metric explicitly supports that interpretation.

## 20-component model

**Model:** `nmf20-2425069e-seg-1-873ab5-776199`

| Topic | Representative terms | Dominant passages | Prominent lectures |
|---|---|---:|---:|
| T00 | know,  okay,  right,  uh,  people,  don,  trump,  um | 164 | 72 |
| T01 | super,  hours,  thousands,  courses,  sure,  buy,  pick,  academic agency | 144 | 70 |
| T02 | price,  market,  prices,  money,  demand,  cost,  costs,  goods | 125 | 63 |
| T03 | europe,  civilization,  west,  western,  french,  german,  spirit,  european | 129 | 28 |
| T04 | stalin,  soviet,  soviet union,  union,  communist,  party,  uh,  trosky | 85 | 18 |
| T05 | elite,  class,  ruling,  elite theory,  ruling class,  theory,  power,  elites | 112 | 41 |
| T06 | book,  uh,  books,  um,  read,  kind,  know,  ve | 96 | 51 |
| T07 | percent,  wages,  rate,  million,  workers,  unemployment,  increase,  economy | 69 | 59 |
| T08 | blair,  tony,  tony blair,  labour,  digital,  id,  prime,  government | 87 | 34 |
| T09 | michael,  special thanks,  crimson,  tragic vision,  spatula,  tragic,  holy spatula,  holy | 42 | 41 |
| T10 | vince,  wrestling,  wwf,  monsoon,  tv,  prime time,  shows,  allstar | 36 | 5 |
| T11 | empire,  british,  british empire,  britain,  war,  imperial,  american,  hilton | 84 | 40 |
| T12 | content,  haiti,  like content,  video subscribe,  buy mug,  channel donating,  consider joining,  content maybe | 41 | 18 |
| T13 | liberalism,  liberal,  state,  human,  nature,  freedom,  political,  free | 156 | 84 |
| T14 | oof,  brutal,  oof brutal,  brutal oof,  brutal brutal,  oof oof,  brutal buy,  buy brutal | 30 | 16 |
| T15 | promo code,  promo,  scorcher,  code,  scorcher scorcher,  code scorcher,  25,  sale | 45 | 33 |
| T16 | rap,  rock,  hip,  hip hop,  hop,  music,  record,  doug | 31 | 9 |
| T17 | women,  men,  sexual,  sex,  masculine,  weininger,  okay,  female | 74 | 31 |
| T18 | know,  like,  kind,  uh,  yeah,  um,  don,  oh | 358 | 110 |
| T19 | liberia,  liberian,  liberians,  tribal,  rubber,  iran,  shah,  firestone | 35 | 28 |

### Model stability

| Comparison | Metric | Value |
|---|---|---:|
| repeat_fit | mean_matched_topic_cosine | 0.908390 |
| repeat_fit | min_matched_topic_cosine | 0.222431 |
| repeat_fit | adjusted_rand_index | 0.762118 |
| repeat_fit | adjusted_mutual_info | 0.798960 |
| repeat_fit | mean_matched_topic_cosine | 0.906009 |
| repeat_fit | min_matched_topic_cosine | 0.067696 |
| repeat_fit | adjusted_rand_index | 0.575871 |
| repeat_fit | adjusted_mutual_info | 0.726290 |
| repeat_fit | mean_matched_topic_cosine | 0.957628 |
| repeat_fit | min_matched_topic_cosine | 0.456661 |
| repeat_fit | adjusted_rand_index | 0.692239 |
| repeat_fit | adjusted_mutual_info | 0.773145 |
| topic_count | coverage_a_in_b | 0.959961 |
| topic_count | coverage_b_in_a | 0.820359 |
| topic_count | co_assignment_a_in_b | 0.696916 |
| topic_count | co_assignment_b_in_a | 0.806719 |
| topic_count | adjusted_rand_index | 0.454744 |
| topic_count | adjusted_mutual_info | 0.670656 |
| alternative_representation | spearman_pairwise_similarity | 0.535140 |

These stability statistics describe comparisons between fitted models or partitions. They should not be interpreted as individual topic confidence scores unless the metric explicitly supports that interpretation.

## 30-component model

**Model:** `nmf30-2425069e-seg-1-873ab5-776199`

| Topic | Representative terms | Dominant passages | Prominent lectures |
|---|---|---:|---:|
| T00 | know,  okay,  uh,  um,  kind,  like,  don,  things | 99 | 49 |
| T01 | super,  hours,  thousands,  courses,  sure,  buy,  pick,  academic agency | 145 | 65 |
| T02 | price,  prices,  market,  demand,  cost,  costs,  supply,  goods | 79 | 45 |
| T03 | europe,  civilization,  west,  western,  french,  german,  spirit,  european | 119 | 10 |
| T04 | stalin,  soviet,  soviet union,  union,  communist,  party,  uh,  trosky | 70 | 11 |
| T05 | elite,  elite theory,  ruling,  class,  ruling class,  theory,  mosca,  pareto | 62 | 23 |
| T06 | book,  books,  uh,  read,  um,  reading,  book uh,  ve | 44 | 21 |
| T07 | oof,  brutal,  oof brutal,  brutal oof,  brutal brutal,  oof oof,  brutal buy,  buy brutal | 14 | 7 |
| T08 | blair,  tony,  tony blair,  labour,  digital,  id,  prime,  digital id | 74 | 25 |
| T09 | michael,  special thanks,  crimson,  tragic vision,  spatula,  holy spatula,  tragic,  holy | 39 | 38 |
| T10 | vince,  wrestling,  wwf,  monsoon,  tv,  prime time,  shows,  allstar | 30 | 4 |
| T11 | empire,  british,  british empire,  britain,  imperial,  war,  american,  hilton | 67 | 31 |
| T12 | haiti,  content,  like content,  video subscribe,  buy mug,  channel donating,  consider joining,  content maybe | 32 | 15 |
| T13 | trump,  israel,  tucker,  donald,  right,  donald trump,  shapiro,  know | 87 | 33 |
| T14 | media,  social media,  social,  television,  spectacle,  twitter,  internet,  mass | 88 | 23 |
| T15 | promo code,  promo,  scorcher,  code,  scorcher scorcher,  code scorcher,  25,  sale | 36 | 23 |
| T16 | cia,  student,  saunders,  wilford,  operations,  america,  civil rights,  involved | 43 | 10 |
| T17 | women,  men,  sexual,  sex,  masculine,  weininger,  female,  feminine | 61 | 16 |
| T18 | rap,  rock,  hip hop,  hip,  hop,  doug,  doug morris,  morris | 29 | 7 |
| T19 | liberia,  liberian,  liberians,  tribal,  rubber,  firestone,  americo,  president | 10 | 4 |
| T20 | yeah,  oh,  london,  mean,  um,  look,  footage,  know | 63 | 30 |
| T21 | filmer,  hobbes,  locke,  authority,  monarchy,  state nature,  king,  nature | 39 | 14 |
| T22 | free,  liberal,  human,  freedom,  classical,  liberalism,  moral,  liberty | 136 | 61 |
| T23 | boomer,  daria,  truth,  boomer truth,  gen,  truth regime,  millennial,  medicine | 64 | 31 |
| T24 | percent,  wages,  rate,  workers,  million,  unemployment,  increase,  economy | 98 | 66 |
| T25 | bowie,  kind,  fisher,  disney,  interesting,  nostalgia,  know,  like | 104 | 21 |
| T26 | money,  banks,  managerial,  firms,  investment,  bank,  company,  investment banks | 61 | 29 |
| T27 | shah,  iran,  iranian,  peasants,  clergy,  peasantry,  reforms,  land | 27 | 12 |
| T28 | enemy,  friend enemy,  enemy distinction,  friend,  schmidt,  lindsay,  distinction,  james lindsay | 43 | 12 |
| T29 | revolutionary,  lenin,  organization,  party,  power,  revolution,  counter,  tactics | 80 | 30 |

### Model stability

| Comparison | Metric | Value |
|---|---|---:|
| repeat_fit | mean_matched_topic_cosine | 0.790308 |
| repeat_fit | min_matched_topic_cosine | 0.078456 |
| repeat_fit | adjusted_rand_index | 0.467074 |
| repeat_fit | adjusted_mutual_info | 0.654836 |
| repeat_fit | mean_matched_topic_cosine | 0.810426 |
| repeat_fit | min_matched_topic_cosine | 0.053555 |
| repeat_fit | adjusted_rand_index | 0.498751 |
| repeat_fit | adjusted_mutual_info | 0.687994 |
| repeat_fit | mean_matched_topic_cosine | 0.817124 |
| repeat_fit | min_matched_topic_cosine | 0.088002 |
| repeat_fit | adjusted_rand_index | 0.458121 |
| repeat_fit | adjusted_mutual_info | 0.644972 |
| alternative_representation | spearman_pairwise_similarity | 0.480411 |

These stability statistics describe comparisons between fitted models or partitions. They should not be interpreted as individual topic confidence scores unless the metric explicitly supports that interpretation.

## Interpretive limitations

The topics are unsupervised statistical components, not verified intellectual categories. Some components may capture promotional material, conversational expressions, or transcription artifacts.

Prominent lecture counts can overlap because a lecture may be associated with several components. These counts should not be summed to estimate the number of distinct lectures.

The representative terms describe the fitted model but do not establish endorsement, historical influence, or causal relationships.
