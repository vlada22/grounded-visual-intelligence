# Article 1 evaluation plan

The evaluation is deliberately small and inspectable. It is intended to support
an applied technical article, not to claim a general benchmark.

## Systems

### Direct multimodal baseline

The model receives uniformly sampled frames and the question. It returns an
answer without access to the Visual Evidence Graph.

### Grounded system

The language model receives the question and provider-neutral tool definitions.
It can query tracks, time ranges, zones, and evidence references. Numeric and
temporal facts are returned by deterministic code.

## Question taxonomy

| Category | Example | Primary score |
| --- | --- | --- |
| Presence | Was a red car visible after 8 seconds? | Exact match |
| Unique count | How many cups appeared? | Absolute error |
| First/last seen | When did the car first appear? | Timestamp error |
| Ordering | Did the car enter before the person? | Exact match |
| Zone visit | Did object 7 enter the marked area? | Event F1 |
| Dwell | How long was the vehicle in the zone? | Duration error |
| Co-occurrence | When were both vehicles visible? | Temporal IoU |
| Evidence | Which frames support the answer? | Evidence precision/recall |

Start with 20-30 questions across two prepared videos. Each question and answer
must be manually verified and stored as a small versioned JSONL dataset.

## Reported measurements

- answer correctness by category;
- numeric mean absolute error;
- median and 95th-percentile timestamp error;
- unsupported-claim rate;
- evidence precision and recall;
- end-to-end latency;
- language-model token use.

## Failure labels

Each incorrect answer should be assigned one primary cause:

- perception miss;
- incorrect track identity;
- tool-selection error;
- unsupported tool capability;
- ambiguous question;
- language-generation error;
- invalid ground truth.

The failure breakdown is more valuable than a single aggregate score because it
shows whether a problem belongs to perception, measurement, orchestration, or
language generation.

