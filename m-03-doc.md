# M03 Experimental Methods

Can intermediate-layer evidence improve complete-label GoEmotions decoding, and can we predict per comment whether to amplify the late change, recover an earlier prediction, or keep the final ranking?

The paper tests decoder-assisted emotion classification with frozen models. The logit lens
and tuned lens supply complete-label trajectories. The Jacobian lens supplies semantic
features that may help the gate choose an action.

## Shared setup

- Use the 28 single-label GoEmotions classes, including neutral, and keep the same complete label paths across decoder comparisons.
- Use final-layer complete-path scoring as the matched baseline. Report strict accuracy, macro-F1, and the counts of wrong-to-correct and correct-to-wrong changes.
- Select layers and contrast strengths on development data. Evaluate the selected rules on untouched data. The gate may use model readouts and comment features at inference time.
- Group duplicate comments when partitioning development data and audit overlap with the
  confirmation set. If previous EXP2 test outcomes informed M03 choices, treat that test as
  exploratory for M03 and confirm on a genuinely untouched reserve or an independent set.
- Start with the pinned EXP2 `google/gemma-2-9b` checkpoint. Verify the public tuned and
  Jacobian lens artifacts against that checkpoint and record their revisions, fit corpora,
  and layer conventions. Any task-specific lens fitting uses training text. The tuned lens
  learns from the frozen model's own token distributions, with candidate prefixes chosen
  independently of the annotations. Keep labels for training the gate and measuring outcomes.
  [Gemma tuned lens](https://huggingface.co/chutommy/tuned-lenses) · [Gemma Jacobian lens](https://huggingface.co/neuronpedia/jacobian-lens/tree/main/gemma-2-9b/jlens/Salesforce-wikitext)

## Scoring and lens checks

- Replay each canonical label and its declared stop event from the exact EXP2 prompt token
  IDs. Normalize token scores over the vocabulary. Verify that all 28 paths decode to the
  intended labels, including multi-token labels and labels with shared prefixes.
- Match the final-block logit-lens readout to the model's native logits on identical prefixes,
  including Gemma's final normalization and soft-capping. On development prompts, check the
  tuned lens's divergence from native next-token predictions at the positions used for
  complete-label scoring. Use raw logit-lens divergence as the reference.
- Check a small set of Jacobian-lens appraisal contrasts on development comments. Use the
  same anchors and token positions with a raw logit-lens readout. Include anchor-word counts
  and inspect examples where the anchors are absent from the comment text.
  [U-Space emotion readout](https://arxiv.org/html/2610.09087v1#A4.SS1)

## Planned experiments

1. **Fixed-sign pilot.** Sweep positive and negative contrast strengths with the logit and
   tuned lenses, including zero and the best fixed intermediate-layer readout. Measure
   which comments each direction repairs or damages. On development data, use the
   annotation to compute the maximum achievable accuracy for the available actions. The
   gap above the best fixed rule estimates the opportunity for per-comment selection.
2. **Per-comment action selection.** Compare the three-action gate with the best fixed
   settings, the adaptive positive-only Sharma adaptation, and a gate whose inputs are
   limited to final scores. Train the action selector on paired correctness outcomes for
   the available actions, with “keep final” as the default when estimated benefit is zero.
   Include two gates restricted to one direction plus “keep final,” trained with the same
   features and selection budget. Compare all methods on identical complete paths.
   Include raw final scores and final scores with regularized class offsets fitted on
   development data to control for gains from changing class priors such as `neutral`.
   A regularized score combiner using the same features and training rows tests the value
   of explicit action selection over direct score fusion.
   Score the original greedy run for context and report its invalid-output rate.
   [Sharma et al.](https://aclanthology.org/2024.emnlp-main.1161/)
3. **Lens ablations.** Compare the full gate with each lens feature group removed in turn,
   using the same gate architecture, training rows, and search budget. Keep the available
   decoder actions and their scoring readout fixed for this comparison. Compare Jacobian
   features with the same number of scores from the same anchors read through the raw
   logit lens. Run logit-only and tuned-only scoring separately in the fixed-sign sweep to
   measure how the choice of scoring readout affects repairs.
4. **Error and prompt controls.** Measure repairs, damages, action coverage, and net gain
   by sign. Inspect `neutral` errors and a few fine-grained confusions chosen on development
   data. Check sensitivity to stop-token scores and label token length. On a prespecified
   subset, rerun the full gate and final-score baseline with a fixed reordering of the
   prompt's label menu. Use the same reordered prompt for both methods.
5. **Replication on a second model family.** Repeat the decisive comparisons on one pinned
   EXP2 checkpoint, `allenai/OLMo-2-1124-7B-Instruct` or `swiss-ai/Apertus-8B-Instruct-2509`.
   The minimum comparison is calibrated final scores, the adaptive positive-only baseline,
   both one-direction gates, the full gate, and the gate without Jacobian features. Use
   checkpoint-specific lenses and apply the same development and evaluation protocol.

## Confirmatory evaluation

- Freeze the actions, layers, anchors, comparison methods, and primary contrasts before
  opening the confirmation set. The main comparisons are full M03 against the strongest
  prespecified baseline chosen on development data, against each one-direction gate, and
  against the gate without Jacobian features. Use development results to select configurations
  within each baseline. Report the adaptive Sharma baseline and calibrated final-only scores
  in the main results regardless of how they rank on development data.
- Report strict accuracy and macro-F1 across the full confirmation set with paired 95%
  confidence intervals. Resample comments in duplicate-text groups and account for the
  planned primary comparisons. Include per-class support, repair and damage counts, and
  the additional inference time per comment. Record offline lens-fitting and gate-training
  costs separately from inference time.
- Evidence for bidirectional routing comes from beating both one-direction gates and
  showing which errors each direction repairs. Evidence for a three-lens gain comes from the
  ablations, especially the improvement from adding Jacobian features to the logit-plus-tuned
  gate. If the evidence supports only one direction or two lenses, scope the paper claim to
  that result.

A claim that a specific internal component causes the loss of evidence would need a causal
intervention and a native-generation rerun. A claim about transfer beyond GoEmotions would
need a second dataset.

## Local working notes

- [M03 proposal](docs/lens-methods.md#m03-learn-when-to-extrapolate-late-progress-and-when-to-recover-earlier-evidence)
- [Complete-label extrapolation design](docs/notes-7.md#3-approach-2--sharma-et-als-layer-extrapolation-over-complete-labels)
