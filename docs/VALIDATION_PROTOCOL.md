# Clinical validation protocol: dental image screening model

Roadmap item 10 in [STRATEGY.md](STRATEGY.md). This is a **study design and regulatory plan**. It isn't a
validation, and no clearance has been sought. Until a study like this is run and an authority clears the
device, DentAgent must present imaging output as *screening support reviewed by a clinician*, never as a
diagnosis. That's how the app works today: findings are proposals a clinician approves.

## 1. Device and intended use

| | |
|---|---|
| Software | DentAgent imaging agent (`agents/imaging_agent.py`, `vision/pipeline.py`) |
| Stage 1 | EfficientNetV2-S "dental image?" gate. P(dental) ≥ 0.08 passes. |
| Stage 2 | YOLO detector, 8 classes: early caries, advanced caries, gingivitis, hypodontia (shown as "Edentulism"), plaque, discoloration, ulcer, abnormal. Confidence threshold 0.10. |
| Intended use (proposed) | Computer-aided **detection** (CADe) on intraoral photographs and intraoral-camera frames. It flags regions for a licensed dentist to review in an outpatient dental setting. The dentist makes the diagnosis. |
| Users | Dentists and dental hygienists |
| Not intended for | Radiographs (not trained on them), autonomous diagnosis, treatment planning, or patients under 12 (not in the training data) |

Locking the intended use matters: the regulatory class, the study endpoints and the claims in the UI all
follow from it.

## 2. Hypotheses and endpoints

**Primary endpoint:** sensitivity and specificity per class, at the patient-image level, against the
reference standard. Run at the deployed operating point (detector confidence 0.10, gate 0.08).

| Class | Minimum sensitivity (lower 95% CI bound) | Minimum specificity (lower 95% CI bound) |
|---|---|---|
| Advanced caries | 0.85 | 0.80 |
| Early caries | 0.70 | 0.80 |
| Gingivitis | 0.75 | 0.75 |
| Ulcer | 0.80 | 0.90 |
| Plaque, discoloration, hypodontia, abnormal | reported, no pass/fail threshold (secondary) | |

The thresholds are proposals to agree with clinical advisors and the regulator in a pre-submission meeting.

**Secondary endpoints:**
- **Gate performance:** the share of dental images wrongly rejected (a missed screening) and of non-dental
  images accepted.
- **Localisation:** box-level detection measured as mAP@0.5 and FROC (false-positive marks per image
  against sensitivity).
- **Reader study:** the change in dentists' sensitivity and specificity *with* the tool compared with
  *without* it. This is what CADe regulators weigh most.
- **Subgroups:** results by camera or device, lighting, patient age band and sex, and by skin and gingival
  pigmentation. Subgroup results must not fall more than 10 points below the overall result.
- **Agreement:** inter-reader agreement (Fleiss' κ) for the reference standard itself.

## 3. Dataset

- **Source:** at least 3 clinics that took no part in training, including at least one in a different
  region or with different equipment, collected prospectively or as a consecutive retrospective series.
  This avoids the bias of hand-picking cases.
- **Size:** the sample is chosen so that each primary class's 95% CI half-width is at most ±0.05 at the
  expected sensitivity.
  - At an expected sensitivity of 0.85, that needs about 196 positive images per class. The formula is
    n = 1.96² · p(1 − p) / 0.05².
  - At roughly 20% prevalence for the rarer classes, that means about 1,000–1,200 images, plus about 300
    non-dental images for the gate.
  - Plan for 1,500 images to allow for exclusions.
- **Independence:** no patient, clinic or capture session overlaps with the training or tuning data. This
  is checked by patient hash and by an image near-duplicate search.
- **De-identification:** strip EXIF/DICOM metadata and keep no faces beyond the oral region, under the
  ethics approval and the DPDP Act / HIPAA as applicable.
- **Ethics:** institutional ethics committee approval. Informed consent for prospective collection, or a
  waiver for retrospective de-identified data.

## 4. Reference standard

- Three calibrated dentists label every image independently, blind to the model's output. Each label is a
  class, a box, and their confidence.
- The reference is the majority vote, with a fourth senior adjudicator for ties.
- For caries, where possible, confirm with clinical examination or bitewing radiographs from the same
  visit, because photo-only labels underestimate early caries.
- Calibration: before labelling, all readers label a training set of 50 images until κ is at least 0.6.

## 5. Reader study (MRMC)

A fully crossed multi-reader multi-case design:
- At least 10 dentists of mixed experience read at least 300 cases.
- Each reads every case twice: unaided, and aided by the tool's boxes, with a washout of at least 4 weeks
  between the two reads and the case order randomised.
- Analyse with the Obuchowski–Rockette method, reporting the difference in AUC and sensitivity with
  readers and cases as random effects.

## 6. Analysis

- Pre-register the statistical analysis plan before unblinding.
- Report sensitivity and specificity with Wilson 95% CIs, the confusion matrix per class, calibration of
  the confidence scores, and every subgroup in §2.
- Analyse failures: report every false negative for advanced caries and ulcer, with the image and the
  reason.
- Lock the software version (git commit and model weight hashes) before the study. Any change to the
  model, the thresholds or the preprocessing during the study voids the results for that version.
- Note on the current prototype: the gate's `best_val_auc` of 0.99998 in `vision/gate_meta.json` comes from
  in-distribution validation. It isn't evidence of real-world performance, and this study replaces it.

## 7. Regulatory route

**India: CDSCO.**
- Under the Medical Devices Rules 2017, software intended for diagnosis or detection is a medical device.
- A CADe tool that supports a clinician's decision is likely **Class B** (low–moderate risk). Confirm this
  against the current CDSCO classification list for software as a medical device (SaMD).
- Steps:
  1. A quality management system to ISO 13485.
  2. Risk management to ISO 14971.
  3. A software lifecycle process to IEC 62304, with this app's safety class decided.
  4. Usability engineering to IEC 62366-1.
  5. Clinical performance evidence: this study.
  6. An application for a manufacturing licence (MD-5/MD-6 for Class A/B through the State Licensing
     Authority), plus a registered importer/agent if made abroad.
- Post-market: a vigilance and complaint process, and periodic safety updates.

**United States: FDA.**
- Dental CADe tools have been cleared through the **510(k)** route as Class II devices, for example under
  product codes for radiological dental image analysis.
- This photo-based tool would need a suitable predicate. If there isn't one, it goes **De Novo**.
- Steps:
  1. Q-Submission (pre-submission) to agree the endpoints and the reader study.
  2. Design controls (21 CFR 820, now aligned with ISO 13485).
  3. Cybersecurity documentation, required for networked devices.
  4. A Predetermined Change Control Plan (PCCP), if the model will be retrained after clearance.
  5. A standalone performance study plus the MRMC reader study.
  6. The 510(k) submission.
- Note: DentAgent's weekly retraining applies to the *no-show* model, not the vision model. Any automatic
  retraining of the vision model would need a cleared PCCP.

**EU (if needed):** MDR Rule 11 software, probably Class IIa. Needs a notified body; the same evidence is
reused.

## 8. What the product must do in the meantime

- Label imaging output "screening support — clinician review required". The verdict and findings are
  approvals, never automatic chart entries. This is already how the app works.
- Make no diagnostic or performance claims in marketing or the UI until cleared.
- Keep the audit trail: who approved each finding (`action.approve` traces, with the approver's name in
  the activity log).
- Track field performance: clinicians' approve/reject decisions on findings are a running proxy for
  precision. Review them monthly per class.
