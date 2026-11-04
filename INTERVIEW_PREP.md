# Interview Preparation Guide

This guide is written for a beginner. Use it to explain what you actually built and measured; do not present yourself as a clinician or deep-learning expert.

## Core concepts in plain language

### What is brain MRI?

Magnetic resonance imaging uses magnetic fields and radio-frequency signals to produce images of soft tissue. Different MRI sequences make tissue and fluid appear with different brightness. That variability is one reason a fixed threshold can work on one scan and fail on another.

### What is a pixel? What is a voxel?

A pixel is one location in a 2D image and stores a value, such as intensity. A voxel is the 3D equivalent: a small volume element. This project first evaluates 2D sagittal slices, then evaluates all voxels in complete 3D NFBS volumes.

### What is segmentation?

Segmentation assigns every pixel to a category. This is binary segmentation: brain or non-brain.

### What is skull stripping, and why is it useful?

Skull stripping creates a mask that keeps brain and rejects skull, skin, eyes, neck, and background. It can simplify registration, visualization, volume measurement, or later segmentation. Errors matter: false negatives remove brain; false positives retain non-brain anatomy.

### What is an image mask?

A mask is an array aligned with the image. Here `1` means the reference says the pixel belongs to brain, and `0` means it does not. Multiplying an image by the mask would keep brain pixels and set everything else to zero.

### What is thresholding?

Thresholding converts continuous intensity into two groups. Values on one side of the threshold become foreground. It is simple and explainable, but it has no direct knowledge of anatomy.

### What is Otsu thresholding?

Otsu chooses the threshold that best separates the intensity histogram into two classes by minimizing variation within each class, equivalently maximizing separation between them. It is automatic, not learned. This project checks both polarities because brain is not always the brighter class.

### What are morphological operations?

They edit binary shapes using a small neighborhood. Erosion shrinks foreground; dilation expands it. Opening uses erosion then dilation to remove small noise. Closing uses dilation then erosion to connect nearby regions and close gaps. Hole filling repairs enclosed background holes.

### What is a connected component?

It is one group of touching foreground pixels. The third baseline prefers a large component near the image center. This often removes separate skull or artifact regions, but can choose the wrong central structure.

### What is a spatial prior?

A spatial prior is an assumption about where a target is likely to be. The optional 3D method assumes a typically positioned brain lies in a superior central ellipsoid after canonical orientation. This greatly reduced neck and peripheral false positives on NFBS, but it can fail when anatomy or field of view differs. It is a transparent heuristic, not anatomical understanding.

### What is ground truth?

Ground truth is the reference mask used for evaluation. It is the best available target, not absolute truth: human definitions and annotation protocols can differ, especially at boundaries.

### What are TP, TN, FP, and FN?

- TP: predicted brain and reference brain.
- TN: predicted non-brain and reference non-brain.
- FP: predicted brain but reference non-brain.
- FN: predicted non-brain but reference brain.

### What is Dice score?

Dice measures overlap: `2TP / (2TP + FP + FN)`. It ranges from 0 for no overlap to 1 for identical masks. The best method's test mean was 0.790, but its worst case was 0.085, so the distribution matters.

### What is IoU?

Intersection over Union is `TP / (TP + FP + FN)`. It divides the shared region by everything covered by either mask. It is stricter numerically than Dice but ranks a fixed set of predictions similarly.

### What are boundary metrics?

Overlap can look similar even when one contour is farther from the reference. Surface Dice asks how much of the two boundaries lies within a tolerance. HD95 is a robust “large boundary error” distance, while ASSD is the average symmetric boundary distance. This project reports HD95 and ASSD in pixels and millimeters using NIfTI header spacing. Lower distance is better; higher surface Dice is better.

### Why can accuracy be misleading?

If most pixels are background, predicting background everywhere can achieve high pixel accuracy while finding no brain. Dice, sensitivity, and precision focus more directly on the target region and its errors.

### What are sensitivity and specificity?

Sensitivity is the fraction of reference brain found: `TP / (TP + FN)`. Specificity is the fraction of non-brain correctly rejected: `TN / (TN + FP)`. A mask can have high precision but terrible sensitivity if it predicts only a tiny unquestionably brain region.

### What is data leakage?

Leakage happens when information from the final test set influences model or parameter choices. It makes results too optimistic. This project selected morphology parameters using 24 official development scans, then evaluated the fixed choice on 558 held-out test scans.

### Why should related images from one scan stay in one split?

Slices from the same scan share anatomy, acquisition settings, and noise. Putting some in development and others in test would make the sets less independent. The official 2D dataset contains one sagittal slice per scan and provides scan-level lists.

### Why start with a baseline before deep learning?

A baseline checks the data loader, masks, metrics, and evaluation design with understandable logic. It provides a number a future U-Net must beat and reveals difficult scan types. Without a baseline, complexity can hide basic pipeline mistakes.

### Why can a classical method fail?

MRI brightness varies by sequence and scanner. Brain, skull, fluid, and artifacts can overlap in intensity. Otsu sees only the histogram; morphology sees shapes locally; the component heuristic assumes the brain is one large central region. None understands anatomy as a whole.

### How could a U-Net improve this project?

A U-Net can learn multi-scale spatial patterns and boundaries from labeled examples rather than applying one intensity rule. It may recognize a whole brain even when intensities are heterogeneous. It also introduces training, more compute, leakage risk, hyperparameters, and a need for stronger external validation. Improvement must be measured, not assumed.

### Why is this not a clinical product?

It is a retrospective educational prototype tested on public 2D slices and one public complete-volume T1 dataset. External NFBS performance was poor, it has catastrophic failure cases, and there is no prospective clinical validation, regulatory review, workflow integration, or evidence for every population and scanner.

## 60-second explanation

“I built a reproducible evaluation of three classical skull-stripping baselines. On 558 held-out SynthStrip slices, mean Dice improved from 0.749 for Otsu to 0.790 with component selection. On complete NFBS volumes, the frozen morphology baseline reached only 0.559 and strongly over-segmented scalp and neck. I then added an explicitly exploratory fourth method: a coarse canonical spatial prior with millimetre-aware morphology. It reached 0.754 mean Dice and improved every NFBS subject over morphology, but NFBS had already motivated that design, so I do not call it fresh external validation. My main lesson is that spatial assumptions can solve a specific failure mode, but the next test must use untouched data.”

## Two-minute detailed explanation

“The project asks how well understandable classical image processing can perform brain extraction before adding deep learning. I downloaded the official 39 MB SynthStrip 2D v1.5 archive, verified its SHA-256, inspected its real NIfTI layout, and used the official scan lists: 24 development and 558 test scans.

“For preprocessing, I clip each image to its 1st and 99th intensity percentiles and scale to zero-to-one. Method 1 uses Otsu's threshold and scores both intensity polarities without looking at the reference mask. Method 2 adds scale-aware opening, closing, hole filling, and small-object removal. Method 3 scores connected components using area and distance from image center, keeps the best one, and performs final cleanup.

“I implemented TP, TN, FP, FN, Dice, IoU, precision, sensitivity, specificity, surface Dice, HD95, and ASSD directly and unit-tested perfect, partial, no-overlap, boundary, and empty cases. HD95 and ASSD are available in pixels and physical millimeters from the NIfTI spacing. A development-only grid selected morphology parameters. On the held-out 558 scans, mean Dice was 0.749, 0.772, and 0.790. Method 3's 95% bootstrap interval was 0.777 to 0.803. It improved precision and specificity but was slightly less sensitive than Method 2 because selecting one component can discard real brain pixels.

“Because the methods process the same scans, I also calculated paired improvements rather than comparing only separate means. Method 3 improved over raw Otsu by 0.041 Dice on average and won on 456 of 558 SynthStrip scans. I generated source- and modality-specific statistics, physical boundary plots, a ranked review report, and a reference-based failure taxonomy.

“For external validation, I used the open NFBS repository of 125 defaced T1 volumes. I first selected each canonical center sagittal slice using image geometry, never the mask, and kept all SynthStrip-tuned settings frozen. Method 3 ranked first there, but mean Dice dropped from 0.790 to 0.537.

“I then built a true 3D path with canonical affine checks, spherical morphology, 26-connected components, physical surface distances, and subject-level bootstrap comparisons. I transferred the scale fractions without tuning on NFBS. Morphology won with mean Dice 0.559 and improved over Otsu by 0.033, but precision was only 0.445 and 124 of 125 masks were categorized as over-segmented. Component selection tied morphology on 124 subjects and failed catastrophically on one. That negative result is useful: more spatial context and a nominally stronger heuristic did not solve the anatomy problem. I would not claim clinical adequacy.”

“As a follow-up, I kept those three methods unchanged and added an optional `spatial_prior` method. It uses only canonical array geometry and voxel spacing: a superior ellipsoid removes implausible neck/peripheral regions, followed by opening and closing in millimetres. Mean Dice rose to 0.754, precision to 0.724, and HD95 fell from 77.86 to 21.94 mm. It beat morphology on all 125 subjects. However, this was designed after inspecting NFBS failures, so it is evidence that the hypothesis is promising—not an unbiased external performance estimate. The next evaluation needs a new untouched dataset.”

## Twelve likely interview questions

1. **Why did you choose classical methods?**  
   I am building understanding in layers. Classical methods let me verify data, metrics, and evaluation while explaining every operation. They also create a meaningful baseline for future deep learning.

2. **Why normalize per image?**  
   MRI intensities do not have a universal scale. Per-image percentile normalization reduces scale differences and limits outliers. Its downside is that it can hide absolute cross-scan intensity information, but these methods do not rely on that information.

3. **How did you prevent test leakage?**  
   I used the dataset authors' scan-level split. I tried morphology settings only on 24 development scans, fixed the configuration, and then ran the 558 SynthStrip test scans. I transferred the scale fractions into 3D without tuning on NFBS and kept each NFBS subject as one evaluation unit.

4. **Why not use accuracy?**  
   Background often dominates. A nearly all-background prediction could look accurate while missing the brain. I report it only as a secondary field and emphasize Dice, IoU, precision, sensitivity, and specificity.

5. **Which method won and why?**  
   It depends on the experiment. `central_component` won the original 2D SynthStrip test at 0.790. Among the three frozen methods, `otsu_morphology` won complete NFBS evaluation at 0.559. The later exploratory `spatial_prior` reached 0.754, but it was designed after NFBS inspection, so it needs untouched external confirmation.

6. **What surprised you?**  
   The severe T2 failures. One case had precision 1.0 but sensitivity 0.045: the small region it predicted was correct, yet it missed almost the entire brain. That made the value of multiple metrics very concrete.

7. **What does the standard deviation tell you?**  
   Method 3's Dice standard deviation was 0.156, so performance varied widely. Mean performance alone is not enough; case plots and minimum values reveal risk.

8. **What would you improve first?**  
   I would reserve a new external dataset before changing parameters. On a declared development set, I would replace the fixed ellipsoid with adaptive head localization and test bias-field correction or watershed refinement. Then I would compare a small U-Net using the same subject-level protocol.

9. **Did you implement the SynthStrip neural model?**  
   No. I used its public 2D evaluation dataset and masks. This version deliberately implements only classical baselines. The original SynthStrip method is deep learning and is cited as the data source.

10. **What do you still not know?**  
    I do not know whether the spatial prior transfers beyond this already-inspected NFBS acquisition. I have not demonstrated generalization across diseases, scanners, orientations, ages, or annotation protocols, nor clinical usefulness or prospective validity.

11. **Are the quality warnings confidence scores?**
    No. They are deterministic geometry and method-disagreement rules. Their thresholds were selected on only 24 development scans, and their observed test behavior is reported honestly. They are not calibrated probabilities and need external validation.

12. **Why did you use a paired analysis?**
    Every method processed the same 558 scans, so each scan acts as its own control. Subtracting scores scan by scan distinguishes a consistent improvement from a mean difference caused by different case mixes. I report effect size, bootstrap interval, wins/losses/ties, and a corrected paired test rather than only a p-value.

## Five intelligent questions to ask the interviewer

1. In your work, which skull-stripping failure modes most strongly affect downstream analysis?
2. How do you define the brain boundary around CSF, dura, lesions, or postoperative anatomy?
3. What kinds of external validation have been most informative across scanners and populations?
4. When has a simple baseline revealed a problem that a more complex model initially hid?
5. For a beginner moving from this project to a U-Net, which evaluation mistake should I be most careful to avoid?

## Glossary

- **Acquisition:** how an MRI scan was collected, including sequence and scanner settings.
- **Baseline:** a simple reference method that a later method should improve upon.
- **Binary mask:** an array containing only two classes, usually 0 and 1.
- **Connected component:** one contiguous foreground region.
- **Contrast:** relative brightness differences between tissues.
- **Deterministic:** the same input and settings produce the same output.
- **Dice:** overlap metric based on twice the intersection divided by total mask size.
- **Erosion/dilation:** operations that shrink/expand binary foreground.
- **False negative:** actual brain that the method missed.
- **False positive:** non-brain that the method kept.
- **Domain shift:** a change between development and deployment data that alters performance.
- **External validation:** evaluating frozen choices on an independent dataset not used for tuning.
- **Ground truth:** reference annotation used for evaluation.
- **IoU:** intersection divided by union.
- **HD95:** robust boundary distance using the 95th percentile; lower is better.
- **ASSD:** average symmetric distance between predicted and reference boundaries.
- **Bootstrap interval:** uncertainty estimate produced by repeatedly resampling scans with replacement.
- **Paired comparison:** comparison of two methods on the same cases.
- **Morphology:** shape-based binary image processing.
- **NIfTI:** a common neuroimaging file format (`.nii` or `.nii.gz`).
- **Normalization:** transforming intensities to a more consistent range.
- **Otsu:** automatic two-class histogram thresholding.
- **Polarity:** which side of a threshold is treated as foreground.
- **Precision:** fraction of predicted brain that is correct.
- **Segmentation:** assigning a class to each pixel or voxel.
- **Sensitivity:** fraction of actual brain that is detected.
- **Specificity:** fraction of non-brain correctly rejected.
- **Surface Dice:** fraction of boundary pixels matching within a chosen tolerance.
- **U-Net:** an encoder-decoder neural network architecture for image segmentation.
- **Voxel:** one volume element in a 3D image.

## Do not claim

- Do not say the system is diagnostic, clinically validated, regulatory approved, or safe for patient care.
- Do not say 0.790 Dice is “good enough” without context; the minimum was 0.085.
- Do not say you trained SynthStrip or implemented its neural network.
- Do not say thresholding understands anatomy.
- Do not generalize beyond the evaluated SynthStrip slices and NFBS 2D/3D constructions, scanners, or populations.
- Do not present the spatial-prior result as untouched external validation; it was motivated by inspected NFBS failures.
- Do not claim deep-learning expertise from this classical project.
- Do not hide the T2 failures or quote only the best cases.
- Do not call warning flags calibrated confidence or proof that an unflagged mask is correct.

## Demonstration checklist

- Activate the environment: `source .venv/bin/activate`.
- Show the project structure and `configs/default.yaml`.
- Run `python -m pytest -q` and point out the metric empty-case tests.
- Optionally run `streamlit run streamlit_app.py` for an interactive public-dataset demonstration.
- Open `outputs/full_run/method_comparison.png`.
- Open `docs/assets/paired_dice_improvement.png` and explain why comparisons are paired.
- Open `docs/assets/modality_dice_comparison.png` and discuss qT1/T2 limitations.
- Open `docs/assets/nfbs_external_method_comparison.png` and explain the 0.790-to-0.537 generalization drop.
- Open `docs/assets/nfbs_3d_method_comparison.png` and explain why morphology, not component selection, won the full-volume run.
- Open `docs/assets/nfbs_3d_spatial_method_comparison.png` and distinguish exploratory improvement from external validation.
- In a spatial-prior review, identify the yellow dashed prior boundary and explain whether it clips brain or excludes non-brain anatomy.
- Open `outputs/nfbs_3d_full/orthogonal_reviews/worst_1_A00060006.png` and identify scalp/neck false positives.
- Open `docs/assets/quality_warning_rates.png` and explain that warnings are conservative.
- Open one best and one worst overlay; explain TP, FP, and FN.
- Show `summary_metrics.csv` and emphasize 558 held-out scans.
- Explain development tuning versus final testing.
- State the best method for each experiment and why the answer changes in 3D.
- State the worst-case Dice and one reason it fails.
- End with the clinical disclaimer and the next U-Net experiment.

