# Interview Preparation Guide

This guide is written for a beginner. Use it to explain what you actually built and measured; do not present yourself as a clinician or deep-learning expert.

## Core concepts in plain language

### What is brain MRI?

Magnetic resonance imaging uses magnetic fields and radio-frequency signals to produce images of soft tissue. Different MRI sequences make tissue and fluid appear with different brightness. That variability is one reason a fixed threshold can work on one scan and fail on another.

### What is a pixel? What is a voxel?

A pixel is one location in a 2D image and stores a value, such as intensity. A voxel is the 3D equivalent: a small volume element. This project uses 2D sagittal slices, so it processes pixels, although the source scans were 3D.

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

It is a retrospective educational prototype tested on 2D slices from two public datasets. External NFBS performance was poor, it has catastrophic failure cases, and there is no prospective clinical validation, regulatory review, workflow integration, or evidence for every population and scanner.

## 60-second explanation

“I built a reproducible evaluation of three classical skull-stripping baselines. On 558 held-out SynthStrip slices, mean Dice improved from 0.749 for Otsu to 0.772 with morphology and 0.790 with central-component selection. I then froze every parameter and evaluated 125 canonical center slices from the independent NFBS dataset. The best mean Dice fell to 0.537, with several zero-overlap cases. A retrospective taxonomy showed under-segmentation, boundary mismatch, and wrong-component selection. My main lesson is that simple spatial cleanup improves an internal baseline, but external validation exposes how poorly intensity heuristics generalize.”

## Two-minute detailed explanation

“The project asks how well understandable classical image processing can perform brain extraction before adding deep learning. I downloaded the official 39 MB SynthStrip 2D v1.5 archive, verified its SHA-256, inspected its real NIfTI layout, and used the official scan lists: 24 development and 558 test scans.

“For preprocessing, I clip each image to its 1st and 99th intensity percentiles and scale to zero-to-one. Method 1 uses Otsu's threshold and scores both intensity polarities without looking at the reference mask. Method 2 adds scale-aware opening, closing, hole filling, and small-object removal. Method 3 scores connected components using area and distance from image center, keeps the best one, and performs final cleanup.

“I implemented TP, TN, FP, FN, Dice, IoU, precision, sensitivity, specificity, surface Dice, HD95, and ASSD directly and unit-tested perfect, partial, no-overlap, boundary, and empty cases. HD95 and ASSD are available in pixels and physical millimeters from the NIfTI spacing. A development-only grid selected morphology parameters. On the held-out 558 scans, mean Dice was 0.749, 0.772, and 0.790. Method 3's 95% bootstrap interval was 0.777 to 0.803. It improved precision and specificity but was slightly less sensitive than Method 2 because selecting one component can discard real brain pixels.

“Because the methods process the same scans, I also calculated paired improvements rather than comparing only separate means. Method 3 improved over raw Otsu by 0.041 Dice on average and won on 456 of 558 SynthStrip scans. I generated source- and modality-specific statistics, physical boundary plots, a ranked review report, and a reference-based failure taxonomy.

“For external validation, I used the open NFBS repository of 125 defaced T1 volumes. I selected each canonical center sagittal slice using image geometry, never the mask, and kept all SynthStrip-tuned settings frozen. Method 3 still ranked first, but mean Dice dropped from 0.790 to 0.537. That result is more scientifically useful than hiding the failure: it demonstrates domain shift and motivates a leakage-safe learned model. I would not claim clinical adequacy.”

## Twelve likely interview questions

1. **Why did you choose classical methods?**  
   I am building understanding in layers. Classical methods let me verify data, metrics, and evaluation while explaining every operation. They also create a meaningful baseline for future deep learning.

2. **Why normalize per image?**  
   MRI intensities do not have a universal scale. Per-image percentile normalization reduces scale differences and limits outliers. Its downside is that it can hide absolute cross-scan intensity information, but these methods do not rely on that information.

3. **How did you prevent test leakage?**  
   I used the dataset authors' scan-level split. I tried morphology settings only on 24 development scans, fixed the configuration, and then ran the 558 test scans.

4. **Why not use accuracy?**  
   Background often dominates. A nearly all-background prediction could look accurate while missing the brain. I report it only as a secondary field and emphasize Dice, IoU, precision, sensitivity, and specificity.

5. **Which method won and why?**  
   `central_component` had the best mean Dice on both datasets: 0.790 on SynthStrip and 0.537 on NFBS. It often removed peripheral false-positive components, but external performance was still poor and it sometimes selected entirely wrong anatomy.

6. **What surprised you?**  
   The severe T2 failures. One case had precision 1.0 but sensitivity 0.045: the small region it predicted was correct, yet it missed almost the entire brain. That made the value of multiple metrics very concrete.

7. **What does the standard deviation tell you?**  
   Method 3's Dice standard deviation was 0.156, so performance varied widely. Mean performance alone is not enough; case plots and minimum values reveal risk.

8. **What would you improve first?**  
   External validation is now complete and revealed a large generalization gap. I would next establish a leakage-safe training and development split for a small U-Net, while preserving NFBS as untouched external evaluation, then add complete 3D evaluation.

9. **Did you implement the SynthStrip neural model?**  
   No. I used its public 2D evaluation dataset and masks. This version deliberately implements only classical baselines. The original SynthStrip method is deep learning and is cited as the data source.

10. **What do you still not know?**  
    I have demonstrated poor transfer to one external T1 dataset, not generalization across all acquisitions. I still have not demonstrated 3D performance, clinical usefulness, or prospective validity. I would seek expert guidance on clinical mask definitions and evaluation design.

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
- Do not generalize beyond the evaluated SynthStrip and NFBS 2D constructions, scanners, or populations.
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
- Open `docs/assets/quality_warning_rates.png` and explain that warnings are conservative.
- Open one best and one worst overlay; explain TP, FP, and FN.
- Show `summary_metrics.csv` and emphasize 558 held-out scans.
- Explain development tuning versus final testing.
- State the best method and one reason it improves.
- State the worst-case Dice and one reason it fails.
- End with the clinical disclaimer and the next U-Net experiment.

