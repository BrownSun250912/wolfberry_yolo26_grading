# Figure captions and text alternatives

## 01_paired_main_results

Fixed new seed25 and historical seed23; paired original-image, source-stratified percentile95 intervals,2000draws; all116val images/518GT; adaptive reusedval, no multiplicity correction. Raw YOLO point uses same fullGT pool.

Alt text: Two forest plots show attribute replacements fail to improve reliably, while output-only prototype refinement improves AP50:95 without changing AP50.

Sources: outputs\reports\attribute_local_20261025\paired_image_intervals.csv

## 02_attribute_factorial

Mean of all3trainingseeds in original2x2 experiment; each cell one fixed replacement policy; equal allocated architecture, held source onlynewtraining; heatmap is descriptive, seeds are not new datasets.

Alt text: Two matched heatmaps retain all factor and intervention combinations, showing clean validation transfer is weaker than internal held performance.

Sources: outputs\reports\attribute_local_20261025\held_pipeline_metrics.csv, outputs\reports\attribute_local_20261025\val_pipeline_metrics.csv

## 03_boundary_dose

Left: equal30epochs, mean/min-max across3fits, not confidenceinterval; held meanIoU measured on originalgrid. Right: all3originalselected-checkpoint seed outcomes,not fixed-dose reselectedval; grades/conf fixed. Historical seeds23/24/25; new25/26/27.

Alt text: Prototype features improve native mask learning despite equal training dose; historical binary refinement remains a strong segmentation control.

Sources: outputs\reports\attribute_local_20261025\prototype_fixed_dose_ledger.csv, outputs\reports\attribute_local_20261025\prototype_val_metrics.csv, outputs\reports\attribute_local_20261025\historical_boundary_control.csv

## 04_stress_all_conditions

Fixedseed25, all10conditions, original518GT denominator percondition; transforms use predicted foreground only. Synthetic label-preservationassumption, detector predictions condition-specific; no robustness dominance claimed. Otherseeds and riskcounts in sourceCSV.

Alt text: A ten-condition heatmap shows severe legacy grading sensitivity persists despite local gains from new attributes.

Sources: outputs\reports\attribute_local_20261025\attribute_stress_metrics.csv

## 05_geometry_sensitivity

Fixed original500matched pairs; no postrefinement cohort reselection. Left3000paired-image percentile95 intervals; IoUpositive is improvement, negative geometryerrors are improvement. Right includes original image diagonal and fixedpixel scales; manualpolygons are not physical metrology.

Alt text: Paired geometry errors improve; boundary quality depends strongly on erosion-band width and the historical control is competitive.

Sources: outputs\reports\attribute_local_20261025\native_geometry_intervals.csv, outputs\reports\attribute_local_20261025\native_geometry_instances.csv

## 06_defect_probability

Fixed original500matched valcohort, available defectlabels only. All occupied15equalwidthbins shown, missingbins leave gaps. No empirical valcalibration fitted. Curves are descriptive, smallbins unstable; counts and numericalECE/Brier/NLL in source tables.

Alt text: Reliability curves separate weighted-loss raw probabilities from analytically prior-corrected values, without claiming calibrated safety.

Sources: outputs\reports\attribute_local_20261025\attribute_calibration_bins.csv

## 07_runtime

24images6/source,3repeats, graphs randomized perimage; GPU synchronized, FP32/conf.001, image decoding/model loading excluded. Pairedimage3000draw percentile95intervals; shared multi-score audit baseline and existing leanRR22 are distinct. Strict replay encodes unusedlocalview for numericalidentity; single-global recomputation is separate faircontrol. Timing on actualdevice, not cachedfeature latency.

Alt text: Actual timing shows old-defect ranking has small overhead, whereas output-mask refinement adds measurable cost; explicit numerical graph variants remain separate.

Sources: outputs\reports\attribute_local_20261025\rank_latency_summary.csv, outputs\reports\attribute_local_20261025\rank_latency_intervals.csv

## 08_rank_information_controls

All3seeds shown; filledpoint fixedseed25, open26/27 inleft. Originaldefect is an adaptive mechanismcontrol, not originalpreselectednewhead family. Right outer sources excluded onlyfrom newlyfittedrank/checkpoints/normalizer; oldencoders saworiginaltrain. Pooled AP recomputed after concatenatingallouter predictions, not averagedsourceAP. Sourcegroup evidence is exploratory, no whole-system OOD claim.

Alt text: Original defect evidence explains the useful rank gain with less added computation; source-group benefits vary and pooled gains are small.

Sources: outputs\reports\attribute_local_20261025\attribute_rank_val_metrics.csv, outputs\reports\attribute_local_20261025\attribute_rank_control_metrics.csv, outputs\reports\attribute_local_20261025\rank_source_metrics.csv, outputs\reports\attribute_local_20261025\rank_source_pooled_metrics.csv

## 09_shape_score_tradeoff

All12fits and RR22, fixedsource split, all3seeds; filledpoint seed25. This two-condition mechanismcontrast is not a globalPareto frontier; prior illuminationnormalizers are stronger on someconditions. All10conditions, including adverse results, retained in sourceCSV. No candidate passed the internalfullpipelinegate.

Alt text: Binary geometry improves one dark-condition result while losing substantial clean accuracy, so it is rejected as a replacement score head.

Sources: outputs\reports\attribute_local_20261025\shape_score_stress_metrics.csv

## 10_qualitative_best_and_worst

GT-only selected largestgain, largestdecline and near-medianIoUchange on fixed500cohort, fixedseed25. These are deliberately selected illustrations, not representative proof. Original RGB pixels retained, viewing crops can be enlarged byrenderer; no recoveredimage detail or photometricenhancement. Dashedgold GTmanualpolygon, solidblue prediction. GT never entersruntime.

Alt text: Three paired native-image crops include both refinement improvement and regression, with unchanged source photography and visibly different contours.

Sources: outputs\reports\attribute_local_20261025\qualitative_selected_cases.csv, outputs\reports\attribute_local_20261025\native_geometry_instances.csv