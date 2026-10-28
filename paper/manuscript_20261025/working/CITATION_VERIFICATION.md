# Independent citation verification

Audit date: 25 October 2026 (Asia/Taipei).

Input: `MANUSCRIPT_EN.md`, initial version containing references [1]–[13]. This verifier received the manuscript and bibliography, without the drafting rationale or experimental evidence map. The audit addresses bibliographic identity and whether each citation supports its associated prose. It does not validate the manuscript's original experimental results. No manuscript, data, model, or archived experiment was modified by this verifier.

## Result and required correction

All 13 cited works were found. No invented work, incorrect author list, wrong year, wrong venue, or incorrect supplied DOI was established. References [1]–[12] are VERIFIED for their present citation uses. Reference [13] has VERIFIED bibliographic identity but an OVERCLAIM / citation-support mismatch at the precise evaluation-threshold statement: the retrieved dataset-paper evidence does not establish the modern ten-threshold definition. The threshold definition itself is directly supported by COCO's official evaluation documentation.

The correction communicated to the drafting agent is to retain [13] as the COCO dataset source and add the official evaluation webpage as [14] at the threshold statement. The drafting agent acknowledged that it would apply this correction in both language versions. This report records the original finding and proposed resolution; the verifier did not edit either manuscript.

Suggested additional reference:

> COCO Consortium. Detection Evaluation. Official dataset evaluation documentation, accessed 25 October 2026. https://cocodataset.org/dataset/detection-eval.htm.

Suggested affected passage: preserve the description of the actual Ultralytics implementation, but attach [14] to the statement that the threshold range follows the COCO convention. [13,14] is acceptable if the author wishes to retain the dataset citation at this occurrence. This does not justify claiming numerical equivalence between Ultralytics AP and every COCO evaluator.

Optional bibliography completion: [4] can add volume **12**. Leave [6] without pages unless the author explicitly chooses between the IEEE and CVF pagination. Its current entry is accurate.

## Method and evidence limits

For every reference, at least three search forms were executed: full title, author plus subject plus year, and core keywords. Searches were followed by retrieval of publisher, conference, author-hosted preprint, DOI registry, or official software/dataset documentation as appropriate. The exact query forms are recorded below. Crossref records for [4] and [6] were also retrieved using their DOI endpoints to check the publisher-deposited identity, authors, year, volume or pagination.

Evidence levels used here:

- **L1**: retrieved full text or official documentation relevant to the cited proposition.
- **L2**: retrieved abstract or directly exposed abstract text from a primary publisher/conference source.
- **L3**: bibliographic metadata only.

Some direct opens returned 403, 429, cache-miss, or CAPTCHA responses. These failures were not interpreted as evidence of nonexistence. Search-indexed text from the same primary publisher/conference pages and successful alternative primary retrievals supplied the evidence identified below. A page listed as L2 was not treated as a fully read paper. No methodological or numerical claim was accepted merely from its title.

Line numbers refer to the initial 394-line English manuscript inspected by this verifier. Later editorial changes may shift them.

## Entry-by-entry verification

### [1] Wang et al., wolfberry recognition and localization

**Status: VERIFIED. Evidence used: L2, primary publisher abstract / PDF first-page text.**

- Matched title: *An Object Feature-Based Recognition and Localization Method for Wolfberry*.
- Matched authors: Renwei Wang; Dingzhong Tan; Xuerui Ju; Jianing Wang. The manuscript initials R., D., X., J. match. A secondary index exposes a shortened author-name variant for Ju; the publisher PDF's Xuerui Ju is the controlling record.
- Matched year and venue: *Sensors*, 2025, 25(11), article 3365.
- Matched DOI: `10.3390/s25113365`.
- Primary sources: [publisher article](https://www.mdpi.com/1424-8220/25/11/3365), [publisher PDF with first-page authors, citation and abstract](https://www.mdpi.com/1424-8220/25/11/3365/pdf?version=1748347285).
- Claim sites: Introduction line 11 and Related Work line 27. Harvesting-oriented recognition/localization and feature-based segmentation in unstructured lighting are expressly described in the abstract. The manuscript does not import its reported accuracy or claim commercial grading validation. No correction required.
- Queries executed: `"An Object Feature-Based Recognition and Localization Method for Wolfberry"`; `Wang Tan Ju Wang wolfberry recognition localization 2025 Sensors`; `wolfberry object feature recognition localization unstructured lighting Sensors`.

### [2] Huang et al., dried-goji spoilage adulteration

**Status: VERIFIED. Evidence used: L2 primary abstract plus complete indexed author/DOI metadata.**

- Matched title: *Detection of Kernel-Level Spoilage Adulteration in Dried Goji Berries Using Zero-Shot Learning and Computer Vision*.
- Matched authors: Ruobin Huang; Yuanning Zhai; Baiwei Sun; Osama Elsherbiny; Lei Zhou; Yiying Zhao.
- Matched date and venue: 17 August 2026, *Foods*, 15(16), article 2869.
- Matched DOI: `10.3390/foods15162869`.
- Sources: [publisher article](https://www.mdpi.com/2304-8158/15/16/2869), [PubMed record with full abstract and author names](https://pubmed.ncbi.nlm.nih.gov/42650563/). The linked PMC full-text page returned a CAPTCHA, so no claim of having read the entire PMC article is made.
- Claim sites: lines 11 and 27. The description as dried-goji spoilage-adulteration assessment is supported. Its freshness rule and segmentation task differ from the current pedicel-aware grades, so the manuscript's stated scope distinction is appropriate. No correction required.
- Queries executed: `"Detection of Kernel-Level Spoilage Adulteration in Dried Goji Berries Using Zero-Shot Learning and Computer Vision"`; `Huang Zhai Sun Elsherbiny Zhou Zhao dried goji spoilage 2026`; `kernel level spoilage adulteration goji berries zero shot learning computer vision`.

### [3] Kusakunniran et al., integrated mangosteen assessment

**Status: VERIFIED. Evidence used: L2 publisher abstract and metadata.**

- Matched title: *Integrating instance segmentation-based sizing and multi-view deep learning framework for mangosteen quality assessment*.
- Matched authors: Worapan Kusakunniran; Suppakorn Pojsomphong; Patiparn Tangmongkolpaisan; Paranee Wannasorn.
- Matched venue and publication status: *Information Processing in Agriculture*, available online 21 September 2026, in press / corrected proof.
- Matched DOI: `10.1016/j.inpa.2026.09.004`.
- Primary source: [Elsevier publisher record and abstract](https://www.sciencedirect.com/science/article/pii/S2214317326001368).
- Claim sites: lines 11 and 27. The abstract expressly combines segmentation-based physical sizing and multi-view appearance classification. The manuscript does not transfer its commercial grades, calibrated physical dimensions or performance numbers into the wolfberry task. No correction required. Do not invent a final volume or page range for an in-press record.
- Queries executed: `"Integrating instance segmentation-based sizing and multi-view deep learning framework for mangosteen quality assessment"`; `Kusakunniran Pojsomphong Tangmongkolpaisan Wannasorn mangosteen 2026`; `mangosteen instance segmentation sizing multiview appearance assessment`.

### [4] Lu et al., pepper impurity prediction

**Status: VERIFIED; optional volume completion. Evidence used: L2 publisher abstract, exposed method text, and Crossref deposit.**

- Matched title: *Impurity rates detection for pepper harvesting based on YOLOv8n-Seg-ASB and random forest*.
- Matched authors: Lijian Lu; Jin Lei; Chenming Cheng; Shiguo Wang; Chengfu Wang; Xinyan Qin.
- Matched venue/year/article: *Smart Agricultural Technology*, 2025, article 101224. The DOI registry additionally supplies volume **12**.
- Matched DOI: `10.1016/j.atech.2025.101224`.
- Sources: [Elsevier article and abstract/method text](https://www.sciencedirect.com/science/article/pii/S2772375525004551), [publisher-deposited Crossref record](https://api.crossref.org/works/10.1016/j.atech.2025.101224).
- Claim sites: lines 11 and 27. The paper combines instance segmentation, Seg-PCA contour-derived geometry and random-forest impurity estimation. This supports the manuscript's three-part description. No performance or latency number is copied; that is appropriate because the retrieved article's abstract and bench-test text expose different timing values. No substantive correction required; adding `12` improves bibliographic completeness.
- Queries executed: `"Impurity rates detection for pepper harvesting based on YOLOv8n-Seg-ASB and random forest"`; `Lu Lei Cheng Wang Qin pepper harvesting impurity random forest 2025`; `pepper impurity rates segmentation geometric feature random forest`.

### [5] He et al., Mask R-CNN

**Status: VERIFIED. Evidence used: L2 CVF abstract and metadata.**

- Matched title: *Mask R-CNN*.
- Matched authors: Kaiming He; Georgia Gkioxari; Piotr Dollár; Ross Girshick.
- Matched venue/year: ICCV, 2017. CVF gives pages **2961–2969**, matching the manuscript's CVF citation.
- Source: [CVF official record](https://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html).
- No DOI is supplied in the manuscript, so no missing DOI was invented or required. CVF and IEEE pagination can differ; the current entry consistently follows its linked CVF source.
- Claim sites: lines 13 and 29. Per-instance masks together with object recognition are explicitly supported by the abstract. No overclaim found.
- Queries executed: `"Mask R-CNN" site:openaccess.thecvf.com/content_iccv_2017`; `He Gkioxari Dollar Girshick Mask R-CNN 2017`; `instance segmentation mask branch bounding box recognition Mask RCNN`.

### [6] Bolya et al., YOLACT

**Status: VERIFIED. Evidence used: L2 CVF abstract plus publisher-deposited Crossref metadata.**

- Matched title: *YOLACT: Real-Time Instance Segmentation*.
- Matched authors: Daniel Bolya; Chong Zhou; Fanyi Xiao; Yong Jae Lee.
- Matched venue/year: IEEE/CVF ICCV, 2019.
- Matched DOI: `10.1109/ICCV.2019.00925` (DOI letter case is immaterial).
- Sources: [CVF official record](https://openaccess.thecvf.com/content_ICCV_2019/html/Bolya_YOLACT_Real-Time_Instance_Segmentation_ICCV_2019_paper.html), [CVF PDF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Bolya_YOLACT_Real-Time_Instance_Segmentation_ICCV_2019_paper.pdf), [publisher-deposited Crossref record](https://api.crossref.org/works/10.1109/ICCV.2019.00925).
- Claim sites: lines 13 and 29. The abstract supports its use as an instance-segmentation precedent. It describes the prototype/instance-coefficient formulation, but the manuscript does not need or overextend those details.
- Pagination caution: CVF's official record and PDF use **9157–9166**, whereas the IEEE DOI deposit gives **9156–9165**. The current manuscript omits pages and therefore has no pagination error. Keep that omission or choose a version consistently; do not treat the two records as different works.
- Queries executed: `"YOLACT: Real-Time Instance Segmentation" site:openaccess.thecvf.com`; `Bolya Zhou Xiao Lee YOLACT 2019 10.1109/ICCV.2019.00925`; `real time instance segmentation prototype masks YOLACT`.

### [7] Ultralytics YOLO26 documentation

**Status: VERIFIED as software documentation. Evidence used: L1 official documentation.**

- Matched corporate author/title: Ultralytics; *Ultralytics YOLO26*.
- Source: [official model documentation](https://docs.ultralytics.com/models/yolo26/). Access date 25 October 2026 is consistent with the audit date.
- Supporting technical source: [official end-to-end detection guide](https://docs.ultralytics.com/guides/end2end-detection).
- No journal, paper DOI or invented bibliographic year is implied by the current documentation entry.
- Claim sites: lines 13 and 29. The official supported-task table includes YOLO26-seg and `yolo26n-seg.pt`; describing it as the implementation family is supported. The locally trained checkpoint, frozen status and experiment-specific output policy remain author-provided implementation facts, outside what a public documentation citation can verify.
- Version caution: the current official model page describes one-to-many prediction as the default and an optional NMS-free one-to-one head. The manuscript identifies its own historical end-to-end configuration rather than making a universal claim about the current default. Preserve that distinction and the recorded package/checkpoint version in reproducibility material.
- Queries executed: `"Ultralytics YOLO26" site:docs.ultralytics.com/models/yolo26/`; `Ultralytics YOLO26 2026 instance segmentation official documentation`; `YOLO26 instance segmentation end to end NMS free prototype masks`.

### [8] Huang et al., Mask Scoring R-CNN

**Status: VERIFIED. Evidence used: L2 official CVF abstract.**

- Matched title: *Mask Scoring R-CNN*.
- Matched authors: Zhaojin Huang; Lichao Huang; Yongchao Gong; Chang Huang; Xinggang Wang.
- Matched venue/year/pages: CVPR 2019, 6409–6418.
- Primary source: [CVF official record](https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.html).
- No DOI is supplied in the manuscript, and the linked official record suffices.
- Claim sites: lines 17, 33 and 305. Classification confidence versus mask quality, and learning mask-quality estimates for ranking, are expressly described by the abstract. The manuscript does not claim priority for a generic quality-aware scoring idea. No correction required.
- Queries executed: `"Mask Scoring R-CNN" site:openaccess.thecvf.com`; `Huang Huang Gong Huang Wang Mask Scoring RCNN 2019`; `mask quality classification confidence mask scoring network IoU`.

### [9] Li et al., Generalized Focal Loss

**Status: VERIFIED. Evidence used: L2 official NeurIPS abstract.**

- Matched title: *Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection*.
- Matched authors: Xiang Li; Wenhai Wang; Lijun Wu; Shuo Chen; Xiaolin Hu; Jun Li; Jinhui Tang; Jian Yang.
- Matched venue/year: *Advances in Neural Information Processing Systems*, volume 33, 2020.
- Primary source: [official proceedings abstract](https://papers.neurips.cc/paper_files/paper/2020/hash/f0bda020d2470f2e74990a07a607ebd9-Abstract.html). The older `papers.nips.cc/paper/2020/hash/…` URL in the manuscript identifies the same proceedings work; using the current canonical URL is optional.
- No DOI is supplied or needed.
- Claim sites: lines 17, 33 and 305. Joint localization-quality/classification representation is directly supported by the abstract. No statement imports its benchmark performance into the current task. No correction required.
- Queries executed: `"Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection"`; `Li Wang Wu Chen Hu Li Tang Yang Generalized Focal Loss 2020 NeurIPS`; `generalized focal loss joint representation localization quality classification`.

### [10] Kirillov et al., PointRend

**Status: VERIFIED. Evidence used: L2 CVF abstract and exposed paper first-page text.**

- Matched title: *PointRend: Image Segmentation as Rendering*. CVF uses title-case `As`; this capitalization difference is not a metadata mismatch.
- Matched authors: Alexander Kirillov; Yuxin Wu; Kaiming He; Ross Girshick.
- Matched venue/year/pages: CVPR 2020, 9799–9808.
- Sources: [CVF official record](https://openaccess.thecvf.com/content_CVPR_2020/html/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.html), [CVF PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.pdf).
- Claim sites: lines 17 and 39. The rendering perspective, point-based predictions and sharper boundaries are supported. The manuscript expressly distinguishes its dense residual head from the complete PointRend algorithm. No correction required.
- Queries executed: `"PointRend: Image Segmentation as Rendering" site:openaccess.thecvf.com`; `Kirillov Wu He Girshick PointRend 2020`; `point based rendering segmentation boundary refinement PointRend`.

### [11] Cheng et al., Boundary IoU

**Status: VERIFIED. Evidence used: L2 official CVF / author-preprint abstract.**

- Matched title: *Boundary IoU: Improving Object-Centric Image Segmentation Evaluation*.
- Matched authors: Bowen Cheng; Ross Girshick; Piotr Dollár; Alexander C. Berg; Alexander Kirillov.
- Matched venue/year/pages: CVPR 2021, 15334–15342.
- Sources: [CVF official record](https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.html), [CVF PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Cheng_Boundary_IoU_Improving_Object-Centric_Image_Segmentation_Evaluation_CVPR_2021_paper.pdf), [author preprint abstract](https://arxiv.org/abs/2103.16562).
- Claim sites: lines 17 and 39. Boundary-focused evaluation is the express topic of the abstract. The paper supports the existence and purpose of the metric; the manuscript's particular pixel radii and image-relative widths remain its own recorded experimental choices. No correction required.
- Queries executed: `"Boundary IoU: Improving Object-Centric Image Segmentation Evaluation"`; `Cheng Girshick Dollar Berg Kirillov Boundary IoU 2021`; `boundary sensitive mask segmentation evaluation boundary intersection union`.

### [12] Guo et al., calibration

**Status: VERIFIED. Evidence used: L2 official PMLR abstract and bibliography.**

- Matched title: *On Calibration of Modern Neural Networks*.
- Matched authors: Chuan Guo; Geoff Pleiss; Yu Sun; Kilian Q. Weinberger.
- Matched venue/year/pages: 34th ICML, PMLR volume 70, 1321–1330, 2017.
- Primary source: [PMLR official record](https://proceedings.mlr.press/v70/guo17a.html).
- Claim sites: lines 17, 35 and 305. The abstract defines the correspondence between probability estimates and correctness likelihood. The manuscript correctly treats a sigmoid, binary-cross-entropy objective and AP improvement as insufficient demonstrations of calibration. Its downstream rejection caveat is a limitation of its own evidence, not a result attributed to Guo et al. No correction required.
- Queries executed: `"On Calibration of Modern Neural Networks" site:proceedings.mlr.press`; `Guo Pleiss Sun Weinberger calibration 2017 ICML`; `probability calibration confidence predicted correctness frequencies neural networks`.

### [13] Lin et al., Microsoft COCO

**Bibliography status: VERIFIED. Citation-use status at original line 148: OVERCLAIM / support mismatch, targeted correction available.**

- Matched title: *Microsoft COCO: Common Objects in Context*.
- Matched ECCV authors: Tsung-Yi Lin; Michael Maire; Serge Belongie; James Hays; Pietro Perona; Deva Ramanan; Piotr Dollár; C. Lawrence Zitnick.
- Matched venue/year/pages: *Computer Vision – ECCV 2014*, 740–755, 2014; LNCS 8693.
- Matched DOI: `10.1007/978-3-319-10602-1_48`.
- Bibliographic primary source: [Springer conference chapter](https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48).
- Additional retrieved source: [author preprint full text](https://arxiv.org/pdf/1405.0312). This resolves to v3 dated 2015 with an expanded author list. That version difference is not a reason to alter the eight-author ECCV citation, which matches the publisher chapter.
- Original citation site, line 148: the clause “the threshold range follows the COCO-style convention [13]” accompanies the exact `0.50:0.05:0.95` range. Springer exposes the dataset abstract, which does not define this range. The retrieved preprint's evaluation section says the detailed evaluation metrics would be added when the server was complete, and searches for `0.95` and `threshold` did not find the modern definition. Therefore this paper alone is insufficient direct support for the precise threshold convention.
- Resolution: add [COCO official Detection Evaluation](https://cocodataset.org/dataset/detection-eval.htm) as [14]. That source explicitly lists ten IoU thresholds and the `iouThrs` parameter. The claim need not be weakened after this citation is added. The manuscript already disclaims byte-for-byte equivalence with other evaluators and should retain that qualification.
- Queries executed: `"Microsoft COCO: Common Objects in Context" site:link.springer.com`; `Lin Maire Belongie Hays Perona Ramanan Dollar Zitnick COCO 2014`; `COCO common objects context average precision mask IoU thresholds .50 .95`.

## Citation-site coverage

| Initial manuscript lines | References | Audit result |
|---|---|---|
| 11 | [1]–[4] | Supported as topic-level summaries; no imported performance claims. |
| 13 | [5]–[7] | Supported instance-segmentation examples. |
| 17 | [8]–[12] | Supported quality-scoring, boundary and calibration context. |
| 27 | [1]–[4] | Supported by retrieved abstracts; task distinctions are appropriately bounded. |
| 29 | [5]–[7] | Supported model-family context; local checkpoint facts require local records. |
| 33 | [8], [9] | Abstracts directly support the stated concepts. |
| 35 | [12] | Calibration characterization and the absence of calibration evidence are appropriate. |
| 39 | [10], [11] | Supported conceptual descriptions; no claim of implementing PointRend. |
| 148 | [13] | Add official evaluation documentation for the exact threshold convention. |
| 305 | [8], [9], [12] | Supported precedent and calibration caveat. |

The citations and reference list are bidirectionally consistent in the inspected version: every [1]–[13] entry has at least one prose occurrence, and no prose citation points outside the list. No other cited sentence was found to exceed the abstract/documentation evidence actually retrieved.

## Scope of the conclusion

This audit establishes reference identity and the adequacy of the cited background statements. It does not certify the accuracy of the source papers, validate the wolfberry dataset or experimental metrics, or assert that all full papers were read. After the COCO documentation citation is added, there is a concrete resolution for the only substantive citation-support issue identified in the initial version.

## Targeted re-verification after the author's correction

Before delivery of this audit, the English manuscript was read again at the affected occurrence and bibliography. Line 148 now cites `[13,14]` for the official COCO evaluation convention, and line 397 adds COCO Consortium, *Detection Evaluation*, with the official route `https://cocodataset.org/#detection-eval`. The route is the official site's navigation entry; its directly retrievable content is `https://cocodataset.org/dataset/detection-eval.htm`, successfully read in this audit. The stated ten-threshold convention is supported there.

Additional verification queries executed for the new documentation entry: `"Detection Evaluation" site:cocodataset.org`; `COCO Consortium evaluation IoU 2026 official`; `COCO ten IoU thresholds .50 .05 .95 evaluation`.

**Final citation status: [1]–[14] VERIFIED for the English manuscript's current uses. The original [13] support mismatch is resolved by [14]. No unresolved citation problem remains.** This targeted recheck does not claim an independent re-audit of the Chinese manuscript or any later manuscript modifications.
