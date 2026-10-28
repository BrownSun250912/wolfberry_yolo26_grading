"""Build a source-backed methods handoff. Never trains or performs inference.

Input audit JSON was collected by the two audit_materials_* scripts. Source
files are copied byte for byte; derivative summaries explicitly name their
sources. This is a writing resource, not a new experiment or manuscript.
"""
from pathlib import Path
import csv, hashlib, json, shutil, re, textwrap
from collections import Counter
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs/reports/materials_audit_20261028'
PACK = ROOT / 'paper/第二章写作素材包_20261028'
FOLDERS = ['00_阅读说明与素材完整性清单','01_数据采集与数据集构建','02_分级依据_标注与预处理','03_基础模型','04_改进模块与整体方法','05_训练环境与训练设置','06_评价指标与协议','07_图片_表格与图注','08_证据索引与待补充问题']
for folder in FOLDERS: (PACK/folder).mkdir(parents=True, exist_ok=True)

def load(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def md(folder, name, content):
    p=PACK/folder/name;p.parent.mkdir(parents=True, exist_ok=True);p.write_text(textwrap.dedent(content).strip()+'\n',encoding='utf-8')
def table(folder,name,rows,columns=None):
    rows=list(rows); p=PACK/folder/name
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns or list(rows[0]));w.writeheader();w.writerows(rows)
    return p

D=load(AUDIT/'data_facts.json'); A=load(AUDIT/'additional_facts.json'); W=load(AUDIT/'weights_metadata.json'); T=load(AUDIT/'training_records.json')
E={}; evidence=[]
def source(path, purpose='', copy=True):
    p=Path(path);p=p if p.is_absolute() else ROOT/p
    rel=p.relative_to(ROOT).as_posix()
    if rel in E:return E[rel]
    eid=f'E{len(evidence)+1:03d}';E[rel]=eid
    row={'证据ID':eid,'项目相对路径':rel,'用途':purpose,'字节数':p.stat().st_size if p.is_file() else '', 'SHA256':sha(p) if p.is_file() else '', '素材包内副本':'','核验状态':'已读取文件或元数据；待作者确认'}
    if not p.is_file():raise FileNotFoundError(p)
    if copy:
        dest=PACK/FOLDERS[8]/'关键证据副本'/rel
        if dest.suffix=='.md':
            # Archival source prose may have links to deliberately unbundled
            # data. Preserve its bytes as a text snapshot, not a new landing page.
            old=dest;dest=dest.with_name(dest.name+'.txt')
            if old.exists():old.unlink()
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        row['素材包内副本']=dest.relative_to(PACK).as_posix()
    evidence.append(row);return eid

SCRIPTS = [
 'build_dataset.ps1','build_four_class_dataset.py','build_post_review_candidate_manifest.py','build_r1_multitask_targets.py',
 'finalize_manual_grading_standard.py','build_size_priority_package.py','build_grading_scheme_candidates.py','run_r0_fixed_prediction_replay.py',
 'extract_features.py','train_frozen_offlineaug_e200.py','report_frozen_offlineaug_e200.py','train_r1a_hierarchical_grader.py',
 'r1_research_runtime.py','r1_qualified_runtime.py','validate_r1_closure_runtime.py','refine_rule_runtime.py','refine_rule_reliability.py',
 'refine_rule_anchored_residual.py','refine_score_residual_ablation.py','refine_residual_runtime.py',
 'joint_pipeline_runtime_20261024.py','joint_pipeline_policy_20261024.py','joint_pipeline_capture_20261024.py',
 'joint_pipeline_rank_fit_20261024.py','joint_pipeline_evaluate_20261024.py',
 'attribute_local_runtime_20261025.py','attribute_local_rank_20261025.py','attribute_local_rank_controls_20261025.py',
 'attribute_local_cache_20261025.py','attribute_local_proto_cache_20261025.py','attribute_local_proto_fit_20261025.py',
 'attribute_local_proto_evaluate_20261025.py','attribute_local_rank_latency_20261025.py',
 'direction_native_inference_20261023.py','direction_experiments_20261023.py',
 'analyze_val_instance_errors.py','analyze_r1_ap_bootstrap.py','final_test_preflight_20261027.py',
 'final_test_evaluation_20261027.py','analyze_final_test_20261027.py',
 'audit_materials_sources_20261028.py','audit_materials_additional_20261028.py',
]
for s in SCRIPTS:source('scripts/'+s,'实现/训练/评价/本轮只读核验脚本')
EXTRA=[
 'configs/grading_four_class_v3_local.yaml','configs/grading_four_class_v3_augmented_delivery_local.yaml',
 'configs/feature_extraction_frozen_v1_train.yaml','configs/feature_extraction_frozen_v1_val.yaml',
 'docs/FEATURE_DEFINITIONS.md','docs/DATASET_AUDIT.md','environment/ENVIRONMENT_REPORT.md',
 'docs/YOLO26_OFFLINEAUG_E200_FROZEN_PROTOCOL_20260918.md','docs/YOLO26_OFFLINEAUG_E200_RESULTS_20260918.md',
 'data/grading_four_class_v3/audit/build_summary.json','data/grading_four_class_v3/audit/image_audit.csv',
 'data/grading_four_class_v3/audit/exact_duplicate_polygons.csv','data/grading_four_class_v3/audit/test_polygon_deduplication.json',
 'data/grading_four_class_v3_valfix_20261006/version.json',
 'data/grading_four_class_v3_augmented_delivery/metadata/segmentation_augmentation_config.json',
 'review/three_person_audit_v1/README_分工总览.md',
 'review/three_person_audit_v1/04_审核结果汇总/README.md',
 'review/three_person_audit_v1/04_审核结果汇总/audit_summary.json',
 'models/experiments/r1_quality_followup_20261009/head_image_split.csv',
 'outputs/strict_rot_exclusion_v2/exclusion_summary.json',
 'outputs/grading_standard_manual_v4_final/grading_standard_config.json',
 'outputs/reports/attribute_local_20261025/FROZEN_CANDIDATES_AND_TEST_PROTOCOL.json',
 'outputs/reports/attribute_local_20261025/runtime_environment.json',
 'outputs/reports/attribute_local_20261025/rank_latency_status.json',
 'outputs/reports/attribute_local_20261025/rank_latency_summary.csv',
 'outputs/reports/attribute_local_20261025/rank_latency_samples.csv',
 'outputs/reports/attribute_local_20261025/attribute_rank_control_ledger.csv',
 'outputs/reports/attribute_local_20261025/prototype_fit_ledger.csv',
 'outputs/reports/final_test_preflight_20261027/test_analysis/analysis_plan.json',
 'outputs/reports/final_test_preflight_20261027/test_analysis/analysis_status.json',
 'models/experiments/attribute_local_20261025/prototype_population.csv',
]
for rel in EXTRA:
    if (ROOT/rel).is_file():source(rel,'配置、审核、运行记录')
for rel in ['data/manifests/dataset_manifest.csv',
 'data/grading_four_class_v3_valfix_20261006/audit/corrections.json',
 'data/grading_four_class_v3_valfix_20261006/audit/r1_multitask_targets.csv',
 'outputs/grading_standard_manual_v4_final/grading_labels_candidate_v3.csv',
 'outputs/strict_rot_exclusion_v2/features_with_manual_rot.csv',
 'data/grading_four_class_v3_augmented_delivery/metadata/segmentation_augmentation_manifest.csv',
 'review/three_person_audit_v1/04_审核结果汇总/test_final_labels_deduplicated.json']:
    source(rel,'仅登记定位与散列；较大数据/人员备注不复制',copy=False)
source('outputs/grading_standard_manual_v4_final/audit/boundary_review_audit.csv','边界标定原审核；派生联接表去除自由备注',False)
for run, rec in T.items():
    for fn in ['args.yaml','results.csv','mask50_selection_trace.csv','training_config.json','training_summary.json','completion.json','history.csv']:
        if (ROOT/run/fn).is_file():source(run+'/'+fn,'逐阶段原始训练记录')
for name,item in W.items():source(item['path'],'检查点元数据；不复制权重',False)

def ev(path,marker=None):
    rel=path.replace('\\','/');eid=E.get(rel) or source(rel,'补充实现证据')
    suffix=''
    if marker:
        lines=(ROOT/rel).read_text(encoding='utf-8-sig').splitlines(); hits=[i+1 for i,l in enumerate(lines) if marker in l]
        if not hits:raise ValueError(f'Anchor missing {rel}: {marker}')
        suffix=f':L{hits[0]}（`{marker}`）'
    return f'[{eid}] `{rel}`{suffix}'

# Derived tables. No checkpoint is reloaded by this package builder.
F0,F1,F2,F3,F4,F5,F6,F7,F8=FOLDERS
stages=[
 ['原始登记图像/单类多边形',1285,5752,'train1028/4593; val128/577; test129/582','单类wolfberry，class0不代表不合格'],
 ['HOLD隔离',74,374,'73数量不一致；1 JSON/TXT几何不一致','从正式导出排除；原文件未删除'],
 ['保留图内精确重复多边形剔除',0,7,'train1；test6','wb_000371及五张test图'],
 ['V3正式原图数据',1211,5371,'train966/4277; val116/518; test129/576','5371=5752−374−1−6'],
 ['当前验证标签覆盖层',116,518,'仅改7个类别/果梗状态；坐标、图像、实例数不变','valfix_20261006，仅val标签覆盖'],
 ['检测器离线增强训练输入',2898,12831,'966原图+1932衍生图','非2898个独立实物/采集事件'],
 ['检测器离线增强包val',116,518,'未增强；原V3等级','训练期选模不采用后来的valfix标签'],
]
table(F1,'数据数量对账.csv',[dict(zip(['阶段','图像数','实例数','划分或变化','说明'],x)) for x in stages])
counts=[]
for version,splits in D['label_counts_actual'].items():
    for split,r in splits.items():counts.append({'版本':version,'集合':split,'图像数':r['images'],'实例数':r['instances'],**{f'class{k}':r['class_counts'][str(k)] for k in range(4)}})
for split,r in D['augmented_actual'].items():counts.append({'版本':'augmented_delivery','集合':split,'图像数':r['images'],'实例数':r['instances'],**{f'class{k}':r['class_counts'].get(str(k),r['class_counts'].get(k,0)) for k in range(4)}})
table(F2,'数据划分与各类实例统计.csv',counts)
shutil.copyfile(AUDIT/'raw_source_inventory.csv',PACK/F1/'原始文件逐图清单_SHA256.csv')
paramcounts={}
for k,v in W.items():
    if 'parameters' in v:paramcounts[k]=v['parameters']
    else:
        shapes=v.get('model_state_shapes',v.get('state_shapes',v.get('head_state_shapes',{})))
        paramcounts[k]=sum(int(np.prod(shape)) for name,shape in shapes.items() if name.endswith(('.weight','.bias')))
table(F3,'权重清单与元数据.csv',[{'组件':k,'项目相对路径':v['path'].replace('\\','/').split('/wolfberry_yolo26_cv/')[-1], 'SHA256':v['sha256'],'字节数':v['bytes'],'检查点epoch字段':v.get('epoch','未记录'),'网络参数量':paramcounts[k],'state张量元素数_含buffer_非参数量':v.get('model_state_elements',v.get('state_elements',v.get('head_state_elements','未单列'))),'角色':'最终主方案' if k in ['detector','attribute_encoder','RR22_residual','old_defect_rank'] else '可选输出细化' if k=='prototype_output_refiner' else '初始化' if k=='pretrained' else '运行兼容依赖/旧分支'} for k,v in W.items()])
assert paramcounts['attribute_encoder']==3067203 and paramcounts['RR22_residual']==8257
for filename in ['weights_metadata.json','training_records.json','detector_layers.json','current_hardware_software.json','additional_facts.json']:
    # additional_facts only contains project paths and aggregate counts, no reviewers.
    shutil.copyfile(AUDIT/filename,PACK/F8/filename)
safe_facts={k:v for k,v in D.items() if k not in ['val_overlay_corrections']}
(PACK/F8/'data_facts_不含人员备注.json').write_text(json.dumps(safe_facts,ensure_ascii=False,indent=2),encoding='utf-8')

acquisition=[
 ('来源属性','本机new/下1285张PNG及同名JSON，labels/下同名单类TXT；自行采集/公开下载/混合未找到记录','需要项目执行者补充原始来源、授权/公开地址'),
 ('品种、鲜/干状态、产地、批次','未找到记录；颜色和外观不能代替溯源凭证','需要项目执行者补充'),
 ('采集地点、时间、环境','未找到可靠采集记录；文件写入时间不等于拍摄时间','需要项目执行者补充'),
 ('设备与镜头','未找到记录；所有1285张PNG无EXIF且无PNG info字段；_Color后缀不能证明设备型号','需要项目执行者补充'),
 ('原始分辨率','逐图实读均1280×720像素','本次raw_source_inventory.csv'),
 ('距离、角度、照明、曝光、白平衡','未找到定量记录；不能以统一拍摄的常见设置代填','需要项目执行者补充'),
 ('背景与摆放','随包4例可见白色圆形容器/托盘、深色外部背景，多果、不同朝向；部分触碰及工具/手遮挡','只对所展示样本直接观察；非全库场景频数'),
 ('物理独立性','1285个不同图像文件；SHA256未见完全重复或跨集合完全重复；未找到实物果ID/连拍序列','不能宣称1285次独立采集或果实无重复'),
 ('来源组','按原始编号001–150、151–500、501–900、901–1300分4组；149/349/390/397张','编号代理分组，不等同经验证采集批次'),
 ('物理标定','calibration.px_per_mm=null；配置写明原果已丢弃、无同平面物理参照；保留px/px²','不能换算毫米或声称满足商品标准的实际粒径'),
]
table(F1,'采集条件与缺口.csv',[{'项目':a,'实证或结论':b,'补充责任或限制':c} for a,b,c in acquisition])

md(F1,'数据采集与构建_逐项核验.md',f'''
# 第一部分：数据采集与数据集构建

这是材料事实清单，不是论文正文。术语：**实例**指一颗被单独标注的果实；**掩膜**指该实例在图像中的像素区域；**GT**指用来训练或评价的人工/规则参考标签，不等于经物理测量证实的商品品质真值。

## A. 来源与采集条件

可直接使用[采集条件表](采集条件与缺口.csv)。原始路径由项目清单明确关联为`../new/NNN_Color.png`、同名JSON及`../labels/NNN_Color.txt`；本轮实际读取了全部1285组文件，不涉及其他个人目录。{ev('data/manifests/dataset_manifest.csv')}；本包[逐图散列清单](原始文件逐图清单_SHA256.csv)。

原始JSON均为`wolfberry`单类多边形，5752个实例。JSON的`version`字段分布为4.0.0-beta.13（50图）、4.0.2（789图）、2.1.0（68图）、4.0.0-beta.9（378图）；这是文件中记录的格式/工具版本字段，不能据此确定每台标注电脑的安装版本。原始TXT的class0表示枸杞；正式四分类数据的class0表示不合格，二者不可混用。

采集方式、品种、鲜干状态、产地、相机、光源、距离、拍摄日期和地点均**未找到记录**，**需要项目执行者补充**。没有设备元数据可用。样本图只能证明其可见背景、摆放和遮挡现象，不能推出整个数据集都采用同一拍摄条件。见本部分表及第07目录图注。

## B. 原始、清洗与最终数量

[完整数量对账](数据数量对账.csv)：1285张/5752个原始多边形 → 74张HOLD（隔离，未删除）含374个多边形 → 另剔除保留图内7条精确重复多边形 → **1211张/5371个实例**。

74张HOLD的原因：73张`original_polygon_count_differs_from_graded_instances`，1张`json_txt_geometry_mismatch`。采用整图完整实例可映射才导出的策略，不能把HOLD中的未分级实例直接当负样本。{ev('data/grading_four_class_v3/audit/build_summary.json')}；{ev('scripts/build_four_class_dataset.py')}。

train重复1条在`wb_000371`；test重复6条在`wb_000240`（2条）、331、356、370、415（各1条）。原始同框重复记录35条与正式剔除7条不是同一集合口径，不能互换。正式5371条标签本轮未见精确同框重复。证据：{ev('data/grading_four_class_v3/audit/exact_duplicate_polygons.csv')}；{ev('data/grading_four_class_v3/audit/test_polygon_deduplication.json')}；第08目录`additional_facts.json`。

原始编号1–1300中未出现145、408、766–775、1070、1174、1265，共15个号码。没有记录证明这些是因何删除的图像，不能据编号断言“原有1300张，清洗删除15张”。当前可核实原始输入就是1285张。

另有预测掩膜特征分析分支：冻结特征实际5088条（train4523、val565），人工腐烂复核表2278条，其中与特征表匹配2268条；匹配后ROT445、NOT_ROT1823、未入复核队列2820，排除445条后留下4643条。它们是**预测实例特征记录**，不是人工多边形5371条的另一种计数；未入队列也不是人工确认健康。{ev('outputs/strict_rot_exclusion_v2/exclusion_summary.json')}。旧5108、447和1831等数字必须带各自“上游/QC前/复核表”口径，不能取代当前匹配计数。

## C. 类别统计及图像独立性

四类为0不合格、1一级、2二级、3三级。train966图/4277实例（372/945/1138/1822）；test129图/576实例（25/126/167/258）。val原V3为116图/518实例（54/115/151/198）；当前评价valfix为54/109/150/205。见第02目录CSV。总类别数在当前train+valfix+test口径为451/1180/1455/2285。

所有1285张PNG按原始字节SHA256核验，无完全重复组、无跨集合完全相同文件。**这不排除同果重复拍摄、近重复视角或连续帧。**清单没有实物果ID和真实批次ID，不能证明5371个标签对应5371颗独立实物。4个source_group只是编号代理。

## D. 原始与衍生图像、标定及可用图

正式1211张是从原始输入继承的整图，未找到它们是离线裁剪合成图的记录；R1的实例ROI裁剪、离线增强1932张、合成水槽压力图属于衍生资料，不能计入新增独立采集图。增强仅来自966张train，val和test无离线增强。正式训练输入为2898图/12831实例而不是2898个独立样本。

像素到长度标定**未完成**：{ev('configs/feature_extraction_frozen_v1_val.yaml','px_per_mm: null')}。轴长单位px，面积px²；尺寸分数是样本相对排序，无毫米单位，不支持内部成分、含水率、糖度、药用成分等质量结论。

随包原始样本、原标注叠图和现成增强对比均在[第07目录](../07_图片_表格与图注/图像说明与中文图注.md)。示例覆盖4个编号来源组；不是随机抽样统计，不能据图计算全库遮挡率或背景复杂度。

## E. 缺失信息

需要执行者补充来源与采集日志、物种/品种与鲜干状态、采集地点日期、设备/镜头及光照设置、实物果是否重复、遮挡目标纳入标准、原始编号缺号原因、数据使用授权。没有这些信息也可先写可核实的数据组织与处理过程，但不能完成可审查的采集条件段落。
''')

md(F2,'分级_标注_预处理_逐项核验.md',f'''
# 第二部分：分级依据、数据标注与预处理

## A. 分级依据与优先级

当前标签顺序是`0=UNQUALIFIED,1=GRADE_1,2=GRADE_2,3=GRADE_3`。采用项目自定义、人工外观审核与冻结相对尺寸规则结合的**操作性等级**；没有找到证明其等同国家/行业鲜果商业分级标准的独立实测记录。配置本身`status=candidate_only_not_final_commercial_truth`。不能把规则名中的final写成商业真值认证。{ev('outputs/grading_standard_manual_v4_final/grading_standard_config.json')}。

| 条件（按先后） | 参考标签/规则 | 推理时的实际判断 | 注意事项 |
|---|---|---|---|
| 腐烂/明显不合格 | 人工ROT复核、后续类别审核 | 最终主方案保留检测器原始class0，不以新缺陷头重建/恢复类别 | R1缺陷头用于排序证据，不是最终缺陷门控替代器 |
| 其余果实缺果梗 | 人工复核或冻结果梗规则来源；部分人工改标 | 缺梗概率与邻域等级证据融合后缺梗→三级 | 缺梗大果也为三级，等级不是纯大小单调排序 |
| 其余可评分有梗果 | 冻结尺寸/形态分数及人工边界意见 | 修正分数与LOW/HIGH、边界邻域证据判1/2/3 | GT存在人工边界覆盖，不是所有标签都能由公式唯一复算 |

ECDF（经验累积分布函数，表示在冻结参考样本中的相对位置）的中位秩定义：

`p_f(x)=(#{{r<x}}+0.5·#{{r=x}})/N_ref`，`N_ref=2649`；`r`是特征f的参考值。

`T_AR=clip(1−2|p_aspect_ratio−0.5|,0,1)`；`H=100(0.50 p_solidity+0.30 T_AR+0.20 p_circularity)`。

`S_rule=100(0.70 p_area+0.10 p_minor_axis)+0.20 H`，分数0–100，无物理单位。

`LOW=37.40298225745565`，`HIGH=65.62684031710077`；在无缺陷且有梗条件下，S≥HIGH一级，LOW≤S<HIGH二级，S<LOW三级。缺梗强制三级、缺陷优先不合格。推理的残差与邻域修正详第04部分。

权重确定：`finalize_manual_grading_standard.py::choose_weights`对area 0.40–0.90步长0.05、minor 0.05–0.40步长0.05及shape余量0.05–0.30搜索，以两边界平均平衡准确率、平均准确率及偏好尺寸权重的顺序决胜。阈值遍历相邻唯一分数中点与边缘候选，以边界人工意见评价。**60条边界审核中排除12条缺梗、4条测量错误，44条有效；实际41条train、3条val。**因此不能声称边界标定完全train-only。没有test参与此阈值拟合的记录。{ev('scripts/finalize_manual_grading_standard.py','def choose_weights')}；第08目录`additional_facts.json`给出逐清单联接后的计数。

参考分布原设计取训练特征；在线实现因原冻结参考文件缺失，从历史候选表的特征/ECDF锚点恢复N=2649的参考数组，**并非重新用当前train/val/test拟合ECDF**。恢复约束与检查见{ev('scripts/run_r0_fixed_prediction_replay.py','DEFAULT_REFERENCE_SIZE')}。这是历史恢复实现，需要如实说明，不能写成“保留了原始完整标定实物”。

18项历史特征包括长轴、短轴、面积、长宽比、圆度、实心度、偏心率，Lab六项均值/标准差、HSV饱和度/亮度均值、暗损伤比、异常颜色比、最大损伤连通区比。最终规则只用面积、短轴、实心度、长宽比、圆度；颜色权重0。记录的动机为照明变化和人工未以颜色区分等级，不能据此补造统一照明实验。定义、像素口径、回退见{ev('docs/FEATURE_DEFINITIONS.md')}及配置副本。没有糖度、水分、营养/药理等独立测量，不能称内在品质预测。

## B. 标注规范、人员与版本

1. 原始JSON是LabelMe/X-AnyLabeling兼容`polygon`，TXT为YOLO分割格式：每行`class x1 y1 x2 y2 ...`，坐标对图像宽高归一化；没有单独人工框标签，检测框可由多边形外接矩形得到。不能把自动外接框当成另一轮框标注。{ev('docs/DATASET_AUDIT.md')}。
2. 果实多边形是实例可见轮廓标注；是否包含完整果梗、是否补全不可见区域、触碰/模糊/工具遮挡时的统一规则**未找到正式规范**，需要原标注者补充。没有果体/果梗语义分区的独立像素标签。单独的stem属性也不是果梗分割掩膜。
3. 腐烂审核2278条表记录（ROT447/NOT_ROT1831），与当前特征匹配2268条（445/1823）；不在队列2820条不能当人工NOT_ROT。果梗审核和边界人工意见用于候选等级；候选实例与原始多边形几何映射、整图一致性审查后才导出V3。{ev('scripts/build_post_review_candidate_manifest.py')}；{ev('scripts/build_four_class_dataset.py')}。
4. 三人分工目录记载主审果梗280、几何10，两名复核者各果梗325、几何13，并各承担test129图/582条初始实例审核。角色可写主审/A/B；具体谁操作、专业背景、盲法、独立性和分歧仲裁准则**需要项目执行者补充**，不能仅凭分工表推断双盲。测试初始582条有303一致/279不一致记录，去重后576条的实际来源：302双人一致、51主审决定、223按冻结基础规则回退。因此不能称test所有标签均双人一致确认。
5. 版本：原始单类 → 四类V3（含HOLD及去重） → valfix_20261006仅val覆盖7条类别/果梗修订（6条一级→三级、1条二级→三级），由1名审核者作出，无第二审核者。几何、图像、split和总实例数不变。检测器及原R1训练早于覆盖层；最终评价使用覆盖层。{ev('data/grading_four_class_v3_valfix_20261006/version.json')}。未复制人员备注，只登记源文件散列。
6. R1 train+val共4795条目标，defect_available全4795，stem/score_available各4369，不合格426条缺梗/分数目标不可用；0占位值必须与availability掩码合用，训练损失只统计可用项。目标的连续分数沿用历史预测掩膜测量/规则，而训练图像ROI后来由人工GT掩膜裁剪；不是逐颗人工实际测长。{ev('scripts/build_r1_multitask_targets.py')}。

## C. 预处理与增强

| 阶段 | 实际操作与参数 | 使用范围/证据 |
|---|---|---|
| 检测器训练 | 640×640 letterbox；颜色/张量处理由Ultralytics；图像转float并/255；mask_ratio=4、overlap_mask=True | train2898；原V3 val116；`train_frozen_offlineaug_e200.py`及args.yaml |
| 检测器正式推理 | imgsz=640为最大边请求，默认rect=True；本次129个test缓存实际网络输入384×640，输出恢复1280×720 | 不能把imgsz640写成每次必为640²；缓存input_hw核验 |
| 原图掩膜 | public polygon →四舍五入像素坐标并裁剪→cv2.fillPoly，图像原尺寸 | `analyze_val_instance_errors.py::polygon_to_mask` |
| R1裁剪 | 外接框上下文padding=max(12,ceil(0.25×长边))；RGB和mask同步，letterbox160²，RGB填114、mask填0 | 推理原预测mask；训练人工GT mask |
| R1归一化 | BGR→RGB，RGB/255，均值[.485,.456,.406]、std[.229,.224,.225]；拼mask为4通道；6维几何以train均值/标准差标准化 | `train_r1a_hierarchical_grader.py`及`r1_research_runtime.py` |
| R1在线增强 | 水平/垂直翻转各0.5，图像和mask同步；对比度U(.85,1.15)，加性亮度U(−12,12) | 仅训练crop；不对val/test随机增强 |
| 规则几何 | 最小实例80px、连通区20px、保留最大连通区；白边修剪深度≤8px、HSV S≤.35/V≥.45、Lab色度≤18，白区≥80px，保留≥200px、最多删除45% | 原掩膜派生几何测量mask，不覆盖最终分割输出；详配置 |

离线增强每张train固定2份，seed=20260914，原图966→train2898；实例4277→12831。val116不增强，test不在增强交付包。各图各版本由全局seed、文件名、增强序号SHA256派生独立随机seed；清单保留实际参数。增强不会生成新的独立果实。

| 参数 | aug01 | aug02 |
|---|---|---|
| 几何 | 必做水平翻转，x→1−x并反转顶点顺序 | 不翻转、不旋转、不裁剪 |
| 亮度因子 | 等概率U(.96,.995)或U(1.005,1.04) | 等概率U(.95,.995)或U(1.005,1.05) |
| 对比度 | U(.96,1.04) | U(.95,1.05) |
| gamma | U(.98,1.02) | U(.97,1.03) |
| Gaussian噪声 | 7.5%，σ=U(.25,1.5)/255 | 15%，σ=U(.25,2)/255 |
| Gaussian模糊 | 7.5%，radius=U(.1,.4) | 10%，radius=U(.1,.5) |
| 噪声/模糊关系 | 互斥；85%两者均无 | 独立，可同时出现 |

顺序为Pillow亮度、对比度 → 浮点/255 → `I^(1/gamma)` → 加噪声、clip/round到uint8 → GaussianBlur。最终检测器训练关闭全部在线几何/HSV/混合增强（degrees、translate、scale、shear、perspective、flip、mosaic、mixup、cutmix、copy_paste、erasing、bgr等均0），不能照抄库默认增强。{ev('data/grading_four_class_v3_augmented_delivery/metadata/segmentation_augmentation_config.json')}。

## D. 数据划分、内部拟合与test历史

源划分脚本seed=20260823，按source_group×实例数档（1–3、4–6、7+）分层，使用System.Random在层内打乱，以80/10/10目标划分；7+拥挤图10张另平衡为8/1/1。原始1028/128/129，V3继承图像归属后排除HOLD为966/116/129。**划分单位是图像，没有实物果/真实批次约束。**同原图增强始终在train，无衍生图跨集合；未证明同果不同照片不跨集合。{ev('scripts/build_dataset.ps1')}。

新增头沿用966张train内部的773 head_fit/193 head_holdout图像角色。RR22残差3053/761条可评分匹配实例；排序8764/2226条非零最终类预测（含低置信/未匹配预测）；prototype3349/837条匹配mask，共4186条。实例数不同因为纳入条件不同，不是重新划分原始全集。原检测器/R1早已训练过整个train，因此该held只对**新增头拟合和选checkpoint**独立，不能称完整系统外部独立验证。{ev('models/experiments/r1_quality_followup_20261009/head_image_split.csv')}。

val经历大量自适应方案选择与标签审核，是开发集。test早期已有基线使用，2026-10-27在冻结候选后完成一次统一终测；不能声称test自项目开始从未使用，也不能把终测后检查当成新一轮选模。第06部分记录评价协议。当前素材整理只读缓存，不重跑test、不生成新训练。

## E–F. 图示、清单和脚本

见第07的实际标注及原图/aug01/aug02对比。完整标签逐文件SHA、图像归属和当前计数在本包CSV；关键脚本/配置副本、原始路径与散列在第08证据索引。全数据与权重不随包复制；如需独立重跑，另行提供完整数据和环境，不能仅凭这个素材包承诺一键训练。
''')

md(F3,'基础模型_结构与证据.md',f'''
# 第三部分：基础模型

## A. 身份、来源和选择理由

| 项目 | 实际信息 | 证据 |
|---|---|---|
| 名称 | YOLO26n-seg，n规模，Ultralytics SegmentationModel，四类实例分割 | best.pt反序列化元数据、安装库8.4.117 |
| 结构 | n=[depth .5,width .25,max_channels1024]；Conv/C3k2/SPPF/C2PSA，三尺度融合，Segment26 | `detector_layers.json`、checkpoint yaml |
| 模型规模 | 3,054,420参数；头nm=32/npr=64、nc=4、stride8/16/32、end2end=True、reg_max=1 | 检查点实际属性；不以其他YOLO版本参数代填 |
| 初始化 | `models/pretrained/yolo26n-seg.pt`，nc80、3,126,280参数；SHA见CSV | 本地通用预训练模型，不是已在本项目训练的best |
| 来源证明边界 | 项目训练记录称官方Ultralytics权重；精确下载URL/时间及完整预训练数据经历未找到记录 | 不补造COCO预训练轮数或下载日期 |
| 项目训练 | 正式离线增强200轮，结构不变，通常训练权重均优化，无最终主干冻结开关 | args.yaml `freeze`及训练脚本；后处理阶段整体冻结 |
| 选用原因 | 项目记录明确将其作为轻量固定结构基线，并避免在已大量复用的val上继续结构追优 | 冻结训练协议；“边缘实时部署已验证”未找到记录 |

轻量、单阶段、同时提供框与mask可作为设计考虑；没有找到采购/部署约束或与所有同代模型公平比较后选型的证据，不能写成已证明其最适合枸杞。

## B. 可以据此绘制的原模型结构

输入3通道RGB训练640²；正式推理保持比例letterbox（已保存test实际384×640）。骨干layer0–10；neck11–22；head23。详[逐层元数据](../08_证据索引与待补充问题/detector_layers.json)。

| layer | 连接/模块 | 作用 |
|---|---|---|
| 0–4 | Conv、Conv、C3k2、Conv、C3k2 | 逐级下采样与局部特征 |
| 5–10 | Conv、C3k2、Conv、C3k2、SPPF、C2PSA | 深层/多尺度上下文 |
| 11–16 | Upsample→Concat(layer6)→C3k2；Upsample→Concat(layer4)→C3k2 | 融合高分辨率细节 |
| 17–22 | Conv→Concat(layer13)→C3k2；Conv→Concat(layer10)→C3k2 | 再下采样融合 |
| 23 | Segment26输入[16,19,22] | 三尺度box、class、mask系数及prototype |

Prototype（掩膜原型）是共享的像素特征基底，每个候选以系数线性组合成软mask；不是为每颗果人工另标的32通道标签。end2end（端到端检测头）在此模型预测阶段采用top-k候选输出，后续重分类仍可能出现同果同类重复；不能仅凭模型名声称无需任何后处理。

## C. 原模型实际问题及后续连接

| 已记录问题 | 可核实线索 | 设计改进方向/身份 |
|---|---|---|
| 定位成功但等级错误，缺梗大果与普通大小等级混合 | val固定匹配原始YOLO416/500正确的历史实例记录；数字仅作定位问题示例 | 外接R1分层属性+尺寸评分；不是更换YOLO分类头 |
| 重分类后原不同类近重合预测成为同类竞争，单纯将confidence置低未物理删除 | `joint_pipeline_policy`真实keep-index NMS与旧输出审计 | 原raw≥.25优先、同最终类别boxIoU≥.7真实删除 |
| 阈值附近分数误差、局部类别证据未利用 | 边界/邻域矩阵实验脚本与错误审计 | 有界残差、局部证据边界融合；作用属设计动机，不推断新效果 |
| 重分类后原类别置信度不反映新类/分割质量 | 排序器不同质量目标及old-defect对照 | 新增22维质量排序器，复用原缺陷信息 |
| mask边界误差可能影响高IoU分割 | prototype拟合与固定剂量对照 | 可选输出边界细化，原mask仍用于分级/排序 |

上述观察来自实际代码/实验而非通用YOLO缺点。未将所有重复框当成同实物已证实，也未将mask细化等同糖度/内在品质改善。

## D. 权重链和最终冻结范围

见[权重清单](权重清单与元数据.csv)（9项，完整SHA256）。最终基线best路径`models/experiments/yolo26n_seg_offlineaug_e200_mask50_seed20260915/weights/best.pt`，SHA`{W['detector']['sha256']}`，完整训练200轮但选中**第16轮**。best已strip，内部epoch=−1不代表训练−1轮。R1独立编码器路径名称含e200，但所用最佳是**第131轮**，checkpoint config仍保留150阶段参数；以实际训练summary和epoch为准。

后续主方案未改YOLO骨干、neck或分割head；YOLO一次forward（前向计算）输出候选、mask、可选prototype，进入外接冻结R1和小型新增头。Strip/DEG/Partial、neck微调、Model Soup属于探索，没有被选入最终主方法。推理所有权重eval冻结；训练新增头不反传进YOLO或R1。

主要实现证据：{ev('scripts/train_frozen_offlineaug_e200.py')}；{ev('scripts/attribute_local_runtime_20261025.py','class RankedPipeline')}；{ev('scripts/joint_pipeline_capture_20261024.py','class CandidateCapture')}。
''')

MODULES = [
 dict(id='M0',name='冻结YOLO26n-seg候选生成',status='原有，最终主方案',
 problem='提供整图多果定位、四类初判和实例掩膜；最终不通过改骨干实现分级改进。',
 input='一张BGR uint8 H×W×3图像，内部转换/letterbox；imgsz640，test缓存实际384×640。',
 process='检测器eval/FP32，conf=.001、max_det=300、retina_masks=True、iou=.7；top-k公共输出的box/class/conf与多边形依次保留。输出多边形栅格化为原生图尺寸M_i。',
 output='N行boxes(N×4)、baseline(N)、raw confidence(N)、原掩膜列表N；可选同次forward的soft logits、32通道prototype和candidate index。',
 fit='历史200轮四类实例分割；后续所有模块训练与最终推理期间完全冻结。',
 parameters='正式训练best16，权重与sha在第03目录；阈值由冻结协议统一，非按单张test调节。',
 fallback='无预测返回空数组；候选/box/mask行数或索引谱系不一致抛错，不猜测匹配。',
 code='scripts/attribute_local_runtime_20261025.py',anchor='self.options=dict',
 weight=W['detector']['path']),
 dict(id='M1',name='独立R1多任务属性编码器与分层证据',status='新增外接，最终主方案冻结',
 problem='等级依赖缺陷、缺梗与相对尺寸，平级四分类不能直接表达条件优先级。',
 input='每个原预测mask的RGB+mask 4×160×160上下文crop，及6维log1p面积/宽/高/crop宽/高、面积占比；训练输入使用GT mask裁剪。',
 process='视觉五级下采样4→32→64→128→192→256，stride2 Conv-BN-SiLU与双卷积残差块，GAP得256；几何6→32→32；拼288→256 SiLU/Dropout.2融合；三个线性头输出defect logit、missing-stem logit、sigmoid×100 score。',
 output='融合特征f_i∈R^256、缺梗概率p_stem、原缺陷logit；旧score作为兼容回退。传递给M2/M3/M4。',
 fit='随机初始化独立网络；weighted BCE缺陷+可用项weighted BCE缺梗+SmoothL1(score/100,beta=.05)；λstem=λscore=1，缺失项按availability屏蔽。缺陷pos_weight10.49731182795699，缺梗5.09204368174727。',
 parameters='历史train4277/val518 GT crops；seed20260929；实际选best131。最终class0仍由检测器原class0门控，R1缺陷头只给排序证据，不重新定义不合格。',
 fallback='空/非法crop不能构造特征时记录NaN；有效raw scope外或score/stem非有限值时M3保留检测器原类别。',
 code='scripts/train_r1a_hierarchical_grader.py',anchor='class R1HierarchicalGrader',
 weight=W['attribute_encoder']['path']),
 dict(id='M2',name='冻结规则评分与有界学习残差RR22',status='规则+新增小头，最终主方案',
 problem='设计上补偿预测mask几何测量误差与固定规则之间的偏差，不提供独立实物尺寸或新标签。',
 input='原预测mask派生的五项几何、冻结ECDF/分级配置、M1的256维融合特征。',
 process='fast_shape_features清理最大连通区并做受限白边修剪→五特征中位秩ECDF→S_rule；拟合头以fit均值/σ标准化f，MLP256→32SiLU→1，δ=.05tanh(MLP)，s=100clip(S_rule/100+δ,0,1)。最后一层零初始化。',
 output='0–100修正分数S_i，最多±5分（端点clip）；交M3和M4。',
 fit='3053 fit/761 held匹配可评分实例；SmoothL1β=.025 + .1meanδ² + .1边界ordinal BCE，边界normalized LOW/HIGH，logit宽度.05；AdamW .001/wd.001，batch128，max60 patience12；best19，实际31。',
 parameters='三种子20261022/23/24作成对实验，部署固定22；残差幅度/ordinal等候选来自train内部与开发实验，最终冻结，不按test选seed。参考分布历史锚点恢复，不在推理重拟合。',
 fallback='直接几何失败(ValueError)回退旧几何回归score并累计rule_fallbacks；有效且非原class0、raw≥.25才施加残差。低置信度不参与分级改变。',
 code='scripts/joint_pipeline_runtime_20261024.py',anchor='def infer_sources',
 weight=W['RR22_residual']['path']),
 dict(id='M3',name='邻域候选证据融合、缺梗门控与等级边界',status='新增确定性决策，最终主方案',
 problem='设计上利用几乎重合候选的互补等级证据，缓解分数阈值附近及缺梗决策冲突。',
 input='N行原框、原top1类别、raw conf，M1缺梗概率及M2分数；无需GT。',
 process='同图boxIoU≥.85邻域中，对每类取该类原top1候选最大raw conf，clip到[1e−4,1−1e−4]得e_ic。用其logit差对LOW/HIGH边界加高斯局部有界修正；用grade3 vs max(grade1,2)的logit差修正missing-stem margin；原class0不恢复不新建。完整公式见关键公式表。',
 output='N行最终labels_all；只改变类别，不移动框、不改mask、还未删行。交M4/M5。',
 fit='无单独梯度训练；系数为开发过程锁定参数。',
 parameters='LOW/HIGH见分级规则，alpha_stem=.5、lambda_b=1.5、sigma_s=2.5分、tanh温度3、邻域IoU=.85；recover=False。不是未经约束概率贝叶斯融合。',
 fallback='raw<.25或score/stem非有限保持baseline；空集合返回空。没有某类邻居则该类证据1e−4。',
 code='scripts/joint_pipeline_policy_20261024.py',anchor='def decide',weight='无权重；常量在代码/冻结协议'),
 dict(id='M4',name='复用原缺陷信息的22维预测质量排序',status='新增小头，最终主方案；未新增缺陷网络',
 problem='设计上为重分类后的合格预测估计mask/类别相容质量，使AP排序和重复预测保留更合理。',
 input='核心14维+邻域4类logit/10+原R1 defect logit/10、二元熵/log2、可用标志、固定0；共22维。可用范围raw≥.25且stem有限。',
 process='fit-only mean/std（std<1e−6置1）→MLP22→16SiLU→1→sigmoid q_i；非零最终类rank_i=raw_conf_i·q_i；最终class0保留raw_conf。原缺陷概率p取原logit的sigmoid。',
 output='N行ranking_all，进入AP置信排序及M5；不直接改label/score/mask。数值是排序分，不宣称概率校准。',
 fit='8764 fit/2226 held非零最终类预测，包含低conf和未匹配。y_i=10个IoU阈值上，与同最终类GT的最大maskIoU达到阈值的平均值；逐候选独立软目标BCE，不是竞争式一对一TP标签。AdamW .003/wd.0001，batch256，max300 patience30，best heldBCE最小，epoch0可选；固定25 best109、stop139。',
 parameters='原缺陷/flag-only两个信息对照各3种子25/26/27；最终选择是经历val反馈后的自适应机制审查，再冻结终测，不能写成预注册唯一主假设。',
 fallback='属性不可用时额外4维全0且标志0；原score/stem非有限按50/.5缺省特征；最终rank必须全部有限，否则报错。',
 code='scripts/attribute_local_rank_20261025.py',anchor='def features',weight=W['old_defect_rank']['path']),
 dict(id='M5',name='重分类后真实同类NMS与索引对齐',status='改造后处理，最终主方案',
 problem='不同原类别的同果近重合候选重分类后可能碰撞；置零置信度仍会在计数/缓存中保留，需真实删除。',
 input='原boxes、M3 labels、M4 rank及raw≥.25布尔scope，所有数组同N且原行序一致。',
 process='NMS（非极大值抑制，即保留优先预测并移除同类高重叠预测）：先raw scope优先，再rank降序，再原索引稳定打破平局；若与任一已保留同最终类框boxIoU≥.7则删除；最终keep_indices按原索引升序返回。',
 output='K≤N个keep索引；所有box/label/score/conf/mask/candidate lineage使用相同keep切片，不只改变confidence。',
 fit='不训练，阈值冻结。',
 parameters='boxIoU=.7，使用框非mask；类别依据重分类后labels。AP仍可保留.001–.25候选，但优先级低于raw≥.25。',
 fallback='空预测返回空keep；长度、有限值检查不通过报错。不同最终类的高重叠候选仍可共存，没有保证“一颗实物仅一条输出”。',
 code='scripts/joint_pipeline_policy_20261024.py',anchor='def keep_indices',weight='无权重'),
 dict(id='M6',name='同次forward的prototype边界细化与双mask分工',status='可选分割变体；主方案不启用',
 problem='设计上只在边界带修正分割轮廓，同时避免把细化mask反馈给原分级链改变其特征。',
    input='只细化raw≥.25且非空mask；ROI按原mask外接框pad=max(12,ceil(.25长边))裁剪，直接resize160²（不是R1的letterbox）；RGB/255不做ImageNet归一化。coarse mask1×160²，soft logit1×160²（clip±8/8），prototype32×40²（tanh(P/4)后resize160²）；soft/proto缓存FP16后头计算转FP32；原生mask与原生边界带。',
 process='36通道(RGB3+soft1+prototype32)→1×1Conv36→32SiLU→16SiLU→1；δ=6tanh输出，最后层零初始化。原生band=dilate9(M)−erode9(M)，半径4px；δ双线性回原ROI，仅band内叠加原mask±2基准logit并>0二值化。',
 output='output_masks细化轮廓用于分割评价/输出；grading_masks保持原mask用于R1/几何/排序；候选和keep索引相同。',
 fit='4186条IoU≥.5的train匹配预测，3349/837按图像角色；GPU FP32 AdamW .001/wd.0001 batch32 max30 patience8；损失加权pixel BCE(1+4band)+Dice(smoothing1)；held原生IoU最高、相对面积误差较小择优，best27，完成30。',
 parameters='hard/soft×无/有prototype，3种子25/26/27共12 fits；家族held平均IoU提高≥.001且面积误差不劣于+.005才合格；固定25。可选身份在test前冻结，不能按test提升为主方案。',
 fallback='同次forward capture索引/公共结果不一致直接抛错；若不启用细化返回原mask。代码未提供细化失败静默回退原mask的统一try/except，不能假称有。',
 code='scripts/attribute_local_proto_fit_20261025.py',anchor='class PointHead',weight=W['prototype_output_refiner']['path']),
]
table(F4,'模块总表.csv',[{'模块ID':m['id'],'名称':m['name'],'身份/采用状态':m['status'],'输入':m['input'],'输出':m['output'],'训练':m['fit'],'代码':m['code'],'权重或配置':m['weight']} for m in MODULES])
for m in MODULES:
    md(F4,f"{m['id']}_{m['name']}.md",f'''
    # {m['id']}：{m['name']}

    | 核验项 | 材料 |
    |---|---|
    | 身份/最终状态 | {m['status']} |
    | 具体问题与设计动机 | {m['problem']} |
    | 输入/来源/维度 | {m['input']} |
    | 内部结构及步骤 | {m['process']} |
    | 输出与去向 | {m['output']} |
    | 训练目标与损失 | {m['fit']} |
    | 阈值、选择与范围 | {m['parameters']} |
    | 异常/缺失回退 | {m['fallback']} |
    | 权重/兼容依赖 | `{m['weight']}` |

    实现证据：{ev(m['code'],m['anchor'])}；源文件逐字节副本见第08索引。设计依据是项目中的错误审计和成对控制，不把未验证的收益写成事实。通用算法名不等于本文原创；具体文献引文应由作者从已共享文献库核对原文后绑定，本包没有替作者补造引用。

    模块连接可据第07的同名Mermaid图绘制；下游均按原candidate索引对齐。主方案M0→M1/M2→M3→M4→M5；M6是可选输出分支，不能并入“所有最终输出均细化”的描述。
    ''')

formulas=[
 ('F01','像素/形态','A=Σ1[M>0]；circularity=4πA_contour/P²；solidity=A_contour/A_convexhull','A为掩膜非零像素；轮廓面积与像素面积有离散差异；轴长px','configs/feature_extraction_frozen_v1_val.yaml；docs/FEATURE_DEFINITIONS.md'),
 ('F02','ECDF','p_f(x)=(Σ1[r<x]+0.5Σ1[r=x])/2649','r为冻结参考特征；p∈[0,1]；原分布通过历史锚点恢复','scripts/run_r0_fixed_prediction_replay.py'),
 ('F03','规则分数','T=clip(1−2|p_AR−.5|,0,1)；H=100(.5p_solidity+.3T+.2p_circularity)；S=100(.7p_area+.1p_minor)+.2H','分数0–100，无长度单位','outputs/grading_standard_manual_v4_final/grading_standard_config.json'),
 ('F04','残差','δ=.05tanh(MLP((f−μ)/σ))；S*=100clip(S/100+δ,0,1)','f融合256；μσ仅fit；δ归一化分数±.05','scripts/refine_score_residual_ablation.py'),
 ('F05','邻域证据','e_ic=max({raw_j:IoU_box(b_i,b_j)≥.85,origclass_j=c}∪{1e−4})；l_ic=log(e_ic/(1−e_ic))','clip e≤1−1e−4；4列独立证据不归一，不是同候选完整softmax分布','scripts/joint_pipeline_policy_20261024.py'),
 ('F06','边界margin','m12=S*−HIGH+1.5exp(−.5((S*−HIGH)/2.5)²)tanh((l_i1−l_i2)/3)；m23同理LOW和l_i2−l_i3','HIGH=65.62684031710077；LOW=37.40298225745565；S与σ单位为分','scripts/joint_pipeline_policy_20261024.py'),
 ('F07','缺梗margin','missing=1[logit(p_stem)+.5(l_i3−max(l_i1,l_i2))≥0]','p_stem为缺梗概率；原class0优先；missing→3；否则m12≥0→1，m23≥0→2，否则3','scripts/joint_pipeline_policy_20261024.py'),
 ('F08','rank软目标','y_i=(1/10)Σ_{τ=.50,.55,…,.95}1[max_{g:class_g=final_i}IoU_mask(g,i)≥τ]','无相容GT时0；每个候选独立，无一对一竞争','scripts/joint_pipeline_rank_fit_20261024.py'),
 ('F09','rank输出','q_i=sigmoid(MLP22→16→1(z_i))；rank_i=raw_i q_i (final_i≠0)，否则raw_i','排序置信分不作概率校准解释；operating scope仍看raw_i≥.25','scripts/attribute_local_rank_20261025.py'),
 ('F10','可选边界','B=dilate9(M)−erode9(M)；Mout=1[(2M−1)·2+resize(6tanh(head))·B>0]','native band半径4px；mask原生坐标；无物理标尺','scripts/attribute_local_proto_fit_20261025.py'),
]
table(F4,'关键公式与符号.csv',[dict(zip(['编号','主题','公式','符号/单位/限制','来源'],r)) for r in formulas])

md(F4,'整体方法_数据流_方案边界.md',f'''
# 第四部分：整体方法与模块连接

## A. 最终身份及模块独立说明

**正式主方案**是固定种子20261025的old_defect_rank（原mask分级/排序/输出），依赖冻结YOLO、R1编码器、RR22规则锚定残差和固定邻域决策/真实NMS；**可选分割变体**仅增加同种子的prototype输出细化。读取本目录M0–M6七份独立卡片及[模块总表](模块总表.csv)。此前20261022的selected learned residual、14维rank等不是20261027终测主方案的完整定义。

术语：**融合特征**是图像与几何经神经网络拼接后的256维向量；**残差**是对已有规则分数的有界加法修正；**logit**是概率p转换的log(p/(1−p))；**NMS**是基于框重叠的预测删除操作；**prototype**是YOLO共享mask基底，不是人工语义通道。

## B. 从输入到输出（按实际调用图）

1. M0对一张图做检测，保留.001候选池（最多300），得到原class/raw_conf/box/公共polygon。原polygon栅格化产生`grading_masks`。需要可选细化时从**同一forward**抓取soft/prototype，并以candidate id与公共结果严格校验，不能最近框随意配对。
2. M1以原mask裁上下文ROI，得到stem概率、原defect logit和fusion256；M2仍对原mask计算显式几何规则并叠加RR22残差。**当前主方案在线确实计算规则几何**；不能沿用早期“完全神经代理消除CPU形态学”的描述。
3. M3按raw≥.25且score/stem有限确定可改标签的集合，原class0保留；邻域证据修正缺梗与大小边界，产生同N行`labels_all`。
4. 可选M6从同候选soft/prototype生成`output_masks`，不反馈到M1–M4。禁用时`output_masks=grading_masks`。这是代码中的两套mask分工，**不保证评价后的条件分级计数完全不变**：改变mask可能改变GT重新匹配对象。
5. M4为固定类别计算22维质量排序；M5用最终类别与rank做真实同类NMS。原框不被重回归；保留数N变为K，所有字段按同keep取片。
6. 返回预测字典，runtime保留完整N行数组与`keep_indices`，下游评价/导出必须统一按keep取K行；当前评价确实只用K行，而不是简单把删除行confidence置0。不能把runtime的完整数组长度直接当最终果实数。人工GT只用于离线训练、匹配和评价，在线不输入GT、真实等级、真实缺陷/果梗标记或真实尺寸。规则参考分布和训练后的权重是冻结常量，不是单张图推理时的人工输入。

## C. 字段、索引与行为变化

| 字段 | 格式 | 改变它的步骤 | 对齐方式 |
|---|---|---|---|
| entry.boxes | N×4 xyxy原图像素 | M0 | 基准行索引i，后续不移动框 |
| entry.baseline / conf | N整数 / N浮点 | M0 | 保留原值，raw scope不能改看rank |
| entry.score / stem / defect_logit | N，失败可NaN | M1/M2 | valid_indices/fusion行计数一致，原索引回填 |
| labels_all | N整数0–3 | M3 | 与base逐行对应 |
| ranking_all | N有限浮点 | M4 | 只改置信排序，不改类别 |
| grading_masks | N个H×W二值mask | M0 raster | M1/M2/M4只读此版本 |
| output_masks | N个H×W二值mask | 可选M6 | 与原mask同候选，边界带内改变像素 |
| keep_indices | K个原行索引 | M5 | 真删后框/类/分数/两套mask共享同keep |
| candidate_indices / source_prediction_indices | N或K个整数 | 捕获/切片 | 捕获时校验原class/conf精确及bbox≤1e−3；不一致报错 |

`CandidateCapture`提取的是同一候选的4个原始类别分数，`raw_same_candidate`是实验开关；**最终M3采用的是同图邻域top1证据**，4列并不构成和为1的类别分布。把“孪生框证据”描述成“同一框的完整多类别概率分布”会混淆两项不同实现。{ev('scripts/joint_pipeline_policy_20261024.py','def evidence')}；{ev('scripts/joint_pipeline_capture_20261024.py','def align')}。

原class0门控在最终主方案中冻结；UQ恢复、缺陷新建、new attribute head等是实验。新排序器复用原defect信息；不是把早期缺陷输出当作新人工标签，也不是同时训练一个新缺陷网络。

## D. 方案、兼容依赖和未解决问题

| 身份 | 内容 | 正文如何处理 |
|---|---|---|
| 最终主方案 | M0–M5，original grading/output masks，固定old-defect rank25与残差22 | 正式方法主体 |
| test前冻结可选变体 | M0–M6，细化只作output | 单列可选分割分支，不能按test优劣重定主次 |
| 运行兼容依赖 | 父类还加载旧geometry score、selected22 learned residual、旧rank08；部分仅计算旧分支或回退 | 明确列出权重链，勿把被覆盖旧rank当最终排序器 |
| 仅做过实验 | Strip/DEG/Partial/neck/soup，直接raw_same_candidate证据，恢复/新建缺陷、新attribute rank、不同残差范围/损失 | 放探索/消融，不能一股脑写成最终网络结构 |
| 未完成/不具备 | 新独立采集批次、mm标定、内在品质真值、完整实物果级去泄漏证明、全库可见实例标注规范 | 缺口，不写成已完成实验 |

最终NMS仍是同**最终类别**框抑制；跨类别重叠可保留，未实现经验证的全局“一果唯一”判定。原检测器已判class0的错误也不会被主方案恢复。边界融合与rank会改变重复竞争/GT匹配，保留候选级class0不等于保证GT层面零误放。部分无效几何回退使用旧score，对其频数不能没有日志就补数字。

图与绘图源见第07目录；全部图标明“根据代码绘制的示意图”，不把示意连接当性能证据。[关键公式与符号CSV](关键公式与符号.csv)给出逐项定义。
''')

training=[
 ['YOLO固定结构检测/分割','2898 train图/12831实例；116原V3 val图/518；四类GT','640²；batch8；workers4','200；无早停；best16','AdamW lr.00125→.0000125线性；wd.0005；momentum.9；warmup3；nbs64','库内E2E实例分割loss；box7.5/cls.5/seg使用box gain，regmax1无有效DFL','通用本地预训练；全检测器优化；后续冻结','20260915单条续训轨迹；第146后中断，第147恢复','val MaskAP50最高；全精度相同后轮覆盖；非默认综合fitness'],
 ['R1属性编码器初训','4277 train/518 val GT crops；defect全有，stem/score可用项','4×160²+6；batch64；workers0；GPU AMP','计划100 patience20；实40；best20','AdamW .001 wd.0001；CosineAnnealing eta_min1e−5；clip5','weighted BCEdefect+availability masked BCEstem+SmoothL1(score/100,beta.05)','独立随机初始化；YOLO不导入','20260929；train RGB/mask同步翻转/亮度对比增强','val crop macroF1，再UQF1、stemF1、较低scoreMAE'],
 ['R1续训40→150','同原4277/518，不使用valfix改标层','同上','总150；禁早停；best131','恢复模型与optimizer；余110轮重建cosine eta_min1e−5','同初训','从初训last40继续；val用于选模','seed+已完成轮数重新播种增强；非无中断等价轨迹','同上'],
 ['R1续训150→200','同原4277/518','同上','总200；禁早停；全链best仍131','恢复模型/optimizer；余50轮重建cosine eta_min1e−5','同初训','从last150继续，best131继承','同一链，不是独立第二/第三随机种子','最终best.pt内部config仍epochs150；以summary核验'],
 ['RR22规则锚定残差','3053 fit/761 held匹配可评分预测；原train773/193图角色','fusion256；batch128；CPU FP32 threads2','max60 patience12；epoch0合格；31完成/best19','AdamW .001 wd.001；无scheduler；clip5','SmoothL1β.025+.1δ²+.1ordinalBCE(width.05)','冻YOLO/R1，MLP256→32→1末层零初始化','22/23/24三种子；固定22部署；无新增在线增强','held scoreMAE，至少改善1e−4'],
 ['原缺陷信息rank','8764 fit/2226 held非零最终类预测；含未匹配/低conf','22维；batch256；CPU FP32','max300 patience30；epoch0；139完成/best109','AdamW .003 wd.0001；无scheduler记录','独立多IoU软质量目标BCEWithLogits','仅训385参数22→16→1，所有上游冻结','25/26/27信息对照；固定25部署；无新增数据增强','held BCE至少改善1e−6；最终家族选择经val反馈后冻结'],
 ['可选soft+prototype输出细化','4186匹配mask，3349 fit/837 held','RGB160²+soft160²+proto32×40²；batch32；GPU FP32','max30 patience8；epoch0；30完成/best27','AdamW .001 wd.0001；无scheduler；clip5','pixelBCE权重1+4band + Dice(smoothing1)','仅训1729参数PointHead，末层零初始化','hard/soft×zero/proto×25/26/27=12fits；固定25','held nativeIoU最大，面积误差次选；家族门槛IoU+.001、面积≤+.005'],
]
table(F5,'分阶段训练参数.csv',[dict(zip(['组件/阶段','数据与标签','输入/批量/精度','轮数与实际完成','优化器与调度','损失','初始化/冻结','随机性/增强/重复','最佳检查点标准'],r)) for r in training])
md(F5,'训练环境_设置与执行记录.md',f'''
# 第五部分：训练环境与分阶段设置

## A. 当前与历史环境必须分开

| 项目 | 本轮2026-10-28只读探测 | 历史可核实记录 |
|---|---|---|
| OS | Windows 11，build10.0.26200 | 初始环境报告同Windows，具体每次历史补丁未记录 |
| CPU/RAM | Intel Core Ultra9 275HX，24逻辑核；OS可见33751425024 B（约31.43 GiB） | 历史每个run未逐项快照，不能用当前探测代替 |
| GPU/显存 | NVIDIA RTX5070 Laptop GPU，8151 MiB（8546484224 B） | e200运行中记录同卡与8151MiB；驱动591.86 |
| Python | 3.12.14 | 早期环境报告3.12.13；e200、R1训练config及后期runtime记录3.12.14 |
| 框架/CUDA | torch2.13.0+cu130，torchvision0.28.0+cu130，编译CUDA13.0 | 驱动支持CUDA13.1不等于torch编译13.1；记录应分开 |
| 模型库 | ultralytics8.4.117 | detector checkpoint.version及训练协议8.4.117 |
| 辅助库 | NumPy2.5.2、pandas3.0.5、SciPy1.18.1、opencv-python5.0.0.93、Pillow12.3 | 本轮值；详细历史不全见runtime_environment和training_config |
| 离线增强生成器 | 不以当前环境替代 | metadata记录Python3.13.5/NumPy2.1.3/Pillow11.1.0，属于单独生成环境 |

本轮环境实测在第08目录`current_hardware_software.json`；历史证据{ev('environment/ENVIRONMENT_REPORT.md')}；{ev('docs/YOLO26_OFFLINEAUG_E200_FROZEN_PROTOCOL_20260918.md')}；{ev('outputs/reports/attribute_local_20261025/runtime_environment.json')}。机器型号/系统当前状态不是每次历史实验运行于同样环境的证明。

## B. 参数表与损失实现

逐组件逐阶段见[分阶段训练参数CSV](分阶段训练参数.csv)，不能合并成一套“epoch200、AdamW.001”。checkpoint与各run原始args/config/history/completion已登记并附小文件副本。

检测器stock criterion为`E2ELoss(v8SegmentationLoss)`，one2many topk10与one2one topk7/topk2=1；训练混合权重起始.8/.2，one2many随epoch降至.1、one2one增至.9。实例mask BCE按GT框区域裁剪并按面积归一化；语义分支如输出proto tuple则含BCE/Dice分支。box gain7.5、cls.5、dfl1.5是args中的系数，但reg_max=1使DFL项不可当作非零有效分布损失。**未提出新YOLO损失**。本机安装`ultralytics/nn/tasks.py::SegmentationModel.init_criterion`与`utils/loss.py`是依据；已在第08另附对应函数行号索引与散列，不能用其他版本公式替代。

R1 loss的score目标先除100；缺梗正类是**缺梗**。R1验证标签是当时V3目标，后来valfix更改不反向改写历史checkpoint选择。RR22与rank内部held仅给新增头选模；整个编码器曾见全train。prototype中GT只作训练/held loss，推理边界带来自预测mask，不来自GT。

## C. 顺序、续训、冻结和共同对照

训练顺序：四类标签/离线增强 → YOLO200轮与best16 → GT裁剪R1初训40轮及续训到150/200、best131 → 冻结上游预测与773/193图角色 → 规则残差/不同score控制 → 邻域决策和真实NMS开发 → 质量rank与信息控制 → 可选prototype2×2×3实验 → 冻结5个终测臂 → 2026-10-27统一test。

R1初训计划100但早停40（patience20）；续训禁用早停，恢复optimizer后按剩余110、50轮重建cosine；固定种子加已完成轮数重设增强随机性，不能表述为从头连续200轮逐步等价。检测器第146后中断147恢复保留optimizer/EMA及选模轨迹，不能虚称无中断。

对照的共同设置是冻结同一检测器/原mask/图像角色/operating raw scope与评价脚本；不同头在相同输入控制中配对初始化/打乱顺序，比较多个固定种子，完整保留失败实验。新增feature位数相同不代表有效信息容量相同；0填充控制必须说明。不同历史训练剂量、conf池、online图或延迟协议不应直接混入最终公平对照。

最终主方案rank25与score22是固定身份，不按test挑最佳seed；三种子是头初始化波动检查，不是三个独立test集合。YOLO/R1只有各一条历史训练链，不能写整个系统已进行3次从头独立重复。

## D. 运行命令与归档定位

以下是**历史调用格式/记录定位**，本次没有执行。不要直接运行会写原目录或重跑test的脚本。

```powershell
# detector；历史初训/续训
.venv/Scripts/python.exe scripts/train_frozen_offlineaug_e200.py
.venv/Scripts/python.exe scripts/train_frozen_offlineaug_e200.py --resume
# R1参数以对应training_config.json为准；不要以脚本默认覆盖实际run
.venv/Scripts/python.exe scripts/train_r1a_hierarchical_grader.py --epochs 100 --patience 20 --seed 20260929
# 新增头脚本会读取既有固定缓存与内部角色
.venv/Scripts/python.exe scripts/refine_rule_anchored_residual.py
.venv/Scripts/python.exe scripts/attribute_local_rank_controls_20261025.py
.venv/Scripts/python.exe scripts/attribute_local_proto_fit_20261025.py
```

R1 resume的完整历史命令文本若不能在日志定位，应写“未找到完整命令记录”，以run的config、summary、history和检查点元数据恢复有效设置，不补造启动参数。原文件路径含本机根目录，在另一台电脑需显式重映射。

第三方修改：本机Ultralytics `utils/ops.py::process_mask_native`有分块显存补丁（32,000,000/(H×W)上限确定每块mask数、分块scale/threshold、再拼回并crop，空输入返回uint8空mask）。YOLO结构不因该补丁改变；复现不能仅写pip install 8.4.117。自定义Mask50 trainer只把保存best的fitness改为val MaskAP50；prototype可选分支临时hook同一native函数，finally恢复。无证据把这些实现称为发布版原封不动。第08附安装源码函数摘录及SHA，不复制整个虚拟环境。
''')

metrics=[
 ['Box/Mask IoU','|P∩G|/|P∪G|','单预测与单GT；mask原生像素，box xyxy连续面积','越大越好','box与mask的IoU不可互换'],
 ['Box/Mask AP50','每类PR积分AP_c(.50)，4类平均','全图所有GT与保留候选；池raw≥.001，按当前rank排序','越大越好','类别相容匹配；不是分级accuracy'],
 ['Box/Mask AP50:95','(1/(4×10))Σ_cΣ_τ AP_c(τ)，τ=.50:.05:.95','4类×10阈值宏平均；不能省略缺失类改变分母','越大越好','AP本身按101点插值PR积分'],
 ['固定scope precision/recall','P_c=TP_c/N_pred,c；R_c=TP_c/N_GT,c','raw≥.25的保留预测，类别相容IoU=.5匹配','越大越好','confidence使用raw筛选，不使用新rank≥.25'],
 ['operating macro F1','(1/4)Σ_c 2TP_c/(N_pred,c+N_GT,c)','同上；每类等权','越大越好','不是conditional accuracy；漏检和多输出均影响'],
 ['定位召回','L/G','原生maskIoU≥.5、忽略类别的Hungarian匹配；L匹配数、G全部GT','越大越好','仅说明定位，不保证等级正确'],
 ['已定位条件分级正确率','C/L','类无关匹配后的正确等级数C/已匹配GT数L','越大越好','不把漏检计入分母，必须同时报L/G'],
 ['全部GT定位且分级正确比例','C/G','同一C除全部GT；与C/L成对报告','越大越好','例如val原YOLO416/500与416/518分母不同，示例不是新结果'],
 ['score MAE/RMSE','Σ|S_pred−S_ref|/N_score；sqrt(Σ误差²/N_score)','只有独立参考score可用且匹配实例；单位分','越小越好','不是N=全部GT；test无独立score目标不生成伪真值'],
 ['不合格误放','E_UQ/G_UQ；另列E_UQ/L_UQ','GT类0匹配到预测1/2/3；明确全GT或已定位分母','越小越好','漏检不计为被接受；总误放不等于新增误放'],
 ['新增误放','#原RR22正确拒绝GT0、候选输出合格/G_UQ','同GT实例ID配对；匹配规则不变','越小越好','相对基准变化；零新增不代表零绝对误放'],
 ['任意合格输出触及UQ','#GT0与任一raw-scope合格maskIoU≥.5/G_UQ','不依赖Hungarian只选一条，检查额外重复合格输出','越小越好','用于补充匹配表可能漏掉的重复输出风险'],
 ['合格误拒','#GT>0匹配到预测0/(G−G_UQ)','匹配实例统计分子，全部合格GT分母','越小越好','缺梗降三级不等于判不合格'],
 ['漏检/重复输出','漏检=G−L；同类NMS删除=N−K','固定scope与全候选分别记录before/after；预测数不是果实数','越小越好但需保留召回','同类boxNMS删除不能当已证明物理重复真值'],
 ['延迟/FPS','逐图ms；FPS=1000/mean(ms)','24张val，每臂每图3次，图像聚类统计；排除decode/load','ms小/FPS大','端到端选定调用图，不等于裸网络耗时或批处理吞吐'],
]
table(F6,'指标定义与分母.csv',[dict(zip(['指标','定义/公式','统计单位/分母','方向','解释限制'],r)) for r in metrics])
protocol=[
 ['候选池','raw conf=.001，max_det=300，imgsz640，retina_masks=True，FP32','scripts/attribute_local_runtime_20261025.py'],
 ['正式后处理','最终类相同且boxIoU≥.7删除；raw≥.25优先、rank降序、原行索引稳定','scripts/joint_pipeline_policy_20261024.py::keep_indices'],
 ['邻域证据','同图boxIoU≥.85；最大该原类别raw conf，不依赖GT','scripts/joint_pipeline_policy_20261024.py::evidence'],
 ['AP匹配','逐IoU阈值类别相容；IoU降序greedy，先预测去重再GT去重；.50:.05:.95','analyze_r1_ap_bootstrap.py::pure_match / BaseValidator.match_predictions'],
 ['AP积分','当前Ultralytics ap_per_class→compute_ap；101点插值，PR单调包络，trapezoid积分','本包安装库摘录/源码SHA'],
 ['工作点匹配','保留预测中原raw≥.25；不看类别，IoU≥.5；linear_sum_assignment最大化Σvalid(1+IoU)','analyze_val_instance_errors.py::optimal_match'],
 ['类别汇总','AP4类宏平均；operating F1按4类等权，micro与macro分开','joint_pipeline_evaluate_20261024.py / final_test_evaluation_20261027.py'],
 ['终测重采样','129原图，4编号组内保留各组大小有放回；5臂共享multiplicity，2000，seed20261027','analyze_final_test_20261027.py'],
 ['区间','2.5/97.5百分位95%点态描述区间；不作多重比较校正，不报显著性p','analysis_plan.json'],
 ['延迟','24图×3重复×10调用图=720记录；warmup3图/臂；CUDA sync夹perf_counter','attribute_local_rank_latency_20261025.py'],
]
table(F6,'评价参数与匹配规则.csv',[dict(zip(['项目','有效参数/规则','来源'],r)) for r in protocol])
md(F6,'评价方法_统计与复现协议.md',f'''
# 第六部分：评价指标与协议

本部分描述方法，不展开性能比较。**AP**（平均精度）是按置信排序的precision–recall曲线面积；**mAP**是不同类别AP的平均；**IoU**是预测与参考的交并比；**Bootstrap**是保留原图整体进行有放回抽样以描述有限样本不确定性。

## A. 指标定义

见[指标定义与分母CSV](指标定义与分母.csv)。G全部GT、L工作点类无关匹配到的GT、C其中等级相同数、N_pred,c为工作点预测类c数量、N_GT,c为该类GT数、TP_c为类别相容匹配TP数。三种比值C/L、C/G、L/G不是同一个指标，AP亦不是其中任何一个。

mask按原图H×W二值区域计算IoU；公共polygon坐标先round、clip再fillPoly。box为连续xyxy面积，不加1。AP对每类每个IoU阈值计算TP/FP序列，用当前Ultralytics的PR包络、101点插值与积分，最终4类等权平均；AP50:95再平均10个阈值。不是把全部实例混成单类micro AP。{ev('scripts/joint_pipeline_evaluate_20261024.py','def evaluate')}。

## B. 两套匹配必须并存

1. **AP匹配**用全部保留候选（原raw池.001），类别必须一致，mask/box IoU分别评价，τ=.50:.05:.95。复用库`match_predictions(use_scipy=False)`的IoU降序贪心去重，先unique预测，再unique GT；类不相容相似度置0。按当前confidence/rank全局降序构造PR。不能换成工作点Hungarian后还称同一AP实现。
2. **工作点分级**用原raw≥.25的保留候选，类别无关maskIoU≥.5。实现`optimal_match`构造valid pair的权重`1+IoU`、无效pair=0，调用`scipy.optimize.linear_sum_assignment`最大化总权重，然后保留有效pair。函数注释写“优先匹配数量”，但代码的加权目标本身不是任意图上严格字典序最大数量的形式化保证；应如实按实现描述，不补一个没有实现的算法保证。
3. 类别相容operating P/R/F1另用AP式IoU=.5匹配，不能把Hungarian的C直接当每类TP。最终`raw≥.25`是原检测置信度scope，不因rank乘小而改变入选分母。

原始YOLO控制不额外NMS；raw+realNMS、RR22、rank、rank+prototype臂统一真实NMS与raw scope优先。四类AP矩阵必须4×10；若重采样缺类不静默改分母，也不反复重抽直到有类。{ev('scripts/analyze_r1_ap_bootstrap.py','def pure_match')}；{ev('scripts/analyze_val_instance_errors.py','def optimal_match')}。

## C. 风险、重复和连续分数

不合格误放是GT0匹配到合格类1–3；分子可以报数量，比例必须明确除全部G_UQ还是已定位L_UQ。新增误放是相对于RR22原先正确拒绝的同一GT0转为合格，零新增不代表绝对误放为零。“任意合格输出触及不合格”检查一个GT0是否被任何合格预测maskIoU≥.5触及，不受单条Hungarian匹配掩盖。合格误拒是GT1–3匹配到0；缺梗降三级不算误拒。

漏检是工作点G−L；多余预测是未匹配输出，不能一律叫重复果。真实NMS以同类box重叠定义删除条件，跨类冲突仍可能保留。应同时报候选before/after与raw scope before/after；没有独立实物唯一标记，不能说所有删除均为物理重复目标。

MAE/RMSE仅在参考score可用且匹配的实例上算，注明N_score、score单位分、GT来源为冻结规则/人工审核而非实物毫米。test如无独立score/果梗目标不根据预测生成参考。原mask和细化mask会导致评价GT匹配变化，条件分级统计不是固定候选标签不变的必然推论。

## D. 重复、置信区间和时序

最终点估计用冻结5臂、test129图/576GT；2026-10-27后续只读取预测缓存做图像配对分层Bootstrap2000次，seed20261027。四个编号来源组内保留原组样本数，有放回抽整图（含该图全部GT和预测）；5臂共享抽样multiplicity。抽到的图像索引排序，保持置信平局的原始顺序；每次重算AP及计数比值。用2.5/97.5百分位作描述性点态95%区间，对同一抽样下的两臂差值给配对区间。保留缺类无定义draw并报告有效数，不重新抽到“好结果”。

主比较rank对RR22的MaskAP50在test前冻结；**具体Bootstrap细节是在看过点估计后、抽样前固定**，不是事前预注册统计方案。无多重比较校正或p值；源组不是真正独立采收批次；图像级区间不能证明实物果/批次独立泛化。三种子小头不等于三套独立测试集。{ev('scripts/analyze_final_test_20261027.py','plan = dict')}。

## E. 效率协议

正式rank延迟实验选择4个source_group各前6张val共24图，10个调用图均预加载，各warmup前3图；每图3次，固定rng20261025打乱每图每轮的调用图顺序，共720条、每臂72条。每次CUDA synchronize前后夹perf_counter，图像已预解码。包含选定调用图的预处理、YOLO、ROI/R1、规则几何、残差/排序、输出mask及NMS，排除磁盘解码、模型加载；不是裸网络forward。FP32、conf.001、maxdet300、retina native mask一致，GPU5070/8151MiB。

报mean/median/std/p95 ms以及1000/mean(ms)的FPS；它不是批处理吞吐。区间以每张图3次的均值为单位，3000次image-cluster paired bootstrap，seed20261025。历史54.51ms等来自不同conf池或调用图，不并入此协议。test replay总执行秒数含缓存/GT/磁盘，不能当推理延迟。{ev('scripts/attribute_local_rank_latency_20261025.py','def main')}；{ev('outputs/reports/attribute_local_20261025/rank_latency_status.json')}。

## F. 可复现步骤、位置和未明确口径

1. 核对第08证据SHA、数据版本、四类顺序、val覆盖层和最终5臂身份；核对实际权重SHA，不根据文件夹名推断轮次。
2. 原历史统一终测入口`final_test_evaluation_20261027.py`，冻结快照/防重复揭盲守卫在`final_test_preflight_20261027.py`；每图共享一次产生正式候选的detector/R1前向（初始化预热除外），再在相同输出上构建各臂。记录路径`outputs/reports/final_test_preflight_20261027/test_replay/`。
3. 从保存的`*_evaluation.pt`、逐图`predictions/*.pt`与metrics/details检查计数；`analyze_final_test_20261027.py`只读缓存重算区间（原脚本有输出目录存在保护）。本包未复制大PT缓存，准确路径登记，不声称离开本机能重新计算所有AP。
4. 延迟脚本/原始720记录及summary在第08小型副本；只做独立环境重现实验时才另设新输出目录，不能直接覆盖现有终测目录。

未明确/需补：所有历史实验是否均使用当前同一补丁与同一安装文件散列，没有逐次快照；早期baseline test的完整启动命令/首次时间需要旧记录补证；不同历史MAE的可用目标集合须逐表带N，不能自动统一；实物果独立性及跨批泛化无法由现有协议证明。上述问题不通过本轮重训练或重新test解决。
''')

# Complete label/file inventory, without copying the full dataset.
labels=[]
for split in ['train','val','test']:
    folder=ROOT/'data'/('grading_four_class_v3_valfix_20261006' if split=='val' else 'grading_four_class_v3')/'labels'/split
    for p in sorted(folder.glob('*.txt')):
        rr=[l.split() for l in p.read_text(encoding='utf-8-sig').splitlines() if l.strip()];c=Counter(int(r[0]) for r in rr)
        labels.append({'image_id':p.stem,'split':split,'标签相对路径':p.relative_to(ROOT).as_posix(),'label_sha256':sha(p),'实例数':len(rr),**{f'class{k}':c[k] for k in range(4)}})
table(F2,'当前正式标签逐文件清单.csv',labels)

# Machine-readable counterparts for method tables and the exact nested config.
def flatrows(obj,prefix=''):
    if isinstance(obj,dict):
        for key,value in obj.items():yield from flatrows(value,prefix+'.'+str(key) if prefix else str(key))
    else:yield prefix,json.dumps(obj,ensure_ascii=False)
import yaml
cfgrows=[]
for rel in ['configs/feature_extraction_frozen_v1_val.yaml','configs/grading_four_class_v3_augmented_delivery_local.yaml','data/grading_four_class_v3_augmented_delivery/metadata/segmentation_augmentation_config.json','outputs/grading_standard_manual_v4_final/grading_standard_config.json']:
    obj=load(ROOT/rel) if rel.endswith('.json') else yaml.safe_load((ROOT/rel).read_text(encoding='utf-8-sig'))
    for key,value in flatrows(obj):cfgrows.append({'项目相对路径':rel,'字段':key,'原记录值_JSON':value,'解释限制':'源配置保留历史状态；最终采用身份以本包说明为准'})
table(F2,'分级与预处理增强配置_逐字段.csv',cfgrows)
table(F2,'等级规则与版本表.csv',[
 {'类别/版本':'class0','定义/用途':'参考为人工ROT等外观不合格；最终在线保留原YOLO class0','优先级/变化':'最高优先，不由新头恢复','证据':'grading config / joint_pipeline_policy'},
 {'类别/版本':'class1','定义/用途':'无缺陷且有梗，高相对尺寸分；基本HIGH以上','优先级/变化':'最终另有局部边界证据修正','证据':'joint_pipeline_policy'},
 {'类别/版本':'class2','定义/用途':'无缺陷且有梗，基本LOW至HIGH区间','优先级/变化':'最终另有局部边界证据修正','证据':'joint_pipeline_policy'},
 {'类别/版本':'class3','定义/用途':'合格缺梗优先；或有梗低尺寸分','优先级/变化':'缺梗大果也为三级','证据':'joint_pipeline_policy'},
 {'类别/版本':'原始单类','定义/用途':'class0=wolfberry，5752 polygon','优先级/变化':'不是不合格标签','证据':'raw JSON/TXT'},
 {'类别/版本':'V3','定义/用途':'1211图5371实例四类GT，train/test当前仍用','优先级/变化':'HOLD74图，精确poly去重7条','证据':'build_summary / 逐文件清单'},
 {'类别/版本':'valfix_20261006','定义/用途':'116图518实例val-only标签覆盖层','优先级/变化':'7条等级/果梗改标；单审，无训练改动','证据':'version.json / corrections.json'},
])
hw=load(AUDIT/'current_hardware_software.json')
hwrows=[{'阶段':'本轮2026-10-28探测','字段':key,'值':value,'来源':'current_hardware_software.json','历史解释':'不能代替所有历史run环境'} for key,value in flatrows(hw)]
hwrows.extend([
 {'阶段':'初始环境报告','字段':'Python','值':'3.12.13','来源':'environment/ENVIRONMENT_REPORT.md','历史解释':'与后期3.12.14并列'},
 {'阶段':'e200运行时记录','字段':'Python/torch/ultralytics/GPU','值':'3.12.14 / 2.13.0+cu130 / 8.4.117 / RTX5070 Laptop 8151MiB','来源':'docs/YOLO26_OFFLINEAUG_E200_FROZEN_PROTOCOL_20260918.md','历史解释':'历史运行报告，非仅当前探测'},
 {'阶段':'离线增强生成环境','字段':'Python/NumPy/Pillow','值':'3.13.5 / 2.1.3 / 11.1.0','来源':'segmentation_augmentation_config.json.versions','历史解释':'单独生成环境'},
])
table(F5,'硬件软件环境与历史差异.csv',hwrows)
table(F3,'基础模型属性.csv',[{'字段':key,'值':value,'来源':'detector best.pt CPU元数据；非重新forward'} for key,value in flatrows({k:v for k,v in W['detector'].items() if k in ['path','sha256','model_type','parameters','yaml','names','stride','end2end','head_attributes','version']})])
featuredefs=[
 ('major_axis','椭圆长轴，失败用最小旋转矩形长边','px','未直接加权'),('minor_axis','椭圆短轴，失败用最小旋转矩形短边','px','ECDF权重.10'),
 ('area','geometry mask非零像素数','px²','ECDF权重.70'),('aspect_ratio','major_axis/minor_axis','无量纲','shape内变换后权重.30'),
 ('circularity','4π连续轮廓面积/周长²','无量纲','shape内权重.20'),('solidity','连续轮廓面积/凸包面积','无量纲','shape内权重.50'),
 ('eccentricity','sqrt(1−(minor_axis/major_axis)²)','无量纲','无最终grade权重'),
 ('L_mean','非高光有效果体CIELAB L*均值','0–100','0'),('L_std','上述L*标准差','0–100','0'),
 ('a_mean','非高光有效果体a*均值','近似−128–127','0'),('a_std','上述a*标准差','CIELAB','0'),
 ('b_mean','非高光有效果体b*均值','近似−128–127','0'),('b_std','上述b*标准差','CIELAB','0'),
 ('S_mean','非高光果体HSV饱和度均值','0–1','0'),('V_mean','非高光果体HSV亮度均值','0–1','0'),
 ('dark_ratio','暗区候选面积/损伤有效区域像素数','0–1','0；疑似区域非人工缺陷GT'),
 ('abnormal_color_ratio','稳健Lab异常色面积/损伤有效区域像素数','0–1','0；疑似区域非人工缺陷GT'),
 ('max_damage_ratio','暗区与异常色并集最大连通区面积/geometry mask面积','0–1','0；疑似区域非人工缺陷GT'),
]
table(F2,'18项历史可解释特征与最终作用.csv',[{'特征':a,'定义':b,'单位':c,'最终规则作用':d,'来源':'docs/FEATURE_DEFINITIONS.md；configs/feature_extraction_frozen_v1_val.yaml；grading_standard_config.json'} for a,b,c,d in featuredefs])

# Environment source function excerpts: retain source SHA/line numbers and short
# relevant functions, not an entire installed dependency or unrelated files.
installed=[('.venv/Lib/site-packages/ultralytics/utils/ops.py',['process_mask_native']),
 ('.venv/Lib/site-packages/ultralytics/utils/metrics.py',['compute_ap','ap_per_class']),
 ('.venv/Lib/site-packages/ultralytics/engine/validator.py',['match_predictions']),
 ('.venv/Lib/site-packages/ultralytics/utils/loss.py',['E2ELoss','v8SegmentationLoss'])]
excerpts=[]
for rel, names in installed:
    p=ROOT/rel
    if not p.exists():continue
    source(rel,'本机依赖实现与补丁；仅附函数摘录，不复制整库',False)
    lines=p.read_text(encoding='utf-8-sig').splitlines()
    for name in names:
        matches=[i for i,l in enumerate(lines) if re.match(r'\s*(?:def|class) '+re.escape(name)+r'\b',l)]
        if not matches:raise ValueError(name)
        start=matches[0];indent=len(lines[start])-len(lines[start].lstrip());end=start+1
        while end<len(lines):
            line=lines[end]
            if line.strip() and len(line)-len(line.lstrip())<=indent and re.match(r'\s*(?:def |class |@)',line):break
            end+=1
        text='\n'.join(f'{i+1:05d}: {lines[i]}' for i in range(start,end))
        dest=PACK/F8/'安装库函数摘录'/f'{p.stem}__{name}.txt';dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(f'来源: {rel}\n源文件SHA256: {sha(p)}\n行号: {start+1}–{end}\n此摘录来自本机安装版本，不保证与官方发行包字节相同。\n\n{text}\n',encoding='utf-8')
        excerpts.append({'源文件':rel,'函数/类':name,'起始行':start+1,'末行':end,'源SHA256':sha(p),'摘录':dest.relative_to(PACK).as_posix()})
table(F8,'安装库与补丁摘录索引.csv',excerpts)

# Actual sample images and overlays, not generated illustrative fruit.
fontpath=Path('C:/Windows/Fonts/msyh.ttc')
def font(size):return ImageFont.truetype(str(fontpath),size) if fontpath.exists() else ImageFont.load_default()
rawmap={r['image_id']:r for r in csv.DictReader((AUDIT/'raw_source_inventory.csv').open(encoding='utf-8-sig'))}
boundary=list(csv.DictReader((ROOT/'outputs/grading_standard_manual_v4_final/audit/boundary_review_audit.csv').open(encoding='utf-8-sig')))
table(F2,'边界标定样本与集合归属_核验.csv',[{'review_id':r['review_id'],'image_id':r['image_id'],'instance_id':r['instance_id'],'split':rawmap[r['image_id']]['split'],'边界组':r['candidate_group'],'人工期望等级':r['manual_expected_grade'],'calibration_eligible':r['calibration_eligible'],'manual_stem_missing_comment':r['manual_stem_missing_comment'],'manual_measurement_error':r['manual_measurement_error']} for r in boundary])
palette=[(220,50,47),(0,114,178),(0,158,115),(204,121,167)]
assets=[];panels=[]
for sample in A['sample_candidates']+[{'image_id':'wb_000069','split':'train'}]:
    iid=sample['image_id'];split=sample['split'];raw=rawmap[iid];src=(ROOT/raw['source_image']).resolve()
    target=PACK/F7/'原始样本'/f'{iid}_{split}_原始.png';target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target)
    lp=ROOT/'data'/('grading_four_class_v3_valfix_20261006' if split=='val' else 'grading_four_class_v3')/'labels'/split/(iid+'.txt')
    im=Image.open(src).convert('RGB');overlay=Image.new('RGBA',im.size,(0,0,0,0));od=ImageDraw.Draw(overlay)
    rr=[l.split() for l in lp.read_text(encoding='utf-8-sig').splitlines() if l.strip()]
    for n,line in enumerate(rr,1):
        c=int(line[0]);pts=np.asarray(line[1:],float).reshape(-1,2)*np.array(im.size)
        pts=[(int(round(x)),int(round(y))) for x,y in pts];od.polygon(pts,fill=(*palette[c],38));od.line(pts+[pts[0]],fill=(*palette[c],255),width=3)
        x,y=min(x for x,y in pts),min(y for x,y in pts);od.rectangle((x,y-28,x+190,y),fill=(255,255,255,225));od.text((x+3,y-26),f'实例{n} / 类{c}',font=font(19),fill=(*palette[c],255))
    annotated=Image.alpha_composite(im.convert('RGBA'),overlay).convert('RGB')
    ap=PACK/F7/'标注示例'/f'{iid}_{split}_当前标签叠图.png';ap.parent.mkdir(parents=True,exist_ok=True);annotated.save(ap)
    labelcopy=ap.with_suffix('.txt');shutil.copyfile(lp,labelcopy)
    row={'图像ID':iid,'集合':split,'原始来源':raw['source_image'],'原图SHA256':raw['image_sha256'],'标签来源':lp.relative_to(ROOT).as_posix(),'标签SHA256':sha(lp),'原图文件':target.relative_to(PACK).as_posix(),'叠图文件':ap.relative_to(PACK).as_posix(),'标注实例数':len(rr),'图性质':'原始图来自实际文件；叠图由实际标签绘制，无模型推理'};assets.append(row)
    tile=annotated.resize((768,432),Image.Resampling.LANCZOS);canvas=Image.new('RGB',(768,480),'white');canvas.paste(tile,(0,48));ImageDraw.Draw(canvas).text((12,8),f'{iid} / {split} / {len(rr)}个标注实例',font=font(23),fill='black');panels.append(canvas)
contact=Image.new('RGB',(1536,960),'white')
for i,im in enumerate(panels[:4]):contact.paste(im,((i%2)*768,(i//2)*480))
contact.save(PACK/F7/'标注实例_四例总览.png')
table(F7,'样本图与标签溯源.csv',assets)

augroot=ROOT/'data/grading_four_class_v3_augmented_delivery';augs=[]
with (augroot/'metadata/segmentation_augmentation_manifest.csv').open(encoding='utf-8-sig',newline='') as f:
    selected=[r for r in csv.DictReader(f) if any('wb_000002' in str(v) for v in r.values())]
table(F7,'增强示例实际参数.csv',selected)
for stem in ['wb_000002','wb_000002__aug01','wb_000002__aug02']:
    src=augroot/'images/train'/(stem+'.png');dest=PACK/F7/'实际增强对比'/(stem+'.png');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
    label=augroot/'labels/train'/(stem+'.txt');shutil.copyfile(label,dest.with_suffix('.txt'))
    im=Image.open(src).convert('RGB').resize((768,432),Image.Resampling.LANCZOS);panel=Image.new('RGB',(768,480),'white');panel.paste(im,(0,48));ImageDraw.Draw(panel).text((12,8),stem,font=font(24),fill='black');augs.append(panel)
ac=Image.new('RGB',(2304,480),'white')
for i,im in enumerate(augs):ac.paste(im,(i*768,0))
ac.save(PACK/F7/'原图与现成增强_实际文件对比.png')

# Structural drawings, explicitly schematic and supported by the .mmd source.
flows={
 '系统总流程':'''flowchart TD
 I[整图RGB/BGR] --> Y[冻结YOLO26n-seg M0]
 Y --> C[框 原类 raw_conf 原mask]
 C --> R[冻结R1属性编码器 M1]
 C --> G[冻结ECDF规则几何]
 R --> F[融合256 缺梗概率 原缺陷logit]
 G --> S[规则锚定残差 M2 仅小头已训练]
 F --> S
 S --> D[邻域证据与分层决策 M3]
 C --> D
 D --> Q[原缺陷信息质量排序 M4 仅小头已训练]
 F --> Q
 Q --> N[同最终类真实NMS M5]
 D --> N
 C --> P[可选 soft+prototype边界细化 M6]
 P --> O[原mask用于分级 细化mask仅用于输出]
 N --> K[统一keep索引切片]
 O --> K
 K --> Z[框 类 分数 rank mask 保留索引]
 classDef frozen fill:#dceaf7,stroke:#0072b2;
 classDef trained fill:#d9f0e5,stroke:#009e73;
 classDef optional fill:#f6e4ef,stroke:#cc79a7,stroke-dasharray:5 3;
 class Y,R,G frozen;
 class S,Q trained;
 class P,O optional;
''',
 'M0_检测器结构':'''flowchart LR
 I[RGB 输入] --> B[Conv C3k2 下采样]
 B --> A[SPPF C2PSA]
 A --> U[上采样 Concat 侧连4与6]
 U --> D[再下采样融合层13与10]
 U --> H[Segment26 输入16 19 22]
 D --> H
 H --> O[boxes classes mask系数 prototype32]
''',
 'M1_属性编码器':'''flowchart LR
 I[RGB+mask 4x160x160] --> V[5级stride2 Conv+Residual<br/>32 64 128 192 256]
 V --> P[GAP 256]
 G[几何6维] --> M[MLP 6-32-32]
 P --> C[Concat288 融合256 Dropout0.2]
 M --> C
 C --> A[缺陷logit]
 C --> S[缺梗概率]
 C --> R[历史score头/回退]
 C --> F[256维复用特征]
''',
 'M2_规则残差':'''flowchart LR
 M[原mask] --> G[受限白边清理 五几何]
 G --> E[冻结ECDF 规则分数]
 F[融合256] --> N[fit归一化 MLP256-32-1]
 N --> D[0.05 tanh 残差]
 E --> S[相加 clip0-1 x100]
 D --> S
''',
 'M3_分层与邻域':'''flowchart TD
 B[原类为0] --> U[保留不合格]
 A[raw>=.25 且score/stem有限 非原class0] --> E[框邻域IoU>=.85 各原类最大raw]
 E --> M[缺梗logit+邻域margin]
 M -->|缺梗| G3[三级]
 M -->|有梗| T[局部高斯边界修正]
 T --> G[一级 二级 或三级]
 X[其余无效候选] --> O[保留原类别]
''',
 'M4_质量排序':'''flowchart LR
 A[核心14] --> C[22维 fit归一化]
 B[邻域4] --> C
 D[原缺陷logit 熵 可用标志 0] --> C
 C --> M[MLP22-16-1 SiLU]
 M --> S[sigmoid x raw_conf]
 S --> O[非零类新rank]
 U[最终class0] --> R[保留raw_conf]
''',
 'M5_NMS与索引':'''flowchart LR
 I[N行框 最终类 rank raw_scope] --> S[scope优先 rank降序 原索引稳定]
 S --> D[同类boxIoU>=.7 真删除]
 D --> K[保留索引按原序返回 K<=N]
 K --> O[所有字段与两套mask共享keep]
''',
 'M6_可选边界':'''flowchart LR
 R[RGB3+soft1+prototype32] --> H[1x1 Conv36-32-16-1]
 H --> D[6tanh delta]
 M[原native mask] --> B[9x9膨胀减腐蚀 半径4px]
 B --> A[native基准logit 加 band乘resize delta]
 D --> A
 A --> O[细化mask仅分割输出]
 M --> G[原mask仍分级与排序]
''',
 '数据构建与划分':'''flowchart TD
 I[1285图 5752单类多边形] --> S[seed20260823 图像分层划分]
 S --> A[train1028 val128 test129]
 A --> Q[整图一致性审核]
 Q --> H[HOLD74图 374实例]
 Q --> D[保留图去精确重复7条]
 D --> F[正式1211图 5371实例]
 F --> T[train966 4277]
 F --> V[val116 518]
 F --> E[test129 576]
 T --> U[离线每图两份 train2898 12831]
 V --> O[仅val 7类修订覆盖层]
 T --> R[新增头图像角色773fit 193held]
'''
}
for name,flow in flows.items():md(F7,'结构图源/'+name+'.mmd',flow)
md(F7,'模块结构图_可渲染.md','# 根据代码绘制的模块示意图\n\n各连接以最终实现为准，非实验结果。冻结、训练小头和可选身份见M0–M6卡片；所有输入输出按原候选索引对齐。\n\n'+'\n\n'.join('## '+name+'\n\n```mermaid\n'+flow.strip()+'\n```' for name,flow in flows.items()))

# A clear independent PNG and SVG system schematic; drawing is not a result.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['svg.fonttype']='none'
fig,ax=plt.subplots(figsize=(15,10));ax.set_xlim(0,15);ax.set_ylim(0,10);ax.axis('off')
nodes={
 'I':(7.5,9.1,3.3,.6,'输入图像 / 无GT','#f5f5f5'),
 'Y':(7.5,7.9,4.6,.8,'M0 冻结YOLO26n-seg\n框 / 原类别 / raw conf / 原mask','#dceaf7'),
 'R':(3.5,6.4,4.6,.8,'M1 冻结R1编码器\nRGB+原mask / 几何6 → fusion256','#dceaf7'),
 'G':(10.5,6.4,4.2,.8,'M2 原mask几何 → 冻结ECDF\n规则分数 + 学习残差±5分','#d9f0e5'),
 'D':(6.8,4.8,6.6,.9,'M3 分层决策 + 邻域证据\n保留原class0 / 缺梗门控 / 尺寸边界','#fff3cd'),
 'Q':(4.4,3.25,5.1,.9,'M4 复用原缺陷信息质量排序\n22 → 16 → 1，合格类rank×raw','#d9f0e5'),
 'P':(11.2,3.25,4.6,.9,'M6 可选soft+prototype边界\n细化仅作output mask','#f6e4ef'),
 'N':(4.4,1.85,5.1,.8,'M5 同最终类真实NMS，IoU≥.7\n统一keep索引，K≤N','#fff3cd'),
 'O':(7.5,.55,8,.65,'最终框 / 类别 / 分数 / rank / mask / 保留索引','#f5f5f5')}
for name,(x,y,w,h,label,color) in nodes.items():
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=.10',edgecolor='#404040',facecolor=color,linewidth=1.3,linestyle='--' if name=='P' else '-'))
    ax.text(x,y,label,ha='center',va='center',fontsize=12)
def arrow(a,b):
    x,y,w,h,*_=nodes[a];xx,yy,ww,hh,*_=nodes[b];ax.annotate('',xy=(xx,yy+hh/2+.11),xytext=(x,y-h/2-.11),arrowprops=dict(arrowstyle='->',color='#555555',lw=1.5))
for a,b in [('I','Y'),('Y','R'),('Y','G'),('R','D'),('G','D'),('D','Q'),('Q','N'),('N','O'),('Y','P'),('P','O')]:arrow(a,b)
ax.annotate('',xy=(8.25,6.4),xytext=(5.9,6.4),arrowprops=dict(arrowstyle='->',color='#555555',lw=1.5))
ax.text(7.05,6.55,'fusion256',ha='center',fontsize=10,color='#444444')
ax.annotate('',xy=(1.75,3.25),xytext=(1.05,6.4),arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.22',color='#555555',lw=1.4))
ax.text(.18,4.8,'原缺陷\nlogit',ha='left',va='center',fontsize=10,color='#444444')
ax.text(.2,9.8,'根据代码绘制的示意图｜蓝：冻结；绿：新增已训练小头；黄：确定性决策；粉：可选',fontsize=11,color='#444444')
fig.savefig(PACK/F7/'系统总流程_代码示意.png',dpi=240,bbox_inches='tight',facecolor='white');fig.savefig(PACK/F7/'系统总流程_代码示意.svg',bbox_inches='tight',facecolor='white');plt.close(fig)

md(F7,'图像说明与中文图注.md','''
# 图像、图表与图注

## 图像性质与处理记录

原图为实际1280×720 PNG，逐字节复制；SHA与原标签路径见`样本图与标签溯源.csv`。叠图只依据正式人工/规则标签TXT绘制半透明填色、边界、实例序号和class编号，没有运行模型，也没有美化或修补果实。四例总览为便于阅读下采样至768×432并加标题；原尺寸单图独立保留。颜色0红/1蓝/2绿/3紫，与类别不是生物颜色映射。

增强图直接取已交付训练图，未重新抽样生成；原图/aug01/aug02及对应标签逐字节复制，实际参数在CSV中。三图拼版仅缩放和加标题，不能当新训练数据计数。没有生成虚构实验图、错误热图或不存在的mask预测。

系统PNG/SVG和所有`.mmd`为“根据代码绘制的示意图”，不是实际网络性能结果；SVG可编辑文本/形状，Mermaid可修改连接。M0–M6各有独立Mermaid图源；总流程PNG是清晰独立预览。示意图没有物理比例尺，不能添加mm标尺。

## 可直接调整的中文图注草稿

| 文件/示例 | 中文图注草稿 | 限制 |
|---|---|---|
| 原始样本/wb_000002_train_原始.png | 枸杞多实例图像示例：白色容器中不同朝向果实，局部可见手和工具遮挡。 | 工具附近局部可见目标是否纳入标注需原标注者说明；不据图宣称全目标完整标注。 |
| 原始样本/wb_000151_train_原始.png | 编号来源组151–500的训练图示例，展示多果摆放与局部接触。 | 编号组不是已验证采集批次。 |
| 原始样本/wb_000501_val_原始.png | 编号来源组501–900的验证图示例，展示外观与朝向差异。 | 不能由单图推断品种或采集时间。 |
| 原始样本/wb_000911_test_原始.png | 编号来源组901–1300的测试图示例，展示近邻实例和视角差异。 | 属已完成终测后的材料图，不参与再次选模。 |
| 标注实例_四例总览.png | 实例分割与外观等级标签示例。半透明区域和轮廓来自当前正式标签，数字依次表示实例行号及外观等级（0不合格、1一级、2二级、3三级）。 | 是参考标签叠图，不是模型预测。val使用valfix，其余V3。 |
| 标注示例/wb_000069_train_当前标签叠图.png | 含四个外观等级的同图标注示例，红、蓝、绿、紫轮廓依次表示不合格、一级、二级和三级。 | 等级来自实际标签；不根据示例外观重新判级或补造腐烂成因。 |
| 原图与现成增强_实际文件对比.png | 训练图像的离线增强示例。由左至右为原始图像、水平翻转与轻度光度扰动版本、仅光度扰动版本。参数来自该图既有增强清单。 | 此实例两个版本均未触发噪声/模糊；不能称本图展示了所有随机增强。 |
| 系统总流程_代码示意.png/.svg | 枸杞实例分割与分层外观分级系统示意。冻结检测器与独立属性编码器提供原始实例及复用特征，规则锚定残差、邻域证据和质量排序依次作用，重分类后真实同类NMS统一保留索引。可选边界分支仅改变输出掩膜。 | 根据代码绘制，颜色表示模块身份，无性能数据。 |

## 表格文件目录

正式表格散布在各材料目录，保持与解释同位置。`图表清单.csv`索引所有CSV、PNG、SVG、Mermaid文件；原始样本、标注叠图、参数表、权重清单、指标表均可独立引用。CSV为UTF-8 BOM，Excel可直接打开编辑。没有将全数据集或全部模型权重打包。
''')

issues=[
 ('Q01','高','原始数据属于自行采集/公开下载/混合？来源、授权是否可发表？','未找到记录','项目执行者/导师','采集部分不能定稿'),
 ('Q02','高','品种、鲜干状态、产地、地点日期、真实采集批次','未找到记录','采集人/项目执行者','不能凭外观和文件日期代填'),
 ('Q03','高','相机镜头、拍摄距离角度、光源曝光及背景搭建','PNG无EXIF与info；未找到记录','采集人','补实际记忆并标明回忆依据，不补常用值'),
 ('Q04','高','实物果重复、连拍/不同视角以及train/test是否同果','只有图像编号与来源代理组；hash无完全重复不等于无同果','采集人/数据管理者','限制独立泛化论证'),
 ('Q05','高','果梗是否纳入polygon、不可见部分是否补全、模糊/接触/局部工具遮挡目标纳入标准','未找到统一规范；代表图局部可见目标需要确认','原标注者/审核者','避免宣称完全可见实例全覆盖'),
 ('Q06','高','44条边界拟合样本混入3条val应怎样诚实报告','已由实际清单联接核验41train+3val','方法负责人/论文作者','不能宣称所有规则阈值仅train拟合；不回改历史'),
 ('Q07','中','标注者背景、独立审核/仲裁细则、工具安装版本','有任务分工和JSON版本，无完整操作日志','原标注者/主审','不能直接称双盲专家一致标注'),
 ('Q08','中','原始编号15个缺号及HOLD74图是否另有修复归档','当前正式导出仍为1211图；缺号原因未找到','数据管理者','不将猜测删除理由写入论文'),
 ('Q09','中','最终test混合来源GT（302双人、51主审、223冻结规则）与历史baseline test使用','当前来源已核验；早期首次访问完整命令/日期需旧日志','执行者/评价负责人','不得称全人工双人确认或从未使用test'),
 ('Q10','中','原ECDF参考值完整原件/精确下载权重记录/全部历史安装散列','当前锚点恢复与预训练SHA可用；部分原件未找到','执行者/代码管理者','复现边界说明；不要补造来源'),
 ('Q11','低','变量命名与最终模块主次是否按此包统一','primary old_defect_rank；prototype optional；原class0冻结','方法负责人/共同作者','统一图、表、正文，不采用早期神经替代规则叙事'),
 ('Q12','已明确限制','毫米标定、内在品质测量、新独立批次','px_per_mm=null，未找到新实物参照或内部品质测量','作者在局限性中披露','不通过写作补造实验'),
]
table(F8,'待补充问题与责任人.csv',[dict(zip(['问题ID','优先级','问题','现有证据/缺失','需要谁补充','写作影响'],r)) for r in issues])

coverage_groups={
 '01':('01_数据采集与数据集构建/数据采集与构建_逐项核验.md',[
 ('自行/公开/混合及来源','未找到原始来源记录，已明确本地原文件映射','Q01'),
 ('品种/鲜干/产地/批次','未找到记录','Q02'),('地点/时间/环境','未找到采集日志，示例背景直接观察','Q02'),
 ('设备/分辨率/距离/角度/光照/摆放','分辨率1280×720实读，其余无定量记录','Q03'),
 ('多果/遮挡/粘连/朝向/背景','随包真实样本可见；无全库频率','Q05'),('原始独立图/实例/每类','1285不同文件5752单类多边形；物理独立性未证','Q04'),
 ('原始和衍生图','原始、GT crop、两份增强、合成压力图区分',''),('清洗/去重/前后数','5752−374−1−6=5371；HOLD不删除原始','Q08'),
 ('重复果/连续帧/批次','hash无完全重复；没有实物关联ID','Q04'),('物理标定','明确没有，只有px/px²','Q12'),
 ('A–E采集表/数量表/类别表/原图图注/缺口','相应CSV、实际PNG、图注与问题表齐备','')]),
 '02':('02_分级依据_标注与预处理/分级_标注_预处理_逐项核验.md',[
 ('等级及定义','四类、优先级和缺梗三级非单调明确',''),('标准/文献/专家/自定义依据','项目操作规则，非商业标准验证',''),
 ('缺陷/梗/大小/形态/颜色','规则仅五几何、颜色权重0；缺陷梗先行',''),('公式/单位/权重/阈值/拟合','完整ECDF与分数；44条含3val','Q06'),
 ('人工/规则/模型标签及复核','腐烂、果梗、边界与混合test来源分开','Q07'),('可见外观及未实测内质','没有内在品质实测','Q12'),
 ('标注工具版本/格式','JSON版本实读；工具安装版本未确认；YOLO polygon','Q07'),('框/掩膜/等级/属性/连续分数','多边形GT、框派生、属性可用标志、规则分数',''),
 ('果梗/遮挡/重叠/模糊边界规范','未找到完整统一规范','Q05'),('人员/复核/分歧修订','分工与混合来源已提取，细则待补','Q07'),
 ('缺失标签和训练处理','426不合格stem/score unavailable，loss屏蔽',''),('标注版本及修改','V3与valfix七类修订，无几何变化',''),
 ('缩放/填充/裁剪/归一化/转换','按detector/R1/规则分阶段表',''),('增强操作/概率/幅度/次数','两份离线精确参数+R1在线+YOLO在线关闭',''),
 ('增强范围与前后数','train966→2898；val/test不增强',''),('三集合图/实例/每类统计','版本分行CSV并非混合',''),
 ('划分单位及seed/分层','图像、20260823、来源号段×实例档','Q04'),('同原图增强防跨集','所有衍生继承train，但同果跨照片不明','Q04'),
 ('新增模块内部fit/held及上下游关系','773/193；各模块实例数和历史encoder见过train已说明',''),('集合使用与test历史','开发val复用、test早期基线与终测已披露','Q09'),
 ('A–F规则/规范/参数/统计/图/清单','CSV+实际标注和增强图+源代码清单','')]),
 '03':('03_基础模型/基础模型_结构与证据.md',[
 ('准确名称/变体/来源/版本','checkpoint实际yaml与库版本','Q10'),('选择原因及设计考虑','有证据与未验证考虑区分',''),
 ('结构/输入/输出','实际24层、三尺度head、native尺寸',''),('预训练/历史checkpoint来源训练经历','9权重路径SHA及metadata；下载原件未找到','Q10'),
 ('训练/冻结范围','基线训练、外接小头阶段冻结；未改骨干',''),('原模型已观察问题/案例','历史匹配、重分类碰撞与边界误差记录',''),
 ('新增模块连接','一次forward、R1独立crop、rank与optionalproto',''),('A–D表/结构/问题/路径','权重CSV、层JSON、Mermaid、问题表','')]),
 '04':('04_改进模块与整体方法/整体方法_数据流_方案边界.md',[
 ('最终模块列表及身份','M0–M5主；M6可选；探索分开',''),('每模块问题/动机/来源','独立卡片，动机非验证收益，未补造文献',''),
 ('每模块输入维度/内部结构/关键参数','七卡片+公式CSV',''),('每模块输出/位置','统一字段表和总流程',''),
 ('每模块训练/目标/损失','独立卡片+训练表',''),('阈值/权重/残差范围确定','开发/内部held/规则校准区分','Q06'),
 ('异常/缺失回退','原类保持、旧score fallback、不可对齐报错、不虚称静默回退',''),('代码/配置/权重位置','逐文件证据+SHA',''),
 ('分层/规则残差/边界邻域/排序旧缺陷/NMS/细化','存在的全部逐模块核查',''),('原mask/细化mask分工','原mask分级，细化mask可选输出；GT重匹配风险',''),
 ('完整推理步骤及传输字段','6步数据流与N/K索引谱系',''),('改变类/score/rank/轮廓/数量分别在哪','字段表明确',''),
 ('同候选全类分布与邻域证据区别','捕获为实验/细化谱系；主融合用邻域top1',''),('训练人工标签与在线需求','在线无GT；训练参考混合来源',''),
 ('输出字段及未解决冲突','跨类重复/原class0错误锁定/非物理独立已写',''),('A–E模块/公式/图/总图/方案差别','七卡片、CSV、9份mmd及PNG/SVG','')]),
 '05':('05_训练环境与训练设置/训练环境_设置与执行记录.md',[
 ('CPU/GPU/显存/RAM/OS','当前probe与历史日志区分',''),('Python/framework/CUDA/library','版本冲突与单独增强环境并列','Q10'),
 ('源码补丁','本机native mask分块补丁和customfitness，附函数摘录',''),('每模块数据/标签/输入/batch','逐阶段参数CSV',''),
 ('epochs/optimizer/lr/schedule/wd/loss','7行各自设置，best16/131等实际',''),('初始化/预训练/冻解','最终冻上游；R1随机独立；小头末层零',''),
 ('种子/重复/早停/选checkpoint','epoch0、patience、续训记录、单系统链限制',''),('增强启用阶段','离线train、R1online、detectoronline0',''),
 ('阶段顺序依赖','按实际阶段顺序列出',''),('对照共同与差异','固定数据/上游，剂量与容量限制',''),('A–D环境表/组件表/顺序/命令日志','已获取，部分原完整命令未找到','Q10')]),
 '06':('06_评价指标与协议/评价方法_统计与复现协议.md',[
 ('检测/分割/分级/效率指标及方向','指标CSV15项',''),('公式/符号/单位/分母','G/L/C、类macro、score有效N',''),
 ('IoU/conf/NMS/maxdet','池.001 scope.25 NMS.7 max300 邻域.85',''),('预测GT匹配及是否考虑类','AP类相容greedy；工作点类无关Hungarian',''),
 ('AP与工作点不同规则','明确不混用',''),('conditional与allGT正确比例','C/L和C/G并列',''),('误放/误拒/新增/any-qualified/漏检/重复','逐项定义与限制',''),
 ('macro/micro总体汇总','4类等权与分母守卫',''),('种子/CI/重采样具体','test2000配对组内原图；时序非预注册；延迟3000',''),
 ('耗时步骤硬件样本重复','24×3×10，sync/warmup与decode排除',''),('评价库版本/修改/脚本','本机8.4.117函数摘录，historical patch边界','Q10'),
 ('A–D定义表/参数匹配表/步骤/未明确','齐备，无新训练/test推理','')]),
}
coverage=[]
for part,(location,items) in coverage_groups.items():
    for i,(question,answer,issue) in enumerate(items,1):coverage.append({'条目ID':part+f'-{i:02d}','写作部分':part,'要求':question,'回答/核验状态':answer,'材料文件':location,'待补问题':issue})
table(F0,'逐项覆盖核对.csv',coverage)
summary=[
 ['一 数据采集与构建','原始文件SHA、1285→1211清洗对账、类别、实际样图','manifest；原JSON/TXT/PNG；build_summary','真实来源/设备/日期/品种/批次/同果关系','采集人/执行者','可写组织清洗，采集段待补'],
 ['二 分级/标注/预处理','公式阈值/混合标签来源/增强参数/版本/划分','规则配置；标签实读；审核记录；增强metadata','统一标注规范/人员背景/44条含3val的报告措辞','原标注者/方法负责人','可起草；上述缺口需正式披露'],
 ['三 基础模型','结构/参数/权重SHA/真实best轮次/问题线索','checkpoint元数据；yaml；results/trace','精确预训练下载凭证、部分历史环境原件','代码管理者','足以写模型主体，来源边界保留'],
 ['四 各模块与整体方法','7模块卡、公式、索引谱系、主/可选/探索区分、结构图','runtime/policy/rank/proto训练源码与冻结协议','不补造跨类唯一果实/零误放保证','方法负责人/共同作者','足以写最终方法；按限制措辞'],
 ['五 环境与训练','7阶段参数、版本冲突、续训、选模、补丁摘录','args/config/history/checkpoint/环境报告','部分原完整命令与历史机器逐次快照','实验执行者','足以写可核实设置；未知项留缺'],
 ['六 评价指标与协议','指标公式/不同匹配/分母/Bootstrap/延迟协议','评价源码；安装库；冻结及analysis计划','早期test首次访问细节；实物独立性不具备','评价负责人/执行者','足以写当前评价方法；历史限制披露'],
]
table(F0,'六部分素材完整性汇总.csv',[dict(zip(['写作部分','已获取材料','证据来源','缺失信息','需要谁补充','是否足以开始写作'],r)) for r in summary])
md(F0,'阅读说明.md','''
# 第二章写作素材包：阅读说明

整理日期：2026-10-28。对象：共同作者撰写“材料与方法”。本包提供经本机文件核验的事实、参数、公式、索引、实际样图和缺口，**不是论文初稿，也不是新增实验报告**。证据由agent读取，仍需作者承担最终核实与署名责任；没有将自动整理标为“人工核实完毕”。

## 建议阅读顺序

1. 先看本目录`六部分素材完整性汇总.csv`和`逐项覆盖核对.csv`。
2. 按01–06逐节读取Markdown，搭配同目录可编辑CSV。尤其确认最终主方案old_defect_rank、可选prototype、valfix标签及训练期旧val的区别。
3. 第07取实际样本、标注、增强对比、中文图注与结构图源；第08按证据ID定位相对路径、字段/函数和SHA。关键小文件有逐字节副本；数据/权重只列准确路径与摘要，不随包整套上传。
4. 把第08问题CSV交给表中责任人补充。明确记录“未找到记录”和“需要项目执行者补充”的项目不可用常见拍摄条件或其他论文设置代填。

## 不应沿用的历史表述

- 1285张不同文件不等于1285次独立采集；5371实例不等于5371颗独立实物。
- 原始class0=wolfberry，四类class0=不合格；切勿混用。
- train966/val116/test129是正式原图；2898为增强训练输入；5088/4643是预测特征分支，不是正式GT实例总数。
- val原V3与valfix类别数不同；训练选模使用前者，当前评价用后者。44条有效边界拟合含3条val，不是全train-only。
- YOLO200轮完成但best16；R1目录e200但best131；不是拿最后一轮作最终模型。
- 最终没有改YOLO骨干，在线仍计算显式几何；原缺陷头复用作rank，最终不合格门控保留原class0。
- 同候选4类原始分数与邻域top1证据不同，主方案使用后者；同类NMS不保证跨类一果唯一。
- prototype是test前锁定的可选输出分支，原mask仍分级；评价重新匹配后计数可能变动。
- val为反复使用开发集；test有早期基线使用，10月27日冻结候选后统一终测。Bootstrap详细计划晚于点估计，不叫全程预注册。

## 文件与复现范围

所有源路径相对`wolfberry_yolo26_cv`项目根，`../new`和`../labels`是原始清单明确关联的兄弟目录。副本中的绝对路径保留历史原样，不代表另一电脑可直接运行。离开本机足以据此撰写并追溯素材，但重训练/全量重算仍需未入包的数据、PT缓存和权重。

本轮只读原始文件、CPU读取可信checkpoint元数据、读取既有test缓存；没有训练、模型推理、再次揭盲、改GT、改权重或改历史代码/配置。新增文件只在素材包与本轮核验脚本/报告目录。所有图注明实际图或代码示意，不生成虚构性能图。CSV为UTF-8 BOM，可由Excel编辑。
''')
md(F8,'证据索引与使用限制.md','''
# 证据索引与待补充问题

`证据索引.csv`给出E编号、项目相对路径、SHA256、大小和包内副本位置。各章节以E编号与字段/函数锚点引用来源；`代码函数行号索引.csv`便于不打开原机定位。源脚本副本不更改路径、参数或既有注释，故可能含历史阶段的状态（例如“test未执行”），应以各阶段日期和当前冻结终测记录解读，不能把旧计划状态当当前状态。

`权重元数据`由本轮CPU读取可信本地checkpoint获得，无forward/fit；state_shapes只列形状，未上传tensor权重。`data_facts_不含人员备注.json`去除个人审核备注整块，完整原文件路径/SHA另登记；不要把该派生摘要当成未经处理的原始审核表。

`原始文件逐图清单_SHA256.csv`（第01）覆盖1285组图像/JSON/TXT；`当前正式标签逐文件清单.csv`（第02）覆盖当前train/V3、val/valfix、test/V3共1211个TXT。目录与副本不是全数据交付。

`待补充问题与责任人.csv`是作者需要补齐的事实，不是要求继续追逐val或重跑test的实验任务。采集回忆如无法找到原始凭证，应明确写成执行者回忆而不是文件记录。

安装库摘录仅保留直接相关函数/类，带本机源文件SHA和原行号。第三方软件原有许可适用；本素材包不替代Ultralytics项目许可或私有数据使用授权。
''')

# Freeze the new builder itself; it is a packaging helper, not a scientific model change.
source(Path(__file__),'本素材包构建脚本；只读来源、派生表格和图',True)
anchors=[]
for r in evidence:
    p=ROOT/r['项目相对路径']
    if p.suffix not in ['.py','.ps1']:continue
    for i,line in enumerate(p.read_text(encoding='utf-8-sig').splitlines(),1):
        hit=re.match(r'\s*(def|class|function)\s+([^(:\s]+)',line)
        if hit:anchors.append({'证据ID':r['证据ID'],'项目相对路径':r['项目相对路径'],'行号':i,'类型':hit[1],'名称':hit[2]})
table(F8,'代码函数行号索引.csv',anchors)
table(F8,'证据索引.csv',evidence)

partslinks='\n'.join(f'- [{f}](./{f}/'+({'00_阅读说明与素材完整性清单':'阅读说明.md','01_数据采集与数据集构建':'数据采集与构建_逐项核验.md','02_分级依据_标注与预处理':'分级_标注_预处理_逐项核验.md','03_基础模型':'基础模型_结构与证据.md','04_改进模块与整体方法':'整体方法_数据流_方案边界.md','05_训练环境与训练设置':'训练环境_设置与执行记录.md','06_评价指标与协议':'评价方法_统计与复现协议.md','07_图片_表格与图注':'图像说明与中文图注.md','08_证据索引与待补充问题':'证据索引与使用限制.md'}[f])+')' for f in FOLDERS)
summarymd='\n'.join('| '+' | '.join(row)+' |' for row in summary)
md('', 'README.md',f'''
# 论文第二章：材料与方法——写作素材包

2026-10-28｜根据实际文件核验｜面向共同作者｜不是论文代写稿

{partslinks}

## 六部分完整性汇总

| 写作部分 | 已获取材料 | 证据来源 | 缺失信息 | 需要谁补充 | 是否足以开始写作 |
|---|---|---|---|---|---|
{summarymd}

请先读第00说明。主要新核验：5752→5371的完整清洗对账、44条边界拟合含3条val、YOLO最佳16/R1最佳131、主方案仍在线规则几何、原class0门控与原缺陷rank的分工、可选输出mask和原分级mask分工。未知采集条件不以其他论文代填。

图像来自实际文件，结构图根据代码绘制；CSV可用Excel编辑。证据副本保持源字节，全数据/权重/PT缓存不入包。完整来源路径/SHA、逐项覆盖表和待补问题见第08与第00。作者需独立确认科学表述及资料使用权。
''')

chartrows=[]
for p in sorted(PACK.rglob('*')):
    if p.is_file() and '关键证据副本' not in p.parts and p.suffix.lower() in ['.csv','.png','.svg','.mmd']:
        chartrows.append({'包内路径':p.relative_to(PACK).as_posix(),'格式':p.suffix,'性质':'根据代码绘制的示意图' if p.suffix in ['.mmd','.svg'] or '代码示意' in p.name else '来自实际文件/标签/参数的派生材料','字节数':p.stat().st_size})
table(F7,'图表清单.csv',chartrows)

# Validate copies and core arithmetic before publishing the package.
assert 5752-374-1-6==5371
assert sum(r['实例数'] for r in labels)==5371 and len(labels)==1211
assert Counter(r['split'] for r in labels)=={'train':966,'val':116,'test':129}
assert A['boundary_eligible_splits']=={'train':41,'val':3}
assert len(MODULES)==7 and len(coverage_groups)==6
for r in evidence:
    assert sha(ROOT/r['项目相对路径'])==r['SHA256'],r['项目相对路径']
    if r['素材包内副本']:assert sha(PACK/r['素材包内副本'])==r['SHA256']
validation={'status':'PASS','scope':'本包材料结构、静态链接、计数与源副本散列；不验证作者采集回忆或新增科学效果','date':'2026-10-28','sections':6,'coverage_items':len(coverage),'source_files':len(evidence),'label_files':len(labels),'formal_instances':5371,'raw_images':1285,'new_training':0,'new_model_inference':0,'image_assets':len(assets),'missing_questions':len(issues),'human_verified':False}
broken=[]
for p in PACK.rglob('*.md'):
    if '关键证据副本' in p.parts:continue
    for link in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
        if link.startswith(('http:','https:','#')):continue
        target=link.split('#')[0]
        if not (p.parent/target).exists():broken.append((str(p),link))
validation['broken_authored_markdown_links']=broken
if broken:raise ValueError(broken)
(PACK/F0/'本轮核验结果.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
manifest=[{'path':p.relative_to(PACK).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(PACK.rglob('*')) if p.is_file() and p.name!='素材包文件清单_SHA256.json']
(PACK/F8/'素材包文件清单_SHA256.json').write_text(json.dumps({'date':'2026-10-28','files':manifest,'scope':'不包含本清单自身'},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({**validation,'pack':str(PACK),'files':len(manifest)+1,'total_MB':round(sum(r['bytes'] for r in manifest)/1e6,2)},ensure_ascii=False,indent=2))

