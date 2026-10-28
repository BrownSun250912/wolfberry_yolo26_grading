"""Audit saved test results and bilingual edits, then write the final delivery report."""
from pathlib import Path
import datetime as dt, hashlib, json, re
import numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/reports/final_test_preflight_20261027'
RUN=BASE/'test_replay'
AN=BASE/'test_analysis'
P=ROOT/'paper/manuscript_20261025'
ARMS=['raw_YOLO','raw_YOLO_realNMS','RR22_reference','original_defect_rank','original_defect_rank_prototype']


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def number_tokens(text):return re.findall(r'-?\d+(?:\.\d+)?',text.replace('−','-'))


def table(text,caption):
    after=text.split(caption,1)[1]
    lines=[];started=False
    for line in after.splitlines():
        if line.startswith('|'):lines.append(line);started=True
        elif started:break
    return [[s.strip() for s in line.strip('|').split('|')] for line in lines[2:]]


def main():
    checks=[]
    def check(name,passed,detail=None):checks.append(dict(check=name,passed=bool(passed),detail=detail))
    status=read(AN/'analysis_status.json');run=read(RUN/'replay_report.json');gate=read(BASE/'VAL_REPLAY_PASS.json')
    frozen=read(BASE/'FROZEN_INPUTS.json');update=read(P/'working/FINAL_TEST_UPDATE_20261027.json')
    m=pd.read_csv(RUN/'metrics.csv').set_index('arm');ci=pd.read_csv(AN/'paired_contrasts.csv')
    changes=read(AN/'refined_matching_changes.json')
    check('Test evaluation completed on all 129 images',run['status']=='PASS' and run['split']=='test' and run['images']==129)
    check('Test evaluation marker exists',read(BASE/'FINAL_TEST_STARTED.json')['output']==str(RUN))
    check('Five fixed arms, no test selection',list(m.index)==ARMS and run['arms']==ARMS)
    check('129 per-image prediction caches',len(list((RUN/'predictions').glob('*.pt')))==129)
    check('All fixed arms contain all 576 GT',(m.GT==576).all())
    check('All fixed arms localize 560 GT',(m.localized==560).all())
    check('One completed frozen-test evaluation pass recorded',run['test_model_inference_runs']==1)
    check('Cached bootstrap complete and point values reproduced',status['status']=='COMPLETE' and status['point_reproduction']=='PASS' and not status['test_inference_repeated'])
    check('All 2000 paired replicates valid',len(ci)==60 and (ci.valid_replicates==2000).all() and (ci.invalid_replicates==0).all())
    for item in frozen['artifacts']+frozen['metadata']:
        check('Frozen dependency '+item['path'],sha(item['path'])==item['sha256'])
    for path,digest in gate['source_hashes'].items():check('Frozen runtime source '+path,sha(path)==digest)
    for item in [r for r in frozen['images'] if r['split']=='test']:
        check('Test image unchanged '+item['image_id'],sha(item['image_path'])==item['image_sha256'])
        check('Test label unchanged '+item['image_id'],sha(item['label_path'])==item['label_sha256'])
    check('Source/manifest hashes verified by test evaluator',run['frozen_inputs_sha256']==sha(BASE/'FROZEN_INPUTS.json') and run['sealed_sources_unchanged'])
    zh=(P/'MANUSCRIPT_ZH.md').read_text(encoding='utf-8');en=(P/'MANUSCRIPT_EN.md').read_text(encoding='utf-8')
    backup=Path(update['backup_directory'])
    check('Original drafts and documentation backed up',all(sha(backup/name)==digest for name,digest in update['before_sha256'].items()))
    for name,digest in update['after_sha256'].items():check('Updated manuscript/document hash '+name,sha(P/name)==digest)
    zref=zh.split('## 参考文献',1)[1];eref=en.split('## References',1)[1]
    check('Chinese references unchanged',zref==(backup/'MANUSCRIPT_ZH.md').read_text(encoding='utf-8').split('## 参考文献',1)[1])
    check('English references unchanged',eref==(backup/'MANUSCRIPT_EN.md').read_text(encoding='utf-8').split('## References',1)[1])
    for lang,text in [('ZH',zh),('EN',en)]:
        check(lang+' contains all 14 original references',len(re.findall(r'^\[\d+\]',text,re.M))==14)
        check(lang+' final test protocol/results included','### 4.4' in text and '### 5.7' in text)
        for token in ['80.72','78.04','69.12','481/560','480/560','1.43','4.08','0.27','1.70','223','25']:
            check(lang+' key result '+token,token in text)
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if target.startswith(('F:/','F:\\')):check(lang+' local artifact '+target,Path(target).is_file())
        check(lang+' balanced display-math delimiters',text.count('$$')%2==0)
    for token in ['尚未完成最终测试','本文也不报告当前锁定流程的test结果','preferred candidate for future test evaluation','have not received their final test evaluation','The next evaluation should execute']:
        check('Obsolete test-status claim removed: '+token,token not in zh+en)
    t8z=table(zh,'表8 ');t8e=table(en,'Table 8.')
    check('Five Table 8 rows in both languages',len(t8z)==len(t8e)==5)
    for i,a in enumerate(ARMS):
        expected=[f"{100*m.loc[a,'Mask_AP50']:.2f}",f"{100*m.loc[a,'Mask_AP50_95']:.2f}",f"{int(m.loc[a,'correct_grade'])}/{int(m.loc[a,'localized'])}",
            f"{100*m.loc[a,'conditional_grade_accuracy']:.2f}",f"{100*m.loc[a,'correct_per_all_GT']:.2f}",f"{100*m.loc[a,'operating_macro_F1_rawscope']:.2f}"]
        check('Table 8 '+a+' values in both languages',t8z[i][1:]==t8e[i][1:]==expected)
    for zcap,ecap,nrows in [('表9 ','Table 9.',6),('表A4 ','Table A4.',4)]:
        zt,et=table(zh,zcap),table(en,ecap)
        check(zcap+' bilingual row count',len(zt)==len(et)==nrows)
        check(zcap+' bilingual numerical cells identical',[number_tokens(' '.join(r[1:])) for r in zt]==[number_tokens(' '.join(r[1:])) for r in et])
    # Independent source checks for the new interval and class tables.
    pairs=[(ARMS[1],ARMS[0]),(ARMS[2],ARMS[1]),(ARMS[3],ARMS[2]),(ARMS[3],ARMS[0]),(ARMS[3],ARMS[1]),(ARMS[4],ARMS[3])]
    t9=table(zh,'表9 ')
    for i,(a,b) in enumerate(pairs):
        ep='Mask_AP50_95' if i==5 else 'Mask_AP50';r=ci[(ci.candidate==a)&(ci.reference==b)&(ci.endpoint==ep)].iloc[0]
        check('Table 9 source values '+str(i),t9[i][2:]==[f'{100*r.point_delta:.2f}',f'{100*r.ci95_low:.2f}',f'{100*r.ci95_high:.2f}'])
    ta4=table(zh,'表A4 ')
    for c in range(4):
        expected=[f"{100*m.loc[a,f'class{c}_Mask_AP50']:.2f} / {100*m.loc[a,f'class{c}_Mask_AP95']:.2f}" for a in [ARMS[0],ARMS[2],ARMS[3],ARMS[4]]]
        check('Table A4 class '+str(c),ta4[c][1:]==expected)
    check('One refined assignment change retained',len(changes['changes'])==1 and changes['changes'][0]['instance_id']=='wb_000911__i001')
    check('Primary/refined correctness and risk not concealed',m.loc[ARMS[3],'correct_grade']==481 and m.loc[ARMS[4],'correct_grade']==480 and m.loc[ARMS[4],'qualified_rejected']==38)
    passed=all(c['passed'] for c in checks)
    draft_validation=dict(status='PASS' if passed else 'FAIL',scope='Final test source integrity, saved-metric reproduction and bilingual test-update checks; not an independent test inference rerun or a new citation verification.',
        check_count=len(checks),failed_checks=[c for c in checks if not c['passed']],checks=checks,
        manuscript_sha256={name:sha(P/name) for name in ['MANUSCRIPT_ZH.md','MANUSCRIPT_EN.md']},generated_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    write(P/'working/FINAL_TEST_DRAFT_VALIDATION_20261027.json',draft_validation)
    fallacies=[
        ('Simpson reversal','CHECKED_WITH_LIMIT','All four source groups have higher correct/all-GT for primary than raw. Per-group AP is not averaged; smallest group lacks UQ, so no universal subgroup-AP claim.'),
        ('Ecological fallacy','DISCLOSED_LIMITATION','Bootstrap unit is image; source ranges are not proven physical-fruit or harvest-batch independence.'),
        ('Selection/Berkson bias','DISCLOSED_LIMITATION','Existing curated dataset, mixed rule/review labels; no claim of population-representative fruit quality.'),
        ('Collider bias','DISCLOSED_LIMITATION','Conditional grading selects localized instances; correct/all-GT and localization are also reported, not only the selected subset.'),
        ('Base-rate neglect','CHECKED','UQ prevalence reported as 25/576; escape 2/25 and qualified rejection denominators retained.'),
        ('Regression to mean','NOT_APPLICABLE_AS_PREPOST_TEST','Paired fixed algorithms on the same frozen images, not a selected extreme-score pre/post intervention.'),
        ('Survivorship bias','CHECKED_WITH_LIMIT','All129 images and576GT retained;16 unlocalized GT counted separately. Conditional560 denominator is explicit.'),
        ('Look-elsewhere effect','DISCLOSED_LIMITATION','Five frozen arms and six contrasts reported, including RR22-minus-NMS CI crossing zero. Pointwise intervals unadjusted; no stars/p-values.'),
        ('Forking paths','DISCLOSED_LIMITATION','Development adaptation and early baseline test use retained; methods and primary endpoint frozen before current test. Bootstrap details recorded after point estimates.'),
        ('Causal overclaim','CHECKED_WITH_LIMIT','Controlled algorithm comparison supports performance differences on this partition, not causal nutrition/commercial sorting outcomes.'),
        ('Reverse causality','NOT_APPLICABLE_TO_FIXED_ALGORITHM_CONTRAST','Prediction precedes GT evaluation; no observational causal direction between fruit attributes and commercial outcomes is asserted.')]
    write(AN/'statistical_interpretation_audit.json',dict(verification_status='ANALYZED',coverage='11/11',
        primary_assumption='Image blocks approximately independent within source ranges; not established for physical fruits/batches.',
        normality_and_equal_variance='Not required for the chosen nonparametric paired image-bootstrap; no t-test/ANOVA performed.',
        missing_data='No image/GT drop, all2000 AP draws contain four target classes; attribute/continuous-score test targets unavailable and omitted.',
        zero_difference='Degenerate zero paired escape difference is an observed invariance, not equivalence or absence of population risk.',
        rows=[dict(fallacy=n,status=s,assessment=a) for n,s,a in fallacies]))
    names=['原YOLO26','原YOLO26＋同类NMS','RR22分层参考','冻结主方案：原缺陷排序','主方案＋输出掩膜细化']
    rows=[]
    for name,a in zip(names,ARMS):
        v=m.loc[a]
        rows.append(f"| {name} | {100*v.Mask_AP50:.4f}% | {100*v.Mask_AP50_95:.4f}% | {int(v.correct_grade)}/{int(v.localized)} | {100*v.operating_macro_F1_rawscope:.4f}% | {int(v.qualified_rejected)} |")
    contrastrows=[]
    cnames=['同类NMS−原YOLO','RR22−同类NMS原YOLO','主方案−RR22（主终点）','主方案−原YOLO','主方案−同类NMS原YOLO','输出细化−主方案']
    for i,(a,b) in enumerate(pairs):
        ep='Mask_AP50_95' if i==5 else 'Mask_AP50';r=ci[(ci.candidate==a)&(ci.reference==b)&(ci.endpoint==ep)].iloc[0]
        contrastrows.append(f"| {cnames[i]} | {'AP50–95' if i==5 else 'AP50'} | {100*r.point_delta:.4f} | [{100*r.ci95_low:.4f}, {100*r.ci95_high:.4f}] |")
    report=f'''# 冻结方案最终测试报告（2026-10-27）

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent；statistical-analysis
- Origin Mode: run + validate（统计解释）
- Origin Date: {dt.datetime.now(dt.timezone.utc).isoformat()}
- Verification Status: ANALYZED（完整val实际重放已通过；test执行一次，后续仅使用缓存复算与分析）
- Version Label: final_test_20261027_v1

## 执行结论

已按用户“继续做吧”的授权执行一次固定方案test评估。129张图像、576个GT全部纳入，五组方案各定位560个实例；没有新训练、修改标签、调整阈值或依据test更换主方案。原始测试报告为PASS。模型、代码和数据哈希检查以及中英文稿更新验证{'通过' if passed else '未通过'}，本次闭环检查{len(checks)}项，失败{len([c for c in checks if not c['passed']])}项。

实际命令：`.venv/Scripts/python.exe -X utf8 scripts/final_test_evaluation_20261027.py --split test --execute-final-test`；工作目录为`{ROOT.as_posix()}`，进程退出码0。主评估使用同次检测前向计算五组方案，第一张图另有预测器初始化预热。全129图的循环墙钟日志为23.5秒，包含GT处理、IoU、掩膜哈希和缓存写入，不能除以129作为论文模型延迟。共享前向的复用不影响冻结预测与各组独立匹配，但不构成五组单独计时。

## 一、最终主表（test）

| 固定方案 | Mask AP50 | Mask AP50–95 | 分级正确/定位 | 工作阈值宏F1 | 匹配合格误拒 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

主方案的条件分级正确率为85.8929%（481/560），正确/全部GT为83.5069%（481/576）；原YOLO分别为81.4286%（456/560）和79.1667%（456/576）。原YOLO到主方案净增加25个正确分级，排序器本身不增加已定位实例的分级正确数，而是改变质量排序与AP。

原YOLOAP50为78.0363%，主方案80.7203%，累计提升2.6840个百分点；相同额外NMS下的增量为1.8337个百分点；主方案相对RR22的增量为0.9719个百分点。开发集对应累计提升3.6196个百分点，不能以开发成绩87.1013%替代最终test成绩。

## 二、配对差值区间

| 固定对照 | 指标 | 点估计（百分点） | 95%描述性区间（百分点） |
|---|---|---:|---|
{chr(10).join(contrastrows)}

主终点在测试前冻结，为主方案相对RR22的Mask AP50。2,000次整图配对、来源分层bootstrap使用seed20261027；四个来源范围有14、36、38、41张图。所有抽样保持五组图像重复次数一致，全部2,000次均包含四个GT类别。点指标及操作风险计数均由缓存复算，与首次结果一致。

具体bootstrap参数在查看test点估计后记录、计算区间前固定；不是事前注册的完整分析。区间为逐项百分位95%描述，未作多重比较校正，不报告p值或显著性星号。它们不消除反复开发、早期test使用或果实/拍摄批次关联。RR22相对同NMS原YOLO的AP50区间跨零，完整保留该结果。

## 三、输出细化的一项负结果

输出细化的AP50不变，AP50–95提升1.0851个百分点，95%区间0.6292至1.6717个百分点；正确分级却由481/560变为480/560。

已核对仅一处类别无关匹配改变：`wb_000911__i001`（GT三级）原匹配候选行1（三级，原IoU0.912555），细化后匹配候选行2（不合格，细化IoU0.918377）。所有候选标签、排序和NMS保留行未改变；变化来自输出几何参与匹配。因此保留原分级输入保证的是候选标签计算一致，不能保证重匹配后的评价正确数一致，也不能把该结果描述成分级完全无损。

## 四、风险与来源审计

- 全部25个不合格GT均定位，五组匹配漏放均为2/25（8%），任一合格输出暴露均为3/25（12%）。主方案相对RR22的新增漏放为0，绝对漏放仍为2。
- 主方案匹配合格误拒37个；输出细化38个。按全部551个合格GT作参考分母分别为6.7151%和6.8966%；匹配子集分母与全部GT分母不能混用。
- 测试标签为302个双审一致、51个人工裁决、223个冻结规则沿用；不是全部双人独立金标准。缺陷、果梗及连续分数独立test目标未核实，相关test MAE/属性准确率不报告。
- 四个来源组中，原YOLO正确数42/123/144/147，主方案45/131/151/154，输出细化45/131/151/153；分母分别51/143/175/191个定位实例。来源组不能冒充独立采收批次；最小组没有不合格GT。
- 对11类统计解释风险进行了逐项审计：报告条件筛选、基率、适应性选择、多重比较和分析单位的局限；未将零差值解释为总体等效，未将现有结果扩展为营养品质或商业分选安全。

## 五、中英文初稿更新

已在原初稿位置补入第4.4节终测协议、第5.7节结果、表8/9与附录表A4，并更新摘要、数据集表、标签来源、讨论和结论。原开发集消融、负结果和14项参考文献保留，引用内容没有改写。当前共有9张正文表、4张附录表；现有方法图与开发实验图继续保留。

- [中文更新稿]({(P/'MANUSCRIPT_ZH.md').as_posix()})
- [英文更新稿]({(P/'MANUSCRIPT_EN.md').as_posix()})
- [本次稿件/结果一致性验证]({(P/'working/FINAL_TEST_DRAFT_VALIDATION_20261027.json').as_posix()})
- [更新前原稿备份]({(backup/'MANUSCRIPT_ZH.md').as_posix()})

终测已完成。后续收敛工作应集中在全文审阅、目标期刊格式、采集设备与标注说明、作者贡献/资助/利益冲突/数据及AI使用声明。不继续把本test用作优化验证集；若提出新方法，应另行定义评价边界。现有证据能支撑的结论是既有数据划分上的固定方案性能差异，不能保证中科院二区录用，也不能据此宣称现场商业可靠性。

## 证据与复核入口

- [首次测试原始指标]({(RUN/'metrics.csv').as_posix()})
- [首次测试完整运行记录]({(RUN/'replay_report.json').as_posix()})
- [按方案保存的实例结果与混淆矩阵]({(RUN/'details.json').as_posix()})
- [配对区间全表]({(AN/'paired_contrasts.csv').as_posix()})
- [各方案绝对指标区间]({(AN/'absolute_intervals.csv').as_posix()})
- [bootstrap方案与来源]({(AN/'analysis_plan.json').as_posix()})
- [掩膜细化匹配变化]({(AN/'refined_matching_changes.json').as_posix()})
- [11项统计解释审计]({(AN/'statistical_interpretation_audit.json').as_posix()})
- [测试前核验报告]({(BASE/'终测前核验报告.md').as_posix()})（历史时点记录，不覆盖为当前状态）
'''
    (BASE/'最终测试报告.md').write_text(report,encoding='utf-8')
    author_note='''# 2026-10-27锁定测试结果与写作口径

最终测试已完成，中英文正文同步更新。当前主结果须使用test的78.04%→80.72% Mask AP50（+2.68个百分点），原缺陷排序相对RR22另贡献0.97个百分点；配对区间分别为[1.43,4.08]和[0.27,1.70]个百分点。分级正确数456/560→481/560，输出细化AP50–95为69.12%，重匹配分级正确数480/560。

开发集83.48%→87.10%仅作为开发证据，不写成test成绩。表8为最终五组主结果，表9为配对区间，附录表A4为test逐类指标。第4.4节交代冻结协议和统计范围，第5.7节完整报告结果与负结果。

不能删去的限制：302双审/51裁决/223规则标签；仅25个不合格实例；早期基线历史test使用；图像分开不等于实物果或采集批次独立；输出细化出现一个匹配误拒增量；同类NMS与排序增量不能全部归为新视觉主干创新。所有区间为逐项描述性区间。

还需作者补齐实际相机/镜头、拍摄距离和光源资料，确认作者单位、贡献、基金、利益冲突、数据开放与AI使用声明，确定期刊后按模板整理。不得根据test重新调参、换种子或挑选新主方案。

'''+f'[最终测试报告]({(BASE/"最终测试报告.md").as_posix()}) · [中文正文]({(P/"MANUSCRIPT_ZH.md").as_posix()}) · [英文正文]({(P/"MANUSCRIPT_EN.md").as_posix()})\n'
    (ROOT/'paper/EXPERIMENT_UPDATE_20261027.md').write_text(author_note,encoding='utf-8')
    # The final inventory records evidence bytes, not a new inference pass.
    inventory=[]
    for folder in [RUN,AN]:
        for p in sorted(folder.rglob('*')):
            if p.is_file():inventory.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    final=dict(status='COMPLETE_TEST_AND_BILINGUAL_UPDATE' if passed else 'FAILED_FINAL_CHECKS',
        test_evaluation_passes_this_stage=1, subsequent_analysis_model_inference=0, primary_arm=ARMS[3],
        test_images=129, test_GT=576, primary_AP50=float(m.loc[ARMS[3],'Mask_AP50']),
        primary_correct=int(m.loc[ARMS[3],'correct_grade']), refined_correct=int(m.loc[ARMS[4],'correct_grade']),
        validation_checks=len(checks),failed_checks=[c for c in checks if not c['passed']],
        primary_contrast=status['primary'], cumulative_contrast=status['cumulative'],
        report=str(BASE/'最终测试报告.md'),report_sha256=sha(BASE/'最终测试报告.md'),
        manuscript_sha256=draft_validation['manuscript_sha256'], artifact_inventory=inventory,
        completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    write(BASE/'FINAL_TEST_STATUS.json',final)
    print(json.dumps({k:v for k,v in final.items() if k!='artifact_inventory'},ensure_ascii=False,indent=2))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
