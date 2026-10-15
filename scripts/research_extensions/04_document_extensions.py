"""Recalculate new extension metrics and install aggregate report/dashboard."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--xgboost-run', type=Path, required=True)
    parser.add_argument('--temporal-run', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import pandas as pd
    from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score, roc_auc_score, average_precision_score, matthews_corrcoef
    metrics = {
        'asd_f1': lambda y, p, s: f1_score(y, p, zero_division=0),
        'macro_f1': lambda y, p, s: f1_score(y, p, average='macro', zero_division=0),
        'precision': lambda y, p, s: precision_score(y, p, zero_division=0),
        'recall': lambda y, p, s: recall_score(y, p, zero_division=0),
        'balanced_accuracy': lambda y, p, s: balanced_accuracy_score(y, p),
        'roc_auc': lambda y, p, s: roc_auc_score(y, s),
        'average_precision': lambda y, p, s: average_precision_score(y, s),
        'mcc': lambda y, p, s: matthews_corrcoef(y, p)}
    root = args.project.resolve()
    reference = pd.read_csv(root / 'dataset/external/asd_feat/derived/e6_inputs/development_metadata.csv').set_index('filename')
    summaries, sources = [], []
    for kind, run in [('xgboost_shap', args.xgboost_run.resolve()), ('temporal', args.temporal_run.resolve())]:
        validation = json.loads((run / 'validation.json').read_text(encoding='utf-8'))
        require(validation['status'] == 'PASS' and validation['official_test_evaluated'] is False, 'Invalid completion evidence.')
        folds = pd.read_csv(run / 'fold_results.csv')
        oof = pd.read_csv(run / 'oof_predictions.csv')
        require(len(folds) == 15 and not folds.duplicated(['model', 'fold']).any(), 'Unexpected fold results.')
        require(folds['participant_overlap'].eq(0).all(), 'Participant overlap detected.')
        require(oof['model'].nunique() == 3 and len(oof) == 4530, 'Unexpected OOF coverage.')
        for model, table in oof.groupby('model'):
            require(table['filename'].is_unique and set(table['filename']) == set(reference.index), 'Recording coverage mismatch.')
            aligned = reference.loc[table['filename']]
            for column in ['sub_id', 'evaluation_fold', 'target']:
                require(np.array_equal(table[column].to_numpy(), aligned[column].to_numpy()), 'OOF metadata mismatch: ' + column)
            require(table['target'].isin([0, 1]).all() and table['prediction'].isin([0, 1]).all(), 'Nonbinary output.')
            require(np.isfinite(table['score']).all() and table['score'].between(0, 1).all(), 'Invalid scores.')
            require(np.array_equal(table['prediction'].to_numpy(), (table['score'].to_numpy() >= 0.5).astype(int)), 'Threshold mismatch.')
            require(table.groupby('sub_id')['evaluation_fold'].nunique().eq(1).all(), 'Participant fold mismatch.')
            values = {metric: [] for metric in metrics}
            for fold in range(1, 6):
                held = table[table['evaluation_fold'] == fold]
                saved = folds[(folds['model'] == model) & (folds['fold'] == fold)]
                require(len(saved) == 1 and len(held) > 0, 'Missing fold.')
                for metric, function in metrics.items():
                    score = float(function(held['target'], held['prediction'], held['score']))
                    require(np.isclose(score, float(saved.iloc[0][metric]), rtol=1e-8, atol=1e-10), 'Metric reproduction failed: ' + model + '/' + metric)
                    values[metric].append(score)
            row = {'experiment': kind, 'model': model}
            for metric, numbers in values.items():
                row[metric + '_mean'] = float(np.mean(numbers))
                row[metric + '_std'] = float(np.std(numbers, ddof=1))
            summaries.append(row)
        if kind == 'xgboost_shap':
            checks = pd.read_csv(run / 'shap_validation.csv')
            require(len(checks) == 15 and checks['shap_reconstruction'].eq('PASS').all() and checks['saved_inference'].eq('PASS').all(), 'SHAP evidence incomplete.')
        else:
            checks = pd.read_csv(run / 'saved_inference_validation.csv')
            require(len(checks) == 15 and checks['saved_inference'].eq('PASS').all(), 'Temporal inference evidence incomplete.')
        sources.append({'experiment': kind, 'run': str(run), 'file_sha256': {name: hashlib.sha256((run / name).read_bytes()).hexdigest() for name in ['validation.json', 'experiment_config.json', 'fold_results.csv', 'oof_predictions.csv']}})
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    backup = root / 'documentation_backups' / ('extensions_' + stamp)
    docs = root / 'docs'
    page = root / 'dashboard/pages/10_Research_Extensions.py'
    targets = [docs / 'RESEARCH_EXTENSION_RESULTS.md', docs / 'RESEARCH_EXTENSION_EVIDENCE.json', page]
    for path in targets:
        if path.exists():
            destination = backup / path.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    docs.mkdir(parents=True, exist_ok=True)
    page.parent.mkdir(parents=True, exist_ok=True)
    evidence = {'status': 'PASS', 'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'team': ['Muhammad Danial', 'Hamza Rasheed', 'Muhammad Freeh'], 'scope': 'Exploratory ASD-FEAT development extensions only', 'metric_recalculation': 'PASS', 'sources': sources, 'summary': summaries, 'official_test_evaluated': False}
    table = pd.DataFrame(summaries)
    columns = ['model', 'asd_f1_mean', 'asd_f1_std', 'precision_mean', 'recall_mean', 'roc_auc_mean', 'average_precision_mean']
    lines = ['# Research extension results', '', 'Team: Muhammad Danial, Hamza Rasheed, Muhammad Freeh.', '', 'Six exploratory candidates were evaluated using the existing five participant-disjoint folds: 1,510 recordings from 259 participants, with 113 positive recordings. Threshold 0.5 was fixed. No official test was evaluated and no frozen model was replaced.', '', '| Model | Mean ASD F1 | SD | Precision | Recall | ROC-AUC | AP |', '|---|---:|---:|---:|---:|---:|---:|']
    for _, row in table[columns].iterrows():
        lines.append('| ' + str(row['model']) + ' | ' + ' | '.join(f'{row[column]:.4f}' for column in columns[1:]) + ' |')
    lines += ['', '## Interpretation', '', 'The temporal matched mean-pooling LR control has higher mean ASD F1 than LSTM and TCN in this fixed configuration. This does not establish that temporal learning is generally ineffective. XGBoost uses raw features, whereas the earlier selected LR uses PCA; algorithm and representation differences must both be considered. Previous reported RGB-plus-behavior LR mean ASD F1 was 0.3142; the new extensions provide no reason to silently replace its frozen artifact.', '', 'SHAP explained all held-out XGBoost predictions in raw logit space, with reconstruction and saved inference checks. Temporal models also passed saved inference and OOF coverage checks. All 30 fold models are new experimental artifacts.', '', 'Stored RGB array order was retained in 32 contiguous bins. Timestamps and audio synchronization were not independently verified. Behavior inputs remain human-coded. SHAP reflects fitted feature attribution, not causal or clinical evidence.', '', 'These extensions were designed after earlier official-test results had been seen. They are exploratory development comparisons, without nested model selection or significance claims. Fold standard deviations are not confidence intervals. Model scores are uncalibrated and do not diagnose autism or determine severity.', '', '## Remaining work', '', 'Dataset permission review, verified literature synthesis and DREAM feasibility remain separate reviews. The research report, presentation and README must be updated to reflect these new findings before final publication.', '', '## Verification', '', 'All 15 fold metric rows per experiment were recalculated from saved OOF predictions. Recording IDs, targets, participant IDs, fold assignments and threshold predictions matched the development metadata. Source hashes are recorded in RESEARCH_EXTENSION_EVIDENCE.json. No models were loaded or fitted during this documentation stage.']
    targets[0].write_text('\n'.join(lines) + '\n', encoding='utf-8')
    targets[1].write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    source = '''"""Aggregate development extension results; no model loading."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st

st.title("Research Extensions: XGBoost, SHAP and Temporal Models")
path = Path(__file__).resolve().parents[2] / "docs/RESEARCH_EXTENSION_EVIDENCE.json"
if not path.is_file():
    st.info("Run the extension documentation stage first.")
    st.stop()
evidence = json.loads(path.read_text(encoding="utf-8"))
if evidence.get("status") != "PASS":
    st.error("Validated extension evidence is unavailable.")
    st.stop()
st.caption("ASD-FEAT exploratory development comparison | 1,510 recordings | 259 participants | five participant folds | threshold 0.5")
summary = pd.DataFrame(evidence["summary"])
columns = ["model", "asd_f1_mean", "asd_f1_std", "precision_mean", "recall_mean", "roc_auc_mean", "average_precision_mean"]
st.dataframe(summary[columns].round(4), hide_index=True, width="stretch")
st.bar_chart(summary.set_index("model")[["asd_f1_mean"]], horizontal=True)
st.write("LSTM and TCN did not exceed the matched mean-pooling LR control in this fixed configuration. Existing frozen models are preserved.")
st.write("XGBoost SHAP reconstruction, temporal saved inference and OOF coverage checks passed. Fold metrics were recalculated from saved predictions.")
st.info("Development research only. Scores do not diagnose autism or determine severity. Behavior features are human-coded; stored RGB order was retained without independently verified timestamps. No official test was evaluated for these extensions.")
st.caption("Team: Muhammad Danial, Hamza Rasheed, Muhammad Freeh")
'''
    compile(source, str(page), 'exec')
    page.write_text(source, encoding='utf-8')
    for path in targets:
        print('Saved:', path)
    print('Metric recalculation and dashboard Python syntax: PASS')
    print('No models loaded or retrained. No predictions changed.')


if __name__ == '__main__':
    main()
