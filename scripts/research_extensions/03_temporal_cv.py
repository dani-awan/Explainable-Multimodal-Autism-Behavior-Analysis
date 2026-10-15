"""Exploratory RGB-sequence plus human-coded behavior CV; development data only."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequence = nn.LSTM(32, 32, batch_first=True)
        self.head = nn.Sequential(nn.Linear(44, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, sequence, behavior):
        _, (hidden, _) = self.sequence(sequence)
        return self.head(torch.cat([hidden[-1], behavior], dim=1)).squeeze(1)


class TCNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Conv1d(32, 32, 3, dilation=d) for d in [1, 2, 4]])
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Sequential(nn.Linear(44, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, sequence, behavior):
        hidden = sequence.transpose(1, 2)
        for dilation, layer in zip([1, 2, 4], self.layers):
            hidden = hidden + self.dropout(torch.relu(layer(nn.functional.pad(hidden, (2*dilation, 0)))))
        return self.head(torch.cat([hidden.mean(dim=2), behavior], dim=1)).squeeze(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--sequence-run', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    from scipy.special import expit
    import joblib
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import (f1_score, precision_score, recall_score, balanced_accuracy_score,
        roc_auc_score, average_precision_score, confusion_matrix, matthews_corrcoef)
    from threadpoolctl import threadpool_limits
    root = args.project.resolve()
    cache = args.sequence_run.resolve()
    input_dir = root / 'dataset/external/asd_feat/derived/e6_inputs'
    validation_path = cache / 'sequence_validation.json'
    require(validation_path.is_file(), 'Sequence validation evidence missing.')
    validation = json.loads(validation_path.read_text(encoding='utf-8'))
    require(validation['status'] == 'PASS' and not validation['official_test_processed'], 'Invalid sequence evidence.')
    for name, key in [('rgb_sequences.npy', 'sequence_sha256'),
                      ('sequence_metadata.csv', 'metadata_sha256'), ('bin_window_counts.npy', 'counts_sha256')]:
        require(sha(cache / name) == validation[key], f'Cache hash mismatch: {name}')
    for name, expected in validation['input_sha256'].items():
        require(sha(root / name) == expected, f'Sequence source input changed: {name}')
    raw = np.load(cache / 'rgb_sequences.npy', mmap_mode='r', allow_pickle=False)
    require(raw.shape == (1510, 32, 1024) and raw.dtype == np.float32, 'Unexpected sequence cache.')
    metadata = pd.read_csv(cache / 'sequence_metadata.csv')
    original_metadata = pd.read_csv(input_dir / 'development_metadata.csv')
    features = pd.read_csv(input_dir / 'development_combined_X.csv')
    schema = json.loads((input_dir / 'input_schema.json').read_text(encoding='utf-8-sig'))
    require(metadata['filename'].notna().all() and metadata['filename'].is_unique, 'Invalid sequence IDs.')
    ids = metadata['filename'].tolist()
    for table in [features, original_metadata]:
        require(table['filename'].is_unique and set(table['filename']) == set(ids), 'Recording mismatch.')
    original_metadata = original_metadata.set_index('filename').loc[ids].reset_index()
    require(metadata.equals(original_metadata), 'Sequence metadata disagreement.')
    features = features.set_index('filename').loc[ids]
    groups = schema['feature_groups']
    behavior = features[groups['behavior']].to_numpy(dtype=np.float64)
    reference_rgb = features[groups['rgb']].to_numpy(dtype=np.float64)
    y = metadata['target'].to_numpy(dtype=np.float32)
    folds = metadata['evaluation_fold'].to_numpy()
    require(set(y) == {0, 1} and int(y.sum()) == 113, 'Invalid target support.')
    require(metadata['sub_id'].notna().all() and metadata['sub_id'].nunique() == 259 and
        metadata.groupby('sub_id')['evaluation_fold'].nunique().eq(1).all() and
        np.isin(folds, [1, 2, 3, 4, 5]).all(), 'Participant folds invalid.')
    require(np.isfinite(behavior).all() and np.isfinite(reference_rgb).all(), 'Non-finite input.')
    for begin in range(0, len(ids), 64):
        require(np.isfinite(raw[begin:begin+64]).all(), 'Non-finite sequence.')
    print('Temporal preflight: PASS | 1510 recordings | 259 participants', flush=True)
    if not args.run:
        print('Inspection only. Use --run to fit new models.')
        return
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = root / 'results/research_extensions/temporal_cv' / stamp
    output.mkdir(parents=True, exist_ok=False)
    config = {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'sequence_run': str(cache),
        'threshold': 0.5, 'epochs': 20, 'batch_size': 64, 'optimizer': 'Adam',
        'learning_rate': 0.001, 'weight_decay': 0.001, 'seed_per_fold': '42 + evaluation fold',
        'device': 'cpu', 'models': ['mean_pool_lr', 'rgb_lstm_behavior', 'rgb_tcn_behavior'],
        'preprocessing': 'Fit RGB input scaler and PCA32 on fit-fold recording RGB means; transform ordered bins; fit projected-step scaler on fit-fold bins; behavior scaler on fit-fold behavior.',
        'control': 'LR receives mean of normalized projected bins plus scaled behavior; this is a new matched representation control, not a rerun of frozen LR.',
        'training': 'Fixed 20 epochs, fit-fold class weights, no evaluation-driven early stopping or tuning.',
        'scope': 'Exploratory development-only comparison; prior official test results already seen.',
        'limitations': ['Stored RGB array order treated as temporal window order; timestamps not validated.',
            '32 bins retain coarse order, not real-time frame resolution.',
            'Human-coded behavior is required; no automatic clinical symptom measurement.',
            'No diagnosis, severity, calibrated probability, or significance claim.'],
        'sequence_validation_sha256': sha(validation_path), 'script_sha256': sha(Path(__file__)),
        'python': sys.version, 'versions': {n: importlib.metadata.version(n) for n in
            ['torch', 'numpy', 'pandas', 'scikit-learn', 'joblib']}}
    (output / 'experiment_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print('Output:', output, flush=True)

    def metrics(truth, score):
        pred = (score >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
        return dict(asd_f1=f1_score(truth, pred, zero_division=0),
            macro_f1=f1_score(truth, pred, average='macro', zero_division=0),
            precision=precision_score(truth, pred, zero_division=0),
            recall=recall_score(truth, pred, zero_division=0),
            balanced_accuracy=balanced_accuracy_score(truth, pred), roc_auc=roc_auc_score(truth, score),
            average_precision=average_precision_score(truth, score), mcc=matthews_corrcoef(truth, pred),
            tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))

    def infer(model, sequence, human):
        model.eval()
        scores = []
        with torch.no_grad():
            for start in range(0, len(sequence), 64):
                logits = model(torch.from_numpy(sequence[start:start+64]),
                    torch.from_numpy(human[start:start+64]))
                scores.append(torch.sigmoid(logits).numpy())
        return np.concatenate(scores)

    rows, oof, checks, losses = [], [], [], []
    for fold in range(1, 6):
        fit = folds != fold
        evaluate = folds == fold
        require(not (set(metadata.loc[fit, 'sub_id']) & set(metadata.loc[evaluate, 'sub_id'])), 'Participant overlap.')
        require(set(y[fit]) == {0, 1} and set(y[evaluate]) == {0, 1}, 'Single-class fold.')
        print(f'Fold {fold}/5: fitting preprocessing on training participants...', flush=True)
        with threadpool_limits(limits=2):
            rgb_scaler = StandardScaler().fit(reference_rgb[fit])
            pca = PCA(n_components=32, svd_solver='randomized', random_state=42)
            pca.fit(rgb_scaler.transform(reference_rgb[fit]))
            projected = np.empty((len(ids), 32, 32), dtype=np.float32)
            for start in range(0, len(ids), 64):
                block = raw[start:start+64]
                projected[start:start+64] = pca.transform(rgb_scaler.transform(
                    block.reshape(-1, 1024))).reshape(len(block), 32, 32)
            step_scaler = StandardScaler().fit(projected[fit].reshape(-1, 32))
            projected = step_scaler.transform(projected.reshape(-1, 32)).reshape(len(ids), 32, 32).astype(np.float32)
            behavior_scaler = StandardScaler().fit(behavior[fit])
            human = behavior_scaler.transform(behavior).astype(np.float32)
        require(np.isfinite(projected).all() and np.isfinite(human).all(), 'Invalid transformed input.')
        joblib.dump(dict(rgb_scaler=rgb_scaler, pca=pca, step_scaler=step_scaler,
            behavior_scaler=behavior_scaler, rgb_columns=groups['rgb'], behavior_columns=groups['behavior'],
            evaluation_fold=fold), output / f'preprocessing_fold{fold}.joblib')
        static = np.concatenate([projected.mean(axis=1), human], axis=1)
        control = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced',
            max_iter=3000, random_state=42))
        with threadpool_limits(limits=2), warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            control.fit(static[fit], y[fit])
        scores = {'mean_pool_lr': control.predict_proba(static[evaluate])[:, 1]}
        control_path = output / f'mean_pool_lr_fold{fold}.joblib'
        joblib.dump(control, control_path)
        require(np.array_equal(joblib.load(control_path).predict_proba(static[evaluate])[:, 1],
            scores['mean_pool_lr']), 'Control round-trip failed.')
        for name, model_class in [('rgb_lstm_behavior', LSTMModel), ('rgb_tcn_behavior', TCNModel)]:
            seed = 42 + fold
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            model = model_class()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
            positive_weight = float((y[fit] == 0).sum() / (y[fit] == 1).sum())
            loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight))
            indices = np.flatnonzero(fit)
            rng = np.random.default_rng(seed)
            for epoch in range(1, 21):
                model.train()
                order = rng.permutation(indices)
                total_loss = 0.0
                for start in range(0, len(order), 64):
                    batch = order[start:start+64]
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(torch.from_numpy(projected[batch]), torch.from_numpy(human[batch]))
                    loss = loss_function(logits, torch.from_numpy(y[batch]))
                    require(torch.isfinite(loss).item(), 'Non-finite training loss.')
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    total_loss += float(loss.item()) * len(batch)
                losses.append(dict(model=name, fold=fold, epoch=epoch, fit_loss=total_loss/len(indices)))
                if epoch % 5 == 0:
                    print(f'Fold {fold}/5 | {name} | epoch {epoch}/20 | fit loss {total_loss/len(indices):.4f}', flush=True)
            score = infer(model, projected[evaluate], human[evaluate])
            path = output / f'{name}_fold{fold}.pt'
            torch.save({'state_dict': model.state_dict(), 'model_name': name, 'threshold': 0.5,
                'evaluation_fold': fold}, path)
            restored = model_class()
            restored.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['state_dict'])
            reproduced = infer(restored, projected[evaluate], human[evaluate])
            require(np.array_equal(reproduced, score), 'Saved temporal model inference mismatch.')
            scores[name] = score
        for name, score in scores.items():
            require(np.isfinite(score).all(), 'Non-finite scores.')
            rows.append({'model': name, 'fold': fold, 'participant_overlap': 0,
                'evaluation_recordings': int(evaluate.sum()), **metrics(y[evaluate], score)})
            table = metadata.loc[evaluate].copy()
            table['model'] = name; table['score'] = score
            table['prediction'] = (score >= 0.5).astype(int)
            oof.append(table)
            checks.append(dict(model=name, fold=fold, saved_inference='PASS'))
        pd.DataFrame(rows).to_csv(output / 'fold_results.csv', index=False)
        pd.DataFrame(losses).to_csv(output / 'fit_losses.csv', index=False)
        print(f'Fold {fold}/5 complete | participant overlap: 0', flush=True)
    predictions = pd.concat(oof, ignore_index=True)
    pooled = []
    for name in config['models']:
        table = predictions[predictions['model'] == name]
        require(len(table) == len(ids) and table['filename'].is_unique and
            set(table['filename']) == set(ids), 'OOF coverage failed.')
        pooled.append({'model': name, **metrics(table['target'], table['score'].to_numpy())})
    predictions.to_csv(output / 'oof_predictions.csv', index=False)
    metrics_columns = ['asd_f1', 'macro_f1', 'precision', 'recall', 'balanced_accuracy', 'roc_auc', 'average_precision', 'mcc']
    summary = pd.DataFrame(rows).groupby('model')[metrics_columns].agg(['mean', 'std'])
    summary.columns = ['_'.join(column) for column in summary.columns]
    summary.to_csv(output / 'cv_summary.csv')
    pd.DataFrame(pooled).to_csv(output / 'pooled_oof_metrics.csv', index=False)
    pd.DataFrame(checks).to_csv(output / 'saved_inference_validation.csv', index=False)
    require(sha(validation_path) == config['sequence_validation_sha256'], 'Sequence evidence changed.')
    evidence = dict(status='PASS', recordings_per_model=1510, participant_overlap=0,
        fold_models=15, oof_coverage='PASS', saved_inference='PASS', official_test_evaluated=False,
        completed_at_utc=datetime.now(timezone.utc).isoformat())
    (output / 'validation.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print('\nMean fold results:\n', summary.round(4).to_string())
    print('\nPooled OOF results:\n', pd.DataFrame(pooled).round(4).to_string(index=False))
    print('\nTemporal CV coverage and saved-model reproduction: PASS')
    print('Saved:', output)
    print('No official test evaluated. Existing frozen models preserved.')


if __name__ == '__main__':
    main()
