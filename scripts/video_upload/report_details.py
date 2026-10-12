"""Plain-language summaries and exact linear logit contributions."""
import numpy as np
from scipy.special import expit


def explain_features(bundle, frame):
    model = bundle['model']
    columns = list(bundle['feature_columns'])
    labels = list(bundle['target_columns'])
    if frame.columns.tolist() != columns or len(frame) != 1:
        raise ValueError('Explanation requires one row in saved feature order.')
    if [name for name, _ in model.steps] != ['scaler', 'classifier']:
        raise ValueError('Unsupported explanation pipeline.')
    scaled = np.asarray(model.named_steps['scaler'].transform(frame), dtype=float)[0]
    classifier = model.named_steps['classifier']
    probabilities = np.asarray(model.predict_proba(frame), dtype=float)[0]
    logits = np.asarray(model.decision_function(frame), dtype=float)[0]
    if len(classifier.estimators_) != len(labels):
        raise ValueError('Classifier and target count differ.')
    explanations = []
    for index, (label, estimator) in enumerate(zip(labels, classifier.estimators_)):
        coefficients = np.asarray(estimator.coef_, dtype=float)
        if coefficients.shape != (1, len(columns)):
            raise ValueError('Unsupported classifier coefficient shape.')
        contributions = scaled * coefficients[0]
        intercept = float(estimator.intercept_[0])
        reconstructed = float(intercept + contributions.sum())
        score = float(expit(reconstructed))
        if not np.isfinite(contributions).all() or not np.isclose(reconstructed, logits[index], rtol=1e-7, atol=1e-8) or not np.isclose(score, probabilities[index], rtol=1e-7, atol=1e-8):
            raise ValueError('Explanation reconstruction failed.')
        order = np.argsort(-np.abs(contributions), kind='stable')[:10]
        explanations.append({
            'label': label, 'intercept': intercept,
            'reconstructed_logit': reconstructed,
            'reconstructed_score': score,
            'score_difference': float(abs(score - probabilities[index])),
            'audio_logit_contribution': float(sum(value for name, value in zip(columns, contributions) if not name.startswith('visual_'))),
            'visual_logit_contribution': float(sum(value for name, value in zip(columns, contributions) if name.startswith('visual_'))),
            'top_features': [{'feature': columns[j], 'logit_contribution': float(contributions[j]), 'direction': 'Raises score' if contributions[j] > 0 else 'Lowers score' if contributions[j] < 0 else 'No contribution'} for j in order],
            'reconstruction': 'PASS'})
    return explanations


def summarize_results(results, threshold):
    selected = sorted((r for r in results if r['model_score'] >= threshold), key=lambda r: -r['model_score'])
    if selected:
        text = f"The model assigned scores at or above {threshold:.2f} to {len(selected)} behavioral labels: " + ', '.join(r['label'] for r in selected) + '.'
    else:
        text = f'No behavioral-label score reached the frozen threshold of {threshold:.2f}. This does not rule out autism or the listed behaviors.'
    return text + ' These are model predictions for this recording, not confirmed observations. Each label is scored separately; scores do not add up to 100%.'
