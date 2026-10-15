"""Summarize existing DREAM quality CSV, correct task categories, and define a pilot cohort."""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_fields(value):
    require(isinstance(value, str), 'Expected saved task metadata string.')
    task = ast.literal_eval(value)
    require(isinstance(task, dict), 'Task metadata is not a dictionary.')
    ability = task.get('ability')
    require(isinstance(ability, str), 'Task ability missing or not a string.')
    return ability.strip(), task.get('difficultyLevel')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--quality-run', type=Path, required=True)
    args = parser.parse_args()
    import pandas as pd
    root = args.project.resolve()
    run = args.quality_run.resolve()
    report_path = run / 'quality_audit.json'
    csv_path = run / 'session_quality.csv'
    original = json.loads(report_path.read_text(encoding='utf-8'))
    require(original['parsed_sessions'] == 3121 and original['failed_sessions'] == 0,
            'Expected completed 3121-session audit.')
    sessions = pd.read_csv(csv_path, keep_default_na=False)
    require(len(sessions) == 3121 and sessions['member'].is_unique, 'Session CSV coverage mismatch.')
    parsed_tasks = [task_fields(value) for value in sessions['task']]
    sessions['task_ability'] = [value[0] for value in parsed_tasks]
    sessions['task_difficulty_level'] = [value[1] for value in parsed_tasks]
    ability_counts = dict(Counter(value if value else 'UNSPECIFIED' for value in sessions['task_ability']))
    require(sum(ability_counts.values()) == 3121, 'Task aggregation coverage failed.')
    actual_quality = sessions['quality_status'].value_counts().to_dict()
    require(actual_quality == original['quality_status_counts'], 'Quality counts disagree with saved audit.')
    require(sessions['participant_id'].nunique() == original['participants_in_parsed_sessions'], 'Participant count mismatch.')
    require(set(sessions['coordinate_lengths_aligned'].astype(str).str.lower()).issubset({'true', 'false'}), 'Invalid alignment flags.')
    aligned = sessions['coordinate_lengths_aligned'].astype(str).str.lower().eq('true')
    frames = pd.to_numeric(sessions['frames_with_any_complete_joint_xyz'], errors='raise')
    condition = sessions['quality_status'].isin(['COMPLETE_NUMERIC', 'PARTIAL_MISSING'])
    named = sessions['task_ability'].isin(['TT', 'JA', 'IM'])
    eligible = aligned & condition & named & frames.ge(32)
    selected = sessions.loc[eligible].copy()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = root / 'results/research_extensions/dream_review' / stamp
    output.mkdir(parents=True, exist_ok=False)
    selected.to_csv(output / 'provisional_activity_cohort.csv', index=False)
    sessions[['member', 'participant_id', 'task_ability', 'task_difficulty_level', 'quality_status']].to_csv(output / 'corrected_task_metadata.csv', index=False)
    cohort = {'status': 'PROVISIONAL_METADATA_SELECTION', 'sessions': len(selected),
        'participants': int(selected['participant_id'].nunique()),
        'ability_counts': selected['task_ability'].value_counts().to_dict(),
        'rule': 'Known TT/JA/IM task; aligned coordinate lengths; COMPLETE_NUMERIC or PARTIAL_MISSING; at least 32 frames with one complete joint XYZ.',
        'interpretation': 'Basic data availability rule, not a validated missingness or tracking-accuracy threshold.',
        'planned_endpoint': 'Exploratory therapy-activity recognition only if subsequent features and participant class coverage support it.',
        'excluded_sessions': int((~eligible).sum()),
        'no_activity_or_autism_model_trained': True}
    evidence = {'status': 'REVIEW_DOCUMENTED', 'reviewed_at_utc': datetime.now(timezone.utc).isoformat(),
        'team': ['Muhammad Danial', 'Hamza Rasheed', 'Muhammad Freeh'],
        'parsed_sessions': 3121, 'participants': original['participants_in_parsed_sessions'],
        'parsing_failures': original['failed_sessions'], 'quality_status_counts': actual_quality,
        'condition_counts': original['condition_counts'], 'task_ability_counts': ability_counts,
        'sessions_with_finite_coordinates': original['sessions_with_finite_coordinates'],
        'sessions_with_complete_joint_xyz': original['sessions_with_any_complete_joint_xyz'],
        'provisional_activity_cohort': cohort,
        'sources': {'quality_run': str(run), 'quality_report_sha256': sha(report_path),
                    'session_csv_sha256': sha(csv_path), 'archive_sha256': original['archive_sha256']},
        'task_summary_correction': 'Group by task.ability rather than by serialized entire task dictionary.',
        'limitations': ['All dataset participants have ASD; RET/SHT are therapy conditions, not diagnostic classes.',
            'Partial coordinate coverage must be represented by masks; missing values were not filled as zero.',
            'Timestamp metadata strings do not establish per-frame timestamp alignment.',
            'No clinical diagnosis/severity experiment or frozen-model external validation was performed.']}
    (output / 'review_evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    docs = root / 'docs'
    page = root / 'dashboard/pages/11_DREAM_Data_Quality.py'
    docs.mkdir(parents=True, exist_ok=True);page.parent.mkdir(parents=True, exist_ok=True)
    targets = [docs / 'DREAM_DATA_QUALITY_REVIEW.md', docs / 'DREAM_DATA_QUALITY_EVIDENCE.json', page]
    backup = root / 'documentation_backups' / ('dream_review_' + stamp)
    for path in targets:
        if path.exists():
            destination = backup / path.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    text = ['# DREAM data-quality review', '', 'Team: Muhammad Danial, Hamza Rasheed, Muhammad Freeh.', '',
        'The verified full release contains 3,121 session JSON files representing 61 participants. Every session parsed successfully. The completed audit is a structural/numeric coverage review, not a training-readiness or diagnostic-validation claim.', '',
        '## Quality findings', '', '| Category | Sessions |', '|---|---:|']
    for key, count in actual_quality.items():
        text.append(f'| {key} | {count} |')
    text += ['', f"Sessions with any finite coordinates: {original['sessions_with_finite_coordinates']}. Sessions with at least one complete joint XYZ observation: {original['sessions_with_any_complete_joint_xyz']}. Complete triplets do not prove tracking accuracy.", '',
        '## Corrected task counts', '', 'The initial audit grouped by the complete task dictionary. This review groups by the ability field instead.', '', '| Ability code | Sessions |', '|---|---:|']
    for key, count in sorted(ability_counts.items()):
        text.append(f'| {key} | {count} |')
    text += ['', 'Blank abilities are labelled UNSPECIFIED, not an invented class. TT, JA and IM labels are therapy activities, not autism diagnosis or severity stages.', '',
        '## Provisional activity cohort', '', f"Selected {cohort['sessions']} sessions from {cohort['participants']} participants using a fixed basic availability rule: {cohort['rule']}", '',
        'Selection uses saved audit metadata and does not clean raw sequences. Actual tracking coverage, feature extraction and participant-disjoint class support still need validation before an activity experiment.', '',
        '## Interpretation', '', 'RET and SHT are therapy conditions among participants with ASD. They cannot be relabelled as autism-positive versus autism-negative. DREAM motion/gaze measurements do not match the AV-ASD or ASD-FEAT frozen input schemas.', '',
        'The time field was a string in the inspected sessions. Per-frame timing and timestamp units were not independently established. No diagnostic model, severity model or external evaluation of either frozen classifier was performed.', '',
        'Dataset source: https://doi.org/10.5878/17p8-6k13. Catalogue license: CC BY-NC-SA 4.0. Local recording-level rows remain in the results directory; this document contains aggregate findings.', '',
        'Original coordinate audit files remain preserved. New corrected metadata, provisional selection and source hashes are stored in the separate review output.']
    targets[0].write_text('\n'.join(text)+'\n', encoding='utf-8')
    targets[1].write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    source = '''"""DREAM aggregate coverage review; no model inference."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st

st.title("DREAM Dataset: Data Quality")
path = Path(__file__).resolve().parents[2] / "docs/DREAM_DATA_QUALITY_EVIDENCE.json"
if not path.is_file():
    st.info("Run the DREAM audit documentation stage first.")
    st.stop()
evidence = json.loads(path.read_text(encoding="utf-8"))
st.caption("Separate external-data investigation; no diagnostic model evaluation")
cols = st.columns(3)
cols[0].metric("Parsed sessions", evidence["parsed_sessions"])
cols[1].metric("Participants", evidence["participants"])
cols[2].metric("Parsing failures", evidence["parsing_failures"])
st.subheader("Coordinate quality findings")
quality = pd.DataFrame(list(evidence["quality_status_counts"].items()), columns=["Category", "Sessions"])
st.dataframe(quality, hide_index=True, width="stretch")
st.bar_chart(quality.set_index("Category"), horizontal=True)
st.subheader("Corrected therapy-task counts")
st.dataframe(pd.DataFrame(list(evidence["task_ability_counts"].items()), columns=["Ability code", "Sessions"]), hide_index=True, width="stretch")
cohort = evidence["provisional_activity_cohort"]
st.write(f"Provisional activity cohort: {cohort['sessions']} sessions from {cohort['participants']} participants.")
st.caption(cohort["rule"])
st.info("Missing coordinates remain missing. The provisional cohort is not a validated training set. All participants have ASD; therapy conditions are not autism-negative labels. No diagnosis or severity predictions are made here.")
st.caption("Team: Muhammad Danial, Hamza Rasheed, Muhammad Freeh")
'''
    compile(source, str(page), 'exec')
    page.write_text(source, encoding='utf-8')
    print('Corrected task counts:', ability_counts)
    print('Provisional activity cohort:', cohort['sessions'], 'sessions |', cohort['participants'], 'participants')
    print('Selected ability counts:', cohort['ability_counts'])
    print('Saved selection:', output / 'provisional_activity_cohort.csv')
    for path in targets:
        print('Saved:', path)
    print('Existing CSV coverage and dashboard syntax: PASS')
    print('No archive reread, coordinate changes or model training.')


if __name__ == '__main__':
    main()
