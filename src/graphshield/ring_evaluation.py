"""Retrospective group evaluation, deliberately separate from graph construction."""

from collections import Counter
from statistics import fmean, median


def maximum_matching(edges):
    """Deterministic maximum-cardinality bipartite matching with augmenting paths.

    No score fitting. Left nodes are case indices, right nodes are group IDs.
    Iterative search avoids recursion limits for long alert histories.
    """
    right_to_left, left_to_right = {}, {}
    for start in sorted(edges):
        stack, seen_left, seen_right = [start], {start}, set()
        parent = {}
        free = None
        while stack and free is None:
            left = stack.pop()
            for right in edges[left]:
                if right in seen_right:
                    continue
                seen_right.add(right)
                parent[right] = left
                previous = right_to_left.get(right)
                if previous is None:
                    free = right
                    break
                if previous not in seen_left:
                    seen_left.add(previous)
                    stack.append(previous)
        while free is not None:
            left = parent[free]
            old = left_to_right.get(left)
            left_to_right[left] = free
            right_to_left[free] = left
            free = old
    return left_to_right


def eligible(case, group, event_groups, config):
    members, truth = set(case['members']), set(group['members'])
    overlap = len(members & truth)
    if not members or not truth:
        return False
    return (overlap / len(members) >= config['match_member_precision']
            and overlap / len(truth) >= config['match_member_recall']
            and group['start_step'] < case['cutoff_step']
            and any(event_groups.get(tx) == group['group_id'] for tx in case['transaction_ids']))


def evaluate(cases, groups, event_groups, config):
    """Matching needs member purity, member coverage AND a planted evidence event.

    Truth can be learned after the alert: this is retrospective scoring, never an
    input to detection. Events in cases have already passed the temporal filters.
    """
    cases = sorted(cases, key=lambda c: (c['cutoff_step'], c['candidate_id']))
    positive = {g['group_id']: g for g in groups if g['is_fraud']}
    benign = {g['group_id']: g for g in groups if not g['is_fraud']}
    edges = {i: sorted(gid for gid, group in positive.items() if eligible(case, group, event_groups, config))
             for i, case in enumerate(cases)}
    assigned = maximum_matching(edges)
    benign_hits = {gid for case in cases for gid, group in benign.items() if eligible(case, group, event_groups, config)}
    matches = []
    for index, gid in sorted(assigned.items()):
        case, group = cases[index], positive[gid]
        overlap = len(set(case['members']) & set(group['members']))
        matches.append({'candidate_id': case['candidate_id'], 'group_id': gid,
                        'cutoff_step': case['cutoff_step'], 'scenario_family': group['scenario_family'],
                        'member_precision': overlap / len(case['members']),
                        'member_recall': overlap / len(group['members']),
                        'delay_from_group_start_steps': case['cutoff_step'] - group['start_step']})
    total, matched = len(cases), len(matches)
    families = {}
    for family in sorted({g['scenario_family'] for g in positive.values()}):
        count = sum(g['scenario_family'] == family for g in positive.values())
        found = sum(m['scenario_family'] == family for m in matches)
        families[family] = {'groups': count, 'matched_groups': found, 'recall': found / count}
    return {
        'alerts': total, 'fraud_groups': len(positive), 'matched_groups': matched,
        'unmatched_alerts': total - matched,
        'case_precision': matched / total if total else None,
        'ring_recall': matched / len(positive) if positive else None,
        'duplicate_positive_alerts': sum(bool(edges[i]) and i not in assigned for i in edges),
        'benign_groups': len(benign), 'benign_groups_alerted': len(benign_hits),
        'benign_group_alert_rate': len(benign_hits) / len(benign) if benign else None,
        'mean_matched_member_precision': fmean(m['member_precision'] for m in matches) if matches else None,
        'mean_matched_member_recall': fmean(m['member_recall'] for m in matches) if matches else None,
        'median_detection_delay_steps': median(m['delay_from_group_start_steps'] for m in matches) if matches else None,
        'by_family': families, 'matches': matches,
    }


def aggregate(results):
    counts = ('alerts', 'fraud_groups', 'matched_groups', 'unmatched_alerts',
              'duplicate_positive_alerts', 'benign_groups', 'benign_groups_alerted')
    output = {key: sum(r[key] for r in results) for key in counts}
    for key, numerator, denominator in (
        ('case_precision', 'matched_groups', 'alerts'), ('ring_recall', 'matched_groups', 'fraud_groups'),
        ('benign_group_alert_rate', 'benign_groups_alerted', 'benign_groups'),
    ):
        output[key] = output[numerator] / output[denominator] if output[denominator] else None
    matches = [m for r in results for m in r['matches']]
    output['mean_matched_member_precision'] = fmean(m['member_precision'] for m in matches) if matches else None
    output['mean_matched_member_recall'] = fmean(m['member_recall'] for m in matches) if matches else None
    output['median_detection_delay_steps'] = median(m['delay_from_group_start_steps'] for m in matches) if matches else None
    families = Counter()
    detected = Counter()
    for result in results:
        for name, data in result['by_family'].items():
            families[name] += data['groups']
            detected[name] += data['matched_groups']
    output['by_family'] = {f: {'groups': n, 'matched_groups': detected[f], 'recall': detected[f] / n}
                           for f, n in sorted(families.items())}
    return output
