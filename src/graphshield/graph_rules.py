"""Observed-only rolling graphs and explicit, uncalibrated investigation scores.

No label, group-membership or scenario-family input is accepted by this module.
"""

import hashlib
import math
from collections import Counter, defaultdict
from statistics import fmean, pstdev

VERSION = 'graph-rules-v1'
POLICIES = ('graph_only', 'graph_device')


def validate_config(config):
    expected = {'version', 'window_steps', 'snapshot_stride', 'minimum_pair_events',
                'minimum_members', 'maximum_members', 'minimum_score', 'alerts_per_snapshot',
                'duplicate_jaccard', 'match_member_precision', 'match_member_recall'}
    if set(config) != expected or config['version'] != VERSION:
        raise ValueError('Unsupported or incomplete graph-rule configuration')
    for key in ('window_steps', 'snapshot_stride', 'minimum_pair_events', 'minimum_members',
                'maximum_members', 'alerts_per_snapshot'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if not 3 <= config['minimum_members'] <= config['maximum_members'] <= 50:
        raise ValueError('Member bounds must be ordered within [3,50]')
    if config['minimum_pair_events'] < 2:
        raise ValueError('Repeated-edge rules require at least two events')
    for key in ('minimum_score', 'duplicate_jaccard', 'match_member_precision', 'match_member_recall'):
        if type(config[key]) not in (float, int) or not math.isfinite(config[key]) or not 0 < config[key] <= 1:
            raise ValueError(f'{key} must lie in (0,1]')


def components(adjacency):
    """Iterative Kosaraju SCCs; also yields weak components for symmetric input."""
    nodes = sorted(set(adjacency) | {v for values in adjacency.values() for v in values})
    reverse = defaultdict(set)
    for u, values in adjacency.items():
        for v in values:
            reverse[v].add(u)
    seen, finished = set(), []
    for node in nodes:
        if node in seen:
            continue
        seen.add(node)
        stack = [(node, iter(sorted(adjacency.get(node, ()))))]
        while stack:
            u, children = stack[-1]
            v = next(children, None)
            if v is None:
                finished.append(u)
                stack.pop()
            elif v not in seen:
                seen.add(v)
                stack.append((v, iter(sorted(adjacency.get(v, ())))))
    seen = set()
    result = []
    for node in reversed(finished):
        if node in seen:
            continue
        seen.add(node)
        group, stack = [], [node]
        while stack:
            u = stack.pop()
            group.append(u)
            for v in sorted(reverse.get(u, ())):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        result.append(tuple(sorted(group)))
    return sorted(result)


def snapshot(transactions, observations, cutoff, config):
    """Both event time and arrival time must precede cutoff; current bucket excluded.

    Each transaction tuple is (id, sender, receiver, amount, step, available_step).
    Observation tuples are (id, account, device, step, available_step).
    Edges introduce observed entities; account/device metadata is not a predictor.
    """
    lower = max(1, cutoff - config['window_steps'])
    tx = sorted((r for r in transactions if lower <= r[4] < cutoff and r[5] < cutoff), key=lambda r: r[0])
    obs = sorted((r for r in observations if lower <= r[3] < cutoff and r[4] < cutoff), key=lambda r: r[0])
    pairs = defaultdict(list)
    incident = defaultdict(list)
    weak = defaultdict(set)
    for row in tx:
        _, u, v, *_ = row
        pairs[u, v].append(row)
        incident[u].append(row)
        if u != v:
            incident[v].append(row)
        weak[u].add(v)
        weak[v].add(u)
    repeat = {pair for pair, events in pairs.items() if len(events) >= config['minimum_pair_events'] and pair[0] != pair[1]}
    outgoing, incoming = defaultdict(set), defaultdict(set)
    for u, v in sorted(repeat):
        outgoing[u].add(v)
        incoming[v].add(u)
    proposed = defaultdict(set)
    rejected = Counter()

    def propose(members, reason):
        members = tuple(sorted(set(members)))
        if config['minimum_members'] <= len(members) <= config['maximum_members']:
            proposed[members].add(reason)
        elif len(members) > config['maximum_members']:
            rejected[reason] += 1

    for group in components(outgoing):
        propose(group, 'repeated_strong_component')
    for node, neighbors in sorted(outgoing.items()):
        propose({node} | neighbors, 'repeated_fan_out')
    for node, neighbors in sorted(incoming.items()):
        propose({node} | neighbors, 'repeated_fan_in')

    device_accounts = defaultdict(set)
    for _, account, device, *_ in obs:
        device_accounts[device].add(account)
    cases = []
    for members, reasons in sorted(proposed.items()):
        member_set = set(members)
        touched = {r[0]: r for a in members for r in incident[a]}
        internal = [r for _, r in sorted(touched.items()) if r[1] in member_set and r[2] in member_set]
        internal_pairs = {(r[1], r[2]) for r in internal}
        repeated_events = sum((r[1], r[2]) in repeat for r in internal)
        amounts = [float(r[3]) for r in internal]
        average = fmean(amounts)
        cv = pstdev(amounts) / average if average else 0.0
        shared = {d: accounts & member_set for d, accounts in device_accounts.items()
                  if len(accounts & member_set) >= 2}
        shared_members = set().union(*shared.values()) if shared else set()
        device_ids = sorted(shared)
        evidence_obs = [r for r in obs if r[1] in member_set and r[2] in shared]
        features = {
            'member_count': len(members), 'internal_events': len(internal),
            'internal_directed_pairs': len(internal_pairs),
            'directed_density': len(internal_pairs) / (len(members) * (len(members) - 1)),
            'repeated_event_fraction': repeated_events / len(internal),
            'event_support': min(1.0, len(internal) / (2 * len(members))),
            'internal_event_share': len(internal) / len(touched),
            'boundary_events': len(touched) - len(internal),
            'internal_amount_total': round(sum(amounts), 2), 'amount_cv': cv,
            'amount_similarity': 1 / (1 + cv),
            'shared_device_count': len(shared),
            'shared_device_member_fraction': len(shared_members) / len(members),
            'max_repeated_in_degree': max(len(incoming[a]) for a in members),
            'max_repeated_out_degree': max(len(outgoing[a]) for a in members),
            'internal_active_steps': len({r[4] for r in internal}),
        }
        graph_score = (.35 * features['repeated_event_fraction'] + .25 * features['event_support']
                       + .25 * features['amount_similarity'] + .15 * features['internal_event_share'])
        device_score = (graph_score + .15 * features['shared_device_member_fraction']) / 1.15
        # IDs break ties deterministically; they do not enter a score or feature.
        signature = hashlib.sha256('\0'.join(members).encode()).hexdigest()[:24]
        cases.append({
            'candidate_id': f'{cutoff}:{signature}', 'cutoff_step': cutoff, 'window_start': lower,
            'members': list(members), 'reasons': sorted(reasons), 'features': features,
            'scores': {'graph_only': graph_score, 'graph_device': device_score},
            'transaction_ids': [r[0] for r in internal], 'shared_device_ids': device_ids,
            'device_observation_ids': [r[0] for r in evidence_obs],
            'first_evidence_step': min(r[4] for r in internal),
            'last_evidence_step': max(r[4] for r in internal),
        })
    summary = {
        'cutoff_step': cutoff, 'window_start': lower, 'transactions': len(tx),
        'device_observations': len(obs), 'active_accounts': len(weak), 'directed_pairs': len(pairs),
        'repeated_directed_pairs': len(repeat),
        'largest_weak_component': max((len(c) for c in components(weak)), default=0),
        'oversized_proposals_skipped': dict(sorted(rejected.items())), 'candidates': len(cases),
    }
    return cases, summary


def jaccard(left, right):
    left, right = set(left), set(right)
    return len(left & right) / len(left | right) if left or right else 1.0


def select_alerts(candidates, config, policy):
    """Chronological fixed-capacity policy. Emitted membership/evidence is immutable.

    Suppress overlapping cases against prior emitted cases for the whole world.
    Candidates below capacity remain eligible at a later snapshot.
    """
    if policy not in POLICIES:
        raise ValueError('Unknown investigation policy')
    by_time = defaultdict(list)
    for case in candidates:
        by_time[case['cutoff_step']].append(case)
    emitted, decisions = [], []
    for cutoff, cases in sorted(by_time.items()):
        used = 0
        for case in sorted(cases, key=lambda c: (-c['scores'][policy], c['candidate_id'])):
            duplicate = None
            if case['scores'][policy] < config['minimum_score']:
                decision = 'below_score'
            else:
                duplicate = next((old['candidate_id'] for old in emitted
                                  if jaccard(old['members'], case['members']) >= config['duplicate_jaccard']), None)
                if duplicate:
                    decision = 'duplicate'
                elif used >= config['alerts_per_snapshot']:
                    decision = 'capacity'
                else:
                    decision = 'alert'
                    emitted.append(case)
                    used += 1
            decisions.append({'candidate_id': case['candidate_id'], 'cutoff_step': cutoff,
                              'decision': decision, 'duplicate_of': duplicate})
    return emitted, decisions
