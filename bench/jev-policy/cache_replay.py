"""Replay a bounded cache trace. This module does not control an actual cache."""
from collections import OrderedDict
from policy import digest


def replay(trace, budget, proposals=None):
    if type(budget) is not int or budget <= 0:
        raise ValueError('The cache budget must be a positive integer.')
    proposals = proposals or {}
    cache = OrderedDict()
    hits, misses, fallbacks, decisions = 0, 0, 0, []
    for index, event in enumerate(trace):
        page, size = event['page'], event['bytes']
        if type(size) is not int or size <= 0:
            raise ValueError('The page size must be a positive integer.')
        if page in cache:
            if cache[page]['bytes'] != size:
                raise ValueError('The same page identifier changed size.')
            hits += 1
            cache.move_to_end(page)
            continue
        misses += 1
        if size > budget:
            continue
        while sum(v['bytes'] for v in cache.values())+size > budget:
            eligible = [key for key in cache if key not in event.get('pinned', [])]
            if not eligible:
                break
            # Only the current request and resident page metadata form the decision snapshot.
            snapshot = {'current_topic': event['topic'], 'resident': dict(cache), 'eligible': eligible}
            proposal = proposals.get(str(index), {})
            confidence = proposal.get('confidence', 0)
            valid = (proposal.get('snapshot_hash') == digest(snapshot) and proposal.get('victim') in eligible
                     and type(confidence) in (int, float) and .8 <= confidence <= 1)
            victim = proposal['victim'] if valid else eligible[0]
            fallbacks += not valid
            decisions.append({'index': index, 'snapshot': snapshot, 'snapshot_hash': digest(snapshot),
                              'victim': victim, 'proposal_accepted': valid})
            del cache[victim]
        if sum(v['bytes'] for v in cache.values())+size <= budget:
            cache[page] = {'bytes': size, 'topic': event['topic']}
    return {'simulation_only': True, 'hits': hits, 'misses': misses, 'fallbacks': fallbacks, 'decisions': decisions}
