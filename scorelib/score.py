"""Functions for scoring paired system/reference RTTM files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from collections import defaultdict, namedtuple

import numpy as np
from scipy.linalg import block_diag

from . import metrics
from .six import iteritems, itervalues
from .utils import groupby

__all__ = ['flatten_labels', 'score', 'turns_to_frames', 'Scores']


def turns_to_frames(turns, score_regions, step=0.010):
    """Return frame-level labels corresponding to diarization.

    Parameters
    ----------
    turns : list of Turn
        Speaker turns. Should all be from single file.

    score_regions : list of tuple
        Scoring regions from UEM.

    step : float, optional
        Frame step size in seconds.
        (Default: 0.01)

    Returns
    -------
    labels : ndarray, (n_frames, n_speakers)
        Frame-level labels. The ``i,j``-th entry of this array is 1 if speaker
        ``j``-th speaker was present at frame ``i`` and 0 otherwise. If no
        speaker turns were passed, the second dimension will be 0.
    """
    file_ids = {turn.file_id for turn in turns}
    if len(file_ids) > 1:
        raise ValueError('Turns should be from a single file.')

    # Create matrix whose i,j-th entry is True IFF the j-th speaker was
    # present at frame i.
    onsets = [turn.onset for turn in turns]
    offsets = [turn.offset for turn in turns]
    speaker_ids = [turn.speaker_id for turn in turns]
    speaker_classes, speaker_class_inds = np.unique(
        speaker_ids, return_inverse=True)
    dur = max(score_offset for score_onset, score_offset in score_regions)
    n_frames = int(dur/step)
    X = np.zeros((n_frames, speaker_classes.size), dtype='int32')
    times = step*np.arange(n_frames)
    bis = np.searchsorted(times, onsets)
    eis = np.searchsorted(times, offsets)
    for bi, ei, speaker_class_ind in zip(bis, eis, speaker_class_inds):
        X[bi:ei, speaker_class_ind] = 1

    # Eliminate frames belonging to non-score regions.
    keep = np.zeros(len(X), dtype=bool)
    for score_onset, score_offset in score_regions:
        bi, ei = np.searchsorted(times, (score_onset, score_offset))
        keep[bi:ei] = True
    X = X[keep, ]

    return X


class Scores(namedtuple(
        'Scores',
        ['file_id', 'der', 'jer'])):
    """Structure containing metrics.

    Parameters
    ----------
    file_id : str
        File id for file scored.

    der : float
        Diarization error rate in percent.

    jer : float
        Jaccard error rate in percent.
    """
    __slots__ = ()


def score(ref_turns, sys_turns, uem, step=0.010, nats=False, jer_min_ref_dur=0.0,
          **kwargs):
    """Score diarization.

    Parameters
    ----------
    ref_turns : list of Turn
        Reference speaker turns.

    sys_turns : list of Turn
        System speaker turns.

    uem : UEM
        Un-partitioned evaluation map.

    step : float, optional
        Frame step size  in seconds. Not relevant for computation of DER.
        (Default: 0.01)

    nats : bool, optional
        If True, use nats as unit for information theoretic metrics.
        Otherwise, use bits.
        (Default: False)

    jer_min_ref_dur : float, optional
        Minimum reference speaker duration in seconds for JER calculation.
        Reference speakers with durations less than ``min_ref_dur`` will be
        excluded for scoring purposes. Setting this to a small non-zero number
        may stabilize JER when the reference segmentation contains multiple
        extraneous speakers.
        (Default: 0.0)

    kwargs
        Keyword arguments to be passed to ``metrics.der``.

    Returns
    -------
    file_scores : list of Scores
        Scores for all files.

    global_scores : Scores
        Global scores.
    """
    if jer_min_ref_dur is not None:
        jer_min_ref_dur = int(jer_min_ref_dur/step)

    # Build contingency matrices.
    file_to_ref_turns = defaultdict(
        list,
        {fid : list(g) for fid, g in groupby(ref_turns, lambda x: x.file_id)})
    file_to_sys_turns =defaultdict(
        list,
        {fid : list(g) for fid, g in groupby(sys_turns, lambda x: x.file_id)})
    file_to_cm = {} # Map from files to contingency matrices used by
                    # clustering metrics.
    file_to_jer_cm = {} # Map from files to contingency matrices used by
                        # JER.
    file_to_ref_durs = {} # Map from files to speaker durations in reference
                          # segmentation.
    file_to_sys_durs = {} # Map from files to speaker durations in system
                          # segmentation.
    for file_id, score_regions in iteritems(uem):
        ref_labels = turns_to_frames(
            file_to_ref_turns[file_id], score_regions, step=step)
        sys_labels = turns_to_frames(
            file_to_sys_turns[file_id], score_regions, step=step)
        file_to_ref_durs[file_id] = ref_labels.sum(axis=0)
        file_to_sys_durs[file_id] = sys_labels.sum(axis=0)
        file_to_jer_cm[file_id] = metrics.contingency_matrix(
            ref_labels, sys_labels)
    #     file_to_cm[file_id] = metrics.contingency_matrix(
    #         flatten_labels(ref_labels), flatten_labels(sys_labels))
    # global_cm = block_diag(*list(itervalues(file_to_cm)))
    # Above line has the undesirable property of claiming silence on
    # different files is a different category. However, leave it in for
    # consistency with how the clustering metrics were computed in DIHARD I.

    # Compute DER. This bit is slow as it relies on NIST's perl script.
    file_to_der, global_der = metrics.der(
        ref_turns, sys_turns, uem=uem, **kwargs)

    # Compute JER.
    file_to_jer, global_jer = metrics.jer(
        file_to_ref_durs, file_to_sys_durs, file_to_jer_cm, jer_min_ref_dur)

    # Compute clustering metrics.
    def compute_metrics(fid, der, jer):

        return Scores(
            fid, der, jer)
    file_scores = []
    for file_id, cm in iteritems(file_to_cm):
        file_scores.append(compute_metrics(
            file_id, cm, file_to_der[file_id], jer=file_to_jer[file_id]))
    global_scores = compute_metrics(
        '*** OVERALL ***', global_der, global_jer)

    return file_scores, global_scores
