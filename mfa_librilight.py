from lhotse.recipes.librilight import _parse_utterance, _prepare_subset
from lhotse.audio import Recording, RecordingSet
from lhotse.qa import fix_manifests, validate_recordings_and_supervisions
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.cut import CutSet, Cut
from lhotse.utils import Pathlike
from lhotse.serialization import load_manifest_lazy_or_eager, load_manifest

import os
import logging
from tqdm import tqdm
from functools import partial
from typing import Optional
from pathlib import Path
from concurrent.futures import Executor, ProcessPoolExecutor, as_completed

def change_prefix(cut, old_prefix, new_prefix):
    old_path = cut.recording.sources[0].source
    new_path = old_path.replace(old_prefix, new_prefix)
    cut.recording.sources[0].source = new_path

    if ',' in cut.id:
        print(cut.id)

    # else:
    #     os.makedirs(output_path, exist_ok=True)

    return cut


def save_text_and_audio(cut: Cut, storage_path, t_id, old_prefix = None, new_prefix=None, **kwargs):
    spk_id = cut.id.split('/')[1]
    storage_subdir = f"{storage_path}/{spk_id}"
    os.makedirs(storage_subdir, exist_ok=True)

    text = cut.supervisions[0].custom["texts"][t_id]
    f_id = ','.join(cut.id.split('/'))
    with open(f"{storage_subdir}/{f_id}.lab", 'w') as f:
        f.write(text)

    if not os.path.exists(f"{storage_subdir}/{f_id}.wav"):
        assert old_prefix is not None
        assert new_prefix is not None
        # cut = change_prefix(cut=cut, old_prefix=old_prefix, new_prefix=new_prefix)
        cut.save_audio(
            storage_path=f"{storage_subdir}/{f_id}.wav",
            **kwargs
        )

    return cut

def save_texts_and_audios(
        cuts: CutSet,
        storage_path: Pathlike,
        t_id: int = 0,
        progress_bar: bool = True,
        num_jobs: Optional[int] = None,
        executor: Optional[Executor] = None,
        shuffle_on_split: bool = True,
        **kwargs,
    ):
    """ modify CutSet.save_audios() into save_texts(CutSet)
    Args:
        t_id: type of transcript.
                0: original transcript, cases and punctuations
                1: ASR predicted, upper_case without punctuations
    """
    from cytoolz import identity

    from lhotse.manipulation import combine

    # Pre-conditions and args setup
    progress = (
        identity  # does nothing, unless we overwrite it with an actual prog bar
    )
    if num_jobs is None:
        num_jobs = 1
    if num_jobs == 1 and executor is not None:
        logging.warning(
            "Executor argument was passed but num_jobs set to 1: "
            "we will ignore the executor and use non-parallel execution."
        )
        executor = None
        
    if executor is None and num_jobs == 1:
        if progress_bar:
            progress = partial(
                tqdm, desc="Storing text transcripts", total=len(cuts)
            )
        return CutSet.from_cuts(
            progress(
                save_text_and_audio(cut=cut, storage_path=storage_path, t_id=t_id, **kwargs)
                for cut in cuts
            )
        )

    # Parallel execution: prepare the CutSet splits
    cut_sets = cuts.split(num_jobs, shuffle=shuffle_on_split)

    # Initialize the default executor if None was given
    if executor is None:
        import multiprocessing

        # The `is_caching_enabled()` state gets transfered to
        # the spawned sub-processes implictly (checked).
        executor = ProcessPoolExecutor(
            max_workers=num_jobs,
            mp_context=multiprocessing.get_context("spawn"),
        )

    # Submit the chunked tasks to parallel workers.
    # Each worker runs the non-parallel version of this function inside.
    futures = [
        executor.submit(
            save_texts_and_audios,
            cs,
            storage_path=storage_path,
            t_id=t_id,
            # Disable individual workers progress bars for readability
            progress_bar=(i==0),
            **kwargs
        )
        for i, cs in enumerate(cut_sets)
    ]

    if progress_bar:
        progress = partial(
            tqdm,
            desc="Storing text transcripts (chunks progress)",
            total=len(futures),
        )

    cuts = combine(progress(f.result() for f in futures))
    return cuts

if __name__ == "__main__":
    old_prefix="download/librilight"
    new_prefix="/datasets/LibriLight"
    output_path = f"{new_prefix}/mfa_data"

    subsets = ["small", "dev", "test_clean", "test_other"]
    # subsets = ["medium"]
    for subset in subsets:
        # can not lazily split with progress bar
        cuts: CutSet = load_manifest(f"data/LibriHeavy/libriheavy_cuts_{subset}.jsonl.gz", CutSet)
        cuts = cuts.filter(lambda c: ',' not in c.id)
        cuts = cuts.map(partial(change_prefix, old_prefix=old_prefix, new_prefix=new_prefix))

        storage_path=f"{output_path}/{subset}"
        save_texts_and_audios(cuts=cuts, storage_path=storage_path, old_prefix=old_prefix, new_prefix=new_prefix, num_jobs=16)