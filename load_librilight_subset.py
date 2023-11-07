import logging
import os
from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse.recipes.librilight import _parse_utterance, _prepare_subset
from lhotse.audio import Recording, RecordingSet
from lhotse.qa import fix_manifests, validate_recordings_and_supervisions
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.cut import CutSet
from lhotse.utils import Pathlike

LIBRILIGHT = ("small", "medium", "large")

LIBRILIGHT_URL = (
    "https://dl.fbaipublicfiles.com/librilight/data/small.tar",
    "https://dl.fbaipublicfiles.com/librilight/data/medium.tar",
    "https://dl.fbaipublicfiles.com/librilight/data/large.tar",
)

def prepare_librilight(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    subsets: Optional[Union[str, Sequence[str]]] = "all",
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Path to the LibriLight dataset.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)

    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    logging.info("Preparing LibriLight...")

    if subsets == "all":
        subsets = LIBRILIGHT
    elif subsets == "small":
        subsets = ["small"]
    elif isinstance(subsets, str):
        subsets = [subsets]

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    manifests = defaultdict(dict)

    for part in tqdm(subsets, desc="Dataset parts"):
        logging.info(f"Processing LibriLight subset: {part}")
        if manifests_exist(
            part=part,
            output_dir=output_dir,
            prefix="librilight",
            suffix="jsonl.gz",
        ):
            logging.info(f"LibriLight subset: {part} already prepared - skipping.")
            continue

        try:
            recording_set, supervision_set = _prepare_subset(part, corpus_dir, num_jobs)
        except:
            continue

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"librilight_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(output_dir / f"librilight_recordings_{part}.jsonl.gz")

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests

if __name__ == "__main__":
    # prepare_librilight("/datasets/LibriLight", "data/LibriLight")

    light_small = read_manifests_if_cached(["small"], "data/LibriLight", "librilight_")
    # {'small': {'recordings': RecordingSet(len=2588), 'supervisions': SupervisionSet(len=2588)}}

    # light_small['small']['recordings'][0]:
    #   Recording(
    #       id='small/32/evening_star_0901_librivox_64kb_mp3/eveningstar_poe_blb_64kb',
    #       sources=[
    #           AudioSource(type='file', channels=[0], source='/datasets/LibriLight/small/32/evening_star_0901_librivox_64kb_mp3/eveningstar_poe_blb_64kb.flac')
    #       ],
    #       sampling_rate=16000,
    #       num_samples=1074150,
    #       duration=67.134375,
    #       channel_ids=[0],
    #       transforms=None
    #   )

    # light_small['small']['supervisions'][0]:
    #   SupervisionSegment(
    #       id='small/32/evening_star_0901_librivox_64kb_mp3/eveningstar_poe_blb_64kb',
    #       recording_id='small/32/evening_star_0901_librivox_64kb_mp3/eveningstar_poe_blb_64kb',
    #       start=0.0,
    #       duration=67.134375,
    #       channel=0,
    #       text=None,
    #       language='English',
    #       speaker='32',
    #       gender=None,
    #       custom=None,
    #       alignment=None
    #   )

    heavy_small = read_manifests_if_cached(["small"], "data/LibriHeavy", "libriheavy_", types=["cuts"])
    # {'small': {'cuts': CutSet(len=122526) [underlying data type: <class 'dict'>]}}

    # heavy_small['small']['cuts'][0]:
    #   MonoCut(
    #       id='small/100/sea_fairies_0812_librivox_64kb_mp3/01_baum_sea_fairies_64kb_0',
    #       start=243.9199981689453,
    #       duration=7.36,
    #       channel=0,
    #       supervisions=[
    #           SupervisionSegment(
    #               id='small/100/sea_fairies_0812_librivox_64kb_mp3/01_baum_sea_fairies_64kb_0',
    #               recording_id='small/100/sea_fairies_0812_librivox_64kb_mp3/01_baum_sea_fairies_64kb',
    #               start=0,
    #               duration=7.36,
    #               channel=0,
    #               text=None,
    #               language='English',
    #               speaker='100',
    #               gender=None,
    #               custom={
    #                   'texts': [
    #                       'The little girl was thoughtful for a moment. "But why do folks dive in the water when the mermaids smile an\' wink?" she asked.',
    #                       'THE LITTLE GIRL WAS THOUGHTFUL FOR A MOMENT BUT WHY DO FOLKS DIVE IN THE WATER WHEN THE MERMAIDS SMILE AND WINK SHE ASKED'
    #                   ], 
    #                   'pre_texts': [
    #                       's born, and the old sailor became very fond of the baby girl. Her real name was Mayre, but when she grew big enough to walk, she took so many busy little steps every day that both her mother and Cap\'n Bill nicknamed her "Trot," and so she was thereafter mostly called. It was the old sailor who taught the child to love the sea, to love it almost as much as he and her father did, and these two, who represented the "beginning and the end of life," became firm friends and constant companions. "Why hasn\'t anybody seen a mermaid and lived?" asked Trot again. "\'Cause mermaids is fairies, an\' ain\'t meant to be seen by us mortal folk," replied Cap\'n Bill. "But if anyone happens to see \'em, what then, Cap\'n?" "Then," he answered, slowly wagging his head, "the mermaids give \'em a smile an\' a wink, an\' they dive into the water an\' gets drownded." "S\'pose they knew how to swim, Cap\'n Bill?" "That don\'t make any diff\'rence, Trot. The mermaids live deep down, an\' the poor mortals never come up again.',
    #                       "H A GRIFFITH FAMILY THIS WAS ABOUT THE TIME TROT WAS BORN AND THE OLD SAILOR BECAME VERY FOND OF THE BABY GIRL HER REAL NAME WAS MARY BUT WHEN SHE GREW BIG ENOUGH TO WALK SHE TOOK SO MANY BUSY LITTLE STEPS EVERY DAY THAT BOTH HER MOTHER AND CAP'N BILL NICKNAMED HER TROT AND SO SHE WAS THEREAFTER MOSTLY CALLED IT WAS THE OLD SAILOR WHO TAUGHT THE GIRL TO LOVE THE SEA TO LOVE IT ALMOST AS MUCH AS HE AND HER FATHER DID AND THESE TOO WHO REPRESENTED THE BEGINNING AND THE END OF LIFE BECAME FIRM FRIENDS AND CONSTANT COMPANIONS WHY HASN'T ANYBODY SEEN A MERMAID AND LIVED ASKED TROT AGAIN CAUSE MERMAIDS IS FAIRIES AND AIN'T MEANT TO BE SEEN BY US MORTAL FOLK REPLIED CAP'N BILL BUT IF ANYONE HAPPENS TO SEE EM WHAT THEN CAP'N THEN HE ANSWERED SLOWLY WAGGING HIS HEAD THE MERMAIDS GIVE EM A SMILE AND A WINK AND THEY DIVES INTO THE WATER AND GETS DROWNDED S'POSE THEY KNOW HOW TO SWIM CAP'N BILL THAT DON'T MAKE ANY DIFFERENCE TROT THE MERMAIDS LIVE DEEP DOWN AND THE POOR MORTALS NEVER COME UP AGAIN"
    #                   ], 
    #                   'begin_byte': 4993,
    #                   'end_byte': 5120
    #               },
    #               alignment=None
    #           )
    #       ],
    #       features=None,
    #       recording=Recording(
    #           id='small/100/sea_fairies_0812_librivox_64kb_mp3/01_baum_sea_fairies_64kb',
    #           sources=[
    #               AudioSource(
    #                   type='file',
    #                   channels=[0],
    #                   source='download/librilight/small/100/sea_fairies_0812_librivox_64kb_mp3/01_baum_sea_fairies_64kb.flac'
    #               )
    #           ],
    #           sampling_rate=16000,
    #           num_samples=9567080,
    #           duration=597.9425,
    #           channel_ids=[0],
    #           transforms=None
    #       ),
    #       custom={
    #           'text_path': 'download/librilight_text/output_text_small_cleaned/Sea Fairies/text.txt'
    #       }
    #   )

    from_cuts = CutSet.from_jsonl('data/LibriHeavy/libriheavy_cuts_small.jsonl.gz')
    print(from_cuts)
    assert from_cuts == heavy_small['small']['cuts']