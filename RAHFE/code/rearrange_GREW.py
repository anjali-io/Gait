import argparse
import os
import shutil
from pathlib import Path

from tqdm import tqdm

TOTAL_Test = 24000
TOTAL_Train = 20000
TOTAL_Distractor = 233857  # approx number of distractor sequences

def rearrange_train(train_path: Path, output_path: Path) -> None:
    progress = tqdm(total=TOTAL_Train)
    for sid in train_path.iterdir():
        if not sid.is_dir():
            continue
        for sub_seq in sid.iterdir():
            if not sub_seq.is_dir():
                continue
            for subfile in os.listdir(sub_seq):
                src = os.path.join(train_path, sid.name, sub_seq.name)
                dst = os.path.join(output_path, sid.name+'train', '00', sub_seq.name)
                os.makedirs(dst,exist_ok=True)
                if subfile not in os.listdir(dst) and subfile.endswith('.png'):
                    os.symlink(os.path.join(src, subfile),
                               os.path.join(dst, subfile))
        progress.update(1)

def rearrange_test(test_path: Path, output_path: Path) -> None:
    # for gallery
    gallery = Path(os.path.join(test_path, 'gallery'))
    probe = Path(os.path.join(test_path, 'probe'))
    progress = tqdm(total=TOTAL_Test)
    for sid in gallery.iterdir():
        if not sid.is_dir():
            continue
        cnt = 1
        for sub_seq in sid.iterdir():
            if not sub_seq.is_dir():
                continue
            for subfile in sorted(os.listdir(sub_seq)):
                src = os.path.join(gallery, sid.name, sub_seq.name)
                dst = os.path.join(output_path, sid.name, '%02d'%cnt, sub_seq.name)
                os.makedirs(dst,exist_ok=True)
                if subfile not in os.listdir(dst) and subfile.endswith('.png'):
                    os.symlink(os.path.join(src, subfile),
                               os.path.join(dst, subfile))
            cnt += 1
            progress.update(1)
    # for probe
    for sub_seq in probe.iterdir():
        if not sub_seq.is_dir():
            continue
        for subfile in os.listdir(sub_seq):
            src = os.path.join(probe, sub_seq.name)
            dst = os.path.join(output_path, 'probe', '03', sub_seq.name)
            os.makedirs(dst,exist_ok=True)
            if subfile not in os.listdir(dst) and subfile.endswith('.png'):
                os.symlink(os.path.join(src, subfile),
                            os.path.join(dst, subfile))
            progress.update(1)

def rearrange_distractor(distractor_path: Path, output_path: Path) -> None:
    """
    Rearrange distractor sequences into the same id/type/seq_name structure used
    by gallery and probe, so that the OpenGait dataloader can load them.

    Each distractor subject is mapped to:
        distractor_<subject_id>/04/<seq_name>/

    The seq_type '04' is used to mark distractor sequences, keeping them
    distinct from:
        gallery  -> seq_type '01' / '02'
        probe    -> seq_type '03'
    """
    print(f'[Distractor] Counting sequences in {distractor_path} ...')
    # The distractor folder may have flat structure: <seq_name>/ with .png files
    # or nested: <subject_id>/<seq_name>/ — we support both.
    total = 0
    entries = list(distractor_path.iterdir())
    nested = any((distractor_path / e.name).is_dir() and
                 any((distractor_path / e.name / s).is_dir()
                     for s in os.listdir(distractor_path / e.name))
                 for e in entries if e.is_dir())

    if nested:
        # Nested: distractor/<subject_id>/<seq_name>/*.png
        for sid in sorted(distractor_path.iterdir()):
            if not sid.is_dir():
                continue
            for sub_seq in sorted(sid.iterdir()):
                if not sub_seq.is_dir():
                    continue
                total += 1

        progress = tqdm(total=total, desc='Distractor (nested)')
        for sid in sorted(distractor_path.iterdir()):
            if not sid.is_dir():
                continue
            subject_tag = 'distractor_' + sid.name
            for sub_seq in sorted(sid.iterdir()):
                if not sub_seq.is_dir():
                    continue
                for subfile in os.listdir(sub_seq):
                    src = os.path.join(distractor_path, sid.name, sub_seq.name)
                    dst = os.path.join(output_path, subject_tag, '04', sub_seq.name)
                    os.makedirs(dst, exist_ok=True)
                    if subfile.endswith('.png') and subfile not in os.listdir(dst):
                        os.symlink(os.path.join(src, subfile),
                                   os.path.join(dst, subfile))
                progress.update(1)
    else:
        # Flat: distractor/<seq_name>/*.png  (each seq is an anonymous subject)
        seqs = [e for e in sorted(distractor_path.iterdir()) if e.is_dir()]
        progress = tqdm(total=len(seqs), desc='Distractor (flat)')
        for idx, sub_seq in enumerate(seqs):
            subject_tag = 'distractor_%06d' % idx
            for subfile in os.listdir(sub_seq):
                src = str(sub_seq)
                dst = os.path.join(output_path, subject_tag, '04', sub_seq.name)
                os.makedirs(dst, exist_ok=True)
                if subfile.endswith('.png') and subfile not in os.listdir(dst):
                    os.symlink(os.path.join(src, subfile),
                               os.path.join(dst, subfile))
            progress.update(1)

    print(f'[Distractor] Done. Sequences written to {output_path}')


def rearrange_GREW(input_path: Path, output_path: Path,
                   distractor_only: bool = False) -> None:
    os.makedirs(output_path, exist_ok=True)

    for folder in input_path.iterdir():
        if not folder.is_dir():
            continue

        print(f'Rearranging {folder}')
        if not distractor_only:
            if folder.name == 'train':
                rearrange_train(folder, output_path)
            if folder.name == 'test':
                rearrange_test(folder, output_path)
        if folder.name == 'distractor':
            rearrange_distractor(folder, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GREW rearrange tool')
    parser.add_argument('-i', '--input_path', required=True, type=str,
                        help='Root path of raw GREW dataset (containing train/, test/, distractor/).')
    parser.add_argument('-o', '--output_path', default='GREW_rearranged', type=str,
                        help='Root path for rearranged output.')
    parser.add_argument('--distractor_only', action='store_true',
                        help='Only rearrange the distractor folder (skip train/test). '
                             'Useful when train/test are already processed.')

    args = parser.parse_args()

    input_path = Path(args.input_path).resolve()
    output_path = Path(args.output_path).resolve()
    rearrange_GREW(input_path, output_path, distractor_only=args.distractor_only)
