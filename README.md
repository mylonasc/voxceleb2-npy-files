## A speaker embedding focused subset of VoxCeleb2

This dataset is a partial re-scrape of the audio in VoxCeleb2 dataset.
The annotations shared are derived from the original ones (without any face-tracking info), but the segments are sorted for each speaker from smallest to largest, as estimated by the maximum timestamp in the annotations. If individual variation is of interest, small segments from each speaker should suffice.

## Usage

```python
from voxceleb2_dataset import VoxCeleb2Dataset

dataset = VoxCeleb2Dataset(cache = '.')

# A list of all the available speakers. They are indexed by integers.
speakers = list(dataset.speaker_set)

# A list of indices for the files corresponding to one speaker:
data_inds = dataset.get_speaker_avail_files(speakers[0])

# Loading the npy files that contain the voice segments for that speaker:
all_wavs_for_speaker = [dataset.read_index(i) for i in data_inds]
```


