## A speaker embedding-focused subset of VoxCeleb2

This dataset is a partial re-scrape of the audio in VoxCeleb2 dataset using `pytube` and some custom functions.

The annotations in `/assets/` folder are derived from the original ones (without any face-tracking info - 28MB instead of ~1.2GB).
The segments are sorted for each speaker from smallest to largest, as estimated by the maximum timestamp in the annotations. 

Please cite the original VoxCeleb2 paper if you use this dataset.
## Setup:

Because github does not allow pushing commits larger than 2GB, the `np_files` folder is split into two folders (`_a` and `_b`) which need to be collated upon cloning. 

In Linux or Mac (or in a bash terminal) run the following:
```bash
cp voxceleb2-npy-files/voxceleb2/np_files_a/*  voxceleb2-npy-files/voxceleb2/np_files_b/ && mv voxceleb2-npy-files/voxceleb2/np_files_b/ voxceleb2-npy-files/voxceleb2/np_files/
mv voxceleb2-npy-files/* .
```


## Usage:
```python

from voxceleb2_dataset import VoxCeleb2Dataset

dataset = VoxCeleb2Dataset(cache_folder = '.')

# A list of all the available speakers. They are indexed by integers.
speakers = list(dataset.speaker_set)

# A list of indices for the files corresponding to one speaker:
data_inds = dataset.get_speaker_avail_inds(speakers[0])

# Loading the npy files that contain the voice segments for that speaker:
all_wavs_for_speaker = [dataset.read_index(i) for i in data_inds]

```

see the notebook for an example where this dataset is used for training speaker embeddings using the wav2vec 2.0 model.


