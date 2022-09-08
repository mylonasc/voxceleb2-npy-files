# Tools for managing the voxceleb2 dataset:
#   * downloading and reducing the annotations
#   * downloading small speech segments and keeping 
#     track of them in a disc cache

import librosa
import os
import numpy as np
try:
    import pytube
except:
    print('pytube not available.')
import json

#---- utils ----

DEFAULT_PREPROC_CONFIG = {'top_k' : 3, 'sr_audio' : 16000}

# Loading some assets:
try:
    annotation_asset_path = './assets' # os.environ['DATASET_UTILS']
    with open(os.path.join(annotation_asset_path, 'voxceleb_youtube_video_speaker_annotations.json'),'r') as f:
        ANNOT_ASSET_DICT = json.load(f)


except:
    raise Exception("Some assets failed to load!\nvoxceleb2 youtube video annotations are missing and the VoxCeleb2Dataset class cannot be used.")

class VoxCeleb2Dataset:
    def __init__(self, cache_folder = '.', preproc_config = DEFAULT_PREPROC_CONFIG):
        """
        A class for downloading VoxCeleb2 and for book-keeping the local cache/in-memory data.

        params: 
            cache_folder: 
              the folder to use for cache

            preproc_config:
              a dict containing some parameters for the preprocessing (how many files to download, sampling rate.).
              defaults to {'top_k' : 3, 'sr_audio' : 16000}
        """

        self._cache_folder = cache_folder 
        self.preproc_config = preproc_config
        self._all_files_cache = None
        self._prepare()

    def _prepare(self):
        top_k = self.preproc_config['top_k']
        sr_audio = self.preproc_config['sr_audio']
        if not os.path.exists(self._cache_folder):
            os.mkdir(self._cache_folder)
            print('created new cache folder at %s'%self._cache_folder)



        self._annot_dict = ANNOT_ASSET_DICT
        vid_sp_idx = {}; # an inverted index for video -> speaker
        for i, o in enumerate(self._annot_dict):
            for a in o:
                vid_sp_idx[a['vid_youtube']] = i

        self.vid_sp_idx = vid_sp_idx

        speaker_set = set()
        fc = self._get_file_cache_data()
        for vid,speaker,segment_idx in fc:
            speaker_set.add(speaker)

        self.speaker_set = speaker_set

        self._vid_segments_data = {}
        for vid_, _, seg_idx in self._all_files_cache:
            if vid_ not in self._vid_segments_data:
                self._vid_segments_data[vid_] = [seg_idx]
            else:
                self._vid_segments_data[vid_].append(seg_idx)

        self._inv_idx_vidseg= {}
        for k , (vid, _, seg) in enumerate(self._all_files_cache):
            self._inv_idx_vidseg[(vid, seg)] = k


        print('read %i files from cache covering %i speakers.'%(len(fc), len(speaker_set)))


    def _get_file_cache_data(self):
        """
        Using the information from the annotation it infers which files
        already exist. This index is not explicitly saved since it is
        easy to infer from the data on disc.

        """

        if self._all_files_cache is None:
            # the following can be large but not too large:
            def _get_chunk_idx_from_fname(fname : str):
                vid_name = fname[0:11]
                return vid_name, self.vid_sp_idx[vid_name], int(fname[12:15])
                        

            all_files = [_get_chunk_idx_from_fname(_fname) for _fname in os.listdir(os.path.join(self._cache_folder, 'voxceleb2','np_files')) if '.npy' == _fname[-4:]]
            self._all_files_cache = all_files

        return self._all_files_cache 

    def read_index(self, idx : int) -> np.ndarray:
        vid, speaker, idx = self._all_files_cache[idx]
        fname = os.path.join(self._cache_folder, 'voxceleb2', 'np_files','%s_%03i.npy'%(vid, idx))
        return np.load(fname), speaker

    def get_speaker_avail_files(self, speaker_index):
        """
        returns the indices of the available files 
        for the requested speaker
        """
        if speaker_index not in self.speaker_set:
            raise Exception('Speaker not available.')
            
        vids = [];
        for elm in self._annot_dict[speaker_index]:
            _vid = elm['vid_youtube']
            if _vid in self._vid_segments_data:
                for _seg in self._vid_segments_data[_vid]:
                    vids.append((_vid, _seg))

        return vids

    def get_speaker_avail_inds(self, speaker_index):
        """
        returns a set of indices, corresponding to 
        `self._all_files_cache` elements.
        """
        f = self.get_speaker_avail_files(speaker_index)
        return [self._inv_idx_vidseg[kk] for kk in f]

    def _download_resample_video_split_segments(self,
                                                speaker_index,
                                                video_integer_index,
                                                del_original_file = True,
                                                cache_output_path = None, 
                                                sr_audio = 16000,
                                                cached_files_downloaded_segments : dict = {}, 
                                                processed_files : set = set(), 
                                                all_speaker_annot_data = None 
                                                ):
        """
        Downloads a video, splits it in the segments where the speaker speaks, and
        caches is to disc.pytube

        params:

        speaker_index:
          the index of the speaker (integer)

        video_integer_index:
          an integer which denotes which video to take.

        del_original_file:
          whether to delete the original file after processing the segments

        cache_output_path:
          where to cache (top-level) the files. assumes the directory exists.

        sr_audio:
          the sampling rate to re-sample the file.

        cahced_files_downloaded_segments:
          a dictionary containing a pointer to the segments where to find the processed and
          resampled segments for loading.pytube

        processed_files:
           a set of files that are already completely processed. (to skip downloading and re-sampling if they are already downloaded)

        """
        raise Exception('Untested - added here as a starting point in case re-downloading the re-sampled files is needed again. ')

        v = all_speaker_annot_data[speaker_index][video_integer_index]
        video_unique_index = v['vid_youtube']

        if video_unique_index in processed_files:
            return

        vid_id, start_stop = v['vid_youtube'], v['start_stop']
        yt = pytube.YouTube('https://youtube.com/watch?v='+vid_id).streams.filter(type='audio')
        y = yt[0] # just the first one of the stream. No check for codec etc. (could be an improvement)
        file_path = y.download(cache_output_path)
        loaded_sampled_audio, _ = librosa.load(file_path, sr = sr_audio) #resampling takes some time (unfortunately)

        segments_dict = {};
        ## -- per segment --
        for i, (seg_start, seg_end) in enumerate(v['start_stop']):

            fname_np = '%s_%03i'%(video_unique_index, i)
            if fname_np in cached_files_downloaded_segments:
                continue

            fname_np_save = os.path.join(*[cache_output_path, 'np_files',fname_np])
            np.save(fname_np_save, loaded_sampled_audio[int(seg_start*sr_audio): int(seg_end * sr_audio)])
            cached_files_downloaded_segments.update({fname_np : (fname_np_save + '.npy', seg_end - seg_start)})

        proccessed_files.update(video_unique_index)

        if del_original_file:
            os.remove(file_path)

