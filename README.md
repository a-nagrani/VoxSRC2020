# VoxSRC-20
In this repository we provide the validation toolkit for the [VoxCeleb Speaker Recognition Challenge 2020](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/competition.html). The challenge consists of two main tasks, speaker verification and speaker diarisation.  

This repository contains submodules. Please use the following command to clone the repository:

```
git clone https://github.com/a-nagrani/VoxSRC2020.git --recursive
```

#### Dependencies
```
pip install -r requirements.txt
```

## Speaker Verification 

Within speaker verification, we have 3 tracks (open, closed, and self-supervised), however the validation and test data is the same for all three tracks. The validation and test data for the challenge both consist of pairs of audio segments, and the task is to determine whether they are from the same speaker or from different speakers. Teams are invited to create a system that takes the test data and produces a list of floating-point scores, with a single score for each pair of segments.

### Validation Data 

In order to make a challenging validation set, we have obtained out of domain data for a subset of the identities in the VoxCeleb1 dataset. These additional audio segments can be downloaded [here](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/data/voxceleb1_cd.zip).

The validation data consists of trial pairs of speech from the identities in the VoxCeleb1 dataset. Each trial pair consists of two single-speaker audio segments and can be found in `data/verif/trials.txt`. 
         
### File Format 
Your ouput file for scoring should be a single space-delimited text files containing pair of audio segments per line, each line containing three fields: 

- ``Score``  --  score; must be a floating point number between the closed interval [0, 1], where 1 means the pair of segments correspond to the same speaker and 0 means the pair of segments correspond to different speakers.
- ``FILE1``  --  name of audio segment; should correspond to trials provided in `data/verif/trials.txt`
- ``FILE2``  --  name of audio segment; should correspond to trials provided in `data/verif/trials.txt`

  
For example: 
```
0.33462736175334906 voxceleb1/id10136/kDIpLcekk9Q/00011.wav voxceleb1/id10136/kTkrpBqwqJs/00001.wav
0.34354131205513616 voxceleb1/id10330/s6ZIytkNB1Q/00004.wav voxceleb1/id10330/2FPHpUr_Bss/00005.wav
0.3652844824796708 voxceleb1/id10934/R1twEbtsG1w/00003.wav voxceleb1/id10453/2q9ZA1_VeGM/00003.wav
```
Also see `data/verif/scores.txt` for an example file. 


The leaderboard for the challenge will compute two metrics, the Detection Cost and the Equal Error Rate (EER). Further details about the metrics are provided below. The file `compute_min_dcf.py` computes the detection cost. With the random scores provided you should get a Detection Cost of 1.0000 

```
python compute_min_dcf.py --p-target 0.05 --c-miss 1 --c-fa 1 data/verif/scores.txt data/verif/trials.txt

```
The file `compute_EER.py` computes the EER.  With the random scores provided you should get an EER of 49.975%
```
python compute_EER.py --ground_truth data/verif/trials.txt --prediction data/verif/scores.txt
```
While the leaderboard will display both metrics, the winners of the challenge will be determined using the Detection Cost ALONE, which is the primary metric. 

### Metrics 

#### Equal Error Rate 
This is the rate used to determine the threshold value for a system when its false acceptance rate (FAR) and false rejection rate (FRR) are equal. 
FAR = FP/(FP+TN) and FRR = FN/(TP+FN)

where

FN is the number of false negatives

FP is the number of false positives

TN is the number of true negatives

TP is the number of true positives

#### Minimum Detection Cost
Compared to equal error-rate, which assigns equal weight to false negatives and false positives, this
error-rate is usually used to assess performance in settings where achieving a low false positive rate is more important than achieving a low false
negative rate. We follow the procedure outlined in Sec 3.1 of the NIST 2018 [Speaker Recognition Evaluation Plan](https://www.nist.gov/system/files/documents/2018/08/17/sre18_eval_plan_2018-05-31_v6.pdf) for AfV trials, and use the following parameters for the cost function: 
1) CMiss (cost of a missed detection) = 1 
2) CFalseAlarm (cost of a spurious detection) = 1 
3) PTarget (a priori probability of the specified target speaker) = 0.05

This is the PRIMARY metric for the challenge.


## Speaker Diarisation 

For speaker diarisation, we only have a single track. The goal is to break up multispeaker segments into sections of "who spoke when". In our case, each multispeaker audio file is independant (i.e. we will treat the sets of speakers in each file as disjoint), and the audio files will be of variable length. Our scoring code is obtained from the excellent [DSCORE repo](https://github.com/nryant/dscore).

### Validation Data 

The validation data can be obtained following the instructions [here](https://github.com/joonson/voxconverse).

### File Format
Your output file for scoring (as well as the ground truth labels for the validation set which we provide) must be a [Rich Transcription Time Marked  (RTTM)](#rttm) file.

Rich Transcription Time Marked (RTTM) files are space-delimited text files
containing one turn per line, each line containing ten fields:

- ``Type``  --  segment type; should always by ``SPEAKER``
- ``File ID``  --  file name; basename of the recording minus extension (e.g.,
  ``abcde``)
- ``Channel ID``  --  channel (1-indexed) that turn is on; should always be
  ``1``
- ``Turn Onset``  --  onset of turn in seconds from beginning of recording
- ``Turn Duration``  -- duration of turn in seconds
- ``Orthography Field`` --  should always by ``<NA>``
- ``Speaker Type``  --  should always be ``<NA>``
- ``Speaker Name``  --  name of speaker of turn; should be unique within scope
  of each file
- ``Confidence Score``  --  system confidence (probability) that information
  is correct; should always be ``<NA>``
- ``Signal Lookahead Time``  --  should always be ``<NA>``

For instance:

    SPEAKER abcde 1   0.240   0.300 <NA> <NA> 3 <NA> <NA>
    SPEAKER abcde 1   0.600   1.320 <NA> <NA> 3 <NA> <NA>
    SPEAKER abcde 1   1.950   0.630 <NA> <NA> 3 <NA> <NA>

If you would like to confirm that your output RTTM file is valid, use the included validate_rttm.py script. We provide an example in `data/diar/baseline.rttm`
```
 python validate_rttm.py data/diar/baseline.rttm
```


The file `compute_diarisation_metrics.py` computes both DER and JER. 

```
python compute_diarisation_metrics.py -r voxconverse/dev/*.rttm -s data/diar/baseline.rttm
```
### Metrics

#### Diarization Error Rate (DER)

The leaderboard will be ranked using the diarization error rate (DER), which
is the sum of

- speaker error  --  percentage of scored time for which the wrong speaker id
  is assigned within a speech region
- false alarm speech  --   percentage of scored time for which a nonspeech
  region is incorrectly marked as containing speech
- missed speech  --  percentage of scored time for which a speech region is
  incorrectly marked as not containing speech

We use a collar of 0.25 seconds and include overlapping speech in the scoring. A score of zero indicates perfect performance and higher scores (which may exceed 100) indicate poorer performance. For more
details, consult section 6.1 of the [NIST RT-09 evaluation plan](https://web.archive.org/web/20100606041157if_/http://www.itl.nist.gov/iad/mig/tests/rt/2009/docs/rt09-meeting-eval-plan-v2.pdf).

#### Jaccard error rate (JER)
We also report Jaccard error rate (JER), a metric introduced for [DIHARD II](https://coml.lscp.ens.fr/dihard/index.html) that is based on the [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index). The Jaccard index is a similarity
measure typically used to evaluate the output of image segmentation systems and
is defined as the ratio between the intersection and union of two segmentations.


## Further Details 

Please refer to the [Challenge Webpage](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/competition.html) for more information about the challenge.

## Acknowledgements 
The code from computing the DetectionCost is largely obtained from the excellent [KALDI toolkit](), and for computing the DER is from the excellent [DSCORE repo](https://github.com/nryant/dscore). Please read their licenses carefully before redistributing. 
