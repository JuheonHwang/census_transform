# Sample stereo pair

`sample/left.png` and `sample/right.png` are a rectified pair of 8-bit, single-channel
active infrared (IR) stereo images taken from the public **GigaDepth** dataset. The
sensor's IR dot projector was on, so a dense dot / speckle pattern is visible in both
images. They are used by the tutorial in place of private capture data.

There is **no ground-truth disparity** for this pair. The source dataset also contains depth
maps (`depth*.png`, 1280 x 960, 16-bit), but their alignment to the IR images was not
verified, so no ground truth is shipped in this repository.

## Source

- Dataset: "GigaDepth: Learning Depth from Structured Light with Branching Neural Networks",
  TU Wien Research Data, published 2022-10-22
- Authors: Simon Schreiberhuber, Jean-Baptiste Weibel, Timothy Patten, Markus Vincze
- DOI: https://doi.org/10.48436/q76vf-y9t57
- Landing page: https://researchdata.tuwien.at/records/q76vf-y9t57
- File used: `dataset_gigadepth_jpg.zip`, the first scene folder under `captured_test/`
  (the dataset's real-world captures), file `ir0.png`. This is a 2432 x 896, 16-bit
  grayscale image that holds both IR cameras side by side.
- Accompanying paper: the dataset's title is also the title of the ECCV 2022 paper below.

```bibtex
@misc{schreiberhuber2022gigadepth_data,
  author    = {Schreiberhuber, Simon and Weibel, Jean-Baptiste and Patten, Timothy and Vincze, Markus},
  title     = {GigaDepth: Learning Depth from Structured Light with Branching Neural Networks},
  year      = {2022},
  publisher = {TU Wien Research Data},
  doi       = {10.48436/q76vf-y9t57},
  url       = {https://researchdata.tuwien.at/records/q76vf-y9t57}
}

@inproceedings{schreiberhuber2022gigadepth,
  author    = {Schreiberhuber, Simon and Weibel, Jean-Baptiste and Patten, Timothy and Vincze, Markus},
  title     = {GigaDepth: Learning Depth from Structured Light with Branching Neural Networks},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2022}
}
```

## Licence

Creative Commons Attribution 4.0 International (CC BY 4.0), as stated in the dataset
record's metadata, https://creativecommons.org/licenses/by/4.0/ . You may share and adapt the
material as long as you give appropriate credit, link to the licence, and indicate changes.
The changes made to the images are listed below. The licence applies to these two image
files only, not to the code in this repository.

## Modifications

The two images in this repository are **modified** from the original `ir0.png`:

1. The 2432 x 896 image was split into two 1216 x 896 halves.
2. The halves were swapped. In the original file the first half (x < 1216) is the
   **right** camera and the second half the **left** camera. This was determined
   empirically from feature matches: `x_left - x_right > 0` holds only after swapping.
3. The 16-bit values were converted to 8 bit with a percentile stretch (0.5th to 99.8th
   percentile mapped to 0..255, clipped), so the brightest dots are saturated.

No other processing (no cropping, resizing, filtering or rectification) was applied.

## What the images are

| Property | Value |
|---|---|
| Sensor | Occipital Structure Core (stereo IR cameras with an IR dot-pattern projector) |
| Illumination | Active: IR dot projector on, dense dot / speckle pattern visible in both images (active IR stereo) |
| Scene | Lab desk with boxes, a calibration target and an ArUco marker |
| Size | 1216 x 896 px each |
| Format | PNG, 8-bit, 1 channel (mode `L`) |
| Rectified | Approximately. SIFT matches show a constant sub-pixel row bias: vertical offset median -0.59 px, 5th-95th percentile -0.94 to -0.30 px |
| Left / right | `left.png` is the left camera, `right.png` the right camera. Matched features move **left** from the left to the right image (`x_left - x_right > 0`), the standard convention, so `d = x_left - x_right`, searching `x_right = x_left - d` |
| Disparity | Matched features: 45 to 102 px (1st-99th percentile), median 90 px. The tutorial pipeline's valid disparities (see below): 39.6 / 56.6 / 100.0 px at the 1st / 50th / 99th percentile |

Calibration values (focal length, baseline) are not shipped with the pair and are not
needed by the tutorial, which stops at disparity.

## Suggested parameters

Measured with the PyTorch pipeline in `pytorch/census_stereo.py`:

```
census 7, aggregation 11, max_disp 128, LR threshold 1
```

With these settings about 90% of the pixels keep a valid disparity after the left-right
check, and the matching takes roughly 12 s on a laptop CPU. The sample's disparities reach
about 100 px, so `max_disp` has to be above that (128 is used here); a smaller value such as
64 would cut off the nearer parts of the scene. These values are a reasonable starting
point, not tuned.

## Obtaining the full dataset

The full dataset is very large (the zip archives are about 463 GB and 187 GB), so the tutorial
does not need it. It can be downloaded from the landing page above. A single frame can be
pulled without downloading the whole archive by issuing HTTP range requests against the zip
file (read the central directory at the end of the archive, then request just the bytes of
the entry you want and inflate it).
