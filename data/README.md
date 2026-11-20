# Sample stereo pair

`sample/left.png` and `sample/right.png` are a rectified pair of 8-bit, single-channel
infrared (near-IR) images from the public dataset *Digiteo_seq3_Stereo*. They are used by
the tutorial in place of private capture data.

There is **no ground-truth disparity** for this pair (the source dataset does not provide any).

## Source

- Dataset: "Digiteo_seq3_Stereo", Mendeley Data, Version 2, 2021
- Authors: Imad El Bouazzaoui, Sergio Rodriguez Florez, Abdelhafid El Ouardi (Paris-Saclay University)
- DOI: https://doi.org/10.17632/c2gtvyxyt7.2
- Landing page: https://data.mendeley.com/datasets/c2gtvyxyt7
- Files used: `Stereo/image_0/frame600.png` (left) and `Stereo/image_1/frame600.png` (right).
  They are copied unmodified (already 8-bit grayscale); only renamed.

```bibtex
@misc{elbouazzaoui2021digiteo,
  author    = {El Bouazzaoui, Imad and Rodriguez Florez, Sergio and El Ouardi, Abdelhafid},
  title     = {Digiteo\_seq3\_Stereo},
  year      = {2021},
  publisher = {Mendeley Data},
  version   = {2},
  doi       = {10.17632/c2gtvyxyt7.2},
  url       = {https://data.mendeley.com/datasets/c2gtvyxyt7}
}
```

## Licence

Creative Commons Attribution 4.0 International (CC BY 4.0),
http://creativecommons.org/licenses/by/4.0 . As stated on the dataset page, you may share,
copy and modify the dataset so long as you give appropriate credit, link to the licence, and
indicate changes. This permits redistributing these two images here with the attribution
above. (The two images are not modified.) The licence applies to these two image files
only, not to the code in this repository.

## What the images are

| Property | Value |
|---|---|
| Sensor | Intel RealSense D435i, left/right infrared imagers (global shutter, near-IR, roughly 850 nm band) |
| Scene | Underground car park (Digiteo building 660, Paris-Saclay); a checkerboard, a pillar and cars in the foreground |
| Size | 1280 x 720 px |
| Format | PNG, 8-bit, 1 channel (mode `L`) |
| Rectified | Yes (D435 factory rectification). Measured: median vertical offset of matched features 0.02 px; 5-95 percentile -0.4 to +0.8 px |
| Left / right | `left.png` = `image_0`, `right.png` = `image_1`. Matched features move **left** from left to right image (x_left - x_right > 0), the standard convention, so `d = x_left - x_right`, searching `x_right = x_left - d` |
| Focal length / baseline | fx = fy = 638.149 px, cx = 639.76, cy = 356.51, baseline about 50 mm (`bf` = 31.90 px*m in the dataset's `param.yaml`) |
| Illumination | Ambient car-park lighting. The dataset page only says "Stereo mode"; no projected dot pattern is visible in the images, so the IR projector appears to be off (not verified from the source). The scene is therefore passive NIR stereo, not active-pattern stereo. |

Depth from disparity: `Z = 638.149 * 0.05 / d` metres (about 1.1 m at d = 28 px, about 7 m at d = 4.6 px).

## Disparity range and suggested parameters

Measured on this pair (SIFT matches with |dy| < 3 px, plus OpenCV SGBM as a cross-check):
disparity is roughly 0.05 to 29.5 px (SIFT min/max); SGBM 1st-99th percentile 4.25 to 27.9 px;
all positive. The baseline is short, so the range is modest.

Suggested starting values (heuristic, from a quick sanity check against SGBM, not tuned):

- `max_disp`: 32 (covers the observed maximum of about 30; use 48 if you want headroom,
  but values above about 32 only add false matches in the textureless concrete)
- Census window: 7 x 7 (5 x 5 also works but is noisier on the low-texture concrete floor and walls)
- Aggregation (box) window: 9 x 9 to 15 x 15 (larger windows gave better agreement with SGBM
  on this smooth, low-texture scene; in a quick check with a plain winner-takes-all
  census + box filter, agreement within 1 px with SGBM was about 0.65 for 7x7/9x9 and 0.75 for 7x7/15x15)
- Expect noise in the dark/bright saturated regions and on the concrete floor; add a
  left-right consistency check and/or uniqueness test in the tutorial.

## Obtaining the full dataset

The full archive (`Digiteo_seq3_Stereo.zip`, about 4.2 GB, 7144 stereo frames, `image_0` /
`image_1` in KITTI-like layout, no registration needed) can be downloaded from the landing
page above ("Download All"). Other frames can also be pulled individually with HTTP range
requests against the ZIP file.
