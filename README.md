# argus

![image](images/banner.jpeg)

__argus__ searches your videos using example images to find and extract matching clips.

_Argus is named after [the greek giant with many eyes](https://en.wikipedia.org/wiki/Argus_Panoptes)._

## Steps
1. Extract sample images from your movie.
2. Move some sample images into the `good` and `bad` subdirectories.
3. Create predictions for frames from your movie.
4. Group sequential frame predictions into clip time ranges.
5. Extract clips from your movie.

## How it Works
An embedding is created for each sample image.
For each frame in the video a new embedding is created and compared against the embeddings of the sample images to make a prediction of good vs bad.
By default the [BioCLIP model](https://github.com/Imageomics/bioclip) is used to create embeddings.

## Requirements
- Python compatible with [PyTorch](https://pytorch.org/get-started/locally/#linux-python)
- [juypter](https://jupyter.org/) - provides an easy way to run the code and view images/videos
- [ffmpeg](https://ffmpeg.org/) - used to extract video clips

## Example
The [BirdVideo notebook](examples/BirdVideo.ipynb) is provided as a complete example that will install the dependcencies.

## License

`argus` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

