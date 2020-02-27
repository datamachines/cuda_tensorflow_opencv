`run_X11CV2test.sh` starts `xeyes` to confirm X11 display is functional and display an image using python3's OpenCV to confirm OpenCV and its python3 bindings are functional. Runs for all containers available from the list in the `Makefile`

`run_nvidiasmi.sh` starts `nvidia-smi` for each `cuda_` and `cudnn_` containers to confirm the GPU passthrough is functional.

Note:
Free for commercial use image found on https://pixabay.com/vectors/test-pattern-tv-tv-test-pattern-152459/
