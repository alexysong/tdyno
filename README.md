**T-Dyno** is a 2D finite-difference time-domain (FDTD) package that is capable of applying dynamic modulations in both the real and the imaginary parts of the permittivity, i.e. index modulations and gain/loss modulations.

## Features
T-Dyno natively supports the following features:
*   point sources

*   Total-field scattered-field (TF/SF) sources and directional plane-wave souces with the following temporal profiles:
    *   continuous waves (cw)
    *   Gaussian pulses
    *   wave packets, i.e. Gaussian modulated cw waves

*   Convolutional perfectly matched layers (CPML)

*   Supported materials:
    *   dispersionless lossy/lossless dielectrics
    *   dispersive dielectrics (Lorentz model)
    *   metal (Drude model)
    *   dynamically modulated refractive index
    *   dynamically modulated gain and loss

*   Shapes:
    *   rectangle
    *   circular
    *   ring
    *   wedge

*   Monitors

    Point monitors, both wave amplitude and power spectral density.

*   User interface

    Simple user interface where you can start, pause, reset, save plot, record videos, and save monitor data.


## Requirements

-   Python 2.7 or 3.6
-   Numpy >= 1.11.3
-   matplotlib >= 2.0.0
-   scipy >= 0.19.0
-   for recording videos, need to install `ffmpeg`.


## Usage
The `examples` in the `examples/` folder are the easiest places to get started with the package. 

*   `fdtd_2d.py`

    It's recommended that you go through `fdtd_2d.py` first, then move on to other examples. 
    
    `fdtd_2d.py` contains a detailed explanation of the basic functionalities of the package. It contains: point sources and a TF/SF source with different source temporal profiles, PML absorbing boundary condition, and different types of materials and geometries to build a structure.
    
    https://user-images.githubusercontent.com/55603472/130169577-fa837496-456e-4b95-bf7c-93633b9a2fc7.mp4

*   `ring_coupler.py`

    This is an example of a ring resonator coupled to input/output waveguides.

*   `waveguide_2d_dynamic_modulation.py` 

    This example contains a waveguide under dynamic modulations. The modulation can be in the refractive index or in the gain/loss.

    This example also contain several point monitors to measure wave intensities and calculate the spectrum.

    https://user-images.githubusercontent.com/55603472/130161582-5258d4f3-cc57-4e5d-8354-ffa2e242c366.mp4

## Citing

If you find T-Dyno useful for your research, we would apprecite you citing our [paper](https://doi.org/10.1103/PhysRevA.99.013824). For your convenience, you can use the following BibTex entry:

```
@article{song2019dynamic,
  title={Direction-dependent parity-time phase transition and nonreciprocal amplification with dynamic gain-loss modulation},
  author={Song, Alex Y. and Shi, Yu and Lin, Qian and Fan, Shanhui},
  journal={Physical Review A},
  volume={99},
  issue={1},
  pages={013824},
  numpages={7},
  year={2019},
  month={Jan},
  publisher={American Physical Society},
  doi={10.1103/PhysRevA.99.013824},
  url={https://link.aps.org/doi/10.1103/PhysRevA.99.013824}
}
```
