# phase_ret
Tools for optical wavefront reconstruction via non-linear optimization. The code implements the algorithm described by Brady & Fienup [1], which relies on intensity measurements in several defocused planes.

## Using CUDA-accelerated transforms on Windows
Installing CUDA, pyCUDA and scikit-cuda on Windows can be a bit troublesome. Here are some tips for a successful setup:

1. After installing the CUDA toolkit, to install pyCUDA and scikit-cuda via pip, make sure that the following directories are present in your PATH environment variable:

    * `C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin`
    * `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin`
    * `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include`
  
    If your install directories differ from defaults, modify the above accordingly.
    
2. The current stable release of scikit-cuda available via pip is 0.5.1. Unfortunately, it does not support CUDA 8.0. One solution is to build the [development release](https://github.com/lebedov/scikit-cuda), which has the issue fixed. Another option is to use the 0.5.1 verson with the following simple hack. In your python directory, go to `Lib\site-packages\skcuda` and replace the line:

        _version_list = [7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.0]
        
    with
        
        _version_list = [8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.0]
        
    in all of the following files: `cublas.py, cudart.py, cufft.py, cusolver.py, cusparse.py`. This will ensure that scikit-cuda can locate correct CUDA DLLs.


[1]  G. R. Brady and J. R. Fienup. Nonlinear optimization algorithm for retrieving the full complex pupil function. *Optics Express*, 14(2):474, 2006. [doi:10.1364/OPEX.14.000474](http://dx.doi.org/10.1103/PhysRevLett.98.060404).
