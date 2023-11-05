<!-- ![FrenetFDA Banner](link_to_your_banner_image) -->

FrenetFDA
=========

**FrenetFDA** is a Python package for funtional and shape data analysis of Euclidean curves in $\mathbb{R}^d$ based on the Frenet-Serret representation. 

The package includes statistical methods for estimating the Frenet curvatures functional parameters of a curve (particularly in 3D, the curvature and torsion functions). 
Several shape analysis methods based on these parameters are then implemented, including the method based on the Square-Root Curvature Transform. 
In addition, this package includes implementations of various statistical methods for estimating the mean shape of a population of curves within the Frenet framework.

Papers: \\
    - [Shape Analysis of Euclidean Curves under Frenet-Serret Framework](https://openaccess.thecvf.com/content/ICCV2023/papers/Chassat_Shape_Analysis_of_Euclidean_Curves_under_Frenet-Serret_Framework_ICCV_2023_paper.pdf) \\
    - [Curvature and Torsion estimation of 3D functional data: A geometric approach to build the mean shape under the Frenet Serret framework](https://arxiv.org/abs/2203.02398)


🔗 Requirements
===============
Python 3.8+ 
<!-- All methods based on the Square-Root Velocity Function representation are done using the p  -->

Install the required packages:

```
$ pip install -r requirements.txt
```

🛠 Installation
===============

Clone the repo and run the following command in the directory to install FrenetFDA:

```
$ python3 setup.py install
```

<!-- ⚡️ Quickstart
============== -->






<!-- preprocessing_euclidean_curve.py

    Class: PreprocessingEuclideanCurve

    Attributs:
        - Y, N, dim, time, time_derivatives, scale_ind, L, arc_s_dot, arc_s, grid_arc_s, arc_lenght_derivatives

    Methods:
        Accessibles: 
            - compute_time_derivatives
            - compute_arc_length
            - compute_arc_length_derivatives
            - compute_Z_Gram_Schmidt
            - compute_Z_local_poly_regregression
            - raw_curvatures_extrinsic_formulas
        Non Accessibles:
            - __constrained_local_polynomial_regression
            - __get_loc_param
            - __GramSchmidt


preprocessing_Frenet_path.py

    Class: PreprocessingFrenetPath

    Attributs:

    Methods:
        Accessibles:
            - raw_curvatures_approx_Frenet_ODE
            - raw_curvatures_local_approx_Frenet_ODE
            - 
        Non Accessibles:
            - __compute_sort_unique_val

    
smoothing_utils.py

    Functions:
        - kernel
        - adaptive_kernel
        - local_polynomial_smoothing
        - compute_weight_neighbors_local_smoothing


iek_filter_smoother_Frenet_state.py

    Class: IEKFilterSmootherFrenetState 

    Attributs:
        - n, dim_g, Sigma, W, gamma, P0, P, C, Q, X, Z, U, list_U, K_pts, F, A, H, L,
        - pred_Z, pred_gamma, pred_P, pred_C, track_Z, track_X, track_Q, track_gamma, track_P
        - smooth_Z, smooth_X, smooth_P_dble, smooth_P, smooth_P_full
    
    Methods:
        Accessibles:
            - tracking
            - smoothing

        Non Accessibles:
            - __propagation_Z
            - __propagation_gamma
            - __propagation_P
            - __propagation_C
            - __propagation_U
            - __predict
            - __update
            - __compute_full_P_smooth -->
            