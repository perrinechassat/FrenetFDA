


preprocessing_euclidean_curve.py

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
            - __compute_full_P_smooth
            