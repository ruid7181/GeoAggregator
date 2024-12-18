class AggregatorHyperParameters:
    # -------------------------------------------------------
    # Main configurations
    seq_len = 25
    d_model = 32
    batch_size = 8
    # -------------------------------------------------------
    # Encoder
    x_embed_dim = 16
    y_embed_dim = 16
    x_n_head = 2
    y_n_head = 2
    # -------------------------------------------------------
    # Decoder
    decoder_lin_dims = [32, 16, 1]   # d_model -> decoder_lin_dims -> 1
    # -------------------------------------------------------


class TabDataColumns:
    shp_atr = ['bathrooms', 'sqft_living', 'sqft_lot',
               'grade', 'condition', 'waterfront',
               'view', 'age']
    shp_spa = ['UTM_X', 'UTM_Y']   # 26910
    shp_y = ['y']

    pm25_atr = ['dem_scaled', 'aod_scaled', 't2m_scaled', 'tp_scaled',
                'dwi_scaled', 'r_scaled', 'wind_scaled']
    pm25_spa = ['proj_x', 'proj_y']
    # pm25_spa = ['longitude', 'latitude']   # Uncomment this line, for the SRGCNN model ONLY
    pm25_y = ['pm25']

    sdoh_atr = ['ep_unem', 'ep_pci', 'ep_nohs', 'ep_sngp',
                'ep_lime', 'ep_crow', 'ep_nove', 'rent_1', 'rntov30p_1',
                'ep_unin', 'ep_minrty', 'ep_age65', 'ep_age17', 'ep_disabl']
    sdoh_spa = ['latitude', 'longitude']
    sdoh_y = ['ep_pov']

    syn_atr = ['x1', 'x2']
    syn_spa = ['coord1', 'coord2']
    syn_y = ['y']


class RegisteredDS:
    tab = ['shp',
           'pm25',
           'syn-gwr-r',
           'syn-sl-r',
           'syn-slx-r',
           'syn-durbin-r',
           'syn-gwr-d',
           'syn-sl-d',
           'syn-slx-d',
           'syn-durbin-d',
           'sdoh']


class XGBHyperparameters:
    # ==================================================================================================== #
    # Synthetic datasets
    # ========================================== X type = rand =========================================== #
    hp_gwr = {'max_depth': 17,
              'learning_rate': 0.019332604858627436,
              'n_estimators': 539,
              'min_child_weight': 10,
              'gamma': 0.5370461838656522,
              'subsample': 0.2624369032557898,
              'colsample_bytree': 0.7695319268224925,
              'reg_alpha': 0.6265942986602789,
              'reg_lambda': 0.3309810120420501,
              'early_stopping_rounds': 51}

    hp_sl = {'max_depth': 5,
             'learning_rate': 0.08504614649852574,
             'n_estimators': 983,
             'min_child_weight': 3,
             'gamma': 0.07133670008573871,
             'subsample': 0.7983705410248836,
             'colsample_bytree': 0.9972362259218397,
             'reg_alpha': 0.8749724301361296,
             'reg_lambda': 0.43759534890334667,
             'early_stopping_rounds': 57}

    hp_slx = {'max_depth': 13,
              'learning_rate': 0.018798242156974766,
              'n_estimators': 398,
              'min_child_weight': 9,
              'gamma': 0.5403236113647621,
              'subsample': 0.08348750739123773,
              'colsample_bytree': 0.8629148393025928,
              'reg_alpha': 0.9139015193014638,
              'reg_lambda': 0.5002986430562034,
              'early_stopping_rounds': 63}

    hp_durbin = {'max_depth': 10,
                 'learning_rate': 0.05763252662136285,
                 'n_estimators': 323,
                 'min_child_weight': 10,
                 'gamma': 0.5446397699193813,
                 'subsample': 0.8093977615483291,
                 'colsample_bytree': 0.6570588063711134,
                 'reg_alpha': 0.28994015380045635,
                 'reg_lambda': 0.38804189286268975,
                 'early_stopping_rounds': 59}

    # ========================================== X type = dem =========================================== #

    hp_d_gwr = {'max_depth': 5,
                'learning_rate': 0.02891079480243005,
                'n_estimators': 343,
                'min_child_weight': 5,
                'gamma': 0.8570702424577318,
                'subsample': 0.49088150103164807,
                'colsample_bytree': 0.7621480884506752,
                'reg_alpha': 0.5636045434264513,
                'reg_lambda': 0.6623859570533317,
                'early_stopping_rounds': 22}

    hp_d_sl = {'max_depth': 14,
               'learning_rate': 0.01944402900288293,
               'n_estimators': 994,
               'min_child_weight': 4,
               'gamma': 0.03213936994531709,
               'subsample': 0.2722958646253648,
               'colsample_bytree': 0.9194444088473708,
               'reg_alpha': 0.7517543024911514,
               'reg_lambda': 0.08057396774389677,
               'early_stopping_rounds': 86}

    hp_d_slx = {'max_depth': 11,
                'learning_rate': 0.01938206828786753,
                'n_estimators': 498,
                'min_child_weight': 9,
                'gamma': 0.457600648760649,
                'subsample': 0.07137893388416572,
                'colsample_bytree': 0.8545643535140672,
                'reg_alpha': 0.4564219083105699,
                'reg_lambda': 0.41509732834187807,
                'early_stopping_rounds': 36}

    hp_d_durbin = {'max_depth': 17,
                   'learning_rate': 0.01918487451271496,
                   'n_estimators': 471,
                   'min_child_weight': 2,
                   'gamma': 0.5184509099584765,
                   'subsample': 0.5305025733606055,
                   'colsample_bytree': 0.9276444256973982,
                   'reg_alpha': 0.9285505911988069,
                   'reg_lambda': 0.1408949837869194,
                   'early_stopping_rounds': 54}

    # ==================================================================================================== #
    # Real-world Tab datasets
    # ==================================================================================================== #
    hp_tab_shp_mae = {'max_depth': 13,
                      'learning_rate': 0.05163244842640724,
                      'n_estimators': 933,
                      'min_child_weight': 10,
                      'gamma': 0.012785240471988908,
                      'subsample': 0.6498766880305861,
                      'colsample_bytree': 0.5690412546972458,
                      'reg_alpha': 0.6283481513672151,
                      'reg_lambda': 0.7737194108730473,
                      'early_stopping_rounds': 31}
    hp_tab_shp_mse = {'max_depth': 19,
                      'learning_rate': 0.02297294697039517,
                      'n_estimators': 960,
                      'min_child_weight': 10,
                      'gamma': 0.011019331721181826,
                      'subsample': 0.37631780470703236,
                      'colsample_bytree': 0.7798344981482365,
                      'reg_alpha': 0.17163023082047466,
                      'reg_lambda': 0.8948321752529722,
                      'early_stopping_rounds': 74,
                      'objective': 'reg:squarederror'}

    hp_tab_shp_no_norm = {'max_depth': 9,
                           'learning_rate': 0.01573446775363104,
                           'n_estimators': 615,
                           'min_child_weight': 4,
                           'gamma': 0.7676949858809861,
                           'subsample': 0.5165344718068131,
                           'colsample_bytree': 0.7745639429133382,
                           'reg_alpha': 0.5414395855351936,
                           'reg_lambda': 0.6244831226316103,
                           'early_stopping_rounds': 41}

    hp_pm25_mse = {'max_depth': 14,
                   'learning_rate': 0.06665824966294237,
                   'n_estimators': 861,
                   'min_child_weight': 5,
                   'gamma': 0.24499896164121665,
                   'subsample': 0.7341601404852086,
                   'colsample_bytree': 0.924711075123758,
                   'reg_alpha': 0.19438359493681082,
                   'reg_lambda': 0.3258115081433695,
                   'early_stopping_rounds': 49,
                   'objective': 'reg:squarederror'}

    hp_sdoh_mse = {'max_depth': 6,
                   'learning_rate': 0.025360335871174856,
                   'n_estimators': 960,
                   'min_child_weight': 3,
                   'gamma': 0.2731382139237552,
                   'subsample': 0.7825324482777349,
                   'colsample_bytree': 0.7873155605413902,
                   'reg_alpha': 0.49356584092922623,
                   'reg_lambda': 0.376541233389748,
                   'early_stopping_rounds': 62,
                   'objective': 'reg:squarederror'}
