
import polars as pl
import numpy as np

# -------------------------------
# add feature columns
# -------------------------------
def add_feature_cols(df_, FEATURE_COLS,remove_col_list):
    additional_cols = list(df_.columns)
    additional_cols = [col for col in additional_cols if not col in remove_col_list]
    FEATURE_COLS.extend(additional_cols)
    return FEATURE_COLS

# -------------------------------
# Merge to taget functions
# -------------------------------
def target_merge_tracking(target_df, tracking_df, FEATURE_COLS_, SHIFT_COLS_,
                          TRACKING_COLS_ = ["game_play", "nfl_player_id", "step", 
                                           "x_position", "y_position", "datetime",
                                           "speed","distance","direction","orientation",
                                           "acceleration","sa", "team", "jersey_number"]):
    target_df = target_df.with_column(pl.concat_str([pl.col("game_play"),
                                                     pl.col("step").cast(str),
                                                     pl.col("nfl_player_id_1"),
                                                    ], sep='_').alias('game_step_player_1'))
    target_df = target_df.with_column(pl.concat_str([pl.col("game_play"),
                                                     pl.col("step").cast(str),
                                                     pl.col("nfl_player_id_2"),
                                                    ], sep='_').alias('game_step_player_2'))

    TRACKING_COLS_.extend(SHIFT_COLS_)
    # print(TRACKING_COLS_)
    tracking_df = tracking_df.select(TRACKING_COLS_)
    tracking_df = tracking_df.with_column(pl.concat_str([pl.col("game_play"),
                                                         pl.col("step").cast(str),
                                                         pl.col("nfl_player_id"),
                                                        ], sep='_').alias('game_step_player'))

    tracking_df = tracking_df.drop(["game_play", "step", "nfl_player_id", "datetime"])

    # merge tracking to target
    for player_id in [1,2]:
        tracking_player = tracking_df.select([pl.all().suffix(f"_{player_id}")])
        target_df = target_df.join(tracking_player, on=[f"game_step_player_{player_id}"], how="left")
        # add features col
        FEATURE_COLS_ = add_feature_cols(tracking_player, FEATURE_COLS_,
                                        [f"game_step_player_{player_id}", f"frame_{player_id}", f"datetime_{player_id}"])
    # drop col
    target_df = target_df.drop(["game_step_player_1", "game_step_player_2"])
    print(len(target_df.columns))
    print("original length", len(target_df))
    return target_df, FEATURE_COLS_

def target_merge_helmet(target_df, helmet_df, FEATURE_COLS):
    print("original length", len(target_df))
    # set merge-key (game_frame_player_1,2) to merge helmet_df
    target_df = target_df.with_column(((pl.col("step")/10*59.94+5*59.94).cast(int)+1).alias("frame"))
    target_df = target_df.with_column(pl.concat_str([pl.col("game_play"),
                                                     pl.col("frame").cast(str),
                                                     pl.col("nfl_player_id_1"),
                                                    ], sep='_').alias('game_frame_player_1'))
    target_df = target_df.with_column(pl.concat_str([pl.col("game_play"),
                                                     pl.col("frame").cast(str),
                                                     pl.col("nfl_player_id_2"),
                                                    ], sep='_').alias('game_frame_player_2'))
    # set merge key
    helmet_df = helmet_df.with_column(pl.concat_str([pl.col("game_play"),
                                                     pl.col("frame").cast(str),
                                                     pl.col("nfl_player_id"),
                                                    ], sep='_').alias('game_frame_player'))

    # merge target df & helmet_df
    player_views = [[1, "Endzone"],[2, "Endzone"], [1, "Sideline"],[2, "Sideline"]]
    for player_id, view in player_views:
        helmet_view = helmet_df.filter(pl.col("view")==view)
        helmet_view = helmet_view[["game_frame_player", "left", "width", "top", "height"]]
        helmet_view = helmet_view.select(pl.all().suffix(f"_{player_id}"))
        helmet_view = helmet_view.select([pl.col(helmet_view.columns[0]), pl.col(helmet_view.columns[1:]).prefix(f"{view[0]}_")])
        target_df = target_df.join(helmet_view, on=f"game_frame_player_{player_id}", how="left")  
        # add features col
        FEATURE_COLS = add_feature_cols(helmet_view, FEATURE_COLS, [f"game_frame_player_{player_id}"])

    print(len(target_df.columns))
    print("original length", len(target_df))
    return target_df, FEATURE_COLS

# -------------------------------
# create features from tracking df
# -------------------------------
def get_tracking_shift(tracking_df_, shift_cols=["x_position", "y_position"], shift_nums=range(-6,6,1)):
    # get shift key
    #     tracking_df = tracking_df.with_column((pl.col('step')/10*59.94+5*59.94 + 5000).alias('frame_add'))
    tracking_df_ = tracking_df_.with_column(pl.concat_str([pl.col("game_play"),
                                                         pl.col("nfl_player_id"),
                                                        ], sep='_').alias('shift_key'))
    # get shift features
    SHIFT_COLS_ = []
    for num in shift_nums:
        tracking_df_ = tracking_df_.with_columns(
                            pl.col(shift_cols).shift(periods=num).over("shift_key").suffix(f"_shift{num}"))
    for col in shift_cols:
        colname = [f"{col}_shift{idx}" for idx in shift_nums]
        SHIFT_COLS_.extend(colname)
                    
    return tracking_df_, SHIFT_COLS_



# -------------------------------
# create features from merged df
# -------------------------------
def create_trackmerged_ftr(target_df, FEATURE_COLS,
                           diff_cols = ['x_position', 'y_position', 'speed', 'distance',
                                        'direction', 'orientation', 'acceleration', 'sa']):
    target_df = target_df.with_column((np.sqrt(np.square(pl.col("x_position_1")-pl.col("x_position_2")) \
                                              + np.square(pl.col("y_position_1")-pl.col("y_position_2"))) \
                                      ).alias("players_dis"))
    target_df = target_df.with_column(pl.col("players_dis").fill_null(0))
    
    # players distance sum(in shift range time : default(-6~6 frames not step))
    players_distance_sum = 0
    for idx in range(-6,6,1):
        players_distance_sum += np.sqrt((target_df[f"x_position_shift{idx}_1"] - target_df[f"x_position_shift{idx}_2"])**2 \
                                       + (target_df[f"y_position_shift{idx}_1"] - target_df[f"y_position_shift{idx}_2"])**2)
    target_df = target_df.with_column(pl.Series("players_distance_sum", players_distance_sum))
    target_df = target_df.with_column(pl.col("players_distance_sum").fill_null(0))
    FEATURE_COLS.append("players_distance_sum")

    # players each axis distance sum(in shift range time : default(-6~6 frames not step))
    for axis in ["x", "y"]:
        axis_distance_1 = 0
        axis_distance_2 = 0
        for idx in range(-6, 5, 1):
            axis_distance_1 += abs(target_df[f"{axis}_position_shift{idx}_1"] - target_df[f"{axis}_position_shift{idx+1}_1"])
            axis_distance_2 += abs(target_df[f"{axis}_position_shift{idx}_2"] - target_df[f"{axis}_position_shift{idx+1}_2"])
        target_df = target_df.with_column(pl.Series(f"{axis}_move_distance_1", axis_distance_1))
        target_df = target_df.with_column(pl.col(f"{axis}_move_distance_1").fill_null(0))
        target_df = target_df.with_column(pl.Series(f"{axis}_move_distance_2", axis_distance_2))
        target_df = target_df.with_column(pl.col(f"{axis}_move_distance_2").fill_null(0))
        FEATURE_COLS.extend([f"{axis}_move_distance_1", f"{axis}_move_distance_2"])

    # players difference ftr (in each step)
    for col in diff_cols:
        colname = f"{col}_diff"
        target_df = target_df.with_column((abs(pl.col(f"{col}_1") - pl.col(f"{col}_2"))).alias(colname))
        target_df = target_df.with_column(pl.col(colname).fill_null(0))
        FEATURE_COLS.append(colname)
        
    return target_df, FEATURE_COLS

def create_helmetmerged_ftr(target_df, FEATURE_COLS_):
    # helmet center distance feature
    for view in ["Endzone", "Sideline"]:
        for p_id in ["1", "2"]: 
            # get helmet center
            target_df = target_df.with_column((pl.col(f"{view[0]}_left_{p_id}") + (pl.col(f"{view[0]}_width_{p_id}")//2)).alias(f"{view[0]}_Wcenter_{p_id}"))
            target_df = target_df.with_column((pl.col(f"{view[0]}_top_{p_id}") + (pl.col(f"{view[0]}_height_{p_id}")//2)).alias(f"{view[0]}_Hcenter_{p_id}"))
        # helmet center distance
        target_df = target_df.with_column((np.sqrt(np.square(pl.col(f"{view[0]}_Wcenter_1") - pl.col(f"{view[0]}_Wcenter_2")) \
                                                  + np.square(pl.col(f"{view[0]}_Hcenter_1") - pl.col(f"{view[0]}_Hcenter_2")))
                                          ).alias(f"{view[0]}_helmet_dis"))

        # GがNanになるので0にしておく
        target_df = target_df.with_column(pl.col(f"{view[0]}_helmet_dis").fill_null(0))
        FEATURE_COLS_.append(f"{view[0]}_helmet_dis")
    
    # helmet cols fillna(0) after get helmet distance 
    helmet_cols = ['E_left_1', 'E_width_1', 'E_top_1', 'E_height_1',
                   'E_left_2', 'E_width_2', 'E_top_2', 'E_height_2', 
                   'S_left_1','S_width_1', 'S_top_1', 'S_height_1', 
                   'S_left_2', 'S_width_2', 'S_top_2', 'S_height_2']
    target_df = target_df.with_column(pl.col(helmet_cols).fill_null(0))

    return target_df, FEATURE_COLS_

def get_categorical_ftr(target_df, FEATURE_COLS_):
    target_df = target_df.with_column((pl.col("nfl_player_id_2")== "G").cast(int).alias("is_ground"))
#     target_df["nfl_player_id_2"] = target_df["nfl_player_id_2"].replace("G", "99999").astype(np.int64) # when inference this is after cnn pred
    target_df = target_df.with_column((1 - ((pl.col("E_width_1")==0) & (pl.col("E_width_2")==0) \
                                          & (pl.col("S_width_1")==0) & (pl.col("S_width_2")==0)).cast(int)).alias("is_helmet"))
    target_df = target_df.with_column((1 - ((pl.col("E_width_1")==0) & (pl.col("E_width_2")==0)).cast(int)).alias("is_E_helmet"))
    target_df = target_df.with_column((1 - ((pl.col("S_width_1")==0) & (pl.col("S_width_2")==0)).cast(int)).alias("is_S_helmet"))
    target_df = target_df.with_column(((pl.col("is_E_helmet")==1) & (pl.col("is_S_helmet")==1)).cast(int).alias("both_helmet"))
    # set team 
    target_df = target_df.with_column(((pl.col("team_1")=="home").cast(int)).alias("team_1"))
    target_df = target_df.with_column(((pl.col("team_2")=="home").cast(int)).alias("team_2"))
    
    target_df = target_df.fill_null(0)
    target_df = target_df.fill_nan(0)
    FEATURE_COLS_.extend(["is_ground", "is_helmet"])
    return target_df, FEATURE_COLS_

# -------------------------------
# rolling function
# -------------------------------
def create_roll_ftr(target_df, FEATURE_COLS_,
                    key_cols = ["contact_id", "game_play", "nfl_player_id_1", 
                                "nfl_player_id_2", "datetime", "step"]):
    roll_df = target_df.select(roll_cols+key_cols)

    roll_df = roll_df.with_column(pl.concat_str([pl.col("game_play"),
                                                 pl.col("nfl_player_id_1"),
                                                 pl.col("nfl_player_id_2"),
                                                ], sep='_').alias('key'))
    # cast datetime
    roll_df = roll_df.with_column(pl.col('datetime').str.strptime(pl.Datetime,
                                                                          fmt="%+",
                                                                          strict=False
                                                                         ).alias('datetime'))
    # groupby rolling
    roll_df = roll_df.groupby_dynamic("step", every="1i", period="6i", by="key", closed="both").agg([pl.col(roll_cols).sum().suffix("_rollsum"), pl.col("contact_id")])
    roll_df = roll_df.with_column(pl.col("contact_id").apply(lambda x:x[0]))
    roll_df = roll_df.drop(["key", "step"])
    target_df = target_df.join(roll_df, on="contact_id", how="left")
    additional_cols = [col+"_rollsum" for col in roll_cols]
    FEATURE_COLS_.extend(additional_cols)
    return target_df, FEATURE_COLS_