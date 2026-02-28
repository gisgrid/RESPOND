#!/bin/bash

# uncomment the following line to preprocess highD data if you want to run the preprocessing step
# python -m respond.highd.preprocessing_highD --data_path=<path_to_highD_data> --output_directory=../preprocessed_highD

preprocessing_dir="../preprocessed_highD"

python -m respond.highd.filter_dangerous_turns --preprocessing_dir=${preprocessing_dir}
python -m respond.highd.llm_decision --preprocessing_dir=${preprocessing_dir} --filename=dangerous_turns_ttc_4.0_cont_0.04.csv --output_file=llm_dec_ttc_4.0.csv
python -m respond.highd.respond_highd_exp_report --preprocessing_dir=${preprocessing_dir} --filename=llm_dec_ttc_4.0.csv --output_file=RESPOND_highD_experiment_report_ttc_4.0.csv

# uncomment the following line to draw car frame animations
# python -m respond.highd.draw_car_frame --data_path=${preprocessing_dir} --filename=dangerous_turns_ttc_4.0_cont_0.04.csv --output_path=<output_directory_for_frames>
