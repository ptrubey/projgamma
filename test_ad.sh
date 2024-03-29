#!/bin/bash

# nohup python test_generic.py ./datasets/ad_cardio_x.csv ./ad/cardio dppprg 50000 20000 30 --eta_rate 1e0 --decluster False > /dev/null 2>&1 &
# # nohup python test_generic.py ./datasets/ad_cover_x.csv ./ad/cover dppprg 50000 20000 30 --eta_rate 1e0  --decluster False > /dev/null 2>&1 &
# nohup python test_generic.py ./datasets/ad_mammography_x.csv ./ad/mammography dppprg 50000 20000 30 --eta_rate 1e0 --decluster False > /dev/null 2>&1 &
# nohup python test_generic.py ./datasets/ad_pima_x.csv ./ad/pima dppprg 50000 20000 30 --eta_rate 1e0 --decluster False > /dev/null 2>&1 &
# nohup python test_generic.py ./datasets/ad_satellite_x.csv ./ad/satellite dppprg 50000 20000 30 --eta_rate 1e0 --decluster False --quantile 0.97 > /dev/null 2>&1 &

# # nohup python test_generic_ad.py ./simulated_ad/ad_sim_m5_c5_x.csv ./simulated_ad/m5_c5 dphprg 50000 20000 30 > /dev/null 2>&1 &
# # nohup python test_generic_ad.py ./simulated_ad/ad_sim_m5_c10_x.csv ./simulated_ad/m5_c10 dphprg 50000 20000 30 > /dev/null 2>&1 &
# # nohup python test_generic_ad.py ./simulated_ad/ad_sim_m10_c5_x.csv ./simulated_ad/m10_c5 dphprg 50000 20000 30 > /dev/null 2>&1 &
# # nohup python test_generic_ad.py ./simulated_ad/ad_sim_m10_c10_x.csv ./simulated_ad/m10_c10 dphprg 50000 20000 30 > /dev/null 2>&1 &

# nohup python test_generic_ad.py ./simulated_ad/ad_sim_m5_c5_x.csv ./simulated_ad/m5_c5 dppprg 50000 20000 30 > /dev/null 2>&1 &
# nohup python test_generic_ad.py ./simulated_ad/ad_sim_m5_c10_x.csv ./simulated_ad/m5_c10 dppprg 50000 20000 30 > /dev/null 2>&1 &
# nohup python test_generic_ad.py ./simulated_ad/ad_sim_m10_c5_x.csv ./simulated_ad/m10_c5 dppprg 50000 20000 30 > /dev/null 2>&1 &
# nohup python test_generic_ad.py ./simulated_ad/ad_sim_m10_c10_x.csv ./simulated_ad/m10_c10 dppprg 50000 20000 30 > /dev/null 2>&1 &

nohup python test_generic.py ./simulated/lnad/test_x.csv ./simulated/lnad/results_mdppprgln.pkl --outcome ./simulated/lnad/test_y.csv --cats \[3,4\] --sphere True



# EOF
