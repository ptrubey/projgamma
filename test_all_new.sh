#!/bin/bash

# nohup python test_generic.py ./datasets/ivt_updated_nov_mar.csv ./output2 dpd 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &
# nohup python test_generic.py ./datasets/ivt_updated_nov_mar.csv ./output2 dpgd 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &
# nohup python test_generic.py ./datasets/ivt_updated_nov_mar.csv ./output2 dppg 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &
# nohup python test_generic.py ./datasets/ivt_updated_nov_mar.csv ./output2 dpprg 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &
#
# nohup python test_generic.py ./datasets/ivt_updated_nov_mar.csv ./output2 dphpg 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &
# nohup python test_generic.py ./datasets/ivt_updated_nov_mar.csv ./output2 dphprg 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &
#
# nohup python test_generic.py ./datasets/ivt_updated_nov_mar.csv ./output2 md 50000 20000 30 --nMix 30 > /dev/null 2>&1 &
# nohup python test_generic.py ./datasets/ivt_updated_nov_mar.csv ./output2 mgd 50000 20000 30 --nMix 30 > /dev/null 2>&1 &
# nohup python test_generic.py ./datasets/ivt_updated_nov_mar.csv ./output2 mpg 50000 20000 30 --nMix 30 > /dev/null 2>&1 &
# nohup python test_generic.py ./datasets/ivt_updated_nov_mar.csv ./output2 mprg 50000 20000 30 --nMix 30 > /dev/null 2>&1 &
#
# nohup python test_generic.py ./datasets/ivt_updated_nov_mar.csv ./output2 mhpg 50000 20000 30 --nMix 30 > /dev/null 2>&1 &
# nohup python test_generic.py ./datasets/ivt_updated_nov_mar.csv ./output2 mhprg 50000 20000 30 --nMix 30 > /dev/null 2>&1 &
#
# nohup python test_generic.py ./datasets/ivt_updated_nov_mar.csv ./output2 vd 50000 20000 30 > /dev/null 2>&1 &
# nohup python test_generic.py ./datasets/ivt_updated_nov_mar.csv ./output2 vgd 50000 20000 30 > /dev/null 2>&1 &
# nohup python test_generic.py ./datasets/ivt_updated_nov_mar.csv ./output2 vpg 50000 20000 30 > /dev/null 2>&1 &
# nohup python test_generic.py ./datasets/ivt_updated_nov_mar.csv ./output2 vprg 50000 20000 30 > /dev/null 2>&1 &
#
# nohup python test_generic.py ./datasets/ivt_updated_nov_mar.csv ./output2 dppn 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &

nohup python test_generic.py ./datasets/ivt_updated_nov_mar.csv ./output2 dpppg 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &
nohup python test_generic.py ./datasets/ivt_updated_nov_mar.csv ./output2 dppprg 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &

nohup python test_generic.py ./datasets/ivt_updated_nov_mar.csv ./output2 dpppgln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &
nohup python test_generic.py ./datasets/ivt_updated_nov_mar.csv ./output2 dppprgln 50000 20000 30 --eta_shape 2 --eta_rate 1e-1 > /dev/null 2>&1 &


# EOF
