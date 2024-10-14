
# MAC
python main.py \
 --model_name gemini_1.5_flash \
 --stage inital \
 --times 1 \
 --num_doctors 4 \
 --n_round 10

# MAC with Specialist
python main_ws.py \
 --model_name gemini_1.5_flash \
 --stage inital \
 --times 1 \
 --num_specialists 4 \
 --n_round 10


 # MAC without Supervisor
python main_wo_supr.py \
 --model_name gemini_1.5_flash \
 --stage inital \
 --times 1 \
 --num_doctors 4 \
 --n_round 10