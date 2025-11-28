import pandas as pd

def build_res(data):
    res = {}
    for scenario in data['name'].unique():
        res[scenario] = [
            0.0,    # jerk
            0.0,    # total_length
            0,      # discontinuity
            0,      # goal_reached
            0       # timeout
        ]
    return res
def read_results_file(file: str, file_out: str) -> dict:
    data = pd.read_csv(file, delimiter=';', skipinitialspace=True)
    
    res_coarse = build_res(data)
    res_optim = build_res(data)
    
    
    lc = 1
    lop = 1
    coarse = False
    for i in range(data['name'].count()):
        scenario = data['name'][i]
        coarse = data['coarse'][i] == 'coarse'
        res = res_coarse if coarse else res_optim
        
        
        res[scenario][2] += 1 if data['tan_discontinuity'][i] == 'yes' else 0
        res[scenario][3] += 1 if data['goal_reached'][i] == 'yes' else 0
        res[scenario][4] += 1 if data['timeout'][i] == 'yes' else 0

        if data['timeout'][i] == 'yes':
            continue
        res[scenario][0] += data['jerk'][i]
        res[scenario][1] += data['total_length'][i]
        
        if coarse:
            lc += 1
        else:
            lop += 1
    
    for scenario in data['name'].unique():
        if lc > 1:
            res_coarse[scenario][0] /= (lc-1)
            res_coarse[scenario][1] /= (lc-1)
        if lop > 1:
            res_optim[scenario][0] /= (lop-1)
            res_optim[scenario][1] /= (lop-1)      
    
    delimiter = ','
    
    with open(f"coarse_{file_out}", "w") as f:
        f.write(f"name{delimiter}jerk{delimiter}total_length{delimiter}discontinuity{delimiter}goal_reached{delimiter}timeout\n")
        res = res_coarse
        for scenario in res.keys():
            f.write(f"{scenario}{delimiter}{res[scenario][0]}{delimiter}{res[scenario][1]}{delimiter}{res[scenario][2]}{delimiter}{res[scenario][3]}{delimiter}{res[scenario][4]}\n")

    with open(f"optim_{file_out}", "w") as f:
        f.write(f"name{delimiter}jerk{delimiter}total_length{delimiter}discontinuity{delimiter}goal_reached{delimiter}timeout\n")
        res = res_optim
        for scenario in res.keys():
            f.write(f"{scenario}{delimiter}{res[scenario][0]}{delimiter}{res[scenario][1]}{delimiter}{res[scenario][2]}{delimiter}{res[scenario][3]}{delimiter}{res[scenario][4]}\n")
        

    # grouped.to_csv("outp.csv", sep=';', index=True, header=True)
    # return grouped.to_dict(orient='records')

read_results_file("../results/kinematic/results_rrt_cpu.csv", "rrt.csv")
#https://www.youtube.com/watch?v=4G5BWRRcSV4
#https://www.youtube.com/watch?v=nvUIukUIOuI
#for i in {1..4}; do python3 ./test_benchmark_cpu.py; done