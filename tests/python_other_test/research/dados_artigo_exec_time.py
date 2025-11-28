import pandas as pd

def read_scenario(data) -> list[float]:    
    scenarios = {}    
    for i in range(data['name'].count()):
        scenario = data['name'][i]
        coarse = data['coarse'][i] == 'coarse'
        
        if scenario not in scenarios:
            scenarios[scenario] = [0.0, 0.0, 0, 0]
        
        if coarse:
            scenarios[scenario][0] += float(data['proc_time_ms'][i])
            scenarios[scenario][2] += 1
        else:
            scenarios[scenario][1] += float(data['proc_time_ms'][i])
            scenarios[scenario][3] += 1
            
  
    for s in data['name'].unique():  
        scenarios[s][0] /= (scenarios[s][2] + 1)
        scenarios[s][1] /= (scenarios[s][3] + 1)
        
        #print (f"mean: {s}: {scenarios[s][0]} coarse, {scenarios[s][1]} optim")
    return scenarios
        
            
            
            
def read_results_file() -> dict:
       
    data = {
        "rrt_star": pd.read_csv(f"../results/kinematic/results_cpu_experiment1.csv", delimiter=';', skipinitialspace=True),
        "rrt": pd.read_csv(f"../results/kinematic/results_rrt_cpu.csv", delimiter=';', skipinitialspace=True),
        "fast_rrt": pd.read_csv(f"../results/kinematic/results_gpu_experiment1.csv", delimiter=';', skipinitialspace=True),
    }

    delimiter = ','
    
    with open(f"timing.csv", "w") as f:   
        for s in data.keys():
            d = read_scenario(data[s])
            f.write(f"name{delimiter}scenario{delimiter}exec_time_ms_coarse{delimiter}exec_time_ms_optim{delimiter}total\n")
            for scenario in d.keys():
                l = d[scenario][0] + d[scenario][1]
                f.write(f"{s}{delimiter}{scenario}{delimiter}{d[scenario][0]:.2f}{delimiter}{d[scenario][1]:.2f}{delimiter}{l:.2f}\n")

       
def read_results_jetson() -> dict:
       
    data = {
        "rrt_star": pd.read_csv(f"../results/kinematic/results_jetson_cpu_experiment1.csv", delimiter=';', skipinitialspace=True),
        "rrt": pd.read_csv(f"../results/kinematic/results_rrt_jetson.csv", delimiter=';', skipinitialspace=True),
        "fast_rrt": pd.read_csv(f"../results/kinematic/results_jetson_gpu_experiment1.csv", delimiter=';', skipinitialspace=True),
    }

    delimiter = ','
    
    with open(f"timing.csv", "w") as f:   
        for s in data.keys():
            d = read_scenario(data[s])
            f.write(f"name{delimiter}scenario{delimiter}exec_time_ms_coarse{delimiter}exec_time_ms_optim{delimiter}total\n")
            for scenario in d.keys():
                l = d[scenario][0] + d[scenario][1]
                f.write(f"{s}{delimiter}{scenario}{delimiter}{d[scenario][0]:.2f}{delimiter}{d[scenario][1]:.2f}{delimiter}{l:.2f}\n")


read_results_file()
#https://www.youtube.com/watch?v=4G5BWRRcSV4
#https://www.youtube.com/watch?v=nvUIukUIOuI
#for i in {1..4}; do python3 ./test_benchmark_cpu.py; done