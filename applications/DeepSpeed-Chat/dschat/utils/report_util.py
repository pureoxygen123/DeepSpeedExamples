import time
import torch
REPORT_MAP = {}
CURRENT_MAP = {}
CURRENT_STEP = 0
CURRENT_START = None
CURRENT_END = None

def start_report(step):
    if(torch.distributed.get_rank()>0):
        return
    print(f'========== [step:{step}] start ==========')
    global CURRENT_STEP,CURRENT_START,CURRENT_MAP
    CURRENT_START = time.time()
    CURRENT_STEP = step
    CURRENT_MAP = {}
    pass

def stop_report():
    if(torch.distributed.get_rank()>0):
        return
    global CURRENT_STEP,CURRENT_START,CURRENT_END
    if CURRENT_START == None:
        return
    CURRENT_END = time.time()
    dt = CURRENT_END - CURRENT_START
    for k,v in  CURRENT_MAP.items():
        print(f'[step:{CURRENT_STEP}][stage:{k}][dt:{round(v, 2)}]')
    print(f'========== [step:{CURRENT_STEP}] end, E2E [dt:{round(dt, 2)}] ==========')
    REPORT_MAP[CURRENT_STEP] = CURRENT_MAP

def reportStatistic(key,dt):
    if(torch.distributed.get_rank()>0):
        return
    global CURRENT_MAP
    if key in CURRENT_MAP:
        old_dt = CURRENT_MAP[key]
        CURRENT_MAP[key] = old_dt + dt
    else:
        CURRENT_MAP[key] = dt