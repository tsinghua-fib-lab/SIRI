import numpy as np

def load_config(cfg_file):
    f = open(cfg_file, 'r')
    config_lines = f.readlines()
    cfgs = { }
    for line in config_lines:
        ps = [p.strip() for p in line.split('=')]
        if (len(ps) != 2):
            continue
        try:
            if (ps[1].find('.') == -1):
                cfgs[ps[0]] = int(ps[1])
            else:
                cfgs[ps[0]] = float(ps[1])
        except ValueError:
            cfgs[ps[0]] = ps[1]
            if cfgs[ps[0]] == 'False':
                cfgs[ps[0]] = False
            elif cfgs[ps[0]] == 'True':
                cfgs[ps[0]] = True
    return cfgs

def load_data(data_file):
    if (data_file[-3:] == 'npz'):
        file = np.load(data_file)
        x, t, y = file['x'], file['t'], file['y']
    elif (data_file[-3:] == 'txt'):
        f = open(data_file, 'r')
        data_lines = f.readlines()
        head = data_lines[0].split()
        n = (int)(head[0])
        n_t = (int)(head[1])
        n_x = (int)(head[2])
        x = np.zeros([n, n_x], dtype = np.float32)
        t = np.zeros([n, n_t], dtype = np.int32)
        y = np.zeros([n], dtype = np.float32)
        for i in range(1, n + 1):
            line = data_lines[i].split()
            for j in range(n_x):
                x[i - 1][j] = (float)(line[j])
            for j in range(n_x, len(line) - 1):
                t[i - 1][(int)(line[j])] = 1
            y[i - 1] = (float)(line[-1])
    return x, t, y