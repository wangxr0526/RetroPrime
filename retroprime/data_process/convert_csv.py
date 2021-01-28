import argparse

import pandas as pd


def read_data(path, dataset):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [x.strip() for x in f.readlines()]
    new_lines = []
    for i, line in enumerate(lines):
        if i != 0:
            if line != '':
                rxn = line.split(',')[-1]
                rts, pds = rxn.split('>>')[0], rxn.split('>>')[-1]
                if rts == pds:
                    continue
                new_lines.append(line + ',' + dataset)
            else:
                pass
        else:
            new_lines.append(line + ',dataset')
    return new_lines


def write_csv(ls, path):
    with open(path, 'w', encoding='utf-8') as f:
        for l in ls:
            f.write(l + '\n')


if __name__ == '__main__':
    opt = argparse.ArgumentParser()
    opt.add_argument('-data_path', default='../../databox/uspto_full')
    args, _ = opt.parse_known_args()
    data_path = args.data_path
    print('reading database...')
    all_data = []
    for i, s in enumerate(['train', 'val', 'test']):
        print(s)
        data = read_data('{}/single/raw_{}.csv'.format(data_path, s), s)
        if i == 0:
            all_data.extend(data)
        else:
            all_data.extend(data[1:])
        write_csv(data, '{}/single/new_raw_{}.csv'.format(data_path, s))
    write_csv(all_data, '{}/databox/data/single/new_raw_all.csv')
