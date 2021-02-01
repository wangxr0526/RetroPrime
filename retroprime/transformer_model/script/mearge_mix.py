import os

def read_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [x.strip() for x in f.readlines()]


def write_txt(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(line + '\n')


if __name__ == '__main__':

    save_mix_path = os.path.join('../experiments/results/USPTO-50K_S2R/predictions_USPTO-50K_S2R_model_step_100000.pt_beam20test_extract_5w_mix_top3_sec.txt')
    split_group = 5
    mix_data = []
    for group in range(split_group):
        # split_mix_path = os.path.join('../experiments/results/USPTO-50K_S2R/predictions_USPTO-50K_S2R_model_step_100000.pt_beam20test_extract_5w_mix_top3_sec_{}.txt'.format(group))
        split_mix_path = os.path.join('../experiments/results/USPTO-50K_S2R/predictions_USPTO-50K_S2R_model_step_100000.pt_beam20test_extract_5w_sec_mix_top3_{}.txt'.format(group))
        split_mix = read_txt(split_mix_path)
        mix_data += split_mix
    assert split_group * len(split_mix) == len(mix_data)
    write_txt(save_mix_path, mix_data)