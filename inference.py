from utils import infer_to_CSV
from util_tools import config_loader, init_seeds, params_count, get_msg_mgr

if __name__ == '__main__':
    model_path = r"result/gaitpart_iter20000_wd0.0_1e51"
    data_path = r"data/test_pkl"
    msg_mgr = get_msg_mgr()
    msg_mgr.init_manager(data_path, False, 100, 0)
    infer_to_CSV(model_path, data_path)