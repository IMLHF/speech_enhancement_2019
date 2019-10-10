import os
# os.environ["OMP_NUM_THREADS"] = "1"
import tensorflow as tf
import collections
from pathlib import Path
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
import sys

from .utils import misc_utils
from .utils import audio
from .inference import build_SMG
from .inference import enhance_one_wav
from .inference import SMG
from .utils.assess.core import calc_pesq, calc_stoi, calc_sdr
from .FLAGS import PARAM

test_processor = 6
smg = None


class TestsetEvalAns(
    collections.namedtuple("TestsetEvalAns",
                           ("pesq_noisy_mean", "pesq_noisy_var",
                            "stoi_noisy_mean", "stoi_noisy_var",
                            "sdr_noisy_mean", "sdr_noisy_var",
                            "pesq_enhanced_mean", "pesq_enhanced_var",
                            "stoi_enhanced_mean", "stoi_enhanced_var",
                            "sdr_enhanced_mean", "sdr_enhanced_var"))):
  pass


class RecordEvalAns(
    collections.namedtuple("TestsetEvalAns",
                           ("pesq_noisy",
                            "stoi_noisy",
                            "sdr_noisy",
                            "pesq_enhanced",
                            "stoi_enhanced",
                            "sdr_enhanced"))):
  pass


def eval_one_record(clean_dir_and_noise_dir, mix_snr, save_dir=None):
  """
  if save_dir is not None: save clean, noise and mixed in save_dir.
  """
  global smg
  if smg is None:
    smg = build_SMG()
  clean_dir, noise_dir = clean_dir_and_noise_dir
  assert Path(clean_dir).exists(), 'clean_dir not exist.'
  assert Path(noise_dir).exists(), 'noise_dir not exist.'
  clean_wav, c_sr = audio.read_audio(clean_dir)
  noise_wav, n_sr = audio.read_audio(noise_dir)
  assert c_sr == n_sr and c_sr == PARAM.sampling_rate, 'Sample_rate error.'
  assert len(clean_wav) > 0 and len(noise_wav) > 0, 'clean or noise length is 0.'

  len_clean = len(clean_wav)
  noise_wav = audio.repeat_to_len(noise_wav, len_clean, False)
  mixed_wav, w_clean, w_noise = audio.mix_wav_by_SNR(clean_wav, noise_wav, mix_snr)
  clean_wav = clean_wav * w_clean
  noise_wav = noise_wav * w_noise
  enhanced_wav = enhance_one_wav(smg, mixed_wav)

  clean_dir_name = Path(clean_dir).stem
  noise_dir_name = Path(noise_dir).stem
  if save_dir is not None:
    audio.write_audio(os.path.join(save_dir, clean_dir_name+'.wav'), clean_wav, PARAM.sampling_rate)
    audio.write_audio(os.path.join(save_dir, clean_dir_name+'_'+noise_dir_name+'.wav'), mixed_wav, PARAM.sampling_rate)
    audio.write_audio(os.path.join(save_dir, clean_dir_name+'_enhanced.wav'), enhanced_wav, PARAM.sampling_rate)

  pesq_noisy = calc_pesq(clean_wav, mixed_wav, PARAM.sampling_rate)
  stoi_noisy = calc_stoi(clean_wav, mixed_wav, PARAM.sampling_rate)
  sdr_noisy = calc_sdr(clean_wav, mixed_wav, PARAM.sampling_rate)
  pesq_enhanced = calc_pesq(clean_wav, enhanced_wav, PARAM.sampling_rate)
  stoi_enhanced = calc_stoi(clean_wav, enhanced_wav, PARAM.sampling_rate)
  sdr_enhanced = calc_sdr(clean_wav, enhanced_wav, PARAM.sampling_rate)

  return RecordEvalAns(pesq_noisy=pesq_noisy, stoi_noisy=stoi_noisy, sdr_noisy=sdr_noisy,
                       pesq_enhanced=pesq_enhanced, stoi_enhanced=stoi_enhanced, sdr_enhanced=sdr_enhanced)


def eval_testSet_by_list(clean_noise_pair_list, mix_snr, save_dir=None):
  """
  if save_dir is not None: save clean, noise and mixed in save_dir.
  """

  func = partial(eval_one_record, mix_snr=mix_snr, save_dir=save_dir)
  job = Pool(test_processor).imap(func, clean_noise_pair_list)
  eval_ans_list = list(tqdm(job, "Testing", len(clean_noise_pair_list), unit="test record", ncols=60))

  # eval_ans_list = []
  # # for clean_dir_and_noise_dir in clean_noise_pair_list:
  # for clean_dir_and_noise_dir in tqdm(clean_noise_pair_list, ncols=100, unit="test record"):
  #   eval_ans = eval_one_record(clean_dir_and_noise_dir, mix_snr, save_dir)
  #   eval_ans_list.append(eval_ans)
  #   # print(eval_ans)
  #   # print("________________________________________________________________________________________________________________")

  pesq_noisy_vec = np.array([eval_ans_.pesq_noisy for eval_ans_ in eval_ans_list], dtype=np.float32)
  stoi_noisy_vec = np.array([eval_ans_.stoi_noisy for eval_ans_ in eval_ans_list], dtype=np.float32)
  sdr_noisy_vec = np.array([eval_ans_.sdr_noisy for eval_ans_ in eval_ans_list], dtype=np.float32)
  pesq_enhanced_vec = np.array([eval_ans_.pesq_enhanced for eval_ans_ in eval_ans_list], dtype=np.float32)
  stoi_enhanced_vec = np.array([eval_ans_.stoi_enhanced for eval_ans_ in eval_ans_list], dtype=np.float32)
  sdr_enhanced_vec = np.array([eval_ans_.sdr_enhanced for eval_ans_ in eval_ans_list], dtype=np.float32)

  return TestsetEvalAns(pesq_noisy_mean=np.mean(pesq_noisy_vec), pesq_noisy_var=np.var(pesq_noisy_vec),
                        stoi_noisy_mean=np.mean(stoi_noisy_vec), stoi_noisy_var=np.var(stoi_noisy_vec),
                        sdr_noisy_mean=np.mean(sdr_noisy_vec), sdr_noisy_var=np.var(sdr_noisy_vec),
                        pesq_enhanced_mean=np.mean(pesq_enhanced_vec), pesq_enhanced_var=np.var(pesq_enhanced_vec),
                        stoi_enhanced_mean=np.mean(stoi_enhanced_vec), stoi_enhanced_var=np.var(stoi_enhanced_vec),
                        sdr_enhanced_mean=np.mean(sdr_enhanced_vec), sdr_enhanced_var=np.var(sdr_enhanced_vec))


def eval_testSet_by_meta(mix_SNR, save_test_records=False):
  """
  save_test_records: if save test answer (clean, mixed, enhanced)
  """

  set_root = misc_utils.datasets_dir().joinpath(PARAM.test_name) # "/xx/datasets/train"
  meta_dir = set_root.joinpath(PARAM.test_name+".meta")
  test_log_file = str(misc_utils.test_log_file_dir())
  misc_utils.print_log("Using meta '%s'\n" % str(meta_dir), test_log_file)

  metaf = meta_dir.open("r")
  meta_list = list(metaf.readlines())
  meta_list = [meta.strip().split("|") for meta in meta_list] # [ (clean, noise), ...]

  if not save_test_records:
    test_records_save_dir = None
  else:
    _dir = misc_utils.test_records_save_dir()
    test_records_save_dir = str(_dir)
    if _dir.exists():
      import shutil
      shutil.rmtree(str(_dir))
    _dir.mkdir()
  tmp = eval_testSet_by_list(meta_list, mix_SNR, test_records_save_dir)
  misc_utils.print_log(str(tmp)+"\n", test_log_file)

def main():
  # eval_testSet_by_meta(-5, True)
  eval_testSet_by_meta(0, True)
  # eval_testSet_by_meta(5, True)
  # eval_testSet_by_meta(10)
  # eval_testSet_by_meta(15)

if __name__ == "__main__":
  misc_utils.initial_run(sys.argv[0].split("/")[-2])

  if len(sys.argv) > 1:
    test_processor = int(sys.argv[1])
  main()
  """
  run cmd:
  `OMP_NUM_THREADS=1 python -m xx._3_eval_metrics 3`
  """
