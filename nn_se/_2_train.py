import tensorflow as tf
import os
import numpy as numpy
import time
import collections
from pathlib import Path


from .models import model_builder
from .models import real_mask_model
from .models import modules
from .dataloader import dataloader
from .utils import misc_utils
from .FLAGS import PARAM


def __relative_impr(prev_, new_, declining=False):
  if declining:
    return (prev_-new_)/(abs(prev_)+abs(new_)+1e-8)
  return (new_-prev_)/(abs(prev_)+abs(new_)+1e-8)


class TrainOutputs(
    collections.namedtuple("TrainOutputs",
                           ("avg_loss", "cost_time", "lr"))):
  pass


def train_one_epoch(sess, train_model, train_log_file):
  tr_loss, i, lr = 0.0, 0, -1
  s_time = time.time()
  minbatch_time = time.time()

  while True:
    try:
      (_, loss, lr, global_step
       ) = sess.run([train_model.train_op,
                     train_model.loss,
                     train_model.lr,
                     train_model.global_step,
                     ])
      tr_loss += loss
      i += 1
      if i % PARAM.batches_to_logging == 0:
        msg = "     Minbatch %04d: loss:%.4f, lr:%.2e, Cost time:%ds.\n" % (
                i, tr_loss/i, lr, time.time()-minbatch_time,
              )
        minbatch_time = time.time()
        misc_utils.print_log(msg, train_log_file)
    except tf.errors.OutOfRangeError:
      break
  e_time = time.time()
  tr_loss /= i
  return TrainOutputs(avg_loss=tr_loss,
                      cost_time=e_time-s_time,
                      lr=lr)


class EvalOutputs(
    collections.namedtuple("EvalOutputs",
                           ("avg_loss", "cost_time"))):
  pass


def eval_one_epoch(sess, val_model):
  val_s_time = time.time()
  total_loss = 0.0
  i = 0
  while True:
    try:
      (loss,
       mag_mse, spec_mse, wavL1, wavL2,
       ) = sess.run([val_model.loss,
                     val_model.mag_mse, val_model.spec_mse, val_model.clean_wav_L1_loss, val_model.clean_wav_L2_loss
                     ])
      # print(loss, mag_mse, spec_mse, wavL1, wavL2, flush=True)
      total_loss += loss
      i += 1
    except tf.errors.OutOfRangeError:
      break

  avg_loss = total_loss / i
  val_e_time = time.time()
  return EvalOutputs(avg_loss=avg_loss,
                     cost_time=val_e_time-val_s_time)


def main():
  config_dir = Path(PARAM.root_dir).joinpath('exp', PARAM().config_name())
  train_log_file = config_dir.joinpath('log', 'train.log')
  ckpt_dir = config_dir.joinpath('ckpt')

  if not train_log_file.parent.exists():
    os.makedirs(str(train_log_file.parent))
  g = tf.Graph()
  with g.as_default():
    with tf.name_scope("inputs"):
      train_inputs = dataloader.get_batch_inputs_from_dataset(PARAM.train_name)
      val_inputs = dataloader.get_batch_inputs_from_dataset(PARAM.validation_name)

    ModelC = model_builder.get_model_class()

    variables = modules.Variables()
    train_model = ModelC(variables, train_inputs.clean, train_inputs.noise, train_inputs.mixed, PARAM.MODEL_TRAIN_KEY)
    # tf.compat.v1.get_variable_scope().reuse_variables()
    val_model = ModelC(variables,val_inputs.clean, val_inputs.noise, val_inputs.mixed, PARAM.MODEL_VALIDATE_KEY)
    init = tf.group(tf.compat.v1.global_variables_initializer(),
                    tf.compat.v1.local_variables_initializer())
    misc_utils.show_variables(train_model.save_variables)
    # misc_utils.show_variables(val_model.save_variables)
  g.finalize()

  config = tf.compat.v1.ConfigProto()
  # config.gpu_options.allow_growth = PARAM.GPU_RAM_ALLOW_GROWTH
  config.gpu_options.per_process_gpu_memory_fraction = 0.45
  config.allow_soft_placement = False
  sess = tf.compat.v1.Session(config=config, graph=g)
  sess.run(init)

  # region validation before training
  sess.run(val_inputs.initializer)
  evalOutputs_prev = eval_one_epoch(sess, val_model)
  misc_utils.print_log("\n\n", train_log_file)
  val_msg = "PRERUN.val> AVG.LOSS:%.4F, Cost itme:%.4Fs.\n" % (evalOutputs_prev.avg_loss,
                                                               evalOutputs_prev.cost_time)
  misc_utils.print_log(val_msg, train_log_file)

  assert PARAM.s_epoch > 0, 'start epoch > 0 is required.'
  model_abandon_time = 0
  for epoch in range(PARAM.s_epoch, PARAM.max_epoch+1):
    misc_utils.print_log("\n\n", train_log_file)
    misc_utils.print_log("  Epoch %03d:\n" % epoch, train_log_file)

    # train
    sess.run(train_inputs.initializer)
    trainOutputs = train_one_epoch(sess, train_model, train_log_file)
    misc_utils.print_log("     Train     > loss:%.4f, Cost time:%ds.\n" % (
        trainOutputs.avg_loss,
        trainOutputs.cost_time),
        train_log_file)

    # validation
    sess.run(val_inputs.initializer)
    evalOutputs = eval_one_epoch(sess, val_model)
    val_loss_rel_impr = __relative_impr(evalOutputs_prev.avg_loss, evalOutputs.avg_loss, True)
    misc_utils.print_log("     Validation> loss:%.4f, Cost time:%ds.\n" % (
        evalOutputs.avg_loss,
        evalOutputs.cost_time),
        train_log_file)

    # save or abandon ckpt
    ckpt_name = PARAM().config_name()+('_iter%04d_trloss%.4f_valloss%.4f_lr%.2e_duration%ds' % (
        epoch, trainOutputs.avg_loss, evalOutputs.avg_loss, trainOutputs.lr,
        trainOutputs.cost_time+evalOutputs.cost_time))
    if val_loss_rel_impr > 0:
      train_model.saver.save(sess, str(ckpt_dir.joinpath(ckpt_name)))
      evalOutputs_prev = evalOutputs
      best_ckpt_name = ckpt_name
      msg = "     ckpt(%s) saved.\n" % ckpt_name
    else:
      model_abandon_time += 1
      # tf.compat.v1.logging.set_verbosity(tf.logging.WARN)
      train_model.saver.restore(sess,
                                str(ckpt_dir.joinpath(best_ckpt_name)))
      # tf.compat.v1.logging.set_verbosity(tf.logging.INFO)
      msg = "     ckpt(%s) abandoned.\n" % ckpt_name
    misc_utils.print_log(msg, train_log_file)

    # start lr halving
    if val_loss_rel_impr < PARAM.start_halving_impr and (not PARAM.use_lr_warmup):
      new_lr = trainOutputs.lr * PARAM.lr_halving_rate
      train_model.change_lr(sess, new_lr)

    # stop criterion
    if epoch >= PARAM.max_epoch or model_abandon_time >= PARAM.max_model_abandon_time:
      msg = "finished, too small learning rate %e.\n" % trainOutputs.lr
      tf.logging.info(msg)
      misc_utils.print_log(msg, train_log_file)
      break

  sess.close()
  msg = '################### Training Done. ###################\n'
  misc_utils.print_log(msg, train_log_file)


if __name__ == "__main__":
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  main()
