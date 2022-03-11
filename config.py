

stru_config={'n_inp':1,
        'struc':[[32,'sigmoid']],#'tanh','relu'
        'order_up':3,
        "order_down":2,
        "derive_order":4,#最高阶导数
        'var_name':'real'}


train_config={
        'CKPT':'ckpt',
        "BATCHSIZE":500,
        "MAX_ITER":5000,
        'STEP_EACH_ITER':1000,
        'STEP_SHOW':30,
        'EPOCH_SAVE':1,
        "LEARNING_RATE":0.00005,
}
