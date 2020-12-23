def extract_log(log_file, new_log_file, key_word):
    f = open(log_file)
    train_log = open(new_log_file, 'w')
    for line in f:
    # 去除多gpu的同步log
        if 'Syncing' in line:
            continue
    # 去除除零错误的log
        if 'nan' in line:
            continue
        if key_word in line:
            train_log.write(line)

    f.close()
    train_log.close()


extract_log('train_log.txt', 'train_log_loss.txt', 'images')  # train_log.txt 用于绘制loss曲线
extract_log('train_log.txt', 'train_log_iou.txt', 'IOU')
