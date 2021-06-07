def log_text(file_path, log):
    if not log.endswith('\n'):
        log += '\n'

    print(log)
    with open(file_path, 'a') as f:
        f.write(log)


def log_args(file_path, args):
    log = f"Args: {args}\n"
    log_text(file_path, log)


def log_train_epoch(file_path, epoch, train_loss, train_accuracy):
    log = f"epoch: {epoch}, Train loss: {train_loss}, Train accuracy: {train_accuracy}\n"
    log_text(file_path, log)


def log_val_epoch(file_path, epoch, val_loss, val_acc):
    log = f"epoch: {epoch}, Val loss: {val_loss}, Val accuracy: {val_acc}\n"
    log_text(file_path, log)


def log_test_metrics(file_path, precision, recall, f1, accuracy, cm):
    log = (f"Precision: {precision}\n"
           f"Recall: {recall}\n"
           f"F1 score: {f1}\n"
           f"Accuracy: {accuracy}\n"
           f"Confusion Matrix:\n{cm}\n")
    log_text(file_path, log)


def log_target_test_metrics(file_path, target, precision, recall, f1):
    log = (f"{target}:\n"
           f"\tPrecision: {round(precision, 4)}\n"
           f"\tRecall: {round(recall, 4)}\n"
           f"\tF1 score: {round(f1, 4)}\n")
    log_text(file_path, log)
