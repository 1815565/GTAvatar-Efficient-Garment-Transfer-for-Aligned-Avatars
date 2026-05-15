import argparse
from config import cfg
import torch
from base import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=str, dest='subject_id')
    parser.add_argument('--fit_pose_to_test', dest='fit_pose_to_test', action='store_true')
    parser.add_argument('--continue', dest='continue_train', action='store_true')

    args = parser.parse_args()
    assert args.subject_id, "Please set subject ID"
    return args

def move_dict_to_device(data, device):
    for key1 in data:
        if isinstance(data[key1], torch.Tensor):
            data[key1] = data[key1].to(device)
        if isinstance(data[key1], dict):
            for key2 in data[key1]:
                if isinstance(data[key1][key2], torch.Tensor):
                    data[key1][key2] =  data[key1][key2].to(device)
    return data


def main():
    args = parse_args()
    cfg.set_args(args.subject_id, args.fit_pose_to_test, args.continue_train)

    trainer = Trainer()
    trainer._make_batch_generator()
    trainer._make_model()
    

    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        trainer.tot_timer.tic()
        trainer.read_timer.tic()

        # if epoch < cfg.end_epoch -1 :
        for itr, data in enumerate(trainer.batch_generator):
            # torch.cuda.empty_cache()
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()
            
            # set stage
            cur_itr = epoch * len(trainer.batch_generator) + itr
            cfg.set_stage(cur_itr)

            # set learning rate
            tot_itr = cfg.end_epoch * len(trainer.batch_generator)
            trainer.set_lr(cur_itr, tot_itr)

            # forward
            trainer.optimizer.zero_grad()
            loss = trainer.model(data, 'train', epoch)
            loss = {k:loss[k].mean() for k in loss}

            # backward
            sum(loss[k] for k in loss).backward()

            # update
            trainer.optimizer.step()
            
            # log
            trainer.gpu_timer.toc()
            screen = [
                'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                'speed: %.2f(%.2fs r%.2f)s/itr' % (
                    trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
                '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                ]
            screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k,v in loss.items()]
            trainer.logger.info(' '.join(screen))
            print(cfg.model_dir)

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()
            cur_itr += 1

        # save model
        trainer.save_model({
            'epoch': epoch,
            'network': trainer.model.module.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
        }, epoch)


        # else:
        #     trainer.add_model()
        #     for itr, data in enumerate(trainer.batch_generator):
        #         data = move_dict_to_device(data, "cuda:0")
        #         # torch.cuda.empty_cache()
        #         trainer.read_timer.toc()
        #         trainer.gpu_timer.tic()

        #         trainer.model2.gaussians.optimizer.zero_grad(set_to_none = True)

        #         loss = trainer.model2(data, 'train')
        #         loss = {k:loss[k].mean() for k in loss}
        #         sum(loss[k] for k in loss).backward()

        #         trainer.model2.gaussians.optimizer.step()

        #         trainer.gpu_timer.toc()
        #         screen = [
        #             'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
        #             'speed: %.2f(%.2fs r%.2f)s/itr' % (
        #                 trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
        #             '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
        #             ]
        #         screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k,v in loss.items()]
        #         trainer.logger.info(' '.join(screen))

        #         trainer.tot_timer.toc()
        #         trainer.tot_timer.tic()
        #         trainer.read_timer.tic()
                

        #     # save model
        #     trainer.save_model({
        #         'epoch': epoch,
        #         'network': trainer.model2.state_dict(),
        #         'optimizer': trainer.optimizer.state_dict(),
        #     }, epoch)
   
if __name__ == "__main__":
    main()
