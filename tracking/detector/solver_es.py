"""DetSolver con early stopping sobre el AP (coco_eval_bbox[0]) del val. No modifica el código original de RT-DETRv2:
subclase de DetSolver cuyo fit() es el original más el criterio de parada. Config (clave opcional en el YAML):

early_stopping:
  patience: 5        # épocas seguidas sin mejora > min_delta antes de parar
  min_delta: 0.001   # mejora mínima de AP (escala 0-1)
  min_epochs: 6      # no parar antes de completar estas épocas

best.pth se guarda en cada nuevo máximo de AP. El estado (mejor AP, época, contador) se guarda en early_stop.json
y se recupera al reanudar con --resume.
"""
import datetime, json, time
import torch

from src.misc import dist_utils
from src.solver.det_solver import DetSolver
from src.solver.det_engine import train_one_epoch, evaluate


class DetSolverES(DetSolver):

    def fit(self):
        print("Start training (con early stopping)")
        self.train()
        args = self.cfg
        es = dict(patience=5, min_delta=0.001, min_epochs=6)
        es.update(self.cfg.yaml_cfg.get('early_stopping', {}) or {})
        print(f'early stopping: {es}')

        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        state_path = self.output_dir / 'early_stop.json' if self.output_dir else None
        st = dict(best_ap=-1.0, best_epoch=-1, wait=0)
        start_epoch = self.last_epoch + 1
        if start_epoch > 0 and state_path is not None and state_path.exists():
            st.update(json.load(open(state_path)))
            print(f'estado de early stopping recuperado: {st}')

        start_time = time.time()
        for epoch in range(start_epoch, args.epoches):
            self.train_dataloader.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                max_norm=args.clip_max_norm, print_freq=args.print_freq, ema=self.ema, scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler, writer=self.writer)

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir:
                paths = [self.output_dir / 'last.pth']
                if (epoch + 1) % args.checkpoint_freq == 0:
                    paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for p in paths:
                    dist_utils.save_on_master(self.state_dict(), p)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                                                  self.val_dataloader, self.evaluator, self.device)

            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{i}', v, epoch)

            ap = float(test_stats['coco_eval_bbox'][0])
            if ap > st['best_ap']:
                if self.output_dir:
                    dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best.pth')
                improved = ap > st['best_ap'] + es['min_delta']
                st['best_ap'], st['best_epoch'] = ap, epoch
            else:
                improved = False
            st['wait'] = 0 if improved else st['wait'] + 1
            print(f'epoch {epoch}: AP={ap:.4f}  mejor={st["best_ap"]:.4f} (época {st["best_epoch"]})  sin mejora={st["wait"]}/{es["patience"]}')

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch, 'n_parameters': n_parameters, 'early_stop': dict(st)}
            if self.output_dir and dist_utils.is_main_process():
                json.dump(st, open(state_path, 'w'))
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                if coco_evaluator is not None and "bbox" in coco_evaluator.coco_eval:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    torch.save(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval" / "latest.pth")

            if st['wait'] >= es['patience'] and (epoch + 1) >= es['min_epochs']:
                print(f'Early stopping en la época {epoch}: sin mejora de AP > {es["min_delta"]} en {es["patience"]} épocas. '
                      f'Mejor AP={st["best_ap"]:.4f} en la época {st["best_epoch"]} (best.pth).')
                break

        print('Training time {}'.format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))
