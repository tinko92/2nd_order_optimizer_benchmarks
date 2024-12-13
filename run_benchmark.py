import torch
import time
from recorder import Recorder
from torchsummary import summary

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

def run_benchmark(model, OptimizerType, CriterionType, dataloader, epochs, seed=0, lr = 1e-3, **optimizer_args):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    optimizer = OptimizerType(model.parameters(), lr=lr, **optimizer_args)
    criterion = CriterionType()
    recorder = Recorder()
    run_id = recorder.run({'label': dataloader.label, 'description': ''}, {'label': model.label, 'description': str(model), 'trainable_params': sum(p.numel() for p in model.parameters())}, {'label': str(OptimizerType), 'lr': str(lr), 'all_parameters': str(optimizer_args)}, {'label': str(CriterionType)}, str(seed), str(epochs), str(dataloader.batch_size))
    min_loss = float('inf')
    max_ram = 0
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                if loss.requires_grad:
                    loss.backward()
                return loss
            loss = closure()
            running_loss += loss.item()
            optimizer.step(closure)
            if i % 10 == 9:    # print every 2000 mini-batches
                print(i, running_loss / 10)
                update_hessian = getattr(optimizer, "update_hessian", None)
                if callable(update_hessian):
                    optimizer.update_hessian()
                recorder.step(run_id, epoch, i, running_loss/10, time.time_ns() // 1_000_000)
                min_loss = min(min_loss, running_loss/10)
                running_loss = 0.0
            if device != 'cpu':
                free, total = torch.cuda.mem_get_info(device)
                max_ram = max(max_ram, total - free)
    recorder.min_loss(run_id, min_loss)
    recorder.max_ram(run_id, max_ram)
    recorder.commit()
