# @title strain ctrain test
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
scaler = torch.GradScaler()

# name = 'violet'
# train_summary_writer = tf.summary.create_file_writer('logs/'+name+'/train')
# test_summary_writer = tf.summary.create_file_writer('logs/'+name+'/test')

def strain(model, dataloader, optim, scheduler=None):
    model.train()
    for i, (x, _) in enumerate(dataloader):
        x = x.to(device).to(torch.bfloat16)
        with torch.autocast(device_type=device, dtype=torch.bfloat16): # bfloat16 float16
            loss = model.loss(x)

            # repr_loss, std_loss, cov_loss = model.loss(x)
            # loss = model.sim_coeff * repr_loss + model.std_coeff * std_loss + model.cov_coeff * cov_loss

        optim.zero_grad()
        scaler.scale(loss).backward()

        # total_norm = 0
        # for p in model.parameters(): total_norm += p.grad.data.norm(2).item() ** 2
        # total_norm = total_norm**.5
        # print('total_norm', total_norm)
        # for p in list(filter(lambda p: p.grad is not None, model.parameters())):
        #     print(p.grad.data.norm(2).item())
        # print("max grad norm", max([p.grad.data.norm(2).item() for p in list(filter(lambda p: p.grad is not None, model.parameters()))]))
        # scaler.unscale_(optim)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5) # 0.5

        scaler.step(optim)
        scaler.update()

        # with torch.no_grad():
        #     m=0.999 # 0.99 m = next(momentum_scheduler)
        #     for param_q, param_k in zip(model.student.parameters(), model.teacher.parameters()):
        #         param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

        if scheduler is not None: scheduler.step()
        # print("strain",loss.item())
        # for param in seq_jepa.context_encoder.cls: print(param.data)
        # for param in seq_jepa.predicter.cls: print(param.data)
        try: wandb.log({"loss": loss.item(), "repr/I": repr_loss.item(), "std/V": std_loss.item(), "cov/C": cov_loss.item()})
        # try: wandb.log({"loss": loss.item()})
        except NameError: pass
        # with train_summary_writer.as_default(): tf.summary.scalar('strain', loss.item(), step=i)
        if i>=500: break


def ctrain(model, classifier, dataloader, coptim, scheduler=None): # train function with automatic mixed precision
    model.eval()
    classifier.train()
    for i, (x, y) in enumerate(dataloader):
        # x, y = x.to(device), y.to(device) # [batch, ]
        x, y = x.to(device).to(torch.bfloat16), y.to(device) # [batch, ] # (id, activity)
        with torch.autocast(device_type=device, dtype=torch.bfloat16): # bfloat16 float16
            with torch.no_grad():
                sx = model(x).detach()
            y_ = classifier(sx)
            loss = F.cross_entropy(y_, y)
        coptim.zero_grad()
        scaler.scale(loss).backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5) # .5
        scaler.step(coptim)
        scaler.update()
        # print("classify",loss.item())
        try: wandb.log({"closs": loss.item()})
        except NameError: pass
        # with test_summary_writer.as_default(): tf.summary.scalar('closs', loss.item(), step=i)
        if i>=100: break


def test(model, classifier, dataloader):
    model.eval()
    classifier.eval()
    correct = 0
    for i, (x, y) in enumerate(dataloader):
        # x, y = x.to(device), y.to(device) # [batch, ]
        x, y = x.to(device).to(torch.float), y.to(device) # [batch, ] # (id, activity)
        with torch.no_grad():
            sx = model(x)
            try:
                rankme = RankMe(sx).item()
                lidar = LiDAR(sx).item()
            except NameError: pass
            y_ = classifier(sx)
        test_loss = F.cross_entropy(y_, y)
        correct += (y==y_.argmax(dim=1)).sum().item()
        if i>=100: break
    # print(correct/len(y))
    print('acc', round(correct/len(dataloader.dataset), 3), 'test_loss', round(test_loss.item()/len(y), 3))
    try: wandb.log({"correct": correct/len(dataloader.dataset)})
    # try: wandb.log({"correct": correct/len(y), "rankme": rankme, "lidar": lidar})
    except NameError: pass

# scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=400, num_training_steps=2000) # https://docs.pytorch.org/torchtune/0.2/generated/torchtune.modules.get_cosine_schedule_with_warmup.html
# for i in range(2000): #
for i in range(10000): # 5000
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # np.random.shuffle(train_indices); np.random.shuffle(val_indices) # for wisdm
    # train_sampler, valid_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(val_indices)
    # # batch_size = 64 #512
    # train_loader = DataLoader(train_data, sampler=train_sampler, pin_memory=True, batch_size=batch_size, num_workers=2, drop_last=True) # num_workers = 4
    # test_loader = DataLoader(train_data, sampler=valid_sampler, pin_memory=True, batch_size=batch_size, num_workers=2, drop_last=True)

    # strain(seq_jepa, train_loader, optim)
    # # strain(seq_jepa, train_loader, optim, scheduler)
    # ctrain(seq_jepa, classifier, train_loader, coptim)
    # test(seq_jepa, classifier, test_loader)

    strain(mae, train_loader, optim)
    ctrain(mae, classifier, train_loader, coptim)
    test(mae, classifier, test_loader)

    # scheduler.step()
    # print('lr', i, optim.param_groups[0]["lr"])
    # strain(violet, train_loader, voptim)
    # ctrain(violet, classifier, train_loader, coptim)
    # test(violet, classifier, test_loader)
