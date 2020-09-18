from tests import test_mnist, test_cifar10

if __name__ == '__main__':
    test_mnist()
    # test_cifar10()


n_epochs = 100
print_every = 200 if CUDA else 2
display_every = 2

for epoch in range(n_epochs):
    # train_acc = 0.
    time_start = time.time()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        masked_inputs = put_mask(inputs, 'gaussian')
        if CUDA:
            masked_inputs, inputs = masked_inputs.cuda(), inputs.cuda()
        optimizer.zero_grad()
        class_probs, recons = net(masked_inputs)
        # acc = torch.mean((labels == torch.max(class_probs, -1)[1]).double())
        # train_acc += acc.data.item()
        loss = (reconstruction_loss(recons, inputs))
        loss.backward()
        optimizer.step()
        if (i+1) % print_every == 0:
            print('[epoch {}/{}, batch {}] train_loss: {:.5f}'.format(epoch + 1, n_epochs, i + 1, loss.data.item()))
    test_acc = 0.
    for j, data in enumerate(testloader, 0):
        inputs, labels = data
        masked_inputs = put_mask(inputs, 'gaussian')
        if CUDA:
            masked_inputs, inputs = masked_inputs.cuda(), inputs.cuda()
        class_probs, recons = net(masked_inputs)
        if (j+1) % display_every == 0:
            display(inputs[0].cpu(), masked_inputs[0].cpu(), recons[0].cpu().detach())
        # acc = torch.mean((labels == torch.max(class_probs, -1)[1]).double())
        # test_acc += acc.data.item()
    print('[epoch {}/{} done in {:.2f}s]'.format(epoch + 1, n_epochs, (time.time() - time_start)))
