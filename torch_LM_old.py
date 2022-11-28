import torch
from torch.autograd.functional import hessian
import numpy as np
import matplotlib.pyplot as plt

class LM(object):

    def __init__(self, function, lambda0, max_iter = 100, epsilon4 = 1e-1, Lup = 11., Ldn = 9., L0 = 1e2):

        self.function = function
        self.lambda_history = [np.copy(lambda0.detach().numpy())]
        self.lambda_new = lambda0
        self.lambda_old = lambda0
        self.max_iter = max_iter
        self.epsilon4 = epsilon4
        self.Lup = Lup
        self.Ldn = Ldn
        self.L = L0
        self.loss_history = []
        self.L_history = [self.L]
        self.iteration = 0

        self.main_loop()
        
    def main_loop(self):

        loss = 1e8
        while self.iteration < self.max_iter and loss >= 1e-10 and torch.all(torch.isfinite(self.lambda_new)):
            print("---------iter---------")
            self.lambda_new.requires_grad = True
            self.lambda_new.grad = None
            loss = self.function(self.lambda_new, nocount = True)
            self.loss_history.append(loss.detach().cpu().item())
            loss.backward()
            # print("loss: ", loss.detach().cpu().item())
            hess = hessian(self.function, self.lambda_new)
            print(hess)
            # print("hess: ", hess.detach().cpu().numpy())
            grad = self.lambda_new.grad
            if np.max(np.abs(grad.detach().numpy())) < 1e-4:
                break
            # print("grad: ", grad.detach().cpu().numpy())
            self.L *= np.exp(np.random.normal(loc = 0, scale = 0.1))
            print("L: ", self.L)
            # for i in [0.0001, 0.001, 0.01,0.1,1,10,100, 1000, 10000, 100000]:
            #     print("test step: ", i, -torch.inverse(hess + i*torch.eye(len(grad)))@grad, -torch.linalg.solve(hess + i*torch.eye(len(grad)), grad))
                # print("test step np: ", i, -np.linalg.solve((hess + i*torch.eye(len(grad))).detach().numpy(), grad.detach().numpy()))
                # print("LM hess: ", hess + i*torch.eye(len(grad)), i*torch.eye(len(grad)))
            h = -torch.linalg.solve(hess + self.L*torch.abs(torch.diag(hess))*torch.eye(len(grad)), grad) #-torch.inverse(hess + self.L*torch.diag(hess))@grad #torch.linalg.solve(hess + self.L*torch.diag(hess), grad) # / (1 + self.L) #/ (self.L*torch.diag(hess))
            print("h: ", h) # , self.L*torch.diag(hess)*torch.eye(len(grad))
            self.lambda_new.requires_grad = False
            print("is maximal: ", torch.linalg.det(hess), torch.diag(hess))
            print("compare loss: ", loss.detach().item(), self.function(self.lambda_new + h, nocount = True).detach().item(), loss.detach().item() - self.function(self.lambda_new + h, nocount = True).detach().item(), torch.dot(h, self.L * (torch.diag(hess) * h) + grad).detach().item()) # (torch.diag(hess) * h)
            rho = (loss.detach().item() - self.function(self.lambda_new + h, nocount = True)) / abs(torch.dot(h, self.L * (torch.abs(torch.diag(hess)) * h) + grad).detach().item()) #(torch.diag(hess) * h)
            if 0 < (loss.detach().item() - self.function(self.lambda_new + h, nocount = True).detach().item()) < 1e-8:
                break
            #print("rho: ", rho, (loss.detach().item() - self.function(self.lambda_new + h)), abs(torch.dot(h, self.L * h + grad).detach().item())) # (torch.diag(hess) * h)
            if rho > self.epsilon4:
                print("accept")
                self.lambda_old = self.lambda_new
                self.lambda_new += h
                self.L = max(1e-9, self.L / self.Ldn)
                # print(self.lambda_new.detach().cpu().numpy())
                # print(h.detach().cpu().numpy())
            else:
                print("reject")
                self.L = min(1e9, self.L * self.Lup)
            self.L_history.append(self.L)
            self.lambda_history.append(np.copy(self.lambda_new.detach().cpu().numpy()))
            self.iteration += 1
        
if __name__ == "__main__":

    def y_hat(x, theta):
        return theta[0] * torch.exp(-x / theta[1]) + theta[2] * x * torch.exp(-x/theta[3]) 

    np.random.seed(10)
    theta_true = torch.tensor([20,10,1,50])
    X = torch.tensor(np.random.uniform(0,100,100))
    Y = torch.tensor(y_hat(X, theta_true).detach().numpy() + np.random.normal(loc = 0, scale = 0.5, size = len(X)))

    plt.scatter(X.detach().numpy(), Y.detach().numpy())
    plt.plot(np.linspace(0,100,100), y_hat(torch.linspace(0,100,100), theta_true).detach().numpy())
    plt.show()
    global call_counter
    call_counter = 0
    def residual(theta, nocount = False):
        # print("theta: ", theta)
        # plt.scatter(X.detach().numpy(), Y.detach().numpy())
        # plt.plot(np.linspace(0,100,100), y_hat(torch.linspace(0,100,100), theta).detach().numpy())
        # plt.show()
        if not nocount:
            global call_counter
            call_counter += 1
        return torch.sum(((Y - y_hat(X, theta))**2)/(0.5**2)) / (100 - 4 + 1)
    
    x0 = torch.tensor([5.,2.,0.2,10.])

    res = LM(residual, x0, max_iter = 100)
    print("call counter: ", call_counter)
    plt.plot(range(len(res.loss_history)), np.log10(np.array(res.loss_history)))
    plt.plot(range(len(res.L_history)), np.log10(np.array(res.L_history)))
    plt.show()
    plt.plot(range(len(res.lambda_history)), np.array(res.lambda_history)[:,0])
    plt.plot(range(len(res.lambda_history)), np.array(res.lambda_history)[:,1])
    plt.plot(range(len(res.lambda_history)), np.array(res.lambda_history)[:,2])
    plt.plot(range(len(res.lambda_history)), np.array(res.lambda_history)[:,3])
    # for t in theta_true.detach().numpy():
    #     plt.axhline(t)
    plt.show()
