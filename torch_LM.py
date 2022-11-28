import torch
#from torch.autograd.functional import hessian
from functorch import hessian
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
        h = torch.zeros(len(self.lambda_new))
        self.lambda_new.requires_grad = True
        while self.iteration < self.max_iter and loss >= 1e-10 and torch.all(torch.isfinite(self.lambda_new)):
            print("---------iter---------")
            if self.iteration > 0:
                h = -torch.linalg.solve(hess + self.L*torch.abs(torch.diag(hess))*torch.eye(len(grad)), grad)
                        
            loss = self.function(self.lambda_new + h)
            self.loss_history.append(loss.detach())
            print(loss)
            self.L_history.append(self.L)
            self.lambda_history.append(np.copy(self.lambda_new.detach().cpu().numpy()))
            if self.iteration > 0:
                if 0 < (self.loss_history[-2] - self.loss_history[-1]) < 1e-6:
                    break
                rho = (self.loss_history[-2] - self.loss_history[-1]) / abs(torch.dot(h, self.L * (torch.abs(torch.diag(hess)) * h) + grad).detach())
                #self.L *= np.exp(np.random.normal(loc = 0, scale = 0.1))
                if rho > self.epsilon4:
                    print("accept")
                    self.lambda_new.requires_grad = False
                    self.lambda_old = self.lambda_new
                    self.lambda_new += h
                    self.lambda_new.requires_grad = True
                    self.lambda_new.grad = None
                    #loss.backward()
                    self.L = max(1e-9, self.L / self.Ldn)
                else:
                    print("reject")
                    self.L = min(1e9, self.L * self.Lup)
                    continue
            else:
                pass
                #loss.backward()

            grad = torch.autograd.grad(loss, self.lambda_new)[0]
            hess = hessian(self.function, self.lambda_new + h)
            # grad = self.lambda_new.grad
            
            self.iteration += 1
        print(loss)
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
